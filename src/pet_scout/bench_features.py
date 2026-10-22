"""Pinned, validated, frozen embeddings with honest local compute measurements."""
import json
import os
import platform
import resource
import time
from datetime import datetime, timezone
import numpy as np
import pandas as pd
from .data import digest, write_json
from .features import sequence_fingerprint

POOLING_VERSION = 'amino-acid-mask-mean-v1'


def check_embedding_cache(path, table, spec, revision):
    meta = json.loads(path.with_suffix('.json').read_text())
    expected = {'model_id': spec['model_id'], 'revision': revision,
                'sequence_sha256': sequence_fingerprint(table), 'pooling_version': POOLING_VERSION}
    if any(meta.get(k) != v for k, v in expected.items()):
        raise ValueError('Embedding cache provenance mismatch.')
    if digest(path) != meta['embedding_sha256']:
        raise ValueError('Embedding file checksum mismatch.')
    with np.load(path, allow_pickle=False) as f:
        vectors, ids = f['embeddings'], f['ids']
    if not np.array_equal(ids, table.sample_id.to_numpy(dtype=str)):
        raise ValueError('Embedding ID order mismatch.')
    if vectors.ndim != 2 or len(vectors) != len(table) or not np.isfinite(vectors).all():
        raise ValueError('Invalid embedding shape or values.')
    return vectors, meta


def amino_acid_mean(hidden, attention_mask, special_mask, lengths):
    """Handle ESM BOS/EOS and T5 EOS without assuming identical token layouts."""
    import torch
    mask = attention_mask.bool() & ~special_mask.bool()
    expected = torch.tensor(lengths, device=mask.device)
    if not torch.equal(mask.sum(dim=1), expected):
        raise ValueError('Tokenizer must yield exactly one non-special token per residue.')
    return (hidden * mask.unsqueeze(-1)).sum(dim=1) / expected.unsqueeze(-1)


def embed_model(root, config, key, device='cpu'):
    # Load the published pinned checkpoint; never start a remote conversion side job.
    # Transformers can otherwise leave a non-daemon auto-conversion thread alive.
    os.environ['DISABLE_SAFETENSORS_CONVERSION'] = '1'
    import torch
    from huggingface_hub import model_info
    from transformers import AutoModel, AutoTokenizer, T5EncoderModel, T5Tokenizer
    spec = next(m for m in config['models'] if m['key'] == key)
    base = root / 'benchmark'
    table = pd.read_csv(base / 'proteins.csv')
    path = base / 'embeddings' / f'{key}.npz'
    path.parent.mkdir(parents=True, exist_ok=True)
    revpath = base / 'revisions' / f'{key}.json'
    if revpath.exists():
        pin = json.loads(revpath.read_text())
        if pin['model_id'] != spec['model_id']:
            raise ValueError('Pinned model differs from benchmark configuration.')
        revision = pin['revision']
    else:
        revision = model_info(spec['model_id']).sha
        write_json(revpath, {'model_id': spec['model_id'], 'revision': revision})
    if path.exists() and path.with_suffix('.json').exists():
        check_embedding_cache(path, table, spec, revision)
        print(f'{key}: validated cached embeddings', flush=True)
        return
    torch.set_num_threads(4)
    started = time.monotonic()
    cpu_start = time.process_time()
    cache = root / '.cache/huggingface'
    options = dict(revision=revision, cache_dir=cache)
    print(f'{key}: loading pinned {spec["model_id"]} on {device}', flush=True)
    if spec['family'] == 't5':
        tokenizer = T5Tokenizer.from_pretrained(spec['model_id'], do_lower_case=False, **options)
        model = T5EncoderModel.from_pretrained(spec['model_id'], dtype=torch.float32,
                                             use_safetensors=False, **options)
    else:
        tokenizer = AutoTokenizer.from_pretrained(spec['model_id'], **options)
        model = AutoModel.from_pretrained(spec['model_id'], add_pooling_layer=False,
                                         dtype=torch.float32, **options)
        if table.length.max() + 2 > model.config.max_position_embeddings:
            raise ValueError('Sequence exceeds model positional limit; truncation is forbidden.')
    model.eval().to(device)
    if device == 'cuda':
        torch.cuda.reset_peak_memory_stats()
    if device == 'mps': torch.mps.synchronize()
    if device == 'cuda': torch.cuda.synchronize()
    loading_seconds = time.monotonic() - started
    infer_start = time.monotonic()
    arrays = []
    batch_size = spec['batch_size']
    with torch.inference_mode():
        for start in range(0, len(table), batch_size):
            seqs = table.sequence.iloc[start:start+batch_size].tolist()
            text = [' '.join(seq) for seq in seqs] if spec['family'] == 't5' else seqs
            inputs = tokenizer(text, padding=True, truncation=False, return_tensors='pt', return_special_tokens_mask=True)
            specials = inputs.pop('special_tokens_mask').to(device)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            hidden = model(**inputs).last_hidden_state
            pooled = amino_acid_mean(hidden, inputs['attention_mask'], specials, [len(s) for s in seqs])
            arrays.append(pooled.float().cpu().numpy())
            if start == 0 or (start // batch_size) % 20 == 0:
                print(f'{key}: {min(start+batch_size,len(table))}/{len(table)} ({time.monotonic()-infer_start:.1f}s)', flush=True)
    if device == 'mps': torch.mps.synchronize()
    if device == 'cuda': torch.cuda.synchronize()
    inference_seconds = time.monotonic() - infer_start
    vectors = np.concatenate(arrays).astype(np.float32)
    if not np.isfinite(vectors).all(): raise ValueError('Nonfinite embeddings.')
    np.savez_compressed(path, embeddings=vectors, ids=table.sample_id.to_numpy(dtype=str))
    rss = resource.getrusage(resource.RUSAGE_SELF).ru_maxrss
    peak_rss_gib = rss / (1024**3 if platform.system() == 'Darwin' else 1024**2)
    write_json(path.with_suffix('.json'), {'model_id': spec['model_id'], 'label': spec['label'],
        'revision': revision, 'sequence_sha256': sequence_fingerprint(table),
        'pooling_version': POOLING_VERSION, 'shape': list(vectors.shape),
        'embedding_sha256': digest(path), 'device': device, 'dtype': 'float32',
        'batch_size': batch_size, 'threads': 4, 'loading_seconds': round(loading_seconds, 3),
        'inference_seconds': round(inference_seconds, 3),
        'cpu_seconds': round(time.process_time()-cpu_start, 3),
        'process_peak_rss_gib': round(peak_rss_gib, 3),
        'cuda_peak_allocated_gib': torch.cuda.max_memory_allocated()/1024**3 if device == 'cuda' else None,
        'parameter_count': sum(p.numel() for p in model.parameters()),
        'checkpoint_disk_bytes': sum(p.stat().st_size for p in (cache / ('models--'+spec['model_id'].replace('/','--')) / 'blobs').glob('*') if p.is_file()),
        'created_utc': datetime.now(timezone.utc).isoformat(),
        'hardware': platform.machine(), 'torch_version': torch.__version__,
        'timing_note': 'One measured run, not a hardware benchmark. Loading includes downloads/cache reads. Peak RSS covers the whole process. Models run in separate CLI processes.',
        'pretraining_note': 'Unknown overlap with pretraining sequences; no assay labels used in frozen embeddings.'})
    check_embedding_cache(path, table, spec, revision)
    print(f'{key}: complete; inference {inference_seconds:.1f}s; process peak RSS {peak_rss_gib:.2f} GiB', flush=True)
