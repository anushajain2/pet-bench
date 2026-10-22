"""Local, pinned ESM inference. Uploaded sequences are never persisted or sent out."""
import os
from functools import lru_cache
import numpy as np
import pandas as pd
from .bench_features import check_embedding_cache, amino_acid_mean

MODEL_ID = 'facebook/esm2_t12_35M_UR50D'
REVISION = '6fbf070e65b0b7291e7bbcd451118c216cff79d8'


@lru_cache(maxsize=2)
def load_local_model(cache):
    os.environ['DISABLE_SAFETENSORS_CONVERSION'] = '1'
    import torch
    from transformers import AutoTokenizer, AutoModel
    options = dict(revision=REVISION, cache_dir=cache, local_files_only=True)
    try:
        tokenizer = AutoTokenizer.from_pretrained(MODEL_ID, **options)
        model = AutoModel.from_pretrained(MODEL_ID, add_pooling_layer=False,
                                          dtype=torch.float32, **options).eval()
    except OSError as error:
        raise ValueError('Pinned ESM-2 35M weights are missing locally. Run the documented model-download command first.') from error
    torch.set_num_threads(4)
    return tokenizer, model


def embed_sequences(root, sequences):
    """Reuse verified public vectors; infer new sequences in memory, CPU only."""
    sequences = list(sequences)
    if not sequences or any(not 1 <= len(s) <= 1022 for s in sequences):
        raise ValueError('Require sequences of 1–1022 amino acids; no truncation.')
    lookup = {}
    path = root/'benchmark/embeddings/esm2_35m.npz'
    if path.exists():
        catalog = pd.read_csv(root/'benchmark/proteins.csv')
        vectors, _ = check_embedding_cache(path, catalog, {'model_id': MODEL_ID}, REVISION)
        lookup = dict(zip(catalog.sequence, vectors))
    missing = list(dict.fromkeys(s for s in sequences if s not in lookup))
    if missing:
        import torch
        tokenizer, model = load_local_model(str(root/'.cache/huggingface'))
        with torch.inference_mode():
            for start in range(0, len(missing), 4):
                seqs = missing[start:start+4]
                inputs = tokenizer(seqs, padding=True, truncation=False,
                                   return_tensors='pt', return_special_tokens_mask=True)
                special = inputs.pop('special_tokens_mask')
                hidden = model(**inputs).last_hidden_state
                pooled = amino_acid_mean(hidden, inputs['attention_mask'], special,
                                         [len(s) for s in seqs]).numpy()
                lookup.update(zip(seqs, pooled))
    result = np.asarray([lookup[s] for s in sequences], dtype=np.float32)
    if result.shape != (len(sequences), 480) or not np.isfinite(result).all():
        raise ValueError('Invalid ESM embedding output.')
    return result
