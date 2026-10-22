"""Target-independent sequence features, with content-addressed provenance."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import time
import numpy as np
import pandas as pd
from Bio.Align import PairwiseAligner, substitution_matrices
from scipy.sparse.csgraph import connected_components
from scipy.sparse import csr_matrix
from .data import AA, write_json


def sequence_fingerprint(table):
    return hashlib.sha256('\n'.join(table.sequence).encode()).hexdigest()


def composition(table):
    return np.array([[seq.count(aa)/len(seq) for aa in AA]+[np.log(len(seq))]
                     for seq in table.sequence], dtype=np.float64)


def make_aligner():
    aligner=PairwiseAligner(mode='global')
    aligner.substitution_matrix=substitution_matrices.load('BLOSUM62')
    aligner.open_gap_score=-10
    aligner.extend_gap_score=-0.5
    return aligner


def alignment_identity(a, b, aligner=None):
    aligner=aligner or make_aligner()
    alignment=aligner.align(a,b)[0]
    # Includes terminal and internal gap columns in the denominator.
    return float(alignment.counts().identities / alignment.shape[1])


def pairwise_identity(table, path):
    meta=path.with_suffix('.json'); key=sequence_fingerprint(table)
    if path.exists() and meta.exists() and json.loads(meta.read_text())['sequence_sha256']==key:
        return np.load(path)
    n=len(table); matrix=np.eye(n); aligner=make_aligner(); seqs=table.sequence.tolist()
    for i in range(n):
        for j in range(i): matrix[i,j]=matrix[j,i]=alignment_identity(seqs[i],seqs[j],aligner)
        if i%25==0: print(f'Aligned {i+1}/{n} sequences',flush=True)
    path.parent.mkdir(parents=True,exist_ok=True); np.save(path,matrix)
    write_json(meta,{'sequence_sha256':key,'method':'global BLOSUM62; gap open -10; gap extension -0.5; identity = identical residues / all alignment columns'})
    return matrix


def sequence_groups(identity, threshold):
    if not 0 < threshold <= 1: raise ValueError('Identity threshold must be in (0,1].')
    _,groups=connected_components(csr_matrix(identity>=threshold),directed=False)
    return groups


def embed(root: Path, table: pd.DataFrame, model_id: str, batch_size=8, device='cpu'):
    # Imported only for this stage. The dashboard and evaluation need no torch process.
    import torch
    from transformers import AutoTokenizer, AutoModel
    from huggingface_hub import model_info
    path=root/'data/processed/embeddings.npz'; meta=path.with_suffix('.json')
    key=sequence_fingerprint(table)
    if path.exists() and meta.exists():
        prev=json.loads(meta.read_text())
        if prev['sequence_sha256']==key and prev['model_id']==model_id:
            print('Using validated cached embeddings',flush=True); return
    revision_file=root/'model_revision.json'
    if revision_file.exists() and json.loads(revision_file.read_text())['model_id']==model_id:
        revision=json.loads(revision_file.read_text())['revision']
    else:
        revision=model_info(model_id).sha
        write_json(revision_file,{'model_id':model_id,'revision':revision})
    cache=root/'.cache/huggingface'; cache.mkdir(parents=True,exist_ok=True)
    tokenizer=AutoTokenizer.from_pretrained(model_id,revision=revision,cache_dir=cache)
    model=AutoModel.from_pretrained(model_id,revision=revision,cache_dir=cache,add_pooling_layer=False)
    model.eval(); model.to(device); torch.set_num_threads(4)
    max_len=int(model.config.max_position_embeddings)-2
    if table.length.max()>max_len:
        raise ValueError(f'Sequence longer than model limit {max_len}; silent truncation is prohibited.')
    vectors=[]; start=time.monotonic()
    with torch.inference_mode():
        for i in range(0,len(table),batch_size):
            sequences=table.sequence.iloc[i:i+batch_size].tolist()
            inputs=tokenizer(sequences,return_tensors='pt',padding=True,truncation=False)
            inputs={k:v.to(device) for k,v in inputs.items()}
            hidden=model(**inputs).last_hidden_state
            # Each representation is mean of amino-acid tokens only, never BOS/EOS/padding.
            for row,seq in enumerate(sequences):
                vectors.append(hidden[row,1:len(seq)+1].mean(dim=0).cpu().numpy())
            print(f'Embedded {min(i+batch_size,len(table))}/{len(table)} ({time.monotonic()-start:.0f}s)',flush=True)
    matrix=np.stack(vectors).astype(np.float32)
    np.savez_compressed(path,embeddings=matrix,ids=table.sample_id.to_numpy(dtype=str))
    write_json(meta,{'sequence_sha256':key,'model_id':model_id,'revision':revision,
        'pooling':'Mean amino-acid tokens, excluding special tokens and padding',
        'device':device,'batch_size':batch_size,'shape':list(matrix.shape),
        'wall_seconds':round(time.monotonic()-start,2),'training':'frozen pretrained model; no assay labels used',
        'pretraining_overlap':'Homologs may have occurred in ESM pretraining; assay holdout is not pretraining decontamination.'})
