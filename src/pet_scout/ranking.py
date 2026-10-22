"""Rank user-provided candidates using measurements from one comparable assay."""
import argparse
import hashlib
import io
import json
import re
from pathlib import Path
import numpy as np
import pandas as pd
from threadpoolctl import threadpool_limits
from .campaign import bootstrap_predict
from .data import AA
from .external import cross_identity, construct_overlap
from .inference import embed_sequences, MODEL_ID, REVISION

CONFIG = {'bootstrap_members': 20, 'ridge_alpha': 100.0, 'seed': 2026}
MAX_BYTES = 2_000_000


def text_input(content):
    if isinstance(content, bytes):
        if len(content) > MAX_BYTES:
            raise ValueError('Upload must be smaller than 2 MB.')
        try:
            return content.decode('utf-8-sig')
        except UnicodeDecodeError as error:
            raise ValueError('Use UTF-8 text files.') from error
    if len(content.encode()) > MAX_BYTES:
        raise ValueError('Upload must be smaller than 2 MB.')
    return content


def read_csv(content):
    try:
        frame = pd.read_csv(io.StringIO(text_input(content)), dtype=str, keep_default_na=False)
    except (pd.errors.ParserError, pd.errors.EmptyDataError) as error:
        raise ValueError('Invalid or empty CSV.') from error
    if any(str(c).endswith('.1') for c in frame.columns):
        raise ValueError('Duplicate or ambiguous column names.')
    return frame


def parse_csv(content):
    frame = read_csv(content)
    required = {'sample_id', 'sequence', 'activity'}
    if not required.issubset(frame.columns):
        raise ValueError('CSV needs sample_id, sequence, activity columns. Leave activity blank for untested candidates.')
    return validate_table(frame)


def parse_fasta(fasta, measurements):
    rows = []
    for line in text_input(fasta).splitlines():
        line = line.strip()
        if not line:
            continue
        if line.startswith('>'):
            if not line[1:].strip():
                raise ValueError('FASTA record needs an ID.')
            rows.append({'sample_id': line[1:].split()[0], 'sequence': ''})
        elif rows:
            rows[-1]['sequence'] += line
        else:
            raise ValueError('FASTA must start with a >sample_id header.')
    if not rows:
        raise ValueError('FASTA is empty.')
    labels = read_csv(measurements)
    if not {'sample_id', 'activity'}.issubset(labels.columns):
        raise ValueError('Measurements CSV needs sample_id and activity columns.')
    labels.sample_id = labels.sample_id.str.strip()
    if labels.sample_id.duplicated().any():
        raise ValueError('Measurement IDs must be unique; average comparable replicates first.')
    frame = pd.DataFrame(rows)
    if not set(labels.sample_id).issubset(set(frame.sample_id)):
        raise ValueError('A measurement ID is absent from the FASTA.')
    return validate_table(frame.merge(labels, on='sample_id', how='left', validate='many_to_one').fillna(''))


def validate_table(frame):
    frame = frame.copy().reset_index(drop=True)
    if not 9 <= len(frame) <= 500:
        raise ValueError('Use 9–500 sequences, including at least 8 measured and 1 untested.')
    frame.sample_id = frame.sample_id.astype(str).str.strip()
    if frame.sample_id.duplicated().any() or not frame.sample_id.map(lambda s: bool(re.fullmatch(r'[A-Za-z0-9][A-Za-z0-9_.:|/\-]{0,99}', s))).all():
        raise ValueError('IDs must be unique, 1–100 characters, start with a letter/digit, and use only letters, digits, _ . : | / - .')
    frame.sequence = frame.sequence.astype(str).map(lambda s: ''.join(s.split()).upper())
    for s in frame.sequence:
        if not 1 <= len(s) <= 1022 or set(s)-set(AA):
            raise ValueError('Use canonical 20-letter protein sequences, 1–1022 residues. No ambiguous residues, gaps, or stops.')
    if frame.sequence.duplicated().any():
        raise ValueError('Duplicate sequences are not independent candidates. Merge comparable replicates first.')
    for column in ('condition', 'units', 'assay'):
        if column in frame and frame[column].astype(str).str.strip().nunique() != 1:
            raise ValueError(f'All rows must have one consistent {column}; do not mix assays.')
    # Only genuinely empty cells mean untested. Strings NaN/NA/inf are invalid data.
    raw = frame.activity.astype(str).str.strip()
    measured = raw.ne('')
    numeric = pd.to_numeric(raw[measured], errors='coerce').to_numpy(dtype=float)
    if not np.isfinite(numeric).all() or (numeric < 0).any():
        raise ValueError('Measured activities must be finite, nonnegative numbers; only blank cells mean untested.')
    if measured.sum() < 8 or (~measured).sum() < 1:
        raise ValueError('Provide at least 8 measured proteins and 1 untested candidate.')
    if np.ptp(numeric) <= 0:
        raise ValueError('Measured activities need variation to learn a ranking.')
    frame['activity'] = np.nan
    frame.loc[measured, 'activity'] = numeric
    observed = frame[measured]
    # Reject trivial tag/truncation retests, but retain real single-residue variants.
    if any(construct_overlap(a, b) for a in frame.loc[~measured, 'sequence'] for b in observed.sequence):
        raise ValueError('An untested sequence is an exact near-full-length substring of a measured construct. Resolve construct/tag duplicates first.')
    return frame.sort_values('sample_id').reset_index(drop=True)


def rank_table(root, table, condition, units, batch_size=3, embedder=embed_sequences):
    if not condition.strip() or not units.strip() or max(len(condition), len(units)) > 300:
        raise ValueError('Describe the assay condition and units (1–300 characters each).')
    # Revalidate numeric callers without interpreting their missing values as strings.
    incoming = table.copy()
    incoming['activity'] = incoming.activity.map(lambda v: '' if pd.isna(v) else str(v))
    table = validate_table(incoming)
    observed = np.flatnonzero(table.activity.notna())
    pool = np.flatnonzero(table.activity.isna())
    if not isinstance(batch_size, int) or not 1 <= batch_size <= len(pool):
        raise ValueError('Batch size must fit the untested candidate pool.')
    x = embedder(root, table.sequence)
    with threadpool_limits(limits=1):
        mean, std = bootstrap_predict(x, observed, table.activity.to_numpy()[observed], pool, CONFIG['seed'], CONFIG)
    similarity = cross_identity(table.iloc[pool], table.iloc[observed])
    nearest = similarity.argmax(axis=1)
    result = pd.DataFrame({'sample_id': table.sample_id.to_numpy()[pool],
        'predicted_log1p_activity': mean, 'bootstrap_std_log1p': std,
        'nearest_measured_id': table.sample_id.to_numpy()[observed[nearest]],
        'nearest_identity': similarity.max(axis=1),
        'nearest_measured_activity': table.activity.to_numpy()[observed[nearest]]})
    result = result.sort_values(['predicted_log1p_activity', 'sample_id'], ascending=[False, True]).reset_index(drop=True)
    result.insert(0, 'ai_rank', np.arange(1, len(result)+1))
    baseline = result.sort_values(['nearest_measured_activity', 'sample_id'], ascending=[False, True]).sample_id
    result['nearest_sequence_rank'] = result.sample_id.map(dict(zip(baseline, range(1, len(baseline)+1))))
    result['rank_disagreement'] = result.nearest_sequence_rank-result.ai_rank
    result['selected_for_batch'] = result.ai_rank <= batch_size
    result['similarity_context'] = np.where(result.nearest_identity < .6, 'Below 60% identity: extrapolation', 'At least 60% identity: still unvalidated')
    payload = table[['sample_id', 'sequence', 'activity']].to_csv(index=False).encode()
    manifest = {'version': '1.0', 'condition': condition.strip(), 'units': units.strip(),
        'model_id': MODEL_ID, 'revision': REVISION, 'configuration': CONFIG,
        'input_sha256': hashlib.sha256(payload).hexdigest(),
        'n_measured': len(observed), 'n_untested': len(pool), 'batch_size': batch_size,
        'selected_ids': result.loc[result.selected_for_batch, 'sample_id'].tolist(),
        'training': 'Only supplied measurements; train-only scaling and fixed bootstrap ridge on log1p activity. No external-study labels or benchmark labels added.',
        'score_meaning': 'Relative ranking score in log1p supplied-activity units; not calibrated activity or success probability. Bootstrap spread is not a confidence interval.',
        'scope': 'Research prioritization for comparable PET assays. Benchmark evidence concerns distinct natural enzymes, not single-mutation effects. No prospective validation.',
        'privacy': 'Sequences processed locally in memory; not included in export. Export contains IDs and nearest measured activities; treat it as private.',
        'ranking_sha256': hashlib.sha256(result.to_csv(index=False).encode()).hexdigest()}
    return result, manifest


def main():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument('--root', type=Path, default=Path.cwd())
    inputs = p.add_mutually_exclusive_group(required=True)
    inputs.add_argument('--csv', type=Path)
    inputs.add_argument('--fasta', type=Path)
    p.add_argument('--measurements', type=Path)
    p.add_argument('--condition', required=True)
    p.add_argument('--units', required=True)
    p.add_argument('--batch-size', type=int, default=3)
    p.add_argument('--output', type=Path, required=True)
    args = p.parse_args()
    if args.fasta and not args.measurements:
        p.error('--fasta requires --measurements')
    try:
        table = parse_csv(args.csv.read_bytes()) if args.csv else parse_fasta(args.fasta.read_bytes(), args.measurements.read_bytes())
        result, manifest = rank_table(args.root.resolve(), table, args.condition, args.units, args.batch_size)
    except ValueError as error:
        p.error(str(error))
    if args.output.exists():
        p.error('Output directory already exists; choose a new directory to avoid overwriting a run.')
    args.output.mkdir(parents=True)
    (args.output/'ranking.csv').write_text(result.to_csv(index=False))
    (args.output/'manifest.json').write_text(json.dumps(manifest, indent=2, allow_nan=False)+'\n')
    print(result[['ai_rank', 'sample_id', 'nearest_identity', 'selected_for_batch']].to_string(index=False))
    print(f'Saved ranking and manifest to {args.output}. Sequences were not exported.')


if __name__ == '__main__':
    main()
