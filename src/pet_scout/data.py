"""Acquire and validate the original ACS source workbook, without imputing assays."""
from __future__ import annotations
import hashlib
import json
from pathlib import Path
import numpy as np
import pandas as pd
import requests

SOURCE_URL = 'https://ndownloader.figshare.com/files/57638729'
SOURCE_MD5 = '67d8c85e6c29ce21de0d5dbf77c427e1'
SOURCE_DOI = '10.1021/acscatal.5c03460'
AA = 'ACDEFGHIKLMNPQRSTVWY'
CONTROLS = {'blank', 'LCC-ICCG', 'sumo_LCC-ICCG', 'RFP'}
TARGET = 'umol_product_per_mg_enzyme'


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_json(path: Path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, allow_nan=False) + '\n')


def download(root: Path):
    path = root / 'data/raw/source.xlsx'
    path.parent.mkdir(parents=True, exist_ok=True)
    if not path.exists():
        response = requests.get(SOURCE_URL, timeout=90)
        response.raise_for_status()
        path.write_bytes(response.content)
    if hashlib.md5(path.read_bytes()).hexdigest() != SOURCE_MD5:
        raise ValueError('Source checksum mismatch; do not silently substitute a changed dataset.')
    write_json(root / 'data/raw/provenance.json', {
        'doi': SOURCE_DOI, 'download_url': SOURCE_URL,
        'figshare_record': 'https://acs.figshare.com/articles/dataset/30044876',
        'filename': 'cs5c03460_si_002.xlsx', 'sha256': digest(path),
        'md5': SOURCE_MD5, 'license': 'CC BY-NC 4.0',
        'license_url': 'https://creativecommons.org/licenses/by-nc/4.0/',
        'citation': 'Norton-Baker et al. (2025), Machine Learning-Guided Identification of PET Hydrolases from Natural Diversity, ACS Catalysis 15, 16070–16083.',
    })
    return path


def normalize_sequence(seq: str) -> str:
    if not isinstance(seq, str):
        raise ValueError('Missing sequence')
    seq = ''.join(seq.split()).upper()
    if not seq or set(seq) - set(AA):
        raise ValueError(f'Invalid amino-acid sequence: {set(seq)-set(AA)}')
    return seq


def prepare(root: Path, config: dict):
    source = download(root)
    proteins = pd.read_excel(source, sheet_name='Rounds')
    assays = pd.read_excel(source, sheet_name='Activity')
    if proteins.sample_id.duplicated().any():
        raise ValueError('Source has duplicate sample IDs; explicit resolution required.')
    proteins['sequence'] = proteins.seq_aa.map(lambda s: ''.join(str(s).split()).upper())
    invalid_ids = proteins.loc[proteins.sequence.map(lambda s: not s or bool(set(s)-set(AA))), 'sample_id'].tolist()
    unknown = set(assays.sample_id) - set(proteins.sample_id) - CONTROLS
    if unknown:
        raise ValueError(f'Unknown assay IDs: {unknown}')
    cond = config['condition']
    candidates = assays[~assays.sample_id.isin(CONTROLS | set(invalid_ids))].copy()
    coverage = candidates.groupby(['temperature_c','pH','substrate']).agg(
        candidates=('sample_id','nunique'), assay_rows=('sample_id','size')).reset_index()
    out = root / 'data/processed'; out.mkdir(parents=True, exist_ok=True)
    coverage.to_csv(out / 'condition_coverage.csv', index=False)
    selected = candidates.loc[
        candidates.temperature_c.eq(cond['temperature_c']) &
        candidates.pH.eq(cond['pH']) & candidates.substrate.eq(cond['substrate'])].copy()
    if selected.empty:
        raise ValueError('No measured candidates for requested condition.')
    if not np.isfinite(selected[TARGET]).all() or (selected[TARGET] < 0).any():
        raise ValueError('Missing, nonfinite, or negative assay values need explicit handling.')
    for key, value in [('reaction_vol_mL', .5), ('PET_loading_mg', 3.5), ('enzyme_loading_ug', 5)]:
        if not np.allclose(selected[key], value):
            raise ValueError(f'Mixed assay setup: {key}')
    # Average repeated assay measurements before any split. Never split replicate rows.
    mean = selected.groupby('sample_id',sort=True).agg(
        activity=(TARGET,'mean'), activity_sd=(TARGET,'std'), n_replicates=(TARGET,'size')).reset_index()
    merged = mean.merge(proteins[['sample_id','sequence','round','accession','tm_1','tm_2']],
                        on='sample_id', validate='one_to_one')
    # Identical construct sequences are one unit. Average per-ID means, retain aliases.
    records=[]
    for seq, frame in merged.groupby('sequence',sort=True):
        first=frame.sort_values('sample_id').iloc[0]
        records.append({'sample_id':first.sample_id, 'aliases':';'.join(sorted(frame.sample_id)),
            'sequence':seq,'sequence_sha256':hashlib.sha256(seq.encode()).hexdigest(),
            'length':len(seq),'activity':float(frame.activity.mean()),
            'activity_sd':float(frame.activity_sd.mean()) if frame.activity_sd.notna().any() else None,
            'n_replicates':int(frame.n_replicates.sum()),'n_source_ids':len(frame),
            'round':';'.join(sorted(frame['round'].dropna().unique())),
            'accession':';'.join(sorted(frame.accession.dropna().unique())),
            'tm_1':float(frame.tm_1.mean()) if frame.tm_1.notna().any() else None,
            'tm_2':float(frame.tm_2.mean()) if frame.tm_2.notna().any() else None})
    table=pd.DataFrame(records).sort_values('sample_id').reset_index(drop=True)
    table.to_csv(out/'candidates.csv',index=False)
    selected.to_csv(out/'assay_rows.csv',index=False)
    audit={'source_sha256':digest(source),'condition':cond,
        'invalid_sequence_ids_excluded':invalid_ids,'source_protein_rows':len(proteins),'source_assay_rows':len(assays),
        'control_assay_rows_excluded':int(assays.sample_id.isin(CONTROLS).sum()),
        'selected_assay_rows':len(selected),'selected_source_ids':len(mean),
        'unique_sequences':len(table),'duplicate_sequences_collapsed':len(mean)-len(table),
        'untested_candidate_ids_at_this_condition':len(set(proteins.sample_id)-CONTROLS-set(mean.sample_id)),
        'single_measurement_sequences':int(table.n_replicates.eq(1).sum()),
        'sequences_with_tm_1':int(table.tm_1.notna().sum()),
        'sequence_length_range':[int(table.length.min()),int(table.length.max())],
        'target_units':'µmol aromatic products / mg enzyme after 48 h',
        'dataset_sha256':digest(out/'candidates.csv'),
        'rules':['Exclude blank, LCC-ICCG, SUMO-LCC-ICCG, RFP controls.',
                 'Use provided seq_aa construct sequences; no unverified tag or domain trimming. Exclude ambiguous amino-acid sequences.',
                 'Average replicates per ID, then average identical-sequence ID means.',
                 'Never label unmeasured assays or failed expression as zero activity.',
                 'Thermal measurements are descriptive only, never model inputs.']}
    write_json(out/'audit.json',audit)
    print(f'Prepared {len(table)} unique sequences / {len(selected)} assay rows',flush=True)
    return table
