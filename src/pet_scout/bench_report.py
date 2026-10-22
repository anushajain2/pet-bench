"""Generate task-specific findings from completed results, without picking favorable tasks."""
import json
import pandas as pd
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt


def markdown_table(frame):
    def value(v):
        if pd.isna(v): return '—'
        return f'{v:.2f}' if isinstance(v, float) else str(v)
    return '\n'.join(['| '+' | '.join(frame.columns)+' |', '| '+' | '.join(['---']*len(frame.columns))+' |'] +
                     ['| '+' | '.join(value(v) for v in row)+' |' for row in frame.itertuples(index=False, name=None)])


def build_benchmark_report(root, config):
    base = root / 'benchmark'
    out = base / 'results'
    manifest = json.loads((out / 'manifest.json').read_text())
    summary = pd.read_csv(out / 'summary.csv', dtype={'training_size': str})
    compute = pd.read_csv(out / 'compute.csv')
    tasks = pd.read_csv(base / 'task_summary.csv')
    primary = summary[(summary.regime == 'grouped') & (summary.training_size == 'all') & (summary.budget == 5)]
    sections = []
    for task in config['tasks']:
        table = primary[primary.task == task['id']].sort_values('selected_mean', ascending=False)
        first = table.iloc[0]
        if first.model == 'Nearest sequence':
            verdict = 'The nearest-sequence baseline has the highest mean selected activity.'
        elif first.min_seed_delta > 0:
            verdict = f'{first.model} exceeds nearest-sequence selection in every evaluated partition seed. This is consistent within these partitions, not independent experimental confirmation.'
        else:
            verdict = f'{first.model} has the highest mean, but its advantage over nearest sequence is not positive in every partition seed.'
        columns = table[['model','selected_mean','random_mean','delta_nearest','min_seed_delta','max_seed_delta','spearman','spearman_valid_folds']].copy()
        columns.columns = ['Method','Selected activity','Random expectation','Difference vs nearest','Min seed difference','Max seed difference','Mean fold Spearman','Valid rank folds / 25']
        sections.append(f'## {task["label"]}\n\n{verdict}\n\n{markdown_table(columns)}')
    fig, axes = plt.subplots(1, len(config['tasks']), figsize=(16, 5.5), sharey=False)
    for ax, task in zip(axes, config['tasks']):
        rows = primary[primary.task == task['id']].sort_values('lift')
        ax.barh(rows.model, rows.lift, color=['#126b54' if name not in ['Nearest sequence','Composition ridge'] else '#93aaa3' for name in rows.model])
        ax.errorbar(rows.lift, range(len(rows)), xerr=[rows.lift-rows.min_seed_lift, rows.max_seed_lift-rows.lift], fmt='none', color='#263e37', capsize=3)
        ax.axvline(0, color='#445f55', linewidth=.8)
        ax.set_title(task['label'], fontweight='bold')
        ax.set_xlabel('Top-5 activity above random (µmol/mg)')
        ax.spines[['top','right']].set_visible(False)
    fig.suptitle('PETBench: sequence-group holdout · all training examples', fontweight='bold')
    fig.text(.5,.01,'Whiskers: seed-level range, not confidence intervals. Different conditions have different candidate pools.',ha='center',fontsize=10)
    fig.tight_layout(rect=(0,.05,1,.95))
    fig.savefig(out / 'benchmark_overview.png', dpi=160, bbox_inches='tight')
    plt.close(fig)
    computeview = compute[['model','inference_seconds','loading_seconds','process_peak_rss_gib','embedding_dimensions']].copy()
    computeview.columns = ['Model','Inference seconds','Load/download seconds','Peak process GiB','Dimensions']
    text = f'''# PETBench v1 — findings and protocol

## What was tested

Which frozen protein representation helps select five PET-degrading enzymes for a
specified temperature and pH? Six methods were evaluated on the same labeled data,
training subsets and held-out pools: nearest sequence, composition ridge, ESM-2 35M,
ESM-2 150M, ESM-2 650M and the ProtT5-XL encoder, each embedding method using ridge.
This compares complete frozen-embedding-plus-ridge pipelines; it does not rank every
possible use of these foundation models or test designed mutations.

Completed {manifest['n_outer_model_fits']:,} outer model fits / predictions and
{manifest['n_predictions']:,} candidate-level held-out predictions. The unit of an
outer fit is one task, split regime, partition seed, fold, training budget and method;
inner ridge validation adds additional fits. The nearest-sequence method needs no
parameter fitting. Paid compute used: $0.

## Data coverage

{markdown_table(tasks[['label','n_proteins','n_assay_rows','n_groups','largest_group','single_measurement_proteins']])}

All tasks use crystalline PET powder, 48 hours, 3.5 mg PET, 5 µg enzyme, 0.5 mL reaction.
Outcome: µmol aromatic products per mg enzyme. Measurements are source assay endpoints,
not industrial recycling yields. Replicates are averaged before splitting. Controls
and ambiguous sequences are excluded. Unmeasured assays are not treated as zero.
The catalog contains {manifest['data_manifest']['n_unique_proteins']} unique sequences;
measurements across conditions are not additional independent proteins.

## Primary comparison

The tables below use sequence-group holdouts, all available training examples, and a
five-candidate selection budget per test pool. Rows are sorted by observed mean for
readability. A first-place mean alone does not establish a reliable model advantage.
Selected activity and differences use µmol/mg. Seed ranges describe split sensitivity.

''' + '\n\n'.join(sections) + f'''

## Learning curves and controls

Training budgets were fixed to 16, 32, 64 and all available examples. Subsets are nested,
drawn without labels, and shared across all six methods. Each numeric budget is used
only if every fold in that task can support it. All three tasks support all budgets.
The `all` budget varies with the outer fold size; it is not a fixed sample count.

Random, size-matched random and sequence-group holdouts are reported separately.
Size-matched random has the same train/test sizes as grouped holdout, but different
candidate composition. It does not isolate a causal effect of sequence novelty.
See `summary.csv` for every task/regime/budget/model and `paired_comparison.csv` for
same-seed differences against nearest sequence. Budgets one and three are secondary
selection analyses. No single pooled winner across conditions is reported.

## Evaluation rules

- Five outer folds, repeated for seeds {config['seeds']}.
- Global sequence groups are connected components at ≥60% identity across the union
  of task proteins. Identity uses global BLOSUM62 alignment, gap-open −10,
  gap-extension −0.5, with all gap columns in the denominator.
- Each task uses only labels measured at that condition. No other-condition labels
  are supplied to models. Related proteins remain together in grouped outer folds.
- Pretrained representations are frozen; amino-acid tokens are mean-pooled, excluding
  special tokens and padding. IDs, sequences, exact model revisions and files are checked.
- Ridge predicts log1p(activity); StandardScaler and alpha selection operate only on
  outer training data. Up to three inner folds use the outer grouping policy.
  The shared alpha grid is {config['ridge_alphas']}. If fewer than two training
  groups remain, alpha 100 is used and recorded instead of leaking across groups.
- Selection uses predicted rank, with candidate ID as the tie-breaker. The outcome
  is mean measured activity among selected candidates. Random expectation is exactly
  the mean of the same test pool; the oracle uses its best measured candidates.
- Aggregate folds equally, then seeds. Small and large pools get equal weight because
  the experimental selection budget is the same per pool. Spearman is undefined for
  constant predictions/outcomes; missing correlations are omitted, never set to zero.
- All checkpoint choices, conditions and budgets were fixed before these new runs.
  The 60°C/pH 7.5 condition was studied earlier; this is an exploratory extension of
  public data, not a prospective blind benchmark.

## Measured compute

{markdown_table(computeview)}

All four checkpoints ran locally, sequentially, on CPU in float32 with four Torch
threads. Each model ran in a separate process. Inference timing excludes loading;
load timing includes downloads/cache access. Peak RSS includes the whole process.
Batch sizes differ, so these are practical costs of this run, not controlled hardware
throughput comparisons. Embeddings cover the same complete protein catalog and are
reused across tasks, seeds and training budgets. Evaluation took
{manifest['evaluation_seconds']:.1f} seconds. Model metadata records precision, revisions,
parameter counts, peak memory, package versions and download sizes.

## Limits

''' + '\n'.join('- '+line for line in manifest['limitations']) + '''

The source paper already studied PET-hydrolase discovery and machine learning.
PETBench provides an auditable comparison on this source; it is not the first PET
benchmark, a discovery of a new enzyme, or evidence that a mutation improves function.
A follow-up prospective or genuinely independent experimental dataset is needed to
test how far these rankings generalize. Choosing methods after viewing this leaderboard
requires fresh evaluation before claiming an unbiased performance estimate.

## Reproduce

From the project directory: `pet-bench all`, then `python -m pytest -q`.
See `BENCHMARK.md` for installation and individual stages. The original PET Scout
study remains under `reports/` and its original `config.json`.

## Attribution

- Norton-Baker et al. (2025), *Machine Learning-Guided Identification of PET Hydrolases
  from Natural Diversity*: https://doi.org/10.1021/acscatal.5c03460
- Experimental workbook: https://acs.figshare.com/articles/dataset/30044876
  (CC BY-NC 4.0 per source record; see `DATA_LICENSE.md`).
- ESM-2: https://github.com/facebookresearch/esm
- ProtT5 encoder: https://huggingface.co/Rostlab/prot_t5_xl_half_uniref50-enc

Code is MIT. Do not treat the code license as replacing dataset or model licenses.
'''
    (out / 'FINDINGS.md').write_text(text)
    print('Wrote benchmark/results/FINDINGS.md and benchmark_overview.png', flush=True)
