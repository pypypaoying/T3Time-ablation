# Text Branch Ablation Runbook

## Quick Smoke Test

```bash
git clone https://github.com/pypypaoying/T3Time-ablation.git
cd T3Time-ablation
conda env create -f env_windows.yaml
conda activate T3Time

bash scripts/run_text_ablation.sh --dry-run
DATASETS=ETTm1 HORIZONS=96 SEEDS=2024 VARIANTS=no_text,zero_text EPOCHS_OVERRIDE=1 MAX_TRAIN_BATCHES=2 MAX_EVAL_BATCHES=2 NUM_WORKERS=0 bash scripts/run_text_ablation.sh
```

`no_text` and `zero_text` do not require precomputed GPT2 embeddings. Use them first to verify that training, logging, and metrics work.

## Generate Embeddings

```bash
DATASETS=ETTm1,ETTh1 HORIZONS=96 bash scripts/run_text_ablation.sh --prepare-embeddings --dry-run
DATASETS=ETTm1,ETTh1 HORIZONS=96 bash scripts/run_text_ablation.sh --prepare-embeddings
```

By default embeddings are stored under:

```text
Embeddings/<dataset>/<train|val|test>/<index>.h5
```

For external storage:

```bash
DATA_ROOT=/path/to/dataset EMBED_ROOT=/path/to/Embeddings bash scripts/run_text_ablation.sh --prepare-embeddings
```

## Must-Run Ablation Matrix

```bash
DATASETS=ETTm1,ETTh1 \
HORIZONS=96,192,336,720 \
SEEDS=2024,2025,2026 \
VARIANTS=full,no_text,zero_text,shift_text \
bash scripts/run_text_ablation.sh --prepare-embeddings --skip-existing
```

Outputs:

```text
Results/text_ablation/<run_id>/best_model.pth
Results/text_ablation/<run_id>/metrics.json
Results/text_ablation/logs/<run_id>.log
Results/text_ablation/runs.csv
Results/text_ablation/summary.csv
Results/text_ablation/summary.md
```

## Optional Diagnostics

Run these after the must-run matrix identifies a promising or suspicious setting:

```bash
DATASETS=ETTm1 HORIZONS=96 SEEDS=2024,2025,2026 VARIANTS=mean_text,shuffle_text,noise_text bash scripts/run_text_ablation.sh --skip-existing
```

## Decision Rule

The text branch is supported only if `full` beats `no_text`, `zero_text`, and `shift_text` on seed-mean MSE/MAE. If `shift_text` or `noise_text` is close to `full`, the result should be reported as text-branch signal being weak or possibly non-semantic.
