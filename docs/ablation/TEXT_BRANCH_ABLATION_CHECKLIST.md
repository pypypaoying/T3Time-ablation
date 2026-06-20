# Text Branch Ablation Checklist

## Setup

- [ ] Clone `pypypaoying/T3Time-ablation` on the server.
- [ ] Create/activate the T3Time conda environment.
- [ ] Confirm datasets exist under `dataset/` or set `DATA_ROOT=/path/to/dataset`.
- [ ] Generate prompt embeddings or confirm `Embeddings/<data>/<split>/*.h5` exists.
- [ ] Run `bash scripts/run_text_ablation.sh --dry-run`.

## Must-Run Campaign

- [ ] `ETTm1` horizons `96,192,336,720`: `full,no_text,zero_text,shift_text`.
- [ ] `ETTh1` horizons `96,192,336,720`: `full,no_text,zero_text,shift_text`.
- [ ] Seeds `2024,2025,2026`, unless doing a smoke run.
- [ ] Summarize with `python scripts/summarize_text_ablation.py --results_dir Results/text_ablation`.
- [ ] Check `summary.md` for variant deltas and missing runs.

## Decision Rules

- [ ] Text branch supported: `full` improves seed-mean test MSE/MAE over `no_text`, `zero_text`, and `shift_text`.
- [ ] Text branch weak or unsupported: `full` is tied with or worse than `no_text`.
- [ ] Text semantics suspect: `shift_text`, `shuffle_text`, or `noise_text` matches `full`.
- [ ] Report negative results explicitly; do not hide failed or contradictory variants.

## Optional Diagnostics

- [ ] Run `mean_text`, `shuffle_text`, and `noise_text` on the strongest and weakest full-vs-ablation settings.
- [ ] Add `ETTm2`, `ETTh2`, `Weather`, and `exchange_rate` if anchor evidence is promising.
- [ ] Inspect runtime/memory by variant to ensure the comparison is fair.
