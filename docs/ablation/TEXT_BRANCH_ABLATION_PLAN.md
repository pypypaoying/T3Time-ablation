# Text Branch Ablation Experiment Plan

**Problem**: Test whether the T3Time prompt/text branch contributes real forecasting signal rather than acting as extra parameters or noisy regularization.
**Method thesis**: If the text branch is useful, the full model should beat architecture-matched and information-destroyed text ablations under the same data split, hyperparameters, seeds, and training budget.
**Date**: 2026-06-20

## Claim Map

| Claim | Why It Matters | Minimum Convincing Evidence | Linked Runs |
| --- | --- | --- | --- |
| C1: Prompt embeddings add forecast-useful information | The paper presents T3Time as tri-modal, so the text/prompt branch must improve prediction beyond time and frequency branches. | `full` has lower test MSE/MAE than `no_text`, `zero_text`, and `shift_text` on at least the anchor datasets/horizons with stable seed averages. | `full`, `no_text`, `zero_text`, `shift_text` |
| C2: Any gain is not just extra CMA/decoder capacity | A text branch that helps only by adding parameters is not a real modality contribution. | `no_text` keeps the rest of the architecture path intact but feeds the decoder from time-frequency features only; `zero_text` keeps the prompt encoder/CMA parameter path active with uninformative embeddings. | `no_text`, `zero_text` |
| Anti-claim: Gains come from leakage or index-specific artifacts in stored embeddings | Precomputed embeddings are loaded by sample index, so accidental sample-specific alignment could inflate results. | `shift_text` mismatches each sample with another prompt embedding while preserving embedding distribution; if `shift_text` remains close to `full`, text semantics are suspect. | `shift_text` |

## Ablation Variants

| Variant | Intervention | Parameters Kept? | Text Information Kept? | Interpretation |
| --- | --- | --- | --- | --- |
| `full` | Original T3Time prompt path. | Yes | Yes | Main reference. |
| `no_text` | Skip prompt encoder/CMA and pass time-frequency features directly to decoder. | No, prompt/CMA inactive | No | Architecture-level necessity of text/CMA path. |
| `zero_text` | Replace loaded prompt embeddings with zeros before prompt encoder/CMA. | Yes | No | Tests whether text content matters with same path active. |
| `mean_text` | Replace embeddings with per-sample mean token/channel value. | Yes | Weak | Checks whether only embedding scale/offset helps. |
| `shuffle_text` | Randomly permute text embeddings inside each mini-batch. | Yes | Mostly no | Cheap online mismatch test; less strict than `shift_text`. |
| `shift_text` | Load prompt embedding from a deterministic offset index in the same split. | Yes | No, distribution preserved | Strong leakage/semantic-control test. |
| `noise_text` | Replace embeddings with Gaussian noise matched to batch mean/std. | Yes | No | Optional stress test for random-text regularization. |

## Execution Blocks

### Block 1: Anchor Text Contribution
- Claim tested: C1 and C2.
- Dataset / split / task: Start with `ETTm1` and `ETTh1`, horizons `96,192,336,720`, official train/val/test split.
- Compared systems: `full`, `no_text`, `zero_text`, `shift_text`.
- Metrics: Test MSE and MAE averaged over all forecast horizons; validation MSE; runtime.
- Setup details: Reuse the repository hyperparameters for each dataset/horizon; run seeds `2024,2025,2026` by default.
- Success criterion: `full` improves mean test MSE over every text-destroyed ablation, with consistent direction on most horizons/seeds.
- Failure interpretation: If `full` is not better than `no_text`, the text branch is not necessary. If `full` is not better than `shift_text`, stored prompt embeddings may be acting as distributional noise or leakage-like sample identifiers.
- Priority: MUST-RUN.

### Block 2: Robustness Across Dataset Families
- Claim tested: C1.
- Dataset / split / task: Add `ETTm2`, `ETTh2`, `Weather`, and `exchange_rate` if compute permits.
- Compared systems: `full`, `no_text`, `zero_text`, `shift_text`.
- Metrics: Same as Block 1.
- Setup details: Same hyperparameter lookup table as official scripts.
- Success criterion: The text contribution is not isolated to one benchmark.
- Failure interpretation: Dataset-specific gains should be reported as conditional, not as a general tri-modal advantage.
- Priority: NICE-TO-HAVE.

### Block 3: Diagnostic Controls
- Claim tested: Anti-claim.
- Dataset / split / task: Run on the anchor dataset/horizon where `full` shows the strongest gain.
- Compared systems: `mean_text`, `shuffle_text`, `noise_text`.
- Metrics: Same as Block 1, plus per-run best epoch.
- Success criterion: Semantically aligned `full` beats all text-corrupted controls.
- Failure interpretation: If corrupted controls match `full`, the text branch is not using prompt semantics reliably.
- Priority: NICE-TO-HAVE.

## Run Order and Milestones

| Milestone | Goal | Runs | Decision Gate | Cost | Risk |
| --- | --- | --- | --- | --- | --- |
| M0 | Verify data, embedding, and metric pipeline | `scripts/run_text_ablation.sh --dry-run`; one `ETTm1` horizon `96`, one seed, one epoch | Logs and `metrics.json` are created for all variants | minutes to hours | Missing embeddings; fixed by `--prepare-embeddings`. |
| M1 | Anchor evidence | `ETTm1,ETTh1` x `96,192,336,720` x 3 seeds x 4 variants | `full` beats text-destroyed variants on seed mean | GPU-days depending on server | High variance; inspect seed std. |
| M2 | Broader support | Add `ETTm2,ETTh2,Weather,exchange_rate` | Same trend holds beyond anchors | GPU-days | Some datasets may need memory-specific batch overrides. |
| M3 | Diagnostics | Run `mean_text,shuffle_text,noise_text` on informative settings | Corrupted text does not match `full` | bounded | If full does not win in M1, diagnostics may be lower value. |

## Modification Plan

Files to change:
- `models/T3Time.py`: add `text_ablation` modes and implement model-level interventions.
- `train.py`: expose root/embedding paths, ablation mode, run id, metrics JSON, and robust final logging.
- `data_provider/data_loader_emb.py`: parameterize dataset and embedding directories; add deterministic shifted embedding loading.
- `storage/store_emb.py`: expose dataset root/output embedding root and avoid hard-coded author paths.
- `scripts/text_ablation_configs.py`: central dataset/horizon hyperparameter table based on official scripts.
- `scripts/run_text_ablation.sh`: server entrypoint for embedding preparation, run matrix execution, and log capture.
- `scripts/summarize_text_ablation.py`: parse run outputs into CSV/Markdown tables.
- `docs/ablation/TEXT_BRANCH_ABLATION_PLAN.md` and `docs/ablation/TEXT_BRANCH_ABLATION_CHECKLIST.md`: durable plan and execution surface.

Input/output format:
- Inputs: CSV files under `dataset/` by default; prompt embeddings under `Embeddings/<data>/<split>/<index>.h5`.
- Per-run outputs: `Results/text_ablation/<run_id>/train.log`, `metrics.json`, and `best_model.pth`.
- Aggregate outputs: `Results/text_ablation/summary.csv` and `summary.md`.

Validation:
- `python train.py --help`
- `python scripts/summarize_text_ablation.py --help`
- `bash scripts/run_text_ablation.sh --dry-run`
- `python -m py_compile train.py models/T3Time.py data_provider/data_loader_emb.py storage/store_emb.py scripts/text_ablation_configs.py scripts/summarize_text_ablation.py`

Rollback risk:
- Changes are additive and default `--text_ablation full` preserves original behavior except for portable default paths and extra metrics logging.
- Existing official shell scripts should still call `train.py` without new required arguments.
