# Text Branch Ablation Brief

## Objective

This experiment tests whether the text branch in T3Time makes a genuine contribution to forecasting accuracy. The key question is not whether the full model can train, but whether correctly matched text embeddings improve forecasts compared with models where the text pathway is removed, emptied, or deliberately mismatched.

## Experimental Setting

Datasets:

| Dataset | Frequency / Domain | Variables | Prediction Lengths |
| --- | --- | ---: | --- |
| `ETTm1` | Electricity transformer, 15-minute | 7 | 96, 192, 336, 720 |
| `ETTm2` | Electricity transformer, 15-minute | 7 | 96, 192, 336, 720 |
| `ETTh1` | Electricity transformer, hourly | 7 | 96, 192, 336, 720 |
| `ETTh2` | Electricity transformer, hourly | 7 | 96, 192, 336, 720 |
| `Weather` | Weather station, 10-minute | 21 | 96, 192, 336, 720 |
| `exchange_rate` | Daily exchange rates | 8 | 96, 192, 336, 720 |
| `ILI` | Weekly influenza-like illness | 7 | 24, 36, 48, 60 |

Splits:
- Use the official T3Time/long-term forecasting train, validation, and test splits implemented in the repository.
- Model selection uses validation loss.
- Final reporting uses test MSE and test MAE averaged across all forecast horizons inside each prediction length.

Seeds:
- Main runs: `2024, 2025, 2026, 2027, 2028`.
- Report mean and standard deviation across seeds.
- If runtime becomes unexpectedly large, first keep all datasets and reduce to seeds `2024, 2025, 2026`; do not reduce the ablation variants before reducing seeds.

Training budget:
- Compute budget is assumed sufficient.
- Use the same hyperparameters as the official T3Time scripts for each dataset and prediction length.
- Generate GPT-2 prompt embeddings once per dataset/split and reuse them across variants.
- Run all variants with the same data split, prediction length, seed, optimizer, learning rate, batch size, model width, and epoch budget.

Primary metrics:
- Test MSE: main metric.
- Test MAE: secondary metric.
- Validation MSE, best epoch, and runtime are recorded for auditability.

## Model Variants

### 1. Full T3Time

This is the original model. It uses three information sources:

- the numerical time-series history;
- the frequency-domain representation of the same history;
- a text prompt embedding generated from the time-series values and timestamps.

The text embedding is passed through the prompt encoder and then fused with the numerical representation through the model's cross-modal alignment module. This is the reference system.

### 2. No-Text Model

This variant removes the text pathway from the forward pass. The model still uses:

- the numerical time-series branch;
- the frequency-domain branch;
- the same decoder and output projection.

However, it does not use the prompt encoder or cross-modal alignment with text. This answers: "Can the model do just as well without any text branch at all?"

If this variant matches or beats Full T3Time, the text branch is not necessary for the tested setting.

### 3. Zero-Text Model

This variant keeps the text pathway structurally present, but replaces every text embedding with zeros before it enters the prompt encoder.

The purpose is to distinguish two possibilities:

- the text branch helps because it carries meaningful information;
- the text branch helps merely because it adds extra layers, parameters, or regularization.

If Full T3Time beats Zero-Text, that supports the claim that text content matters. If they are similar, the text pathway may be acting mostly as extra architecture rather than as a meaningful modality.

### 4. Shifted-Text Model

This variant keeps real GPT-2 text embeddings, but intentionally assigns each training/test sample the embedding from a different sample in the same split.

This preserves the overall distribution, scale, and type of text embeddings, but breaks the correct match between the time-series window and its text description.

This answers: "Does the model need the correct text for the correct sample, or does any plausible-looking text embedding work?"

If Shifted-Text performs close to Full T3Time, the text branch may not be using sample-specific semantic information. If Full T3Time clearly beats Shifted-Text, that supports genuine text alignment.

### 5. Mean-Text Diagnostic

This variant replaces each text embedding with a constant embedding equal to its own average value. It preserves a rough embedding scale but removes token-level and variable-level detail.

This is a diagnostic, not a main baseline. It tests whether coarse embedding magnitude is enough to explain the gain.

### 6. Shuffled-Text Diagnostic

This variant randomly permutes text embeddings within each mini-batch. The model sees real embeddings, but they usually belong to the wrong samples.

This is a cheaper online mismatch check. It is less controlled than Shifted-Text, but useful for diagnosing whether performance depends on correct sample-text pairing.

### 7. Noise-Text Diagnostic

This variant replaces the text embedding with Gaussian noise matched to the batch mean and standard deviation.

It tests whether the text branch behaves like a noise injection or regularization mechanism. If Noise-Text is close to Full T3Time, the text branch should not be interpreted as a meaningful semantic branch.

## Main Comparison Matrix

Must-run variants:

| Variant | Purpose | Main Interpretation |
| --- | --- | --- |
| Full T3Time | Reference model | Best case for text contribution |
| No-Text | Remove text pathway | Tests whether text pathway is necessary |
| Zero-Text | Keep pathway but remove text content | Tests whether text content matters |
| Shifted-Text | Keep real embeddings but mismatch samples | Tests whether correct text-sample alignment matters |

Optional diagnostic variants:

| Variant | Purpose |
| --- | --- |
| Mean-Text | Tests whether embedding scale/average is enough |
| Shuffled-Text | Tests online wrong-text sensitivity |
| Noise-Text | Tests whether random embedding-like noise explains the gain |

## Reporting Plan

For each dataset and prediction length, report:

- seed-mean test MSE and test MAE for every variant;
- standard deviation across seeds;
- absolute and percentage difference from Full T3Time;
- best validation epoch and average runtime as audit fields.

Recommended main table:

| Dataset | Pred Len | Full MSE | No-Text MSE | Zero-Text MSE | Shifted-Text MSE | Full vs Best Ablation |
| --- | ---: | ---: | ---: | ---: | ---: | ---: |

Recommended appendix table:

| Dataset | Pred Len | Variant | MSE mean | MSE std | MAE mean | MAE std | Relative MSE vs Full |
| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: |

## Decision Rule

The text branch is considered useful only if Full T3Time is consistently better than all three core controls: No-Text, Zero-Text, and Shifted-Text.

Interpretation:

- Full beats No-Text: the text pathway adds value beyond the time and frequency branches.
- Full beats Zero-Text: the text content matters, not just extra layers.
- Full beats Shifted-Text: the correct pairing between sample and text matters.
- Full does not beat No-Text: the text branch is not needed.
- Full beats No-Text but not Zero-Text: gains may come from architecture or regularization, not text information.
- Full beats Zero-Text but not Shifted-Text: text embeddings may help through distributional effects rather than correct semantic alignment.

## Recommended Execution

Smoke test:

```bash
DATASETS=ETTm1 HORIZONS=96 SEEDS=2024 VARIANTS=no_text,zero_text EPOCHS_OVERRIDE=1 MAX_TRAIN_BATCHES=2 MAX_EVAL_BATCHES=2 NUM_WORKERS=0 bash scripts/run_text_ablation.sh
```

Full main experiment:

```bash
DATASETS=ETTm1,ETTm2,ETTh1,ETTh2,Weather,exchange_rate \
HORIZONS=96,192,336,720 \
SEEDS=2024,2025,2026,2027,2028 \
VARIANTS=full,no_text,zero_text,shift_text \
bash scripts/run_text_ablation.sh --prepare-embeddings --skip-existing
```

For `ILI`, use its own horizons:

```bash
DATASETS=ILI \
HORIZONS=24,36,48,60 \
SEEDS=2024,2025,2026,2027,2028 \
VARIANTS=full,no_text,zero_text,shift_text \
bash scripts/run_text_ablation.sh --prepare-embeddings --skip-existing
```

Diagnostic experiment:

```bash
DATASETS=ETTm1,ETTh1 \
HORIZONS=96,336 \
SEEDS=2024,2025,2026,2027,2028 \
VARIANTS=mean_text,shuffle_text,noise_text \
bash scripts/run_text_ablation.sh --skip-existing
```
