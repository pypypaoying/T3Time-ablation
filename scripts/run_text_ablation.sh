#!/usr/bin/env bash
set -euo pipefail

usage() {
  cat <<'USAGE'
Run T3Time text-branch ablations.

Examples:
  bash scripts/run_text_ablation.sh --dry-run
  DATASETS=ETTm1 HORIZONS=96 SEEDS=2024 VARIANTS=full,no_text,zero_text,shift_text bash scripts/run_text_ablation.sh
  DATASETS=ETTm1,ETTh1 HORIZONS=96,192,336,720 SEEDS=2024,2025,2026 bash scripts/run_text_ablation.sh --prepare-embeddings

Environment variables:
  DATASETS      Comma-separated dataset names. Default: ETTm1,ETTh1
  HORIZONS      Comma-separated pred_lens. Default: 96,192,336,720
  VARIANTS      Comma-separated modes. Default: full,no_text,zero_text,shift_text
  SEEDS         Comma-separated seeds. Default: 2024,2025,2026
  DATA_ROOT     Dataset directory. Default: ./dataset
  EMBED_ROOT    Embedding directory. Default: ./Embeddings
  RESULTS_DIR   Output directory. Default: ./Results/text_ablation
  CUDA_VISIBLE_DEVICES GPU ids. Default: 0
  EPOCHS_OVERRIDE Override config epochs, useful for smoke tests.
  MAX_TRAIN_BATCHES Limit train batches per epoch for pipeline smoke tests.
  MAX_EVAL_BATCHES  Limit val/test batches for pipeline smoke tests.
  NUM_WORKERS   DataLoader workers. Default: 10
  DEVICE        Training device argument. Default: cuda
  PYTHON        Python executable. Default: python

Options:
  --prepare-embeddings  Generate missing embeddings before training.
  --dry-run             Print commands without running them.
  --skip-existing       Skip runs with an existing metrics.json.
  --help                Show this message.
USAGE
}

PREPARE_EMBEDDINGS=0
DRY_RUN=0
SKIP_EXISTING=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --prepare-embeddings) PREPARE_EMBEDDINGS=1; shift ;;
    --dry-run) DRY_RUN=1; shift ;;
    --skip-existing) SKIP_EXISTING=1; shift ;;
    --help|-h) usage; exit 0 ;;
    *) echo "Unknown argument: $1" >&2; usage; exit 2 ;;
  esac
done

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
cd "${REPO_ROOT}"

export PYTHONPATH="${REPO_ROOT}:${PYTHONPATH:-}"
export PYTORCH_CUDA_ALLOC_CONF="${PYTORCH_CUDA_ALLOC_CONF:-expandable_segments:True}"
export CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-0}"

PYTHON_BIN="${PYTHON:-python}"
DATASETS_CSV="${DATASETS:-ETTm1,ETTh1}"
HORIZONS_CSV="${HORIZONS:-96,192,336,720}"
VARIANTS_CSV="${VARIANTS:-full,no_text,zero_text,shift_text}"
SEEDS_CSV="${SEEDS:-2024,2025,2026}"
DATA_ROOT="${DATA_ROOT:-./dataset}"
EMBED_ROOT="${EMBED_ROOT:-./Embeddings}"
RESULTS_DIR="${RESULTS_DIR:-./Results/text_ablation}"
NUM_WORKERS="${NUM_WORKERS:-10}"
DEVICE="${DEVICE:-cuda}"
MAX_TRAIN_BATCHES="${MAX_TRAIN_BATCHES:-0}"
MAX_EVAL_BATCHES="${MAX_EVAL_BATCHES:-0}"

IFS=',' read -r -a DATASET_LIST <<< "${DATASETS_CSV}"
IFS=',' read -r -a HORIZON_LIST <<< "${HORIZONS_CSV}"
IFS=',' read -r -a VARIANT_LIST <<< "${VARIANTS_CSV}"
IFS=',' read -r -a SEED_LIST <<< "${SEEDS_CSV}"

mkdir -p "${RESULTS_DIR}/logs"

run_or_print() {
  if [[ "${DRY_RUN}" == "1" ]]; then
    printf '[dry-run] %q ' "$@"
    printf '\n'
  else
    "$@"
  fi
}

config_field() {
  local data_path="$1"
  local pred_len="$2"
  local field="$3"
  "${PYTHON_BIN}" scripts/text_ablation_configs.py --data_path "${data_path}" --pred_len "${pred_len}" --field "${field}"
}

expected_embedding_count() {
  local data_path="$1"
  local split="$2"
  local seq_len="$3"
  local pred_len="$4"
  "${PYTHON_BIN}" -c "from data_provider.data_loader_save import Dataset_ETT_hour, Dataset_ETT_minute, Dataset_Custom; data='${data_path}'; cls={'ETTh1': Dataset_ETT_hour, 'ETTh2': Dataset_ETT_hour, 'ETTm1': Dataset_ETT_minute, 'ETTm2': Dataset_ETT_minute}.get(data, Dataset_Custom); ds=cls(root_path='${DATA_ROOT}', flag='${split}', size=[int('${seq_len}'), 0, int('${pred_len}')], data_path=data); print(len(ds))"
}

actual_embedding_count() {
  local split_dir="$1"
  if [[ ! -d "${split_dir}" ]]; then
    echo 0
    return
  fi
  find "${split_dir}" -maxdepth 1 -name '*.h5' | wc -l | tr -d ' '
}

ensure_embeddings() {
  local data_path="$1"
  local seq_len="$2"
  local pred_len="$3"
  for split in train val test; do
    local split_dir="${EMBED_ROOT}/${data_path}/${split}"
    local expected_count
    local actual_count
    expected_count="$(expected_embedding_count "${data_path}" "${split}" "${seq_len}" "${pred_len}")"
    actual_count="$(actual_embedding_count "${split_dir}")"
    if [[ "${actual_count}" -ge "${expected_count}" && "${expected_count}" -gt 0 ]]; then
      echo "[embeddings] found ${split_dir} (${actual_count}/${expected_count})"
      continue
    fi
    echo "[embeddings] generating ${data_path}/${split} (seq_len=${seq_len}, pred_len=${pred_len}, have=${actual_count}, need=${expected_count})"
    run_or_print "${PYTHON_BIN}" storage/store_emb.py \
      --data_path "${data_path}" \
      --root_path "${DATA_ROOT}" \
      --embed_root_path "${EMBED_ROOT}" \
      --input_len "${seq_len}" \
      --output_len "${pred_len}" \
      --divide "${split}" \
      --device "${DEVICE}" \
      --num_workers "${NUM_WORKERS}"
  done
}

for data_path in "${DATASET_LIST[@]}"; do
  for pred_len in "${HORIZON_LIST[@]}"; do
    seq_len="$(config_field "${data_path}" "${pred_len}" seq_len)"
    batch_size="$(config_field "${data_path}" "${pred_len}" batch_size)"
    num_nodes="$(config_field "${data_path}" "${pred_len}" num_nodes)"
    learning_rate="$(config_field "${data_path}" "${pred_len}" learning_rate)"
    channel="$(config_field "${data_path}" "${pred_len}" channel)"
    e_layer="$(config_field "${data_path}" "${pred_len}" e_layer)"
    d_layer="$(config_field "${data_path}" "${pred_len}" d_layer)"
    dropout_n="$(config_field "${data_path}" "${pred_len}" dropout_n)"
    epochs="$(config_field "${data_path}" "${pred_len}" epochs)"
    epochs="${EPOCHS_OVERRIDE:-${epochs}}"

    if [[ "${PREPARE_EMBEDDINGS}" == "1" ]]; then
      ensure_embeddings "${data_path}" "${seq_len}" "${pred_len}"
    fi

    for variant in "${VARIANT_LIST[@]}"; do
      for seed in "${SEED_LIST[@]}"; do
        run_id="${data_path}_s${seq_len}_p${pred_len}_${variant}_seed${seed}"
        run_dir="${RESULTS_DIR}/${run_id}"
        log_file="${RESULTS_DIR}/logs/${run_id}.log"

        if [[ "${SKIP_EXISTING}" == "1" && -f "${run_dir}/metrics.json" ]]; then
          echo "[skip-existing] ${run_id}"
          continue
        fi

        echo "[run] ${run_id}"
        cmd=(
          "${PYTHON_BIN}" train.py
          --data_path "${data_path}"
          --root_path "${DATA_ROOT}"
          --embed_root_path "${EMBED_ROOT}"
          --batch_size "${batch_size}"
          --num_nodes "${num_nodes}"
          --seq_len "${seq_len}"
          --pred_len "${pred_len}"
          --epochs "${epochs}"
          --seed "${seed}"
          --channel "${channel}"
          --learning_rate "${learning_rate}"
          --dropout_n "${dropout_n}"
          --e_layer "${e_layer}"
          --d_layer "${d_layer}"
          --num_workers "${NUM_WORKERS}"
          --device "${DEVICE}"
          --text_ablation "${variant}"
          --run_id "${run_id}"
          --save "${RESULTS_DIR}"
          --max_train_batches "${MAX_TRAIN_BATCHES}"
          --max_eval_batches "${MAX_EVAL_BATCHES}"
        )

        if [[ "${DRY_RUN}" == "1" ]]; then
          printf '[dry-run] '
          printf '%q ' "${cmd[@]}"
          printf '> %q 2>&1\n' "${log_file}"
        else
          "${cmd[@]}" > "${log_file}" 2>&1
        fi
      done
    done
  done
done

if [[ "${DRY_RUN}" == "0" ]]; then
  "${PYTHON_BIN}" scripts/summarize_text_ablation.py --results_dir "${RESULTS_DIR}"
fi
