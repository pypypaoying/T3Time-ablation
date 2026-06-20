from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path


FIELDS = [
    "run_id",
    "data_path",
    "pred_len",
    "seed",
    "text_ablation",
    "test_mse",
    "test_mae",
    "best_valid_loss",
    "best_epoch",
    "avg_train_time_per_epoch",
    "avg_val_time_per_epoch",
]


def mean(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    return sum(values) / len(values) if values else None


def std(values):
    values = [v for v in values if v is not None and not math.isnan(v)]
    if len(values) <= 1:
        return 0.0 if values else None
    mu = mean(values)
    return math.sqrt(sum((v - mu) ** 2 for v in values) / (len(values) - 1))


def fmt(value, digits=6):
    if value is None:
        return ""
    return f"{value:.{digits}f}"


def load_rows(results_dir: Path):
    rows = []
    for path in sorted(results_dir.glob("*/metrics.json")):
        with path.open("r", encoding="utf-8") as f:
            payload = json.load(f)
        payload["_metrics_path"] = str(path)
        rows.append(payload)
    return rows


def write_csv(path: Path, rows):
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=FIELDS + ["metrics_path"])
        writer.writeheader()
        for row in rows:
            writer.writerow({field: row.get(field, "") for field in FIELDS} | {"metrics_path": row.get("_metrics_path", "")})


def aggregate(rows):
    grouped = defaultdict(list)
    for row in rows:
        key = (row["data_path"], int(row["pred_len"]), row["text_ablation"])
        grouped[key].append(row)

    full_by_setting = {}
    for (data_path, pred_len, variant), group_rows in grouped.items():
        if variant == "full":
            full_by_setting[(data_path, pred_len)] = {
                "mse": mean([r["test_mse"] for r in group_rows]),
                "mae": mean([r["test_mae"] for r in group_rows]),
            }

    summary = []
    for (data_path, pred_len, variant), group_rows in sorted(grouped.items()):
        mse_values = [float(r["test_mse"]) for r in group_rows]
        mae_values = [float(r["test_mae"]) for r in group_rows]
        full = full_by_setting.get((data_path, pred_len), {})
        mse_mean = mean(mse_values)
        mae_mean = mean(mae_values)
        full_mse = full.get("mse")
        full_mae = full.get("mae")
        summary.append(
            {
                "data_path": data_path,
                "pred_len": pred_len,
                "text_ablation": variant,
                "n": len(group_rows),
                "mse_mean": mse_mean,
                "mse_std": std(mse_values),
                "mae_mean": mae_mean,
                "mae_std": std(mae_values),
                "delta_mse_vs_full": None if full_mse is None else mse_mean - full_mse,
                "delta_mae_vs_full": None if full_mae is None else mae_mean - full_mae,
                "rel_mse_pct_vs_full": None if not full_mse else 100.0 * (mse_mean - full_mse) / full_mse,
                "rel_mae_pct_vs_full": None if not full_mae else 100.0 * (mae_mean - full_mae) / full_mae,
            }
        )
    return summary


def write_summary_csv(path: Path, summary):
    fields = [
        "data_path",
        "pred_len",
        "text_ablation",
        "n",
        "mse_mean",
        "mse_std",
        "mae_mean",
        "mae_std",
        "delta_mse_vs_full",
        "delta_mae_vs_full",
        "rel_mse_pct_vs_full",
        "rel_mae_pct_vs_full",
    ]
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fields)
        writer.writeheader()
        for row in summary:
            writer.writerow(row)


def write_markdown(path: Path, summary, rows):
    lines = [
        "# T3Time Text Branch Ablation Summary",
        "",
        f"Runs found: {len(rows)}",
        "",
        "| Dataset | Pred Len | Variant | N | MSE mean | MSE std | MAE mean | MAE std | Delta MSE vs full | Delta MAE vs full | Rel MSE % | Rel MAE % |",
        "| --- | ---: | --- | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
    ]
    for row in summary:
        lines.append(
            "| {data_path} | {pred_len} | {text_ablation} | {n} | {mse_mean} | {mse_std} | {mae_mean} | {mae_std} | {dmse} | {dmae} | {rmse} | {rmae} |".format(
                data_path=row["data_path"],
                pred_len=row["pred_len"],
                text_ablation=row["text_ablation"],
                n=row["n"],
                mse_mean=fmt(row["mse_mean"]),
                mse_std=fmt(row["mse_std"]),
                mae_mean=fmt(row["mae_mean"]),
                mae_std=fmt(row["mae_std"]),
                dmse=fmt(row["delta_mse_vs_full"]),
                dmae=fmt(row["delta_mae_vs_full"]),
                rmse=fmt(row["rel_mse_pct_vs_full"], 3),
                rmae=fmt(row["rel_mae_pct_vs_full"], 3),
            )
        )
    lines.extend(
        [
            "",
            "Interpretation guide:",
            "- Positive delta means the variant is worse than `full`; this supports a useful text branch.",
            "- `no_text` isolates architecture-level text/CMA necessity.",
            "- `zero_text` keeps the text path active but removes text content.",
            "- `shift_text` preserves embedding distribution but mismatches semantics/sample index.",
            "- If corrupted text variants match `full`, the prompt branch is not providing reliable semantic signal.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--results_dir", type=Path, default=Path("./Results/text_ablation"))
    args = parser.parse_args()

    rows = load_rows(args.results_dir)
    if not rows:
        raise SystemExit(f"No metrics.json files found under {args.results_dir}")

    write_csv(args.results_dir / "runs.csv", rows)
    summary = aggregate(rows)
    write_summary_csv(args.results_dir / "summary.csv", summary)
    write_markdown(args.results_dir / "summary.md", summary, rows)

    print(f"Wrote {args.results_dir / 'runs.csv'}")
    print(f"Wrote {args.results_dir / 'summary.csv'}")
    print(f"Wrote {args.results_dir / 'summary.md'}")


if __name__ == "__main__":
    main()
