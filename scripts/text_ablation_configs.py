"""Official T3Time hyperparameter table used by text-branch ablation scripts."""

from __future__ import annotations

import argparse
import json


CONFIGS = {
    "ETTm1": {
        96: dict(num_nodes=7, seq_len=96, batch_size=64, learning_rate=1e-4, channel=128, e_layer=1, d_layer=2, dropout_n=0.5, epochs=150),
        192: dict(num_nodes=7, seq_len=96, batch_size=32, learning_rate=1e-4, channel=64, e_layer=1, d_layer=2, dropout_n=0.65, epochs=150),
        336: dict(num_nodes=7, seq_len=96, batch_size=32, learning_rate=1e-4, channel=64, e_layer=1, d_layer=2, dropout_n=0.65, epochs=150),
        720: dict(num_nodes=7, seq_len=96, batch_size=16, learning_rate=1e-4, channel=64, e_layer=2, d_layer=2, dropout_n=0.7, epochs=100),
    },
    "ETTm2": {
        96: dict(num_nodes=7, seq_len=96, batch_size=16, learning_rate=1e-4, channel=64, e_layer=1, d_layer=1, dropout_n=0.6, epochs=150),
        192: dict(num_nodes=7, seq_len=96, batch_size=16, learning_rate=1e-4, channel=64, e_layer=1, d_layer=1, dropout_n=0.5, epochs=150),
        336: dict(num_nodes=7, seq_len=96, batch_size=16, learning_rate=1e-4, channel=64, e_layer=1, d_layer=1, dropout_n=0.7, epochs=150),
        720: dict(num_nodes=7, seq_len=96, batch_size=16, learning_rate=1e-4, channel=16, e_layer=1, d_layer=1, dropout_n=0.55, epochs=150),
    },
    "ETTh1": {
        96: dict(num_nodes=7, seq_len=96, batch_size=256, learning_rate=1e-4, channel=256, e_layer=1, d_layer=1, dropout_n=0.4, epochs=150),
        192: dict(num_nodes=7, seq_len=96, batch_size=32, learning_rate=1e-4, channel=256, e_layer=1, d_layer=2, dropout_n=0.6, epochs=150),
        336: dict(num_nodes=7, seq_len=96, batch_size=16, learning_rate=1e-4, channel=64, e_layer=1, d_layer=2, dropout_n=0.7, epochs=120),
        720: dict(num_nodes=7, seq_len=96, batch_size=32, learning_rate=1e-4, channel=64, e_layer=3, d_layer=4, dropout_n=0.5, epochs=150),
    },
    "ETTh2": {
        96: dict(num_nodes=7, seq_len=96, batch_size=256, learning_rate=1e-4, channel=64, e_layer=1, d_layer=1, dropout_n=0.25, epochs=150),
        192: dict(num_nodes=7, seq_len=96, batch_size=256, learning_rate=1e-4, channel=128, e_layer=3, d_layer=5, dropout_n=0.25, epochs=150),
        336: dict(num_nodes=7, seq_len=96, batch_size=256, learning_rate=1e-4, channel=128, e_layer=3, d_layer=5, dropout_n=0.4, epochs=150),
        720: dict(num_nodes=7, seq_len=96, batch_size=256, learning_rate=1e-4, channel=128, e_layer=3, d_layer=5, dropout_n=0.4, epochs=150),
    },
    "exchange_rate": {
        96: dict(num_nodes=8, seq_len=96, batch_size=128, learning_rate=1e-4, channel=16, e_layer=1, d_layer=2, dropout_n=0.5, epochs=120),
        192: dict(num_nodes=8, seq_len=96, batch_size=128, learning_rate=1e-4, channel=16, e_layer=1, d_layer=2, dropout_n=0.5, epochs=120),
        336: dict(num_nodes=8, seq_len=96, batch_size=128, learning_rate=1e-4, channel=8, e_layer=1, d_layer=1, dropout_n=0.3, epochs=150),
        720: dict(num_nodes=8, seq_len=96, batch_size=40, learning_rate=1e-4, channel=8, e_layer=1, d_layer=1, dropout_n=0.01, epochs=200),
    },
    "Weather": {
        96: dict(num_nodes=21, seq_len=96, batch_size=32, learning_rate=1e-3, channel=64, e_layer=6, d_layer=2, dropout_n=0.1, epochs=20),
        192: dict(num_nodes=21, seq_len=96, batch_size=32, learning_rate=1e-4, channel=32, e_layer=1, d_layer=2, dropout_n=0.1, epochs=150),
        336: dict(num_nodes=21, seq_len=96, batch_size=32, learning_rate=1e-4, channel=64, e_layer=1, d_layer=6, dropout_n=0.5, epochs=150),
        720: dict(num_nodes=21, seq_len=96, batch_size=64, learning_rate=1e-4, channel=128, e_layer=1, d_layer=1, dropout_n=0.25, epochs=150),
    },
    "ILI": {
        24: dict(num_nodes=7, seq_len=36, batch_size=16, learning_rate=1e-4, channel=32, e_layer=1, d_layer=1, dropout_n=0.1, epochs=100),
        36: dict(num_nodes=7, seq_len=36, batch_size=16, learning_rate=1e-4, channel=64, e_layer=1, d_layer=1, dropout_n=0.1, epochs=150),
        48: dict(num_nodes=7, seq_len=36, batch_size=32, learning_rate=2.5e-3, channel=128, e_layer=1, d_layer=1, dropout_n=0.5, epochs=150),
        60: dict(num_nodes=7, seq_len=36, batch_size=32, learning_rate=1e-4, channel=64, e_layer=1, d_layer=1, dropout_n=0.1, epochs=150),
    },
}


def get_config(data_path: str, pred_len: int) -> dict:
    try:
        cfg = CONFIGS[data_path][int(pred_len)].copy()
    except KeyError as exc:
        raise SystemExit(f"No text-ablation config for data_path={data_path!r}, pred_len={pred_len!r}") from exc
    cfg["data_path"] = data_path
    cfg["pred_len"] = int(pred_len)
    return cfg


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data_path", required=True)
    parser.add_argument("--pred_len", required=True, type=int)
    parser.add_argument("--field", default="", help="return only one field")
    args = parser.parse_args()

    cfg = get_config(args.data_path, args.pred_len)
    if args.field:
        print(cfg[args.field])
    else:
        print(json.dumps(cfg, sort_keys=True))


if __name__ == "__main__":
    main()
