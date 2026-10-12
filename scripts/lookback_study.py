#!/usr/bin/env python3
"""Lookback-length (seq_len) study.

Fixes pred_len and sweeps seq_len in {96, 192, 336, 720} on Solar and ECL.
Models: CSMamba, iTransformer, Mamba_fft (FLDMamba).
iTransformer / FLDMamba use patience=3 by default (faster); CSMamba uses --patience.

Example:
  python scripts/lookback_study.py --gpu 0
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

MODEL_DISPLAY = {
    "CSMamba": "CSMamba",
    "iTransformer": "iTransformer",
    "Mamba_fft": "FLDMamba",
}

# Faster early-stop for the two baselines the user asked to speed up.
MODEL_PATIENCE = {
    "CSMamba": None,  # use --patience
    "iTransformer": 3,
    "Mamba_fft": 3,
}

DATASET_PRESETS = {
    "Solar": {
        "root_path": "./dataset/Solar/",
        "data_path": "solar_AL.txt",
        "data": "Solar",
        "pred_len": 96,
        "enc_in": 137,
        "dec_in": 137,
        "c_out": 137,
        "use_norm": 1,
        "d_model": 512,
        "d_ff": 512,
        "d_state": 16,
        "e_layers": 2,
        "learning_rate": 0.0005,
        "batch_size": 32,
        "dropout": 0.1,
    },
    "ECL": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "pred_len": 96,
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
        "use_norm": 1,
        "d_model": 512,
        "d_ff": 512,
        "d_state": 16,
        "e_layers": 1,
        "learning_rate": 0.0005,
        "batch_size": 16,
        "dropout": 0.1,
    },
}


def parse_metrics(log_text: str):
    mse = mae = None
    matches = list(re.finditer(
        r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)",
        log_text,
    ))
    if matches:
        mse = float(matches[-1].group(1))
        mae = float(matches[-1].group(2))
    return mse, mae


def model_extras(model: str, seq_len: int):
    if model == "Mamba_fft":
        return {"i_or_cos": 1, "base": seq_len}
    return {"i_or_cos": 0, "base": 0}


def resolve_patience(args, model: str) -> int:
    override = MODEL_PATIENCE.get(model)
    return args.patience if override is None else override


def build_cmd(args, model, dataset, preset, seq_len):
    extras = model_extras(model, seq_len)
    patience = resolve_patience(args, model)
    model_id = f"{dataset}_{seq_len}_{preset['pred_len']}_lookback"
    label_len = min(args.label_len, seq_len)
    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", "1",
        "--root_path", preset["root_path"],
        "--data_path", preset["data_path"],
        "--model_id", model_id,
        "--model", model,
        "--data", preset["data"],
        "--features", "M",
        "--seq_len", str(seq_len),
        "--label_len", str(label_len),
        "--pred_len", str(preset["pred_len"]),
        "--e_layers", str(preset["e_layers"]),
        "--d_layers", "1",
        "--enc_in", str(preset["enc_in"]),
        "--dec_in", str(preset["dec_in"]),
        "--c_out", str(preset["c_out"]),
        "--d_model", str(preset["d_model"]),
        "--d_ff", str(preset["d_ff"]),
        "--d_state", str(preset["d_state"]),
        "--n_heads", "8",
        "--factor", "1",
        "--embed", "timeF",
        "--learning_rate", str(preset["learning_rate"]),
        "--batch_size", str(preset["batch_size"]),
        "--dropout", str(preset["dropout"]),
        "--use_norm", str(preset["use_norm"]),
        "--itr", "1",
        "--des", "Lookback",
        "--class_strategy", "projection",
        "--i_or_cos", str(extras["i_or_cos"]),
        "--base", str(extras["base"]),
        "--gpu", str(args.gpu),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(patience),
        "--noise_percent", "0",
    ]
    if model == "CSMamba":
        cmd.extend([
            "--use_dvr", "1",
            "--use_ms_scan", "0",
            "--use_freq_init", "0",
            "--use_bi_mamba", "0",
            "--use_time_marks", "0",
            "--use_tle", "1",
            "--use_trend_seasonal", "1",
            "--use_spectral_mod", "1",
        ])
    return cmd


def _ascii_table(headers, str_rows):
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt(headers), sep] + [fmt(r) for r in str_rows])


def make_long_table(rows):
    headers = ["model", "dataset", "seq_len", "pred_len", "mse", "mae", "status"]
    str_rows = []
    for r in rows:
        str_rows.append([
            r["model"],
            r["dataset"],
            str(r["seq_len"]),
            str(r["pred_len"]),
            "-" if r["mse"] is None else f'{r["mse"]:.6f}',
            "-" if r["mae"] is None else f'{r["mae"]:.6f}',
            r["status"],
        ])
    return _ascii_table(headers, str_rows)


def make_pivot_table(rows):
    """Per dataset: rows=seq_len, columns=model MSE/MAE."""
    datasets, seqs, models = [], [], []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
        if r["seq_len"] not in seqs:
            seqs.append(r["seq_len"])
        if r["model"] not in models:
            models.append(r["model"])
    seqs = sorted(seqs)
    lookup = {(r["dataset"], r["model"], r["seq_len"]): r for r in rows}

    blocks = []
    for dataset in datasets:
        headers = ["seq_len"]
        for m in models:
            headers.extend([f"{m}_mse", f"{m}_mae"])
        str_rows = []
        for sl in seqs:
            row = [str(sl)]
            for m in models:
                r = lookup.get((dataset, m, sl))
                if r is None or r["mse"] is None:
                    row.extend(["-", "-"])
                else:
                    row.extend([f'{r["mse"]:.6f}', f'{r["mae"]:.6f}'])
            str_rows.append(row)
        blocks.append(f"[{dataset}]\n" + _ascii_table(headers, str_rows))
    return "\n\n".join(blocks)


def write_result_file(path: Path, args, rows, started_at):
    with path.open("w", encoding="utf-8") as f:
        f.write("Lookback-Length Study\n")
        f.write("Fix pred_len, sweep seq_len (lookback).\n")
        f.write(f"models: {args.models}\n")
        f.write(f"datasets: {args.datasets}\n")
        f.write(f"seq_lens: {args.seq_lens}\n")
        f.write(
            "patience: CSMamba={} | iTransformer/FLDMamba=3\n".format(args.patience)
        )
        f.write(f"started: {started_at}\n")
        f.write(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"gpu: {args.gpu}\n\n")
        f.write("=== Long table ===\n")
        f.write(make_long_table(rows))
        f.write("\n\n=== Pivot table ===\n")
        f.write(make_pivot_table(rows))
        f.write("\n")


def run_one(cmd, log_path: Path):
    proc = subprocess.run(
        cmd,
        cwd=str(ROOT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
        check=False,
    )
    log_text = proc.stdout or ""
    log_path.write_text(log_text, encoding="utf-8")
    mse, mae = parse_metrics(log_text)
    status = "ok"
    if proc.returncode != 0:
        status = f"fail({proc.returncode})"
    elif mse is None:
        status = "no_metric"
    return mse, mae, status


def main():
    parser = argparse.ArgumentParser(description="Lookback-length study")
    parser.add_argument("--datasets", type=str, default="Solar,ECL")
    parser.add_argument(
        "--models",
        type=str,
        default="CSMamba,iTransformer,Mamba_fft",
    )
    parser.add_argument("--seq_lens", type=str, default="96,192,336,720")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5,
                        help="patience for CSMamba; iTransformer/FLDMamba forced to 3")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--output", type=str, default="results/lookback_length.txt")
    parser.add_argument("--dry_run", action="store_true")
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    seq_lens = [int(x) for x in args.seq_lens.split(",") if x.strip()]
    for d in datasets:
        if d not in DATASET_PRESETS:
            raise SystemExit(f"Unknown dataset {d}. Choose from {list(DATASET_PRESETS)}")
    for m in models:
        if m not in MODEL_DISPLAY:
            raise SystemExit(f"Unknown model {m}. Choose from {list(MODEL_DISPLAY)}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    planned = []
    for dataset in datasets:
        preset = DATASET_PRESETS[dataset]
        for seq_len in seq_lens:
            for model in models:
                planned.append({
                    "model": MODEL_DISPLAY[model],
                    "dataset": dataset,
                    "seq_len": seq_len,
                    "pred_len": preset["pred_len"],
                    "mse": None,
                    "mae": None,
                    "status": "planned",
                })
    print(f"Total runs: {len(planned)}")
    print(make_long_table(planned))
    if args.dry_run:
        write_result_file(out_path, args, planned, started_at)
        print(f"Dry-run table written to {out_path}")
        return

    rows = []
    for dataset in datasets:
        preset = DATASET_PRESETS[dataset]
        for seq_len in seq_lens:
            for model in models:
                display = MODEL_DISPLAY[model]
                patience = resolve_patience(args, model)
                print("=" * 80)
                print(
                    f"[lookback] {display} / {dataset} "
                    f"seq_len={seq_len} pred_len={preset['pred_len']} patience={patience}"
                )
                print("=" * 80)
                cmd = build_cmd(args, model, dataset, preset, seq_len)
                print(" ".join(cmd))
                log_path = out_path.with_suffix(
                    f".{display}.{dataset}.sl{seq_len}.log"
                )
                t0 = time.time()
                mse, mae, status = run_one(cmd, log_path)
                print(f"done in {time.time()-t0:.1f}s -> mse={mse} status={status}")
                rows.append({
                    "model": display,
                    "dataset": dataset,
                    "seq_len": seq_len,
                    "pred_len": preset["pred_len"],
                    "mse": mse,
                    "mae": mae,
                    "status": status,
                })
                write_result_file(out_path, args, rows, started_at)
                print(make_long_table(rows))

    write_result_file(out_path, args, rows, started_at)
    print("Done.")
    print(f"Final table: {out_path}")
    print(make_pivot_table(rows))


if __name__ == "__main__":
    main()
