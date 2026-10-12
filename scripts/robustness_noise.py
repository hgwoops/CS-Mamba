#!/usr/bin/env python3
"""Input-noise robustness study for CSMamba and baselines.

Train once per (model, dataset) on clean data, then evaluate with
noise_percent in {0, 5, 10, 15, 20}.

Models:
  CSMamba, iTransformer, Mamba_fft (FLDMamba), S_Mamba

Example:
  python scripts/robustness_noise.py --datasets PEMS04,ECL \\
    --models CSMamba,iTransformer,Mamba_fft,S_Mamba --gpu 0
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

# CLI model name -> display name in the result table
MODEL_DISPLAY = {
    "CSMamba": "CSMamba",
    "iTransformer": "iTransformer",
    "Mamba_fft": "FLDMamba",
    "S_Mamba": "S-Mamba",
}

DATASET_PRESETS = {
    "PEMS04": {
        "root_path": "./dataset/PEMS/",
        "data_path": "PEMS04.npz",
        "data": "PEMS",
        "seq_len": 96,
        "pred_len": 12,
        "enc_in": 307,
        "dec_in": 307,
        "c_out": 307,
        "use_norm": 0,
        "d_model": 1024,
        "d_ff": 1024,
        "d_state": 16,
        "e_layers": 1,
        "learning_rate": 0.0002,
        "batch_size": 16,
        "dropout": 0.1,
    },
    "ECL": {
        "root_path": "./dataset/electricity/",
        "data_path": "electricity.csv",
        "data": "custom",
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 321,
        "dec_in": 321,
        "c_out": 321,
        "use_norm": 1,
        "d_model": 512,
        "d_ff": 512,
        "d_state": 16,
        "e_layers": 2,
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


def model_extras(model: str, preset: dict):
    """Extra CLI args that only some models need."""
    if model == "Mamba_fft":
        # FLDMamba Fourier/Laplace bases (see README)
        return {"i_or_cos": 1, "base": preset["seq_len"]}
    return {"i_or_cos": 0, "base": 0}


def setting_name(args, model, dataset, preset, ii=0):
    extras = model_extras(model, preset)
    model_id = f"{dataset}_{preset['seq_len']}_{preset['pred_len']}_robust"
    return (
        f"{model_id}_{model}_{preset['data']}_M_"
        f"ft{preset['seq_len']}_sl{args.label_len}_ll{preset['pred_len']}_"
        f"pl{preset['d_model']}_dm{args.n_heads}_nh{preset['e_layers']}_"
        f"el{args.d_layers}_dl{preset['d_ff']}_df{args.factor}_"
        f"fc{args.embed}_eb{args.distil}_dt{args.des}_"
        f"{args.class_strategy}_{extras['i_or_cos']}_{extras['base']}_{ii}"
    )


def build_cmd(args, model, dataset, preset, is_training, noise_percent):
    extras = model_extras(model, preset)
    model_id = f"{dataset}_{preset['seq_len']}_{preset['pred_len']}_robust"
    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", str(is_training),
        "--root_path", preset["root_path"],
        "--data_path", preset["data_path"],
        "--model_id", model_id,
        "--model", model,
        "--data", preset["data"],
        "--features", "M",
        "--seq_len", str(preset["seq_len"]),
        "--label_len", str(args.label_len),
        "--pred_len", str(preset["pred_len"]),
        "--e_layers", str(preset["e_layers"]),
        "--d_layers", str(args.d_layers),
        "--enc_in", str(preset["enc_in"]),
        "--dec_in", str(preset["dec_in"]),
        "--c_out", str(preset["c_out"]),
        "--d_model", str(preset["d_model"]),
        "--d_ff", str(preset["d_ff"]),
        "--d_state", str(preset["d_state"]),
        "--n_heads", str(args.n_heads),
        "--factor", str(args.factor),
        "--embed", args.embed,
        "--learning_rate", str(preset["learning_rate"]),
        "--batch_size", str(preset["batch_size"]),
        "--dropout", str(preset["dropout"]),
        "--use_norm", str(preset["use_norm"]),
        "--itr", "1",
        "--des", args.des,
        "--class_strategy", args.class_strategy,
        "--i_or_cos", str(extras["i_or_cos"]),
        "--base", str(extras["base"]),
        "--gpu", str(args.gpu),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--noise_percent", str(noise_percent),
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


def make_long_table(rows):
    headers = ["model", "dataset", "noise%", "mse", "mae", "status"]
    str_rows = []
    for r in rows:
        str_rows.append([
            r["model"],
            r["dataset"],
            f'{r["noise"]:g}',
            "-" if r["mse"] is None else f'{r["mse"]:.6f}',
            "-" if r["mae"] is None else f'{r["mae"]:.6f}',
            r["status"],
        ])
    return _ascii_table(headers, str_rows)


def make_pivot_table(rows):
    """Wide table: one row per (dataset, noise), columns per model MSE/MAE."""
    models = []
    for r in rows:
        if r["model"] not in models:
            models.append(r["model"])
    keys = []
    for r in rows:
        key = (r["dataset"], r["noise"])
        if key not in keys:
            keys.append(key)

    lookup = {(r["model"], r["dataset"], r["noise"]): r for r in rows}
    headers = ["dataset", "noise%"]
    for m in models:
        headers.extend([f"{m}_mse", f"{m}_mae"])

    str_rows = []
    for dataset, noise in keys:
        row = [dataset, f"{noise:g}"]
        for m in models:
            r = lookup.get((m, dataset, noise))
            if r is None or r["mse"] is None:
                row.extend(["-", "-"])
            else:
                row.extend([f'{r["mse"]:.6f}', f'{r["mae"]:.6f}'])
        str_rows.append(row)
    return _ascii_table(headers, str_rows)


def _ascii_table(headers, str_rows):
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    lines = [fmt(headers), sep]
    lines.extend(fmt(row) for row in str_rows)
    return "\n".join(lines)


def write_result_file(path: Path, args, rows, started_at):
    with path.open("w", encoding="utf-8") as f:
        f.write("Robustness Study (input Gaussian noise)\n")
        f.write("noise = (noise%/100) * per-variate temporal std * N(0,1)\n")
        f.write(f"models: {args.models}\n")
        f.write(f"datasets: {args.datasets}\n")
        f.write(f"noise levels: {args.noise_levels}\n")
        f.write(f"started: {started_at}\n")
        f.write(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"gpu: {args.gpu}\n\n")
        f.write("=== Long table ===\n")
        f.write(make_long_table(rows))
        f.write("\n\n=== Pivot table (MSE/MAE by model) ===\n")
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


def load_existing_csmamba_rows(path: Path):
    """Parse previously written CSMamba-only table if present."""
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    # old format without model column
    for m in re.finditer(
        r"^(PEMS04|ECL)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+(\S+)",
        text,
        flags=re.M,
    ):
        rows.append({
            "model": "CSMamba",
            "dataset": m.group(1),
            "noise": float(m.group(2)),
            "mse": float(m.group(3)),
            "mae": float(m.group(4)),
            "status": m.group(5),
        })
    # new long format
    for m in re.finditer(
        r"^(CSMamba|iTransformer|FLDMamba|S-Mamba)\s+\|\s+(PEMS04|ECL)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+([0-9.]+)\s+\|\s+(\S+)",
        text,
        flags=re.M,
    ):
        rows.append({
            "model": m.group(1),
            "dataset": m.group(2),
            "noise": float(m.group(3)),
            "mse": float(m.group(4)),
            "mae": float(m.group(5)),
            "status": m.group(6),
        })
    # de-dup keep last
    uniq = {}
    for r in rows:
        uniq[(r["model"], r["dataset"], r["noise"])] = r
    return list(uniq.values())


def main():
    parser = argparse.ArgumentParser(description="Robustness noise study")
    parser.add_argument("--datasets", type=str, default="PEMS04,ECL")
    parser.add_argument(
        "--models",
        type=str,
        default="CSMamba,iTransformer,Mamba_fft,S_Mamba",
        help="comma-separated CLI model names",
    )
    parser.add_argument("--noise_levels", type=str, default="0,5,10,15,20")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=str, default="results/robustness_noise.txt")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument("--des", type=str, default="Robust")
    parser.add_argument("--class_strategy", type=str, default="projection")
    parser.add_argument("--skip_train", action="store_true")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="keep already finished rows from output txt (skip those model/dataset/noise)",
    )
    args = parser.parse_args()
    args.distil = True

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    noise_levels = [float(x) for x in args.noise_levels.split(",") if x.strip()]

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

    rows = load_existing_csmamba_rows(out_path) if args.reuse_existing else []
    done = {(r["model"], r["dataset"], r["noise"]) for r in rows if r["status"] == "ok"}

    for model in models:
        display = MODEL_DISPLAY[model]
        for dataset in datasets:
            preset = DATASET_PRESETS[dataset]
            setting = setting_name(args, model, dataset, preset)
            ckpt = ROOT / "checkpoints" / setting / "checkpoint.pth"

            need_train = not args.skip_train
            # If all noise levels already done for this pair, skip entirely
            if all((display, dataset, n) in done for n in noise_levels):
                print(f"[skip] {display} / {dataset} already complete")
                continue

            if need_train and not ((display, dataset, 0.0) in done and ckpt.exists()):
                print("=" * 80)
                print(f"[train] {display} / {dataset}")
                print("=" * 80)
                cmd = build_cmd(args, model, dataset, preset, is_training=1, noise_percent=0.0)
                print(" ".join(cmd))
                log_path = out_path.with_suffix(f".{display}.{dataset}.train.log")
                t0 = time.time()
                mse, mae, status = run_one(cmd, log_path)
                print(f"train done in {time.time()-t0:.1f}s -> mse={mse} status={status}")
                if 0.0 in noise_levels:
                    row = {
                        "model": display,
                        "dataset": dataset,
                        "noise": 0.0,
                        "mse": mse,
                        "mae": mae,
                        "status": status,
                    }
                    rows = [r for r in rows if not (r["model"] == display and r["dataset"] == dataset and r["noise"] == 0.0)]
                    rows.append(row)
                    done.add((display, dataset, 0.0))
                    write_result_file(out_path, args, rows, started_at)
            elif not ckpt.exists() and args.skip_train:
                print(f"[warn] missing ckpt for {display}/{dataset}: {ckpt}")

            test_levels = [
                n for n in noise_levels
                if not (n == 0.0 and need_train and (display, dataset, 0.0) in done)
            ]
            # always re-test non-zero; for zero if already recorded from train, skip
            test_levels = [n for n in test_levels if (display, dataset, n) not in done]

            for noise in test_levels:
                print("=" * 80)
                print(f"[test] {display} / {dataset} noise={noise:g}%")
                print("=" * 80)
                if not ckpt.exists():
                    rows.append({
                        "model": display,
                        "dataset": dataset,
                        "noise": noise,
                        "mse": None,
                        "mae": None,
                        "status": "no_ckpt",
                    })
                    write_result_file(out_path, args, rows, started_at)
                    continue
                cmd = build_cmd(args, model, dataset, preset, is_training=0, noise_percent=noise)
                print(" ".join(cmd))
                log_path = out_path.with_suffix(f".{display}.{dataset}.noise{noise:g}.log")
                mse, mae, status = run_one(cmd, log_path)
                rows.append({
                    "model": display,
                    "dataset": dataset,
                    "noise": noise,
                    "mse": mse,
                    "mae": mae,
                    "status": status,
                })
                done.add((display, dataset, noise))
                write_result_file(out_path, args, rows, started_at)
                print(make_long_table(rows))

    write_result_file(out_path, args, rows, started_at)
    print("Done.")
    print(f"Final table: {out_path}")
    print(make_pivot_table(rows))


if __name__ == "__main__":
    main()
