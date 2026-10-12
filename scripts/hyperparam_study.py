#!/usr/bin/env python3
"""CSMamba hyperparameter study runner.

Runs a grid of configs sequentially, parses mse/mae/gpu cost from each run,
and writes an ASCII table to a txt file.

Example:
  python scripts/hyperparam_study.py --dataset PEMS04 --gpu 0
"""

from __future__ import annotations

import argparse
import itertools
import json
import os
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]

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
        "base_d_model": 1024,
        "base_d_ff": 1024,
        "base_lr": 0.0002,
        "base_e_layers": 1,
        "base_d_state": 16,
        "base_batch_size": 16,
        "base_dropout": 0.1,
    },
    "PEMS07": {
        "root_path": "./dataset/PEMS/",
        "data_path": "PEMS07.npz",
        "data": "PEMS",
        "seq_len": 96,
        "pred_len": 12,
        "enc_in": 883,
        "dec_in": 883,
        "c_out": 883,
        "use_norm": 0,
        "base_d_model": 512,
        "base_d_ff": 512,
        "base_lr": 0.0002,
        "base_e_layers": 1,
        "base_d_state": 16,
        "base_batch_size": 16,
        "base_dropout": 0.1,
    },
    "ETTh1": {
        "root_path": "./dataset/ETT-small/",
        "data_path": "ETTh1.csv",
        "data": "ETTh1",
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 7,
        "dec_in": 7,
        "c_out": 7,
        "use_norm": 1,
        "base_d_model": 256,
        "base_d_ff": 256,
        "base_lr": 0.0001,
        "base_e_layers": 2,
        "base_d_state": 16,
        "base_batch_size": 32,
        "base_dropout": 0.1,
    },
    "weather": {
        "root_path": "./dataset/weather/",
        "data_path": "weather.csv",
        "data": "custom",
        "seq_len": 96,
        "pred_len": 96,
        "enc_in": 21,
        "dec_in": 21,
        "c_out": 21,
        "use_norm": 1,
        "base_d_model": 512,
        "base_d_ff": 512,
        "base_lr": 0.00005,
        "base_e_layers": 1,
        "base_d_state": 8,
        "base_batch_size": 32,
        "base_dropout": 0.1,
    },
}


def parse_list(text: str, cast=float):
    return [cast(x.strip()) for x in text.split(",") if x.strip()]


def build_grid(args, preset):
    """One-factor-at-a-time grid around the dataset baseline."""
    base = {
        "d_model": preset["base_d_model"],
        "d_ff": preset["base_d_ff"],
        "d_state": preset["base_d_state"],
        "e_layers": preset["base_e_layers"],
        "learning_rate": preset["base_lr"],
        "batch_size": preset["base_batch_size"],
        "dropout": preset["base_dropout"],
    }

    if args.mode == "full":
        keys = ["d_model", "d_state", "e_layers", "learning_rate", "batch_size", "dropout"]
        values = [
            parse_list(args.d_model, int),
            parse_list(args.d_state, int),
            parse_list(args.e_layers, int),
            parse_list(args.learning_rate, float),
            parse_list(args.batch_size, int),
            parse_list(args.dropout, float),
        ]
        configs = []
        for combo in itertools.product(*values):
            cfg = dict(zip(keys, combo))
            cfg["d_ff"] = cfg["d_model"]
            configs.append(cfg)
        return configs

    # default: one-factor-at-a-time (OAT)
    configs = [dict(base)]
    sweeps = {
        "d_model": parse_list(args.d_model, int),
        "d_state": parse_list(args.d_state, int),
        "e_layers": parse_list(args.e_layers, int),
        "learning_rate": parse_list(args.learning_rate, float),
        "batch_size": parse_list(args.batch_size, int),
        "dropout": parse_list(args.dropout, float),
    }
    for key, candidates in sweeps.items():
        for val in candidates:
            if val == base[key]:
                continue
            cfg = dict(base)
            cfg[key] = val
            if key == "d_model":
                cfg["d_ff"] = val
            configs.append(cfg)
    return configs


def format_cfg_id(cfg):
    return (
        f"dm{cfg['d_model']}_ds{cfg['d_state']}_el{cfg['e_layers']}_"
        f"lr{cfg['learning_rate']}_bs{cfg['batch_size']}_do{cfg['dropout']}"
    )


def build_cmd(args, preset, cfg, run_id):
    model_id = f"{args.dataset}_{preset['seq_len']}_{preset['pred_len']}_hp_{run_id}"
    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", "1",
        "--root_path", preset["root_path"],
        "--data_path", preset["data_path"],
        "--model_id", model_id,
        "--model", "CSMamba",
        "--data", preset["data"],
        "--features", "M",
        "--seq_len", str(preset["seq_len"]),
        "--pred_len", str(preset["pred_len"]),
        "--e_layers", str(cfg["e_layers"]),
        "--enc_in", str(preset["enc_in"]),
        "--dec_in", str(preset["dec_in"]),
        "--c_out", str(preset["c_out"]),
        "--d_model", str(cfg["d_model"]),
        "--d_ff", str(cfg["d_ff"]),
        "--d_state", str(cfg["d_state"]),
        "--learning_rate", str(cfg["learning_rate"]),
        "--batch_size", str(cfg["batch_size"]),
        "--dropout", str(cfg["dropout"]),
        "--use_norm", str(preset["use_norm"]),
        "--itr", "1",
        "--des", "HP",
        "--gpu", str(args.gpu),
        "--use_dvr", "1",
        "--use_ms_scan", "0",
        "--use_freq_init", "0",
        "--use_bi_mamba", "0",
        "--use_time_marks", "0",
        "--use_tle", "1",
        "--use_trend_seasonal", "1",
        "--use_spectral_mod", "1",
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
    ]
    return cmd, model_id


def parse_metrics(log_text: str):
    mse = mae = None
    # Prefer the final test metric line written by Exp.test
    matches = list(re.finditer(r"mse:([0-9.eE+-]+),\s*mae:([0-9.eE+-]+)", log_text))
    if matches:
        mse = float(matches[-1].group(1))
        mae = float(matches[-1].group(2))

    fit_time = peak_alloc = None
    m = re.search(
        r"GPU cost \| fit:\s*([0-9.]+)s.*?peak allocated:\s*([0-9.]+)\s*GiB",
        log_text,
        flags=re.S,
    )
    if m:
        fit_time = float(m.group(1))
        peak_alloc = float(m.group(2))

    # fallback: read last gpu_cost.json if present
    return mse, mae, fit_time, peak_alloc


def find_latest_gpu_cost(model_id: str):
    results_dir = ROOT / "results"
    if not results_dir.exists():
        return None, None
    candidates = sorted(
        results_dir.glob(f"{model_id}*/gpu_cost.json"),
        key=lambda p: p.stat().st_mtime,
        reverse=True,
    )
    if not candidates:
        return None, None
    try:
        data = json.loads(candidates[0].read_text())
        return data.get("fit_time_s"), data.get("peak_allocated_gib")
    except Exception:
        return None, None


def make_table(rows):
    headers = [
        "id", "d_model", "d_state", "e_layers", "lr", "batch", "dropout",
        "mse", "mae", "fit_s", "peak_GiB", "status",
    ]
    str_rows = []
    for r in rows:
        str_rows.append([
            str(r["id"]),
            str(r["d_model"]),
            str(r["d_state"]),
            str(r["e_layers"]),
            f'{r["learning_rate"]:g}',
            str(r["batch_size"]),
            f'{r["dropout"]:g}',
            "-" if r["mse"] is None else f'{r["mse"]:.6f}',
            "-" if r["mae"] is None else f'{r["mae"]:.6f}',
            "-" if r["fit_s"] is None else f'{r["fit_s"]:.1f}',
            "-" if r["peak_GiB"] is None else f'{r["peak_GiB"]:.2f}',
            r["status"],
        ])

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
    best = None
    for r in rows:
        if r["mse"] is None:
            continue
        if best is None or r["mse"] < best["mse"]:
            best = r

    with path.open("w", encoding="utf-8") as f:
        f.write("CSMamba Hyperparameter Study\n")
        f.write(f"dataset: {args.dataset}\n")
        f.write(f"mode: {args.mode}\n")
        f.write(f"started: {started_at}\n")
        f.write(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"gpu: {args.gpu}\n")
        f.write("\n")
        f.write(make_table(rows))
        f.write("\n\n")
        if best is not None:
            f.write(
                "best by mse: "
                f"id={best['id']} d_model={best['d_model']} d_state={best['d_state']} "
                f"e_layers={best['e_layers']} lr={best['learning_rate']} "
                f"batch={best['batch_size']} dropout={best['dropout']} "
                f"mse={best['mse']:.6f} mae={best['mae']:.6f}\n"
            )
        else:
            f.write("best by mse: none\n")


def main():
    parser = argparse.ArgumentParser(description="CSMamba hyperparameter study")
    parser.add_argument("--dataset", type=str, default="PEMS04",
                        choices=sorted(DATASET_PRESETS.keys()))
    parser.add_argument("--mode", type=str, default="oat", choices=["oat", "full"],
                        help="oat=one-factor-at-a-time (recommended); full=cartesian product")
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=50)
    parser.add_argument("--patience", type=int, default=5)
    parser.add_argument("--output", type=str, default=None,
                        help="result txt path; default results/hyperparam_<dataset>_<time>.txt")

    # searchable values
    parser.add_argument("--d_model", type=str, default="512,1024")
    parser.add_argument("--d_state", type=str, default="8,16,32")
    parser.add_argument("--e_layers", type=str, default="1,2")
    parser.add_argument("--learning_rate", type=str, default="0.0001,0.0002,0.0005")
    parser.add_argument("--batch_size", type=str, default="8,16,32")
    parser.add_argument("--dropout", type=str, default="0.0,0.1")
    parser.add_argument("--dry_run", action="store_true",
                        help="only print planned configs, do not train")
    args = parser.parse_args()

    preset = DATASET_PRESETS[args.dataset]
    configs = build_grid(args, preset)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_path = Path(args.output) if args.output else ROOT / "results" / f"hyperparam_{args.dataset}_{stamp}.txt"
    out_path.parent.mkdir(parents=True, exist_ok=True)

    print(f"Dataset: {args.dataset}")
    print(f"Mode: {args.mode}")
    print(f"Total runs: {len(configs)}")
    print(f"Output: {out_path}")
    print(make_table([{
        "id": i + 1,
        **cfg,
        "mse": None,
        "mae": None,
        "fit_s": None,
        "peak_GiB": None,
        "status": "planned",
    } for i, cfg in enumerate(configs)]))
    print()

    if args.dry_run:
        write_result_file(out_path, args, [{
            "id": i + 1,
            **cfg,
            "mse": None,
            "mae": None,
            "fit_s": None,
            "peak_GiB": None,
            "status": "planned",
        } for i, cfg in enumerate(configs)], started_at)
        print(f"Dry-run table written to {out_path}")
        return

    rows = []
    for i, cfg in enumerate(configs, start=1):
        run_tag = format_cfg_id(cfg)
        cmd, model_id = build_cmd(args, preset, cfg, run_tag)
        print("=" * 80)
        print(f"[{i}/{len(configs)}] {run_tag}")
        print(" ".join(cmd))
        print("=" * 80)

        t0 = time.time()
        status = "ok"
        try:
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_text = proc.stdout or ""
            # also dump per-run log
            log_path = out_path.with_suffix(f".run{i:02d}.log")
            log_path.write_text(log_text, encoding="utf-8")

            mse, mae, fit_s, peak = parse_metrics(log_text)
            if fit_s is None or peak is None:
                fit2, peak2 = find_latest_gpu_cost(model_id)
                fit_s = fit_s if fit_s is not None else fit2
                peak = peak if peak is not None else peak2
            if proc.returncode != 0:
                status = f"fail({proc.returncode})"
            elif mse is None:
                status = "no_metric"
        except Exception as e:
            log_text = str(e)
            mse = mae = fit_s = peak = None
            status = "error"
            (out_path.with_suffix(f".run{i:02d}.log")).write_text(log_text, encoding="utf-8")

        elapsed = time.time() - t0
        row = {
            "id": i,
            **cfg,
            "mse": mse,
            "mae": mae,
            "fit_s": fit_s if fit_s is not None else elapsed,
            "peak_GiB": peak,
            "status": status,
        }
        rows.append(row)
        write_result_file(out_path, args, rows, started_at)
        print(make_table(rows))
        print(f"Checkpoint table saved: {out_path}\n")

    print("Done.")
    print(f"Final table: {out_path}")


if __name__ == "__main__":
    main()
