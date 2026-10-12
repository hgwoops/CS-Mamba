#!/usr/bin/env python3
"""Scalability study vs dataset cardinality (number of variates).

Figure-style protocol:
  - datasets: PEMS04, PEMS08
  - batch_size fixed at 32
  - cardinality in {20%,40%,60%,80%,100%} of sensors/variates
  - metric: per-epoch training time (s)
  - models: CSMamba (Ours), Mamba_fft (FLDMamba), PatchTST

Example:
  python scripts/scalability_study.py --gpu 0
  python scripts/scalability_study.py --dry_run
"""

from __future__ import annotations

import argparse
import re
import subprocess
import sys
import time
from datetime import datetime
from pathlib import Path

import numpy as np


ROOT = Path(__file__).resolve().parents[1]
CACHE_DIR = ROOT / "dataset" / "PEMS" / "cardinality_cache"

MODEL_DISPLAY = {
    "CSMamba": "Ours",
    "Mamba_fft": "FLDMamba",
    "PatchTST": "PatchTST",
}

DATASET_PRESETS = {
    "PEMS04": {
        "root_path": "./dataset/PEMS/",
        "full_data_path": "PEMS04.npz",
        "data": "PEMS",
        "full_enc_in": 307,
        "seq_len": 96,
        "pred_len": 12,
        "use_norm": 0,
        "d_model": 512,
        "d_ff": 512,
        "d_state": 16,
        "e_layers": 2,
        "learning_rate": 0.0002,
        "dropout": 0.1,
    },
    "PEMS08": {
        "root_path": "./dataset/PEMS/",
        "full_data_path": "PEMS08.npz",
        "data": "PEMS",
        "full_enc_in": 170,
        "seq_len": 96,
        "pred_len": 12,
        "use_norm": 0,
        "d_model": 512,
        "d_ff": 512,
        "d_state": 16,
        "e_layers": 2,
        "learning_rate": 0.0002,
        "dropout": 0.1,
    },
}


def n_vars_for(full_n: int, cardinality_pct: float) -> int:
    return max(1, int(round(full_n * cardinality_pct / 100.0)))


def ensure_cardinality_npz(dataset: str, preset: dict, cardinality_pct: float):
    """Slice the first K% of variates into a cached npz; return (rel_path, n_vars)."""
    full_n = preset["full_enc_in"]
    n_vars = n_vars_for(full_n, cardinality_pct)
    CACHE_DIR.mkdir(parents=True, exist_ok=True)
    out_name = f"{dataset}_c{int(cardinality_pct)}.npz"
    out_path = CACHE_DIR / out_name
    if not out_path.exists():
        src = ROOT / "dataset" / "PEMS" / preset["full_data_path"]
        arr = np.load(src)["data"]  # (T, N, C)
        if arr.shape[1] < n_vars:
            raise RuntimeError(f"{src} has {arr.shape[1]} vars, need {n_vars}")
        subset = arr[:, :n_vars, :]
        np.savez_compressed(out_path, data=subset)
        print(f"[cache] wrote {out_path} shape={subset.shape}")
    return f"cardinality_cache/{out_name}", n_vars


def model_extras(model: str, preset: dict):
    if model == "Mamba_fft":
        return {"i_or_cos": 1, "base": preset["seq_len"]}
    return {"i_or_cos": 0, "base": 0}


def build_cmd(args, model, dataset, preset, cardinality_pct, data_path, n_vars):
    extras = model_extras(model, preset)
    model_id = f"{dataset}_c{int(cardinality_pct)}_{preset['seq_len']}_{preset['pred_len']}_scale"
    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", "1",
        "--root_path", preset["root_path"],
        "--data_path", data_path,
        "--model_id", model_id,
        "--model", model,
        "--data", preset["data"],
        "--features", "M",
        "--seq_len", str(preset["seq_len"]),
        "--label_len", str(args.label_len),
        "--pred_len", str(preset["pred_len"]),
        "--e_layers", str(preset["e_layers"]),
        "--d_layers", "1",
        "--enc_in", str(n_vars),
        "--dec_in", str(n_vars),
        "--c_out", str(n_vars),
        "--d_model", str(preset["d_model"]),
        "--d_ff", str(preset["d_ff"]),
        "--d_state", str(preset["d_state"]),
        "--n_heads", "8",
        "--factor", "1",
        "--embed", "timeF",
        "--learning_rate", str(preset["learning_rate"]),
        "--batch_size", str(args.batch_size),
        "--dropout", str(preset["dropout"]),
        "--use_norm", str(preset["use_norm"]),
        "--itr", "1",
        "--des", "Scale",
        "--class_strategy", "projection",
        "--i_or_cos", str(extras["i_or_cos"]),
        "--base", str(extras["base"]),
        "--gpu", str(args.gpu),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
        "--noise_percent", "0",
        "--patch_len", str(args.patch_len),
        "--stride", str(args.stride),
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


def parse_epoch_times(log_text: str):
    epoch_trains = [float(x) for x in re.findall(
        r"Epoch:\s*\d+\s+train time:\s*([0-9.]+)s", log_text
    )]
    avg = peak = None
    m = re.search(
        r"GPU cost \| fit:\s*([0-9.]+)s\s*\|\s*avg epoch:\s*([0-9.]+)s\s*"
        r"\|\s*peak allocated:\s*([0-9.]+)\s*GiB",
        log_text,
    )
    if m:
        avg = float(m.group(2))
        peak = float(m.group(3))
    steady = None
    if len(epoch_trains) >= 2:
        steady = sum(epoch_trains[1:]) / len(epoch_trains[1:])
    elif epoch_trains:
        steady = epoch_trains[0]
    if avg is None and epoch_trains:
        avg = sum(epoch_trains) / len(epoch_trains)
    return {
        "epoch_trains": epoch_trains,
        "avg_epoch_s": avg,
        "steady_epoch_s": steady,
        "peak_gib": peak,
    }


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
    headers = [
        "model", "dataset", "card%", "n_vars",
        "epoch_s", "steady_epoch_s", "peak_GiB", "status",
    ]
    str_rows = []
    for r in rows:
        str_rows.append([
            r["model"],
            r["dataset"],
            f'{r["cardinality"]:g}',
            str(r["n_vars"]),
            "-" if r["avg_epoch_s"] is None else f'{r["avg_epoch_s"]:.2f}',
            "-" if r["steady_epoch_s"] is None else f'{r["steady_epoch_s"]:.2f}',
            "-" if r["peak_gib"] is None else f'{r["peak_gib"]:.2f}',
            r["status"],
        ])
    return _ascii_table(headers, str_rows)


def make_pivot_table(rows):
    datasets, cards, models = [], [], []
    for r in rows:
        if r["dataset"] not in datasets:
            datasets.append(r["dataset"])
        if r["cardinality"] not in cards:
            cards.append(r["cardinality"])
        if r["model"] not in models:
            models.append(r["model"])
    cards = sorted(cards)
    lookup = {(r["dataset"], r["model"], r["cardinality"]): r for r in rows}

    blocks = []
    for dataset in datasets:
        headers = ["card%", "n_vars"] + [f"{m}_epoch_s" for m in models]
        str_rows = []
        for card in cards:
            sample = next(
                (lookup[(dataset, m, card)] for m in models if (dataset, m, card) in lookup),
                None,
            )
            n_vars = sample["n_vars"] if sample else "-"
            row = [f"{card:g}", str(n_vars)]
            for m in models:
                r = lookup.get((dataset, m, card))
                if r is None or r["steady_epoch_s"] is None:
                    row.append("-")
                else:
                    row.append(f'{r["steady_epoch_s"]:.2f}')
            str_rows.append(row)
        blocks.append(f"[{dataset}]\n" + _ascii_table(headers, str_rows))
    return "\n\n".join(blocks)


def write_result_file(path: Path, args, rows, started_at):
    with path.open("w", encoding="utf-8") as f:
        f.write("Scalability Study (training time vs cardinality of variates)\n")
        f.write("cardinality = fraction of sensors/variates kept (first K%)\n")
        f.write("steady_epoch_s: mean train time excluding epoch 1 (warmup)\n")
        f.write(f"batch_size: {args.batch_size}\n")
        f.write(f"models: {args.models}\n")
        f.write(f"datasets: {args.datasets}\n")
        f.write(f"cardinalities: {args.cardinalities}\n")
        f.write(f"train_epochs: {args.train_epochs}\n")
        f.write(f"started: {started_at}\n")
        f.write(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"gpu: {args.gpu}\n\n")
        f.write("=== Long table ===\n")
        f.write(make_long_table(rows))
        f.write("\n\n=== Pivot (steady epoch time, s) ===\n")
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
    parsed = parse_epoch_times(log_text)
    status = "ok"
    if proc.returncode != 0:
        status = f"fail({proc.returncode})"
    elif parsed["avg_epoch_s"] is None:
        status = "no_metric"
    return parsed, status


def load_existing_rows(path: Path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    for m in re.finditer(
        r"^(Ours|FLDMamba|PatchTST)\s+\|\s+(PEMS04|PEMS08)\s+\|\s+"
        r"([0-9.]+)\s+\|\s+(\d+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)",
        text,
        flags=re.M,
    ):
        avg = None if m.group(5) == "-" else float(m.group(5))
        steady = None if m.group(6) == "-" else float(m.group(6))
        peak = None if m.group(7) == "-" else float(m.group(7))
        rows.append({
            "model": m.group(1),
            "dataset": m.group(2),
            "cardinality": float(m.group(3)),
            "n_vars": int(m.group(4)),
            "avg_epoch_s": avg,
            "steady_epoch_s": steady,
            "peak_gib": peak,
            "status": m.group(8),
        })
    uniq = {}
    for r in rows:
        uniq[(r["model"], r["dataset"], r["cardinality"])] = r
    return list(uniq.values())


def main():
    parser = argparse.ArgumentParser(description="Scalability study vs cardinality")
    parser.add_argument("--datasets", type=str, default="PEMS04,PEMS08")
    parser.add_argument(
        "--models",
        type=str,
        default="CSMamba,Mamba_fft,PatchTST",
        help="CSMamba=Ours, Mamba_fft=FLDMamba, PatchTST",
    )
    parser.add_argument("--cardinalities", type=str, default="20,40,60,80,100")
    parser.add_argument("--batch_size", type=int, default=32)
    parser.add_argument("--train_epochs", type=int, default=3)
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--patch_len", type=int, default=16)
    parser.add_argument("--stride", type=int, default=8)
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--output", type=str, default="results/scalability.txt")
    parser.add_argument("--dry_run", action="store_true")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="keep finished rows in output and skip those model/dataset/card runs",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    cards = [float(x) for x in args.cardinalities.split(",") if x.strip()]

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

    rows = load_existing_rows(out_path) if args.reuse_existing else []
    done = {
        (r["model"], r["dataset"], r["cardinality"])
        for r in rows if r["status"] == "ok"
    }

    planned = []
    for dataset in datasets:
        preset = DATASET_PRESETS[dataset]
        for card in cards:
            n_vars = n_vars_for(preset["full_enc_in"], card)
            for model in models:
                display = MODEL_DISPLAY[model]
                planned.append({
                    "model": display,
                    "dataset": dataset,
                    "cardinality": card,
                    "n_vars": n_vars,
                    "avg_epoch_s": None,
                    "steady_epoch_s": None,
                    "peak_gib": None,
                    "status": "planned",
                })
    print(f"Planned slots: {len(planned)} | already done: {len(done)}")
    print(make_long_table(planned))
    if args.dry_run:
        for dataset in datasets:
            preset = DATASET_PRESETS[dataset]
            for card in cards:
                ensure_cardinality_npz(dataset, preset, card)
        write_result_file(out_path, args, planned, started_at)
        print(f"Dry-run table written to {out_path}")
        return

    for dataset in datasets:
        preset = DATASET_PRESETS[dataset]
        for card in cards:
            data_path, n_vars = ensure_cardinality_npz(dataset, preset, card)
            for model in models:
                display = MODEL_DISPLAY[model]
                if (display, dataset, card) in done:
                    print(f"[skip] {display} / {dataset} / {card:g}%")
                    continue
                print("=" * 80)
                print(f"[scale] {display} / {dataset} / {card:g}% ({n_vars} vars)")
                print("=" * 80)
                cmd = build_cmd(args, model, dataset, preset, card, data_path, n_vars)
                print(" ".join(cmd))
                log_path = out_path.with_suffix(
                    f".{display}.{dataset}.c{int(card)}.log"
                )
                t0 = time.time()
                parsed, status = run_one(cmd, log_path)
                print(
                    f"done in {time.time()-t0:.1f}s -> "
                    f"steady={parsed['steady_epoch_s']} status={status}"
                )
                rows.append({
                    "model": display,
                    "dataset": dataset,
                    "cardinality": card,
                    "n_vars": n_vars,
                    "avg_epoch_s": parsed["avg_epoch_s"],
                    "steady_epoch_s": parsed["steady_epoch_s"],
                    "peak_gib": parsed["peak_gib"],
                    "status": status,
                })
                done.add((display, dataset, card))
                write_result_file(out_path, args, rows, started_at)
                print(make_pivot_table(rows))

    write_result_file(out_path, args, rows, started_at)
    print("Done.")
    print(f"Final table: {out_path}")


if __name__ == "__main__":
    main()
