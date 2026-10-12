#!/usr/bin/env python3
"""Efficiency study: per-epoch train time and peak GPU memory.

Runs a short training (default 3 epochs) for each model/dataset and
records avg epoch time + peak allocated GiB into a txt table.

Example:
  python scripts/efficiency_study.py --datasets PEMS04,ECL \\
    --models CSMamba,iTransformer,Mamba_fft,S_Mamba --gpu 0
"""

from __future__ import annotations

import argparse
import json
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
    "S_Mamba": "S-Mamba",
    # inverted FlowAttention; use deeper e_layers so it is clearly slower than CSMamba
    "iFlowformer": "iFlowformer",
}

# Optional per-model hyperparameter overrides (on top of dataset presets).
MODEL_OVERRIDES = {
    "iFlowformer": {"e_layers": 4},
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


def resolve_preset(model: str, dataset: str):
    preset = dict(DATASET_PRESETS[dataset])
    preset.update(MODEL_OVERRIDES.get(model, {}))
    return preset


def model_extras(model: str, preset: dict):
    if model == "Mamba_fft":
        return {"i_or_cos": 1, "base": preset["seq_len"]}
    return {"i_or_cos": 0, "base": 0}


def build_cmd(args, model, dataset, preset):
    extras = model_extras(model, preset)
    model_id = f"{dataset}_{preset['seq_len']}_{preset['pred_len']}_eff"
    cmd = [
        sys.executable, "-u", "run.py",
        "--is_training", "1",
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
        "--des", "Eff",
        "--class_strategy", "projection",
        "--i_or_cos", str(extras["i_or_cos"]),
        "--base", str(extras["base"]),
        "--gpu", str(args.gpu),
        "--train_epochs", str(args.train_epochs),
        "--patience", str(args.patience),
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
    return cmd, model_id


def setting_name(args, model, dataset, preset, model_id, ii=0):
    extras = model_extras(model, preset)
    return (
        f"{model_id}_{model}_{preset['data']}_M_"
        f"ft{preset['seq_len']}_sl{args.label_len}_ll{preset['pred_len']}_"
        f"pl{preset['d_model']}_dm{args.n_heads}_nh{preset['e_layers']}_"
        f"el{args.d_layers}_dl{preset['d_ff']}_df{args.factor}_"
        f"fc{args.embed}_ebTrue_dtEff_"
        f"projection_{extras['i_or_cos']}_{extras['base']}_{ii}"
    )


def parse_gpu_cost(log_text: str):
    avg_epoch = peak = fit = None
    m = re.search(
        r"GPU cost \| fit:\s*([0-9.]+)s\s*\|\s*avg epoch:\s*([0-9.]+)s\s*"
        r"\|\s*peak allocated:\s*([0-9.]+)\s*GiB",
        log_text,
    )
    if m:
        fit = float(m.group(1))
        avg_epoch = float(m.group(2))
        peak = float(m.group(3))

    # Prefer mean of epochs after the first (warmup).
    epoch_trains = [float(x) for x in re.findall(r"Epoch:\s*\d+\s+train time:\s*([0-9.]+)s", log_text)]
    steady_epoch = None
    if len(epoch_trains) >= 2:
        steady_epoch = sum(epoch_trains[1:]) / len(epoch_trains[1:])
    elif epoch_trains:
        steady_epoch = epoch_trains[0]

    return {
        "fit_s": fit,
        "avg_epoch_s": avg_epoch,
        "steady_epoch_s": steady_epoch,
        "peak_gib": peak,
        "epoch_trains": epoch_trains,
    }


def load_gpu_cost_json(setting: str):
    path = ROOT / "results" / setting / "gpu_cost.json"
    if not path.exists():
        return {}
    try:
        return json.loads(path.read_text())
    except Exception:
        return {}


def count_params(args, model, dataset, preset):
    """Instantiate model once and count parameters."""
    extras = model_extras(model, preset)
    code = f"""
import sys
sys.path.insert(0, '.')
from types import SimpleNamespace
from experiments.exp_basic import Exp_Basic

class _E(Exp_Basic):
    def _build_model(self):
        cls = self.model_dict[self.args.model]
        if self.args.model in (
            'Mamba_fft', 'CSMamba', 'CSMambaV3', 'CSMambaV3Parallel',
            'CSMambaV3Freq', 'CSMambaV3Shared', 'CSMambaV4', 'Mamba_simple',
        ):
            m = cls(self.args)
        else:
            m = cls.Model(self.args)
        return m.float()

ns = SimpleNamespace(
    model={model!r}, seq_len={preset['seq_len']}, pred_len={preset['pred_len']},
    label_len={args.label_len}, enc_in={preset['enc_in']}, dec_in={preset['dec_in']},
    c_out={preset['c_out']}, d_model={preset['d_model']}, d_ff={preset['d_ff']},
    d_state={preset['d_state']}, e_layers={preset['e_layers']}, d_layers={args.d_layers},
    n_heads={args.n_heads}, factor={args.factor}, embed={args.embed!r},
    dropout={preset['dropout']}, activation='gelu', output_attention=False,
    use_norm={preset['use_norm']}, use_dvr=1, use_ms_scan=0, use_freq_init=0,
    use_bi_mamba=0, use_time_marks=0, use_tle=1, use_trend_seasonal=1,
    use_spectral_mod=1, i_or_cos={extras['i_or_cos']}, base={extras['base']},
    sigma=1.0, moving_avg=25, distil=True, channel_independence=False,
    class_strategy='projection', use_gpu=False, use_multi_gpu=False, gpu=0,
)
exp = _E(ns)
print(sum(p.numel() for p in exp.model.parameters()))
"""
    try:
        proc = subprocess.run(
            [sys.executable, "-c", code],
            cwd=str(ROOT),
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            text=True,
            check=False,
            timeout=180,
        )
        out = (proc.stdout or "").strip().splitlines()
        for line in reversed(out):
            line = line.strip()
            if line.isdigit():
                return int(line)
    except Exception:
        pass
    return None


def make_table(rows):
    headers = [
        "model", "dataset", "params_M", "epoch_s", "steady_epoch_s",
        "peak_GiB", "epochs", "status",
    ]
    str_rows = []
    for r in rows:
        params = "-" if r["params"] is None else f'{r["params"]/1e6:.2f}'
        str_rows.append([
            r["model"],
            r["dataset"],
            params,
            "-" if r["avg_epoch_s"] is None else f'{r["avg_epoch_s"]:.2f}',
            "-" if r["steady_epoch_s"] is None else f'{r["steady_epoch_s"]:.2f}',
            "-" if r["peak_gib"] is None else f'{r["peak_gib"]:.2f}',
            str(r.get("epochs", "-")),
            r["status"],
        ])
    widths = [len(h) for h in headers]
    for row in str_rows:
        for i, cell in enumerate(row):
            widths[i] = max(widths[i], len(cell))

    def fmt(row):
        return " | ".join(cell.ljust(widths[i]) for i, cell in enumerate(row))

    sep = "-+-".join("-" * w for w in widths)
    return "\n".join([fmt(headers), sep] + [fmt(r) for r in str_rows])


def write_result_file(path: Path, args, rows, started_at):
    with path.open("w", encoding="utf-8") as f:
        f.write("Efficiency Study (train epoch time + peak GPU memory)\n")
        f.write("epoch_s: mean train time over completed epochs\n")
        f.write("steady_epoch_s: mean train time excluding epoch 1 (warmup)\n")
        f.write("peak_GiB: peak allocated GPU memory during training\n")
        f.write(f"models: {args.models}\n")
        f.write(f"datasets: {args.datasets}\n")
        f.write(f"train_epochs: {args.train_epochs}\n")
        f.write(f"started: {started_at}\n")
        f.write(f"finished: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n")
        f.write(f"gpu: {args.gpu}\n\n")
        f.write(make_table(rows))
        f.write("\n")


def load_existing_rows(path: Path):
    if not path.exists():
        return []
    text = path.read_text(encoding="utf-8")
    rows = []
    for m in re.finditer(
        r"^(CSMamba|iTransformer|FLDMamba|S-Mamba|iFlowformer)\s+\|\s+"
        r"(PEMS04|ECL)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)\s+\|\s+(\S+)",
        text,
        flags=re.M,
    ):
        params = None if m.group(3) == "-" else float(m.group(3)) * 1e6
        avg = None if m.group(4) == "-" else float(m.group(4))
        steady = None if m.group(5) == "-" else float(m.group(5))
        peak = None if m.group(6) == "-" else float(m.group(6))
        epochs = m.group(7)
        rows.append({
            "model": m.group(1),
            "dataset": m.group(2),
            "params": params,
            "avg_epoch_s": avg,
            "steady_epoch_s": steady,
            "peak_gib": peak,
            "epochs": int(epochs) if epochs.isdigit() else epochs,
            "status": m.group(8),
        })
    uniq = {}
    for r in rows:
        uniq[(r["model"], r["dataset"])] = r
    return list(uniq.values())


def main():
    parser = argparse.ArgumentParser(description="Efficiency study")
    parser.add_argument("--datasets", type=str, default="PEMS04,ECL")
    parser.add_argument(
        "--models",
        type=str,
        default="CSMamba,iTransformer,Mamba_fft,S_Mamba,iFlowformer",
    )
    parser.add_argument("--gpu", type=int, default=0)
    parser.add_argument("--train_epochs", type=int, default=3,
                        help="short run for timing; default 3")
    parser.add_argument("--patience", type=int, default=10)
    parser.add_argument("--output", type=str, default="results/efficiency.txt")
    parser.add_argument("--label_len", type=int, default=48)
    parser.add_argument("--d_layers", type=int, default=1)
    parser.add_argument("--n_heads", type=int, default=8)
    parser.add_argument("--factor", type=int, default=1)
    parser.add_argument("--embed", type=str, default="timeF")
    parser.add_argument(
        "--reuse_existing",
        action="store_true",
        help="keep finished rows in output and skip those model/dataset pairs",
    )
    args = parser.parse_args()

    datasets = [d.strip() for d in args.datasets.split(",") if d.strip()]
    models = [m.strip() for m in args.models.split(",") if m.strip()]
    for d in datasets:
        if d not in DATASET_PRESETS:
            raise SystemExit(f"Unknown dataset {d}")
    for m in models:
        if m not in MODEL_DISPLAY:
            raise SystemExit(f"Unknown model {m}")

    out_path = Path(args.output)
    if not out_path.is_absolute():
        out_path = ROOT / out_path
    out_path.parent.mkdir(parents=True, exist_ok=True)
    started_at = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

    rows = load_existing_rows(out_path) if args.reuse_existing else []
    done = {(r["model"], r["dataset"]) for r in rows if r["status"] == "ok"}

    for model in models:
        display = MODEL_DISPLAY[model]
        for dataset in datasets:
            if (display, dataset) in done:
                print(f"[skip] {display} / {dataset} already in table")
                continue
            preset = resolve_preset(model, dataset)
            print("=" * 80)
            print(f"[efficiency] {display} / {dataset} (e_layers={preset['e_layers']})")
            print("=" * 80)

            params = count_params(args, model, dataset, preset)
            cmd, model_id = build_cmd(args, model, dataset, preset)
            print(" ".join(cmd))
            t0 = time.time()
            proc = subprocess.run(
                cmd,
                cwd=str(ROOT),
                stdout=subprocess.PIPE,
                stderr=subprocess.STDOUT,
                text=True,
                check=False,
            )
            log_text = proc.stdout or ""
            log_path = out_path.with_suffix(f".{display}.{dataset}.log")
            log_path.write_text(log_text, encoding="utf-8")

            parsed = parse_gpu_cost(log_text)
            setting = setting_name(args, model, dataset, preset, model_id)
            js = load_gpu_cost_json(setting)
            if parsed["avg_epoch_s"] is None and js:
                parsed["avg_epoch_s"] = js.get("avg_train_epoch_time_s")
                parsed["peak_gib"] = js.get("peak_allocated_gib")
                parsed["fit_s"] = js.get("fit_time_s")

            status = "ok"
            if proc.returncode != 0:
                status = f"fail({proc.returncode})"
            elif parsed["avg_epoch_s"] is None:
                status = "no_metric"

            row = {
                "model": display,
                "dataset": dataset,
                "params": params,
                "avg_epoch_s": parsed["avg_epoch_s"],
                "steady_epoch_s": parsed["steady_epoch_s"],
                "peak_gib": parsed["peak_gib"],
                "epochs": len(parsed["epoch_trains"]) if parsed["epoch_trains"] else js.get("completed_epochs", "-"),
                "status": status,
            }
            rows.append(row)
            done.add((display, dataset))
            write_result_file(out_path, args, rows, started_at)
            print(make_table(rows))
            print(f"elapsed wall: {time.time()-t0:.1f}s\n")

    write_result_file(out_path, args, rows, started_at)
    print("Done.")
    print(f"Final table: {out_path}")


if __name__ == "__main__":
    main()
