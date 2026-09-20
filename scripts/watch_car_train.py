"""Snapshot the live car YOLO run. Agent uses this after a 30-minute sleep."""

from __future__ import annotations

import argparse
import csv
import subprocess
import time
from dataclasses import dataclass
from datetime import datetime, timedelta
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
RUNS = ROOT / "runs" / "detect"
HISTORY = RUNS / "watch_history.txt"
TARGET_EPOCHS = 500
PATIENCE = 100
MAP_KEY = "metrics/mAP50-95(B)"


@dataclass(frozen=True, slots=True)
class Row:
    epoch: int
    elapsed_s: float
    precision: float
    recall: float
    map50: float
    map5095: float
    box_loss: float
    cls_loss: float


def newest_results_csv() -> Path:
    found = sorted(RUNS.glob("*/results.csv"), key=lambda p: p.stat().st_mtime)
    if not found:
        raise FileNotFoundError(f"no results.csv under {RUNS}")
    return found[-1]


def load_rows(path: Path) -> list[Row]:
    text = path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    rows: list[Row] = []
    reader = csv.DictReader(text.splitlines())
    for raw in reader:
        try:
            rows.append(
                Row(
                    epoch=int(float(raw["epoch"])),
                    elapsed_s=float(raw["time"]),
                    precision=float(raw["metrics/precision(B)"]),
                    recall=float(raw["metrics/recall(B)"]),
                    map50=float(raw["metrics/mAP50(B)"]),
                    map5095=float(raw[MAP_KEY]),
                    box_loss=float(raw["train/box_loss"]),
                    cls_loss=float(raw["train/cls_loss"]),
                )
            )
        except (KeyError, ValueError):
            continue
    return rows


def python_command_lines() -> list[str]:
    completed = subprocess.run(
        [
            "powershell",
            "-NoProfile",
            "-Command",
            "Get-CimInstance Win32_Process -Filter \"Name='python.exe'\" "
            "| Select-Object -ExpandProperty CommandLine",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    return [line.strip() for line in completed.stdout.splitlines() if line.strip()]


def training_alive() -> bool:
    needles = ("models.run_car_train", "models.run_car_train", "ultralytics")
    return any(any(n in line for n in needles) for line in python_command_lines())


def gpu_line() -> str:
    completed = subprocess.run(
        [
            "nvidia-smi",
            "--query-gpu=name,utilization.gpu,memory.used,memory.total",
            "--format=csv,noheader",
        ],
        capture_output=True,
        text=True,
        check=False,
    )
    if completed.returncode != 0:
        return "GPU: nvidia-smi unavailable"
    return "GPU: " + completed.stdout.strip()


def sec_per_epoch(rows: list[Row]) -> float | None:
    if len(rows) < 2:
        return None
    window = rows[-11:] if len(rows) > 11 else rows
    dt = window[-1].elapsed_s - window[0].elapsed_s
    de = window[-1].epoch - window[0].epoch
    if de <= 0 or dt <= 0:
        return None
    return dt / de


def render(run_dir: Path, rows: list[Row]) -> str:
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    alive = training_alive()
    lines = [
        f"[{now}] car train watch",
        f"run: {run_dir.name}",
        f"process: {'ALIVE' if alive else 'STOPPED'}",
        gpu_line(),
    ]
    if not rows:
        lines.append("results.csv empty / not started")
        return "\n".join(lines)

    last = rows[-1]
    best = max(rows, key=lambda r: r.map5095)
    spe = sec_per_epoch(rows)
    since_best = last.epoch - best.epoch
    lines += [
        f"epoch: {last.epoch} / {TARGET_EPOCHS}  ({last.epoch / TARGET_EPOCHS:.0%})",
        f"last:  P {last.precision:.4f}  R {last.recall:.4f}  mAP50 {last.map50:.4f}  mAP50-95 {last.map5095:.4f}",
        f"best:  epoch {best.epoch}  mAP50 {best.map50:.4f}  mAP50-95 {best.map5095:.4f}  (stale {since_best} ep, patience {PATIENCE})",
        f"loss:  box {last.box_loss:.4f}  cls {last.cls_loss:.4f}",
        f"wall:  {timedelta(seconds=int(last.elapsed_s))}",
    ]
    if spe is not None:
        remain = max(0, TARGET_EPOCHS - last.epoch)
        eta = timedelta(seconds=int(remain * spe))
        lines.append(f"pace:  {spe:.0f} s/epoch  ETA if 500: {eta}")
    if not alive:
        lines.append("NOTE: training process not found")
    elif since_best >= PATIENCE:
        lines.append("NOTE: patience window exhausted, may early-stop soon")
    return "\n".join(lines)


def snapshot(run: Path | None) -> str:
    csv_path = run / "results.csv" if run is not None else newest_results_csv()
    report = render(csv_path.parent, load_rows(csv_path))
    HISTORY.parent.mkdir(parents=True, exist_ok=True)
    with HISTORY.open("a", encoding="utf-8") as fh:
        fh.write(report + "\n---\n")
    latest = RUNS / "watch_latest.txt"
    latest.write_text(report + "\n", encoding="utf-8")
    return report


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Snapshot car YOLO training")
    parser.add_argument("--run", type=Path, default=None, help="specific run dir")
    parser.add_argument("--loop", type=int, default=0, metavar="SECONDS", help="repeat every N seconds")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    while True:
        print(snapshot(args.run), flush=True)
        if args.loop <= 0:
            return
        time.sleep(args.loop)


if __name__ == "__main__":
    main()
