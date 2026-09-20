"""本机车检测一次启动器。正规入口仍是 python -m models.train car。

NVIDIA CUDNN v9.22 on PATH mismatches PyTorch's bundled 9.2 — strip it.
batch=2 left the 8GB 5070 ~40% busy; batch=4 + more workers is the speed fix.
"""
from __future__ import annotations

import os
from pathlib import Path

os.environ["PATH"] = os.pathsep.join(
    part for part in os.environ.get("PATH", "").split(os.pathsep) if "CUDNN" not in part.upper()
)

import torch
from ultralytics import YOLO

from models.schema import TrainJob
from models.stages import DetectStage
from models.trainer import build_train_kwargs

ROOT = Path(__file__).resolve().parents[1]


def main() -> None:
    torch.backends.cudnn.benchmark = True
    job = TrainJob(
        stage=DetectStage.CAR,
        data=ROOT / "models" / "configs" / "data_car.yaml",
        model=str(ROOT / "weights" / "yolo26s.pt"),
        device="0",
        batch=4,
        workers=6,
        project=ROOT / "runs" / "detect",
        name="car-yolo26s-fast",
    )
    if "yolo26s" not in job.weights.replace("\\", "/").split("/")[-1]:
        raise RuntimeError("car stage must stay on yolo26s")
    kwargs = build_train_kwargs(job)
    kwargs["batch"] = 4
    kwargs["workers"] = 6
    kwargs["deterministic"] = False
    kwargs["plots"] = False
    kwargs["cache"] = "disk"
    model = YOLO(job.weights)
    results = model.train(**kwargs)
    best = Path(str(results.save_dir)) / "weights" / "best.pt"
    print(f"best weights: {best}")


if __name__ == "__main__":
    main()
