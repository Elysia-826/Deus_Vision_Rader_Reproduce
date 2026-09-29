"""MobileNetV3-Small 微调循环。torch 只在 train() 里进。"""

from __future__ import annotations

import sys
from pathlib import Path
from typing import TYPE_CHECKING

from models.pattern.classes import CLASS_NAMES, PatternId

if TYPE_CHECKING:
    import torch
    from torch.utils.data import DataLoader
from models.pattern.samples import Sample, load_split, resolve_pattern_root
from models.pattern.schema import PatternTrainJob


def inverse_frequency_weights(counts: tuple[int, ...]) -> tuple[float, ...]:
    """total / (C * n_c)。0 样本类权重为 0。"""
    total = sum(counts)
    n_classes = len(counts)
    return tuple(0.0 if count == 0 else total / (n_classes * count) for count in counts)


def count_by_pattern(samples: tuple[Sample, ...]) -> tuple[int, ...]:
    counts = [0] * len(PatternId)
    for item in samples:
        counts[int(item.pattern)] += 1
    return tuple(counts)


def resolved_workers(workers: int | None) -> int:
    if workers is not None:
        return workers
    return 0 if sys.platform == "win32" else 4


def train(job: PatternTrainJob) -> Path:
    """返回 best checkpoint 路径。不在 import 时拉 GPU。"""
    import os

    # 本机 NVIDIA CUDNN 与 PyTorch 自带版本冲突时，从 PATH 拿掉独立 CUDNN。
    os.environ["PATH"] = os.pathsep.join(
        part for part in os.environ.get("PATH", "").split(os.pathsep) if "CUDNN" not in part.upper()
    )
    import torch
    from torch import nn
    from torch.optim import Adam
    from torch.optim.lr_scheduler import ReduceLROnPlateau
    from torch.utils.data import DataLoader

    from models.pattern.dataset import PatternDataset
    from models.pattern.model import PatternBackbone, build_classifier
    from models.pattern.transforms import train_transform, val_transform

    root = resolve_pattern_root(job.data)
    train_samples, val_samples = load_split(root, val_ratio=job.val_ratio, seed=job.seed)
    device = _torch_device(job.device)
    model = build_classifier(
        job.backbone,
        pretrained=job.pretrained and job.init_weights is None,
    ).to(device)
    if job.init_weights is not None:
        ckpt = torch.load(job.init_weights, map_location=device, weights_only=False)
        state = ckpt["model_state_dict"] if isinstance(ckpt, dict) and "model_state_dict" in ckpt else ckpt
        model.load_state_dict(state)
    counts = count_by_pattern(train_samples)
    criterion: nn.Module
    if job.use_class_weight:
        weight = torch.tensor(inverse_frequency_weights(counts), dtype=torch.float32, device=device)
        criterion = nn.CrossEntropyLoss(weight=weight)
    else:
        criterion = nn.CrossEntropyLoss()
    optimizer = Adam(model.parameters(), lr=job.learning_rate, weight_decay=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, mode="min", factor=0.3, patience=4)

    workers = resolved_workers(job.workers)
    train_loader = DataLoader(
        PatternDataset(train_samples, train_transform(job.imgsz)),
        batch_size=job.batch,
        shuffle=True,
        num_workers=workers,
    )
    val_loader = DataLoader(
        PatternDataset(val_samples, val_transform(job.imgsz)),
        batch_size=job.batch,
        shuffle=False,
        num_workers=workers,
    )

    run_dir = job.project / job.name
    ckpt_dir = run_dir / "models"
    ckpt_dir.mkdir(parents=True, exist_ok=True)
    ckpt_name = (
        "best_efficientnet_b0.pth"
        if job.backbone is PatternBackbone.EFFICIENTNET_B0
        else "best_mobilenetv3_small.pth"
    )
    best_path = ckpt_dir / ckpt_name
    best_acc = -1.0

    print(f"root={root} train={len(train_samples)} val={len(val_samples)} device={device}")
    print("train counts:", dict(zip(CLASS_NAMES, counts, strict=True)))

    for epoch in range(job.epochs):
        train_loss, train_acc = _run_epoch(model, train_loader, criterion, device, optimizer)
        val_loss, val_acc = _run_epoch(model, val_loader, criterion, device, None)
        scheduler.step(val_loss)
        print(
            f"epoch {epoch + 1}/{job.epochs}  "
            f"train loss {train_loss:.4f} acc {train_acc:.4f}  "
            f"val loss {val_loss:.4f} acc {val_acc:.4f}"
        )
        if val_acc > best_acc:
            best_acc = val_acc
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "best_val_acc": best_acc,
                    "class_names": CLASS_NAMES,
                    "imgsz": job.imgsz,
                    "backbone": job.backbone,
                },
                best_path,
            )
            print(f"saved best val_acc={best_acc:.4f} -> {best_path}")
    return best_path


def _torch_device(name: str) -> torch.device:
    import torch

    if name == "cpu":
        return torch.device("cpu")
    if name.isdigit():
        return torch.device(f"cuda:{name}")
    return torch.device(name)


def _run_epoch(
    model: torch.nn.Module,
    loader: torch.utils.data.DataLoader[tuple[torch.Tensor, int]],
    criterion: torch.nn.Module,
    device: torch.device,
    optimizer: torch.optim.Optimizer | None,
) -> tuple[float, float]:
    import torch

    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    correct = 0
    seen = 0
    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for images, targets in loader:
            images = images.to(device)
            targets = targets.to(device)
            if optimizer is not None:
                optimizer.zero_grad()
            logits = model(images)
            loss = criterion(logits, targets)
            if optimizer is not None:
                loss.backward()
                optimizer.step()
            total_loss += float(loss.item())
            predicted = logits.argmax(dim=1)
            correct += int((predicted == targets).sum().item())
            seen += int(targets.size(0))
    return total_loss / max(len(loader), 1), correct / max(seen, 1)
