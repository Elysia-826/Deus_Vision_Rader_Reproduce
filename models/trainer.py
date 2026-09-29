"""真正调 Ultralytics 的地方。CLI / 测试都走这里，不要在别处 new YOLO()。

合并顺序：configs/*.yaml 打底 → TrainJob 里的 CLI 覆盖盖上去。
这样港科大超参是默认，命令行只改你这次想动的项。
"""

from __future__ import annotations

from pathlib import Path

from models.data_yaml import load_cfg_overrides, package_file, resolve_data_yaml
from models.schema import TrainJob


def build_train_kwargs(job: TrainJob) -> dict[str, str | int | float | bool]:
    """拼 model.train(**kwargs)。单测只测这个，避免真的拉 GPU。"""
    recipe = job.recipe
    cfg_path = package_file(recipe.cfg_yaml)
    kwargs = load_cfg_overrides(cfg_path)
    data_yaml = resolve_data_yaml(job.data, recipe)
    kwargs["data"] = str(data_yaml)
    kwargs["epochs"] = job.resolved_epochs()
    kwargs["imgsz"] = job.resolved_imgsz()
    kwargs["device"] = job.device
    kwargs["project"] = str(job.project)
    kwargs["name"] = job.run_name
    kwargs["pretrained"] = job.pretrained
    kwargs["resume"] = job.resume
    # batch / workers 配方 yaml 里已有默认；只有 CLI 显式传了才覆盖
    if job.batch is not None:
        kwargs["batch"] = job.batch
    if job.workers is not None:
        kwargs["workers"] = job.workers
    return kwargs


def train(job: TrainJob) -> Path:
    """微调 YOLO26，返回 best.pt。ultralytics 延迟导入，没装也能跑单测。"""
    import os

    # 本机独立 CUDNN 和 PyTorch 自带版本冲突时，AMP / conv2d 会直接炸。
    os.environ["PATH"] = os.pathsep.join(
        part for part in os.environ.get("PATH", "").split(os.pathsep) if "CUDNN" not in part.upper()
    )
    from ultralytics import YOLO

    kwargs = build_train_kwargs(job)
    # YOLO("yolo26s.pt")：本地没有就下官方预训练，再按 data.yaml 的 nc 重建检测头
    model = YOLO(job.weights)
    results = model.train(**kwargs)
    save_dir = Path(str(results.save_dir))
    return save_dir / "weights" / "best.pt"
