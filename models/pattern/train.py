"""第三级 CLI。必须显式给 --data，避免空命令误开训。

必须在仓库根目录：

    python -m models.pattern.train --help
    python -m models.pattern.train --data "C:/Users/YQS/Downloads/OneDrive_3_2026-8-21/RM2025-Armor-Pattern-Public-Dataset"
"""

from __future__ import annotations

from pathlib import Path

import typer

from models.pattern.errors import PatternDatasetError
from models.pattern.schema import DEFAULT_PATTERN_DATA, PatternTrainJob
from models.pattern.trainer import train

app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    data: Path = typer.Option(
        ...,
        "--data",
        help=f"公开集根目录或 train/val。例: {DEFAULT_PATTERN_DATA}",
    ),
    epochs: int = typer.Option(100, help="默认 100，对齐港科大"),
    batch: int = typer.Option(32, help="默认 32"),
    imgsz: int = typer.Option(64, help="默认 64"),
    learning_rate: float = typer.Option(0.0003, "--lr", help="Adam lr"),
    val_ratio: float = typer.Option(0.2, help="类文件夹布局时的分层验证比例"),
    seed: int = typer.Option(42, help="split / 增强种子"),
    device: str = typer.Option("0", help="cuda 编号 / cpu"),
    project: Path = typer.Option(Path("runs/pattern"), help="输出根目录"),
    name: str = typer.Option("mobilenetv3-small", help="本次 run 名"),
    workers: int | None = typer.Option(None, help="DataLoader workers；默认 Windows=0"),
    pretrained: bool = typer.Option(True, help="ImageNet 预训练"),
    class_weight: bool = typer.Option(True, help="inverse-frequency 类权重"),
) -> None:
    """训 MobileNetV3-Small 图案分类器。确认参数后再带 --data 跑。"""
    job = PatternTrainJob(
        data=data,
        epochs=epochs,
        batch=batch,
        imgsz=imgsz,
        learning_rate=learning_rate,
        val_ratio=val_ratio,
        seed=seed,
        device=device,
        project=project,
        name=name,
        workers=workers,
        pretrained=pretrained,
        use_class_weight=class_weight,
    )
    try:
        best = train(job)
    except PatternDatasetError as exc:
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.secho(f"best weights: {best}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
