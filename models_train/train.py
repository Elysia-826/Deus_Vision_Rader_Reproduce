"""命令行入口：训练 RM2025 车辆 / 装甲板检测器，骨干 YOLO26。

必须在仓库根目录跑，保证 `import models` 能找到包：

    python -m models.train car --data models/datasets/car.yaml
    python -m models.train armor --data D:/rm/armor --imgsz 192 --batch 64
"""

from __future__ import annotations

from pathlib import Path

import typer

from models.schema import DatasetConfigError, TrainJob
from models.stages import DetectStage
from models.trainer import train

# 没给子命令就打印帮助，避免空跑进 Ultralytics
app = typer.Typer(add_completion=False, no_args_is_help=True)


@app.command()
def main(
    stage: DetectStage = typer.Argument(help="car=全图车辆, armor=ROI 装甲板"),
    data: Path = typer.Option(..., "--data", help="YOLO data.yaml，或含 images/labels 的数据集根目录"),
    model: str | None = typer.Option(None, help="覆盖权重，默认 car=yolo26s / armor=yolo26n"),
    epochs: int | None = typer.Option(None, help="覆盖轮数，默认 500"),
    imgsz: int | None = typer.Option(None, help="覆盖输入边长，默认 car=1280 / armor=192"),
    batch: int | None = typer.Option(None, help="batch，-1=AutoBatch"),
    device: str = typer.Option("0", help="cuda 编号 / cpu / 0,1"),
    project: Path = typer.Option(Path("runs/detect"), help="Ultralytics 工程目录"),
    name: str | None = typer.Option(None, help="project 下的本次 run 名"),
    resume: bool = typer.Option(False, help="从上次 checkpoint 续训"),
    workers: int | None = typer.Option(None, help="DataLoader worker 数"),
) -> None:
    """训一个 HKUST 风格检测阶段。Typer 会把 stage 字符串解析成 DetectStage。"""
    job = TrainJob(
        stage=stage,
        data=data,
        model=model,
        epochs=epochs,
        imgsz=imgsz,
        batch=batch,
        device=device,
        project=project,
        name=name,
        resume=resume,
        workers=workers,
    )
    try:
        best = train(job)
    except DatasetConfigError as exc:
        # 数据路径问题用退出码 2，和训练中途崩溃的 1 分开
        typer.secho(str(exc), fg=typer.colors.RED, err=True)
        raise typer.Exit(code=2) from exc
    typer.secho(f"best weights: {best}", fg=typer.colors.GREEN)


if __name__ == "__main__":
    app()
