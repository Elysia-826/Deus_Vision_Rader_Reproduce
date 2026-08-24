"""把 Ultralytics YOLO .pt 导出为 TensorRT .engine。填好 PT_PATH 后直接运行。"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# ========== 填这里 ==========
PT_PATH = r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\model\armor_best.pt"
ENGINE_PATH = ""  # 空则写到与 .pt 同目录、同名 .engine
IMGSZ = 192  # 装甲板 192，车辆 1280
DEVICE = 0
HALF = True  # TensorRT FP16
WORKSPACE = None  # GiB；None 用 Ultralytics 默认
# ===========================


def main() -> None:
    pt_path = Path(PT_PATH).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"pt 不存在: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"输入必须是 .pt: {pt_path}")

    out_path = Path(ENGINE_PATH).expanduser() if ENGINE_PATH else pt_path.with_suffix(".engine")
    print(f"导出 TensorRT engine: {pt_path} -> {out_path}")

    export_kwargs: dict[str, object] = {
        "format": "engine",
        "imgsz": IMGSZ,
        "device": DEVICE,
        "half": HALF,
    }
    if WORKSPACE is not None:
        export_kwargs["workspace"] = WORKSPACE

    exported = YOLO(str(pt_path)).export(**export_kwargs)
    exported_path = Path(str(exported)).resolve()
    if exported_path != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(out_path)
        print(f"已移动到: {out_path}")
    else:
        print(f"已保存: {exported_path}")


if __name__ == "__main__":
    main()
