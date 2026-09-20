"""把 Ultralytics YOLO .pt 导出为 .onnx。填好 PT_PATH 后直接运行。"""

from __future__ import annotations

from pathlib import Path

from ultralytics import YOLO

# ========== 填这里 ==========
PT_PATH = r"C:\Users\YQS\Desktop\DEUS_VISION_RADER_TEST_reproduce\weights\armor_best.pt"
ONNX_PATH = ""  # 空则写到与 .pt 同目录、同名 .onnx
IMGSZ = 192  # 装甲板 192，车辆 1280
DEVICE = 0
OPSET = 17
SIMPLIFY = True
HALF = False  # ONNX 一般先出 FP32；要 FP16 再改 True
# ===========================


def main() -> None:
    pt_path = Path(PT_PATH).expanduser().resolve()
    if not pt_path.is_file():
        raise FileNotFoundError(f"pt 不存在: {pt_path}")
    if pt_path.suffix.lower() != ".pt":
        raise ValueError(f"输入必须是 .pt: {pt_path}")

    out_path = Path(ONNX_PATH).expanduser() if ONNX_PATH else pt_path.with_suffix(".onnx")
    print(f"导出 ONNX: {pt_path} -> {out_path}")

    exported = YOLO(str(pt_path)).export(
        format="onnx",
        imgsz=IMGSZ,
        device=DEVICE,
        opset=OPSET,
        simplify=SIMPLIFY,
        half=HALF,
    )
    exported_path = Path(str(exported)).resolve()
    if exported_path != out_path.resolve():
        out_path.parent.mkdir(parents=True, exist_ok=True)
        exported_path.replace(out_path)
        print(f"已移动到: {out_path}")
    else:
        print(f"已保存: {exported_path}")


if __name__ == "__main__":
    main()
