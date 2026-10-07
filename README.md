# Deus Vision Radar Reproduce

港科大 ENTERPRIZE [RM2025-Radar-Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm) 单目雷达复现。骨干 YOLOv12 → YOLO26，图案 MobileNet-V2 → EfficientNet-B0。

当前能跑：车辆 YOLO → 装甲 YOLO → 图案分类；仿真射线定位或上场四点单应；相机画面内嵌小地图。

操作步骤见 [docs/radar_operation.md](docs/radar_operation.md)。

还没做：mesh 射线、级联跟踪、海康拉流、裁判串口。

## 目录

```
detect/            车 → 装甲推理（BoT-SORT 在 detect/trackers/）
models/            YOLO26 + EfficientNet-B0 训练
locate/            像素 → 场地米制 → 俯视图（射线 / 地标单应）
scripts/           可视化、仿真启动、标定
tools/             pt → onnx / TensorRT
tests/             pytest
docs/              操作手册
dataset/           划分脚本（图像不进 git）
weights/           本地权重（不进 git）
scene/             RMUC2026 Blender 仿真（CAD/贴图不进 git）
```

## 和港科大的差异

| | 港科大 | 本仓 |
|---|---|---|
| 车检测 | YOLOv12-s @ 1280 | YOLO26s @ 1280 |
| 装甲检测 | YOLOv12-n @ 192 | YOLO26n @ 192 |
| 图案 | MobileNet-V2，带颜色 | EfficientNet-B0 @ 96，6 类 `(1,2,3,4,S,Q)` |
| 上场定位 | 像素射线打场地 mesh | 仿真射线；真机四点地标单应 |

## 命令（仓库根目录）

```bash
python -m models.train car --data models/configs/data_car.yaml
python -m models.train armor --data models/configs/data_armor.yaml
python scripts/visualize_two_stage.py
python scripts/visualize_minimap.py
python scripts/run_radar.py
python scripts/calibrate_landmarks.py
pytest
```

Windows 仿真：

```bat
scripts\start_sim.cmd
```

小地图在 OpenCV 窗口 `radar` 右下角。上场把 `scripts/run_radar.py` 的 `LOCATE_MODE` 改成 `homography`，先跑 `calibrate_landmarks.py`。

权重从 `weights/` 读，不要提交。
