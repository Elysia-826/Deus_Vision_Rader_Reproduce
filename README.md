# Deus Vision Radar Reproduce

港科大 ENTERPRIZE [RM2025-Radar-Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm) 单目雷达复现。骨干 YOLOv12 → YOLO26。当前在 `main`，图案是 MobileNetV3-Small @ 64。

当前能跑：车辆 YOLO → 装甲 YOLO → 图案分类；仿真射线定位或上场四点单应；相机画面内嵌小地图；固定兵种槽级联跟踪（身份历史投票 + 转弯自适应过程噪声）。

操作步骤见 [docs/radar_operation.md](docs/radar_operation.md)。

## 分支

- `main`：上场图案是 MobileNetV3-Small @ 64，权重 `weights/pattern_best.engine`。
- `pattern-efficientnet-b0`：图案对照实验，EfficientNet-B0 @ 96，权重 `weights/pattern_efficientnet_b0.engine`。检测链其余部分与 main 相同。

还没做：海康拉流、裁判串口。

## 目录

```
detect/            车 → 装甲推理（BoT-SORT 在 detect/trackers/）
models/            YOLO26 + MobileNetV3 训练
locate/            像素 → 场地米制 → 俯视图（射线 / 地标单应 / PnP 标定）
referee/           裁判串口协议骨架 + stub 裁判（RM2026 V1.3.0，未接真机）
track/             槽位级联跟踪 + 战略猜点
scripts/           可视化、仿真启动、标定、打分
tools/             pt → onnx / TensorRT
tests/             pytest
docs/              操作手册
dataset/           划分脚本（图像不进 git）
weights/           本地权重（不进 git）
scene/             RMUC2026 Blender 仿真（CAD/贴图不进 git）
refs/              官方协议/规则手册本地留档（不进 git）
```

## 和港科大的差异

| | 港科大 | 本仓 |
|---|---|---|
| 车检测 | YOLOv12-s @ 1280 | YOLO26s @ 1280 |
| 装甲检测 | YOLOv12-n @ 192 | YOLO26n @ 192 |
| 图案 | MobileNet-V2，带颜色 | `main`：MobileNetV3-Small @ 64；`pattern-efficientnet-b0`：EfficientNet-B0 @ 96。都是 6 类 `(1,2,3,4,S,Q)` |
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

打分（验收与回归）：

```bash
python scripts/eval_foot_ray.py --weights <best.pt>     # 新车权重 1.2 m 底边射线验收
python scripts/score_identity_log.py [log.jsonl]        # 发布点 vs 真值：召回/误差/野点
python scripts/score_stage_funnel.py                    # GT→车→装甲→图案 分级漏斗
python scripts/score_referee.py [log.jsonl]              # stub 裁判基准：易伤秒/准确率/P
python scripts/validate_pnp.py                           # PnP 标定精度（噪声→端到端误差）
```

裁判 stub 基准（规则手册 5.6.6 计分公式的离线复刻）：

```bash
RADAR_REFEREE_STUB=1 RADAR_SEND_LOST=1 python scripts/run_radar.py   # 回放 + stub 打分
python scripts/score_referee.py                                       # 出易伤报告
```

质量门 A/B 开关：`RADAR_SEND_LOST=1`（LOST 外推也发 0x0305）、`RADAR_GUESS=1`
（LOST 坐标换战略猜点）。patrol 序列实测：外推 +67% 易伤秒（7.0→11.7s）；
猜点 −100%（短遮挡场景外推本来就准，固定战略点全错）——所以猜点默认关。

已知问题（stub 基准量化，按伤害排序）：

1. **装甲 YOLO 队伍颜色在仿真域 ~40% 误读**：真数据训练的模型在仿真渲染
   上红蓝灯条判读不稳（红蓝标签 273:106 严重偏红）。红蓝英雄/工程互吃观测
   是 E1/A1 误差的主因。下一攻坚：收割仿真装甲 ROI + GT 队伍标签，重训
   装甲 YOLO（管线同 pattern：收割 → 合并 → 训练 → 基准）。
2. 图案角色混淆集中在 2→1（42 次）和 3→1（21 次）@ conf 0.91——相邻
   数字在远距小 crop 上难分。投票器能吸收但拉低出生质量。
3. 无装甲观测 ~40%：影子喂最近的槽，真槽和幻影都靠它续命（S 幻影已由
   重训消灭，但影子机制仍在）。

已修复（2026-10 攻坚）：

- **图案模型 S 淞遗**：仿真兵种被高置信读成 S（91 次/200 帧 @ conf 0.94）
  → 渲染域收割 641 张 + DIGIT_N 40→200 重训 MobileNetV3 → S 淞遗归零，
  幻影哨兵槽消灭，正确读 75→203，conf 分离度显著改善。
- **仿真灯条爆白**：AgX 色调映射 + 自发光 ×40 把灯条去饱和成白（蓝车
  71% 被读红）→ Standard 视图变换 + 强度 6 + 纯净发射色 → 可见灯条
  饱和度恢复（B-R=±100），dead 率 50%→12%。
- 基准注意：新基线 7.1s（simfix 模型 + 修复后渲染），旧 11.7s 是 S 淞遗
  意外掩护出的虚高。

基准视频流：`generate_radar_car_sim.py --motion patrol --dt 0.1 [--spin 6.0]` 渲染 10 Hz 固定步长序列（每帧带真值 json），`run_radar.py` 用 `RADAR_SOURCE=images RADAR_HEADLESS=1 RADAR_IMAGES_DIR=... RADAR_IDENTITY_LOG=...` 离线回放打分。同一份序列两次运行逐字节一致。

Windows 仿真：

```bat
scripts\start_sim.cmd
```

小地图在 OpenCV 窗口 `radar` 右下角。上场把 `scripts/run_radar.py` 的 `LOCATE_MODE` 改成 `homography`，先跑 `calibrate_landmarks.py`。

权重从 `weights/` 读，不要提交。
