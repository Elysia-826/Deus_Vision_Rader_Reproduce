# 雷达视觉操作手册

> 版本：v2.0 | 仓库：Deus Vision Radar Reproduce | 最后更新：2026-10
>
> 配套文档：
> [架构深讲](architecture.md) · [训练指南](training_guide.md) · [接手清单](handover_checklist.md) · [2027 迁移计划](migration_2027.md) · [裁判协议笔记](referee_protocol_notes.md) · [Docker 指南](../docker/README.md)

---

## 目录

1. [概述](#1-概述)
2. [准备](#2-准备)
3. [三分钟跑通](#3-三分钟跑通)
4. [仿真联调](#4-仿真联调)
5. [录像回放与打分跑批](#5-录像回放与打分跑批)
6. [上场标定 —— PnP（推荐）](#6-上场标定--pnp推荐)
7. [上场标定 —— 单应（备选）](#7-上场标定--单应备选)
8. [小地图标定](#8-小地图标定)
9. [接真实摄像头](#9-接真实摄像头)
10. [打分与诊断工具箱](#10-打分与诊断工具箱)
11. [stub 裁判基准](#11-stub-裁判基准)
12. [环境变量总表](#12-环境变量总表)
13. [故障排查表](#13-故障排查表)
- [附录 A：命令速查](#附录-a命令速查)

---

## 1. 概述

### 1.1 系统干什么

单目相机 → 三级检测 → 场地定位 → 槽位跟踪 → 裁判标记（0x0305）/ 哨兵引导（0x0301）。
检测和定位不绑定 Blender：图源换成录像或海康相机后，后面是同一条链。

```mermaid
graph LR
    SRC["图源<br/>blender/video/images/hik"] --> CAR["车 YOLO<br/>1280·conf0.45"]
    CAR --> ARMOR["装甲 YOLO<br/>192·conf0.3"]
    ARMOR --> PAT["图案分类<br/>MobileNetV3 @64"]
    PAT --> LOC["定位<br/>ray / homography"]
    LOC --> TRK["槽位级联跟踪<br/>身份投票+发布门"]
    TRK --> MINI["小地图显示"]
    TRK --> REF["裁判 0x0305 /<br/>哨兵引导 0x0301"]
```

### 1.2 坐标系与单位（全仓唯一权威定义）

| 项 | 约定 |
|---|---|
| 场地尺寸 | 28 m × 15 m，`locate/field.py`：x ∈ [-14, 14]，y ∈ [-7.5, 7.5] |
| 场地原点 | 场地中心，**+x 指蓝侧**（蓝雷达约 x = 13~14.6 m） |
| 单位 | 米（m）、秒（s）；裁判协议用 cm（见 [协议笔记 §4](referee_protocol_notes.md)） |
| 裁判图坐标 | 原点在红方补给站场角，+X 向蓝方，+Y 待联调核对（`referee/commands.py` 的 `_OUR_PLUS_Y_IS_REFEREE_PLUS_Y`，一个布尔全局翻转） |

### 1.3 三级检测链

| 级 | 模型 | 输入 | 阈值 | 输出 |
|---|---|---|---|---|
| 车 | YOLO26s @ 1280 | 全图 | conf 0.45 / max_det 10 | 车框（含 BoT-SORT 轨 id） |
| 装甲 | YOLO26n @ 192 | 车框 ROI | conf 0.3 / max_det 1 | 装甲框 + 红/蓝/dead 类 |
| 图案 | MobileNetV3-Small @ 64 | 装甲框 ROI | — | 6 类 `(1, 2, 3, 4, S, Q)` |

> 为什么装甲级不开跟踪：裁剪 ROI 没有帧间连续性，persist 只会串 ID；
> 为什么车级开 BoT-SORT：雷达相机俯仰扫场，ByteTrack 没有 GMC，全图一平移就集体换 ID。
> 但主循环 `run_radar.py` 里 `persist_tracks=False`——跨帧关联由槽位跟踪器拥有，检测内跟踪只作帧内参考。

### 1.4 槽位体系

- 固定兵种槽：红/蓝 × `1/2/3/4/S` 共 **10 个运动槽**，标签形如 `R3` / `B1` / `BS`。
- 前哨 `Q` 不进运动池、不进 0x0305。
- 槽生命周期：`INACTIVE → TENTATIVE → CONFIRMED → LOST`（`track/types.py::TrackPhase`）。
- 0x0305 共 12 槽：敌 E1/E2/E3/E4/E6/E7 + 己 A1/A2/A3/A4/A6/A7（数字是协议机器人 ID：1 英雄、2 工程、3/4 步兵、6 空中、7 哨兵；空中槽视觉不占）。

### 1.5 定位模式（改 `scripts/run_radar.py` 顶部常量 `LOCATE_MODE`）

| 模式 | 开关 | 原理 | 何时用 |
|---|---|---|---|
| 射线 | `LOCATE_MODE = "ray"` | 车底边像素沿相机射线打场地 mesh（`field_mesh.npz`）或 z=0 平面 | Blender 仿真（自带 K/R/t）或 **真机 PnP 标定后（推荐）** |
| 单应 | `LOCATE_MODE = "homography"` | ≥4 个地面点求画面→场地透视变换 | 备选：无可靠内参或需要快速恢复时 |

### 1.6 小地图点型（OpenCV 窗口 **radar** 右下角内嵌）

| 点型 | 含义 |
|---|---|
| 纯色实心圆 + 标签 | 这帧真的看见了这辆车（CONFIRMED） |
| 空心圆（背景透出）+ `标签?` | LOST 外推中：画面上没这辆车，位置是卡尔曼猜的 |
| （无点） | 外推漂出场地自动停发——车不可能在场外 |

---

## 2. 准备

### 2.1 环境依赖

| 依赖 | 用途 | 备注 |
|---|---|---|
| Python 3.11+（含 `enum.StrEnum`） | 全仓 | |
| OpenCV (`cv2`) | 图像 IO、显示、VideoCapture | |
| Ultralytics | YOLO 训练/推理（`.pt` 与 `.engine`） | |
| PyTorch | 图案分类器训练、`.pt` 推理 | |
| TensorRT | `.engine` 推理 | 无 TensorRT 时自动落回 `.pt`；engine 缺失时自动编译（见 §2.2） |
| pydantic | 地标/单应 json 校验 | |
| Blender 4.x | 仿真图源 | 默认路径 `D:\Program Files\blender\blender.exe`（`start_sim.ps1` 里改） |

- 所有命令在**仓库根目录**执行。
- Linux GPU 环境见 [docker/README.md](../docker/README.md)（`docker compose run --rm radar …`）。

### 2.2 权重清单（`weights/`，不进 git）

`detect/weights.py`：**engine 优先于 pt**——同卡不要混 PyTorch cuDNN 和 TensorRT。

| 文件 | 级 | 查找顺序（`detect/runtime.py`） |
|---|---|---|
| `car_best.engine` / `car_best.pt` | 车 | `car_best.engine` → `car_best.pt` → `runs/detect` 下最近训练 → `yolo26s.pt` |
| `armor_best.engine` / `armor_best.pt` | 装甲 | `armor_best.engine` → `armor_best.pt` |
| `pattern_best.engine` / `pattern_best.pth` | 图案 | `pattern_best.engine` → `pattern_best.pth` |

**engine 缺失自动编译**：默认三件套的 `.engine` 不在而同名 `.pt`/`.pth` 在时，检测器加载阶段自动在**子进程**调 `tools/pt_to_engine.py` / `tools/pattern_to_engine.py` 重编（FP16，需数分钟，编译日志直接进当前终端）。CUDA 不可用或编译失败则静默回退 `.pt`/`.pth` 加载，不中断。`RADAR_AUTO_ENGINE=0` 关闭。

### 2.3 标定文件清单（`locate/calib/`）

| 文件 | 谁写 | 谁读 |
|---|---|---|
| `landmarks.json` | 手改（**当前为 RM2027 6 点 3D 表**） | `calibrate_pnp.py` / `calibrate_landmarks.py` |
| `landmarks_2026.json` | 手改（RM2026 地面点表，单应参考） | 备查 |
| `Camera_Radar_Blue.json` / `Camera_Radar_Red.json` | Blender `dump_radar_cameras.py` 导出 | 仿真定位；真机 PnP 的内参模板 |
| `Camera_Field.json` | `calibrate_pnp.py` 产物（**不进 git**） | 真机 `LOCATE_MODE = "ray"` 位姿 |
| `match_homography.json` | `calibrate_landmarks.py` 产物（**不进 git**） | `LOCATE_MODE = "homography"` |
| `minimap.json` | `calibrate_minimap.py` 产物 | 小地图米制→像素 |
| `field_mesh.npz` | `scene/armor_assets/export_field_mesh.py` 导出 | 射线定位的场地 mesh |

### 2.4 Windows 执行策略

`.ps1` 被禁止时：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_sim.ps1
```

或直接用 `scripts\start_sim.cmd`。

---

## 3. 三分钟跑通

按"你想干什么"分流：

| 目标 | 命令 | 详节 |
|---|---|---|
| 看仿真全链路（双窗口） | `scripts\start_sim.cmd` | §4 |
| 无窗口跑基准回放 | `RADAR_SOURCE=images RADAR_HEADLESS=1 python scripts/run_radar.py` | §5 |
| 只验证软件没坏 | `pytest` | — |
| 验证定位精度上限 | `python scripts/validate_pnp.py` | §6.5 |

纯软件冒烟（无 GPU 也能跑的测试）：

```bash
pytest                # tests/ 按模块分组：detect / locate / models / referee / track / dataset
```

> `pytest.ini` 已配 `pythonpath = .`，仓库根目录直接跑即可。

---

## 4. 仿真联调

### 4.1 目标

验证 检测 → 定位 → 跟踪 → 小地图 全链路在不出仿真环境的情况下正常工作。

### 4.2 一键启动

```bat
scripts\start_sim.cmd
```

等价于手动开两个终端：

```bash
# 终端 1（Blender 出图）
& "D:\Program Files\blender\blender.exe" scene/RMUC2027_V1.0.0_radar_fixed.blend --python scene/armor_assets/radar_sim_loop.py -- --motion patrol

# 终端 2（检测 + 小地图）
python scripts/run_radar.py
```

`start_sim.ps1` 支持 `-Motion off|patrol`（默认 `patrol`）。

### 4.3 握手机制

- Blender 端（`radar_sim_loop.py`）每渲染一帧就写 `scene/locate_out/live_frame.png` + `live_meta.json`（含 frame / camera / robots 真值）。
- 检测端轮询该文件，**帧号变化才处理**。
- 实时源帧率约 **0.4 fps**——帧间隔远超裁判 0.5 s 中断线，实时跑 `RADAR_REFEREE_STUB=1` 的易伤数字只作链路演示；**基准结论以 10 Hz 录像回放为准**（§5）。

### 4.4 现象验收

1. 终端刷完 `run_radar waiting for frames…` 后出现 OpenCV 窗口 **radar**。
2. 主画面应是偏暗的雷达相机图（不是灰视口），有车框、装甲 ROI、车号。
3. 小地图（主画面右下角半透明）只标**认出图案**的车号（`R3` / `B1` / `S`），没图案的车不打点。
4. 终端逐帧打印 `[frame] cars N slots M`。
5. 按 `q` 退出检测端；Blender 窗口自行关闭。

### 4.5 常见故障

| 现象 | 处理 |
|---|---|
| 禁止运行脚本 | 用 `start_sim.cmd` 或 Bypass 跑 ps1（§2.4） |
| 只有 Blender、没有小地图 | 看检测终端报错；小地图窗口名是 **radar**，不在 Blender 里 |
| `cars 0`、灰画面 | 出图必须是雷达相机 EEVEE 渲染帧，不是视口截图；确认 `live_frame.png` 在更新 |
| 小地图无点 | 图案没认出则不标；先看主画面有没有 `R3/B1` 标签 |

---

## 5. 录像回放与打分跑批

### 5.1 images 序列格式

`RADAR_SOURCE=images` 读一个**帧序列目录**（默认 `scene/locate_out/videoseq`，基准用 `videoseq_dwell`）：

```
videoseq_dwell/
├── 0000.jpg      # 渲染帧
├── 0000.json     # 同名真值：{frame, camera, dt, robots:[{name, team, x, y, z}]}
├── 0001.jpg
├── 0001.json
└── ...
```

- 固定步长回放：`dt` 取 json 里的值（默认 0.1 s，即 10 Hz）。
- 序列由 `scene/armor_assets/generate_radar_car_sim.py` 渲染（`--motion patrol --count 200 --dt 0.1 [--spin 6.0]`）。

### 5.2 打分跑批

```bash
# Windows PowerShell
$env:RADAR_SOURCE="images"; $env:RADAR_IMAGES_DIR="scene/locate_out/videoseq_dwell"; $env:RADAR_HEADLESS="1"
python scripts/run_radar.py

# Linux / bash
RADAR_SOURCE=images RADAR_IMAGES_DIR=scene/locate_out/videoseq_dwell RADAR_HEADLESS=1 python scripts/run_radar.py
```

- `RADAR_HEADLESS=1`：不弹窗不抢焦点，适合后台跑批。
- 跑完落盘两份日志（均在 `scene/locate_out/`）：
  - `identity_log.jsonl`——逐帧 `{frame, gt:[{name,team,x,y,z}], pred:[{slot,x,y,state}]}`（`track/log.py`）
  - `referee_log.jsonl`——仅 `RADAR_REFEREE_STUB=1` 时（§11）

### 5.3 单视频文件

```bash
RADAR_SOURCE=video python scripts/run_radar.py
```

读 `scripts/RM_TestVideo.mp4`（`run_radar.py` 顶部 `VIDEO_PATH` 可改）。无真值、无日志，仅看链路。

---

## 6. 上场标定 —— PnP（推荐）

### 6.1 目标

由地标点击解出完整相机外参 R/t，写出 `locate/calib/Camera_Field.json`。之后 `LOCATE_MODE = "ray"` 直接走场地网格射线。相对单应：**PnP 无 z=0 平面假设**，斜坡/高地下车不再系统性偏移。

### 6.2 前提

- 雷达架稳，画面不再动（碰过相机就要重标）。
- 一张高清标定帧（仿真可用 `scene/locate_out/live_frame.jpg`）。
- 相机内参 json（含 `image_size` / `K` / `dist`，格式同 `Camera_Radar_Blue.json`；仿真相机 `dist` 全 0，真机要真实畸变值）。

### 6.3 地标表（当前 `locate/calib/landmarks.json`，RM2027）

**6 点非共线 3D 表**——三个高度层、y 轴 ±7 m 展开，PnP 条件数好：

| 顺序 | id | 中文 | (x, y, z) m | 点击目标 |
|---|---|---|---|---|
| 1 | `outpost_blue` | 蓝前哨站顶 | (5.5, 0, 1.2) | **建筑顶** |
| 2 | `outpost_red` | 红前哨站顶 | (-5.5, 0, 1.2) | **建筑顶** |
| 3 | `energy` | 能量机关中心 | (0, 0, 1.5) | **机关中心（最高点）** |
| 4 | `hangar_left` | 左机库地面 | (0, 7, 0.5) | 机库平台面 |
| 5 | `hangar_right` | 右机库地面 | (0, -7, 0.5) | 机库平台面 |
| 6 | `fortress_red` | 红方堡垒 | (-11, -5, 0.5) | 堡垒顶面 |

> ⚠️ **PnP 点顶部，单应点地面**：PnP 的 (x,y,z) 是"地面足迹 + 高度"，点建筑**顶/尖**才匹配；
> 单应（§7）只用 (x,y)，点**地面落点**才匹配。同一张表、两种工具、两种点击目标，别混。
>
> ⚠️ **换年规则图时只改这份 json** 的 `x/y/z`，与官方俯视图对齐，不改点击工具。

### 6.4 操作步骤

```bash
# 1.（可选但强烈推荐）先看精度预期：仿真真值基准，无需图像
python scripts/validate_pnp.py

# 2. 点击求解
python scripts/calibrate_pnp.py path\to\frame.jpg --intrinsics <内参.json>
#    可选：--landmarks locate/calib/landmarks.json   换地标表
#    可选：--out locate/calib/Camera_Field.json      换产物路径（默认即此）
```

1. 弹出标定帧窗口，按 `landmarks.json` 的 `click_order` 顺序依次点 6 个地标（点**顶部**）。
2. 每次点击后弹出 **4 倍放大镜**（最近邻放大，像素边界清晰）：`WASD` 单像素微调（`Shift`+方向 = 8px），`Enter`/`n`/空格 确认，`z`/`ESC` 取消本次点击——把手抖"差 3~5 px"压成亚像素级操作，残差余量更大。
3. 主窗口 `z` 撤销上一个点，回车求解并保存。
4. 工具会报告重投影残差：**> 12 px 直接视为点错**（`MAX_RESIDUAL_PX = 12.0`），残差明显就重点。

### 6.5 验收

- **先跑 `validate_pnp.py` 建立预期**：0/2/5/10 px 点击噪声下的旋转/平移/端到端误差表。10 px 糟糕点击下端到端 **P90 ≈ 0.24 m**（准确线 0.8 m，余量充足）。
- 求解残差 < 12 px。
- 主观验证：图像中心投到场地应落在能量机关附近（蓝侧雷达 x 为正）；已知停在前哨旁的车，小地图应落在对应前哨附近。

### 6.6 换入运行

```python
# scripts/run_radar.py 顶部
LOCATE_MODE = "ray"                      # 本来就是默认
CAMERA_BLUE_JSON = _ROOT / "locate" / "calib" / "Camera_Field.json"   # ← 指向 PnP 产物
```

> 与仿真共用 `"ray"` 代码路径：`poses` 字典里把真机位姿 JSON 换成 `Camera_Field.json` 即可。
> 真机帧若只有一侧相机，保留另一个槽位指旧文件也无妨——按帧 json 的 `camera` 名选位姿，选不到回落 Blue。

### 6.7 常见故障

| 现象 | 可能原因 | 处理 |
|---|---|---|
| 残差 > 12 px | 点错地标 / 坐标表与实物不符 | 对照 §6.3 表重点；核对当年官方尺寸 |
| `click_order needs at least 4 landmarks` | 地标表被改坏 | 恢复 json，至少 4 点且不全共线 |
| 解出外参明显不对 | 地标全在中线附近（共线），PnP 病态 | 使用带机库/堡垒的 6 点表；必须加非共线点 |
| 小地图整体偏一侧 | +y 方向与官方俯视图不符 / 点错了顶还是地面 | 按已知车位核对；坐标定义见 §1.2 |

---

## 7. 上场标定 —— 单应（备选）

### 7.1 何时用

没有可靠内参（无法做 PnP）时的降级方案；或 PnP 现场翻车需要 5 分钟快速恢复。**单应假设车永远在 z=0 平面**，斜坡/高地上的车会系统性偏移。

### 7.2 操作

```bash
python scripts/calibrate_landmarks.py                    # 自动找 live_frame.jpg / locate_frame.png
python scripts/calibrate_landmarks.py path\to\frame.jpg  # 指定标定帧
```

1. 按窗口提示依次点 **建筑在地面上的底座/落点**（**不要点建筑顶部**——单应没有高度维度，点顶部会引入视差误差；地面点参考 `landmarks_2026.json` 的思路）。
2. `z` 撤销，回车保存 → 写出 `locate/calib/match_homography.json`。
3. `run_radar.py` 顶部改 `LOCATE_MODE = "homography"`。
4. 车底边像素经单应到场地，再画到俯视图。

### 7.3 验收

- 图像中心投到场地应接近能量机关附近（蓝侧雷达 x 为正）。
- 已知停在前哨旁的车，小地图应落在对应前哨附近。
- 点位整体偏 → 十有八九点了建筑顶部：改点地面落点。

---

## 8. 小地图标定

`locate/calib/minimap.json` 把场地米制映射到官方俯视图像素。**场地还是 28×15 时不用重做**；换官方俯视图素材（`scene/materials/rmuc_fig4_1_topdown.png`）才需要：

```bash
python scripts/calibrate_minimap.py
```

按提示点俯视图**场地外框**四角：左上 → 右上 → 右下 → 左下，回车写 `minimap.json`。

---

## 9. 接真实摄像头

### 9.1 图源

```bash
# USB 直连（海康 MV-CH120 系列，设备号 0）
RADAR_SOURCE=hik RADAR_HIK_URL=0 python scripts/run_radar.py

# RTSP（网络相机）
RADAR_SOURCE=hik RADAR_HIK_URL=rtsp://user:pass@192.168.1.64:554/... python scripts/run_radar.py
```

- 纯数字当设备号，其余当 RTSP URL（`run_radar.py` 自动判别）。
- **断流自动重连**：读帧失败等 1 s 重开，再失败等 2 s 循环，不退出（仅 hik 源）。

### 9.2 分辨率一致性（关键）

**运行分辨率必须与标定那一帧一致**（当前相机 MV-CH120-60UC + 8 mm 镜头，4096×3000）。中途换分辨率 = 内参/外参全部作废。

### 9.3 上场推荐顺序

1. 先 `RADAR_SOURCE=video` / `images` 用赛场录像把检测和标定跑通；
2. 现场做 PnP 标定（§6）；
3. 再切 `hik` 源——检测/小地图/裁判链路不用改。

---

## 10. 打分与诊断工具箱

所有工具读 `run_radar.py` 落盘的日志，**不看窗口**，可离线重跑。都支持可选位置参数指定日志路径，默认读 `scene/locate_out/` 下对应文件。

### 10.1 底边射线验收 —— `eval_foot_ray.py`（换车模型的门禁）

```bash
python scripts/eval_foot_ray.py --weights <新best.pt> [--old-weights weights/car_best.pt]
```

held-out 仿真帧上：预测框底边打 mesh，与标签底边比对。**过线才允许替换 `weights/car_best.engine`，不过线不换**。通过判据：有误差样本、无漏检、无场外点、无超量检测（`_passed` 逻辑）。

### 10.2 身份召回 —— `score_identity_log.py`

```bash
python scripts/score_identity_log.py [identity_log.jsonl]
```

> 别和 `score_identity_switches.py`（§10.6）混淆——本文件答"点得准不准"
> （召回/误差/野点），那个答"槽位身份稳不稳"（每台真值的身份切换次数）。

GT 先用蓝相机投影过滤"画面内可见"，再与发布点最近邻配对（门限 2.5 m）。输出：可见 GT 数、发布数（tentative/confirmed 分开）、配对率、误差分布、野点数。

> **最近邻不能当发布门的成功线**：槽名错了但位置贴近某台车时，最近邻会把
> 错身份算成命中。审发布门用 §10.7 的 `score_false_reports.py`（按身份判）。

### 10.3 分级漏斗 —— `score_stage_funnel.py`

```bash
python scripts/score_stage_funnel.py
```

在直播分布帧上数：GT → 检出车 → 带装甲 → 图案可用 → 射线落点 → 对上 GT。**看链路到底死在哪一级**。链路与 `run_radar.py` 完全一致（同 engine、同 mesh），只是不进跟踪器。

### 10.4 stub 裁判基准 —— `score_referee.py`

见 §11。

### 10.5 诊断四模式 —— `diagnose.py`

```bash
python scripts/diagnose.py --mode conf       # 图案分类器：每次预测 vs 真值（角色/队伍/置信度）
python scripts/diagnose.py --mode lamp       # 装甲 YOLO：红蓝/dead 颜色误读率 + 置信度分离
python scripts/diagnose.py --mode confusion  # 每个 0x0305 槽实际在跟哪台 GT（吃车/被吃关系）
python scripts/diagnose.py --mode digits     # 图案模型 1↔3 数字混淆矩阵（尺寸 vs 错误率）
python scripts/diagnose.py --seq <目录> --mode conf   # 换序列（默认 videoseq_dwell）
```

### 10.6 其他工具

| 工具 | 用途 |
|---|---|
| `score_identity_switches.py [log]` | 数每个真值 name 的身份切换次数（仿真 JSONL；与 §10.2 召回工具勿混淆） |
| `verify_locate_gt.py` | GT→投影→射线应回到同一点，纯几何自检，不跑检测 |
| `debug_minimap_grid.py` | 小地图网格对齐调试 |
| `visualize_two_stage.py` / `visualize_minimap.py` | 单帧可视化与绘制入口（权重加载在 `detect/runtime.py`） |
| `infer_yolo_trt.py` | TensorRT 推理最小冒烟（默认指向 `weights/armor_best.engine` + `scripts/RM_TestVideo.mp4`，可在文件头改） |

### 10.7 虚报审计 —— `score_false_reports.py`（发布门口径）

```bash
python scripts/score_false_reports.py [identity_log.jsonl ...]   # 多个日志逐个单独报数
```

**口径**（必须按身份，不能用最近邻）：

- **虚报**：pred 的槽名（R3/B1/BS…）必须对上**同队同兵种**的真值，且这台车在
  **2.5 m 内**。对不上、或 2.5 m 内没有这台车——哪怕位置贴着另一台车——记 1 次虚报。
- **漏报**：蓝相机可见的真值（与 §10.2 同一投影过滤）里，该帧没有任何同名槽发布，记 1 次。
- 每套序列单独报数，不混总分。

**基线**（2026-10-24，`main` @ 5289489，未动任何门；序列：`scene/locate_out/{videoseq, videoseq_dwell, videoseq_2027_dwell}`，各 200 帧 @10 Hz）：

| 序列 | 发布数（conf/lost/tent） | 虚报 | 漏报 | 误差中位 | 误差最差 |
|---|---|---|---|---|---|
| videoseq | 570（252/222/96） | 370 | 642 | 1.084 m | 2.484 m |
| videoseq_dwell | 640（246/221/173） | 330 | 599 | 1.160 m | 2.480 m |
| videoseq_2027_dwell | 799（542/204/53） | 386 | 442 | 0.166 m | 2.485 m |

**收门 A/B**（基线虚报 > 0，允许收门；一次只改一个数，同批序列重跑对比）：

| 档（相对基线单值改动） | 虚报 v/d/2027 | 漏报 v/d/2027 | 判定 |
|---|---|---|---|
| BIRTH_CONF 0.5→0.7 | 379/335/349 | 629/632/522 | 几乎无效，弃 |
| BIRTH_CONF 0.5→0.9 | 326/295/197 | 642/595/**630** | 2027 漏报 630 > 上限 530，弃 |
| FRACTION 0.7→0.6 | 354/320/348 | 661/617/522 | **采用** |
| FRACTION 0.7→0.5 | 313/289/320 | 713/660/**570** | 2027 漏报 570 > 上限 530，弃 |
| MIN_VOTES 3→2 | =基线 | =基线 | 数学上无效（出生票必同意），弃 |

**采用档**：`track/cascade.py::_PUBLISH_GATE_FRACTION` 0.7 → 0.6，其余不动。

| 序列 | 虚报（基线→采用） | 漏报（基线→采用） | 漏报 +20% 上限 |
|---|---|---|---|
| videoseq | 370 → **354** | 642 → 661 | 770 ✓ |
| videoseq_dwell | 330 → **320** | 599 → 617 | 718 ✓ |
| videoseq_2027_dwell | 386 → **348** | 442 → 522 | 530 ✓（已贴近） |

**上限声明**：固定回放上虚报**做不到 0**，本轮停在 0.6 这一档。结构原因（审计
证据见 `score_false_reports.py` 按槽/按状态分解）：

1. **角色票看不见队伍错认**：R1 槽吃蓝英雄的观测时，图案「1」读得完全正确
   ——角色票同意、队伍错了，`_identity_publishable` 只看 role 不看 team，
   队伍错认（含 LOST 外推继承的错认）从这道门漏过去。
2. **LOST 外推漂移**：正确身份的槽丢车后外推 1.2 s，车一转弯坐标就漂出
   2.5 m——发布的是「对的槽名、错的位置」，按口径记虚报。
3. **错读出生本身高置信**：BIRTH_CONF 0.9 能把虚报压到 197–326，但 2027
   的正确读数置信度低，出生门一刀切把自己人也切了（漏报 630），不可用。

漏报上升上限**写死**：2027 序列漏报 ≤ 530（基线 442 + 20%）；0.6 档实测 522，
只剩 8 帧余量——任何更强的收门都会先爆漏报。

复现（确定性流水线，两次运行日志 MD5 一致）：

```bash
RADAR_SOURCE=images RADAR_HEADLESS=1 RADAR_IMAGES_DIR=scene/locate_out/videoseq_dwell \
RADAR_IDENTITY_LOG=<临时文件> python scripts/run_radar.py
python scripts/score_false_reports.py <临时文件>
```

#### 10.7.1 会上裁判的虚报分解（0.6 档复查，2026-10-24）

测量于 `false-report-gate` @ `c0ce08f`（FRACTION 0.6，未再动任何门），三套
序列重跑，默认 `RADAR_SEND_LOST=0` / `RADAR_GUESS=0`。**会上裁判的虚报 =
confirmed 虚报**——`marks_from_publishes` 里 TENTATIVE 一律不发、LOST 需
`include_lost`（§11.4 默认关），lost/tentative 的虚报默认到不了 0x0305。
重跑日志与 10.7 A/B 日志逐字节一致（MD5）。

**videoseq**（`scene/locate_out/videoseq`，200 帧，published 548，misses 661）：

| 按态 | confirmed | lost | tentative |
|---|---|---|---|
| 虚报 | **130（会上裁判）** | 192 | 32 |
| 正确发布误差 | med 0.764 / max 2.484 m（n=108） | med 1.978 / max 2.449 m（n=22） | med 1.084 / max 2.293 m（n=64） |

confirmed 虚报三类（互斥）：**a 队伍认错 58** · **b 同名超距 67** · **c 场上无此车 5**。

**videoseq_dwell**（`scene/locate_out/videoseq_dwell`，200 帧，published 622，misses 617）：

| 按态 | confirmed | lost | tentative |
|---|---|---|---|
| 虚报 | **100（会上裁判）** | 180 | 40 |
| 正确发布误差 | med 0.949 / max 2.480 m（n=131） | med 1.248 / max 2.404 m（n=38） | med 1.281 / max 2.346 m（n=133） |

confirmed 虚报三类（互斥）：**a 队伍认错 15** · **b 同名超距 71** · **c 场上无此车 14**。

**videoseq_2027_dwell**（`scene/locate_out/videoseq_2027_dwell`，200 帧，published 709，misses 522）：

| 按态 | confirmed | lost | tentative |
|---|---|---|---|
| 虚报 | **189（会上裁判）** | 141 | 18 |
| 正确发布误差 | med 0.130 / max 2.477 m（n=288） | med 0.706 / max 2.485 m（n=38） | med 0.313 / max 2.137 m（n=35） |

confirmed 虚报三类（互斥）：**a 队伍认错 67** · **b 同名超距 112** · **c 场上无此车 10**。

**汇总与决策**：

| 序列 | confirmed 虚报 | a 队伍认错 | b 同名超距 | c 无此车 | 多数类 |
|---|---|---|---|---|---|
| videoseq | 130 | 58（44.6%） | **67（51.5%）** | 5 | b |
| videoseq_dwell | 100 | 15（15%） | **71（71%）** | 14 | b |
| videoseq_2027_dwell | 189 | 67（35.4%） | **112（59.3%）** | 10 | b |

三套序列 confirmed 虚报都远超「可忽略」线（每套 ≤5），但多数类**全部是 b
（同名超距），不是 a（队伍认错）**——按既定规则**不加队伍票静默门**，
`_identity_publishable` 保持只看 role。即使把 b 类里「贴反队异兵种车」的也
划进队伍错认，dwell 也只占 31%，换一种读法结论不变。

b 类根因细分（confirmed 槽当帧刚吸收过观测，位置必在检出上；数字 = videoseq /
dwell / 2027）：

| b 类贴着谁 | videoseq | dwell | 2027 | 说明 |
|---|---|---|---|---|
| 同队异兵种车 | 31 | 49 | 82 | 兵种认错（1↔3 数字混淆既知问题） |
| 反队异兵种车 | 31 | 16 | 28 | 队伍+兵种双错 |
| 2.5 m 内没车 | 5 | 6 | 2 | 真位置漂移只占 2–8% |

**结论**：会上裁判的虚报根因是**出生/吸附时身份认错（兵种为主），不是位置漂移，
也主要不是队伍认错**——队伍票门对多数类无效，未采用，`track/cascade.py` 本轮未改。
修复依赖图案分类器重训（2027 点阵屏样式未发布，见遗留项 1），不是收门能解决的。

### 10.8 confirmed 标记 d 分级准线 —— `score_false_reports.py`（accuracy-line）

给「默认会发给裁判的 confirmed 标记」定准线：每条 confirmed 发布算它到
**同队同兵种真值**的距离 d，按规则手册 5.6.6 的裁判 d 分级（阈值 import
`referee/stub.py` 的 `ACCURATE_M=0.8 / SEMI_ACCURATE_M=1.6`，同源不改）：

- **准确** d < 0.8 m；**半准确** 0.8 ≤ d < 1.6 m；**错误** d ≥ 1.6 m 或场上没有这台车。

只数 state=confirmed——lost/tentative 默认不进 0x0305（§10.7.1），不混进准线。
d 中位/最差只算**有定义**的 d（同名真值在场；「没有这台车」的错误没有 d，
单独按条数报）。测量于 `accuracy-line-audit`（自 `false-report-gate` @
`d1cebc5` 拉），2026-10-24，当前代码重跑三套序列，日志与 §10.7.1 逐字节
一致（MD5）——没动任何门、任何权重。

| 序列（`scene/locate_out/`） | confirmed 条数 | 准确 | 半准确 | 错误 | d 中位 | d 最差 | 错误=同名 ≥1.6 m | 错误=没有这台车 |
|---|---|---|---|---|---|---|---|---|
| videoseq | 238 | 57（23.9%） | 29（12.2%） | **152（63.9%）** | 2.687 m | 13.754 m | 147 | 5 |
| videoseq_dwell | 231 | 56（24.2%） | 61（26.4%） | **114（49.4%）** | 1.362 m | 16.852 m | 100 | 14 |
| videoseq_2027_dwell | 477 | 257（53.9%） | 15（3.1%） | **205（43.0%）** | 0.382 m | 13.068 m | 195 | 10 |

（d 中位/最差的有定义条数：233 / 217 / 467。）

**决策与根因**（不改代码，`locate/` 未动）：

1. **错误档主要是身份认错，不是位置漂移**。发布贴着谁已钉进打分脚本
   （§10.9 混淆表，可复现）：confirmed 发布里贴错（2.5 m 内最近真值是
   别的车）162/116/206 条，附近没车只有 5/6/2 条——发布几乎全部压在
   某台真车身上、而同名真值很远，是上一节已经结案的兵种认错，d 分级
   只是换裁判的尺子再量一遍。（本条原引用临时脚本的 93%/82%/94% 与
   「没车 11/21/13」，已被 §10.9 的脚本化口径修正替代。）
2. **射线没有系统性偏出 0.8 m，指不出 locate/ 的具体错误**。身份对的
   confirmed 点（d < 2.5 m）均值偏移向量仅 (+0.04,−0.09) / (+0.03,+0.25) /
   (−0.05,−0.01) m，模长 0.093 / 0.255 / 0.050 m——散布大、方向无偏，
   是逐点噪声（检出框 + 卡尔曼滞后），不是系统性落点偏移。2027 序列
   身份对的点 89.2% 落准确档（均值 d 0.317 m），证明当前外参/mesh/底边
   射线链路健康；两套 2026 序列散布更大（准确率 42.7–52.8%）是 2026
   渲染对 2027 标定的年代错配（README 基准历史已注明两套序列不可横比），
   不是 locate/ 代码错误。修定位的条件不成立。
3. **结论：准线失败在身份，不在射线**。不修定位、不训 pattern、不为
   准确档好看去收任何门——BIRTH_CONF / FRACTION / 发布门 / 猜点 /
   哨兵 / 离线优先全部未动。错误档压下去依赖图案分类器重训（遗留项 1）。

### 10.9 confirmed 混淆表 —— 发布槽 → 实际贴着的车（identity-confusion）

把「confirmed 标记贴着哪台真值」钉进 `score_false_reports.py`（stick_class），
替换上一轮不可复现的临时脚本。口径：每条 confirmed 发布，在**全部真值**里
找 2.5 m 内最近的一台车（不管同名不同名）——贴错记一对「发布槽 → 实际槽」，
贴对不进表，附近没车单独计数；一条发布只进一类，不用 0.8 m 再分一次。
只数 confirmed；槽名保持 R1/B3/RS 原样。测量于 `identity-confusion-table`
（自 `accuracy-line-audit` @ `0f70838` 拉），2026-10-24，当前代码重跑三套
序列，日志与 §10.8 逐字节一致（MD5）。

**videoseq**（`scene/locate_out/videoseq`，200 帧）：

| confirmed | 贴错 | 贴对 | 附近没车 |
|---|---|---|---|
| 238 | **162** | 71 | 5 |

贴错构成（互斥）：同队异兵种 58 · 同兵种反队 44 · 又反队又异兵种 60

贴错对（降序，前 10）：`B1→R1` 21 · `B2→R3` 19 · `B1→R2` 14 · `B3→B2` 11 ·
`B3→R3` 10 · `R1→B1` 10 · `R2→R3` 7 · `R3→R2` 6 · `B2→B1` 5 · `B2→R1` 5

**videoseq_dwell**（`scene/locate_out/videoseq_dwell`，200 帧）：

| confirmed | 贴错 | 贴对 | 附近没车 |
|---|---|---|---|
| 231 | **116** | 109 | 6 |

贴错构成（互斥）：同队异兵种 74 · 同兵种反队 8 · 又反队又异兵种 34

贴错对（降序，前 10）：`B2→B3` 41 · `B3→R1` 10 · `B1→R3` 7 · `R4→B3` 6 ·
`RS→R3` 6 · `R3→R2` 5 · `B3→B1` 4 · `B3→R3` 4 · `R2→R3` 4 · `R3→R1` 4

**videoseq_2027_dwell**（`scene/locate_out/videoseq_2027_dwell`，200 帧）：

| confirmed | 贴错 | 贴对 | 附近没车 |
|---|---|---|---|
| 477 | **206** | 269 | 2 |

贴错构成（互斥）：同队异兵种 148 · 同兵种反队 17 · 又反队又异兵种 41

贴错对（降序，前 10）：`B1→B3` 71 · `R1→R3` 24 · `R2→R1` 23 · `R1→B2` 21 ·
`B3→R3` 12 · `R3→B1` 12 · `R1→R2` 7 · `R3→R1` 7 · `B1→R1` 4 · `R2→B1` 3

**三套横查**：

- **没有一对在三套序列里都是第一或第二名**（videoseq 前二 `B1→R1`/`B2→R3`，
  dwell 前二 `B2→B3`/`B3→R1`，2027 前二 `B1→B3`/`R1→R3`，交集为空）——
  「单对训练目标」条款不触发，不为任何一对现在收数或开训。但三套的
  贴错全部以**兵种混淆**为主（构成里同队异兵种 + 又反队又异兵种占大头，
  1↔3 英雄/步兵混认反复出现：`B1→B3`、`R1→R3`、`B2→B3`），与 §10.7.1/
  §10.8 的结论一致：修这一列靠等点阵屏样式后的图案分类器重训，不是
  收门、不是改定位。
- 上一轮 §10.8 引用的「错误档 93%/82%/94% 贴着另一台车、没车 11/21/13」
  出自临时脚本；脚本化后发现临时脚本把「无同名真值但贴着别的车」错计进
  了「没车」桶。可复现口径见上表（附近没车仅 5/6/2），结论只强不弱，
  §10.8 原句已改指本表。

---

## 11. stub 裁判基准

### 11.1 是什么

`referee/stub.py` 把规则手册 5.6.6 的"被标记进度"P 计分公式做成回放可对打的一端——不是真裁判仿真器（无图传/UI/复活），只实现基准相关部分：

- d 分级：准确 <0.8 m，半准确 <1.6 m，错误 ≥1.6 m（对 `set_truth` 注入的真值算）
- 连击 x：准确 +1.0 / 半准确 +0.5（上次好则累进），错误 -0.8（上次坏则连扣）
- 中断：累计 0.5 s 没收到该机器人坐标 → 追加一次"错误"
- P ∈ [0, 150]；敌方 P≥100 记易伤秒；**多台同时易伤不叠加**（工会时间取并集）
- 1 Hz 回发 0x020C（易伤掩码）/ 0x020E（双倍机会数/触发中）
- 金标测试：`tests/referee/test_stub.py::test_rulebook_example` 逐值对照规则手册 worked example

### 11.2 跑法

```bash
# 1. 回放 + stub 打分（headless 跑批）
RADAR_SOURCE=images RADAR_IMAGES_DIR=scene/locate_out/videoseq_dwell \
RADAR_HEADLESS=1 RADAR_REFEREE_STUB=1 RADAR_SEND_LOST=1 python scripts/run_radar.py

# 2. 出报告
python scripts/score_referee.py [referee_log.jsonl]
```

`run_radar.py` 结束时打印 `union_vuln_s` 摘要并提示 score 命令。

### 11.3 报告怎么读

```
slot   P  vuln_s  acc semi err intr  d_med     ← 每槽终值 P / 累计易伤秒 / 各判定计数 / 误差中位数
mark frames  X/Y  accurate % semi % error %    ← 发布帧占比与准确率
interrupts   N
union vuln   Ns of Ms   double chances K       ← 主基准指标：工会易伤秒（敌方任一槽易伤的并集时长）
```

- 槽名 `E1/E2/E3/E4/E6/E7` 为敌方（1 英雄 2 工程 3/4 步兵 6 空中 7 哨兵）。
- **主基准指标 = union vuln**：当前最优 17.2 s（2026 场地 dwell 序列）。

### 11.4 质量门（A/B 实验）

| 开关 | 行为 | 结论 |
|---|---|---|
| `RADAR_SEND_LOST=0`（默认） | 只发 CONFIRMED 槽 | 没把握就不发；(0,0) 不扣进度，错发比静默代价高 |
| `RADAR_SEND_LOST=1` | LOST（卡尔曼外推）也发 | **实测 +67% 易伤秒**；外推误差中位数 <0.8 m 准确线时才值得开 |
| `RADAR_GUESS=1` | LOST 坐标换战略猜点 | 短遮挡场景有害，默认关 |
| `RADAR_SENTRY_GUIDE=1` | 0x0301 把敌方位置发己方哨兵 | 走 `send_sentry_guidance` |

TENTATIVE 槽一律不发（两帧确认前身份不可信）。stub 的比赛阶段压缩由 `RADAR_PHASE_SCALE` 控制：1.0 = 真实时长（准备 3 min + 自检 15 s + 倒计时 5 s + 战斗 7 min）；0.05 → 准备 9 s + 战斗 21 s。

---

## 12. 环境变量总表

全部在 `scripts/run_radar.py` 读取；未列出的开关（如 `LOCATE_MODE`）是代码常量，改文件顶部。

| 变量 | 默认 | 说明 |
|---|---|---|
| `RADAR_SOURCE` | `blender` | 图源：`blender` / `video` / `images` / `hik` |
| `RADAR_HEADLESS` | `0` | 1 = 不弹窗口（跑批） |
| `RADAR_IMAGES_DIR` | `scene/locate_out/videoseq` | images 源目录（基准用 `videoseq_dwell`） |
| `RADAR_HIK_URL` | `0` | USB 设备号或 RTSP 地址 |
| `RADAR_REFEREE_STUB` | `0` | 1 = 启用 stub 裁判（写 referee_log） |
| `RADAR_REFEREE_LOG` | `scene/locate_out/referee_log.jsonl` | stub 裁判日志路径 |
| `RADAR_SEND_LOST` | `0` | 1 = LOST 外推槽也发 0x0305（§11.4） |
| `RADAR_GUESS` | `0` | 1 = LOST 坐标换战略猜点 |
| `RADAR_PHASE_SCALE` | `0.05` | stub 比赛阶段时间压缩比（1.0 = 真实时长） |
| `RADAR_SENTRY_GUIDE` | `0` | 1 = 启用 0x0301 哨兵引导 |
| `RADAR_IDENTITY_LOG` | `scene/locate_out/identity_log.jsonl` | identity 日志路径 |
| `RADAR_AUTO_ENGINE` | `1` | 0 = 关闭"engine 缺失自动编译"（检测器加载阶段读取，见 §2.2） |

> 导出工具的环境变量（`YOLO_PT` / `YOLO_ENGINE` / `PATTERN_PT` 等）见[训练指南](training_guide.md)。

---

## 13. 故障排查表

| # | 现象 | 可能原因 | 诊断方法 | 修复建议 |
|---|---|---|---|---|
| 1 | 禁止运行脚本 | PowerShell 执行策略限制 | 报错含 "cannot be loaded" | `start_sim.cmd`，或 `powershell -ExecutionPolicy Bypass -File …` |
| 2 | 只有 Blender 没有雷达窗口 | 检测终端崩了 / 窗口被最小化 | 看检测终端最后几行 | 窗口名是 **radar**；按报错对号本表 |
| 3 | `cars 0`、画面灰 | 图源不是雷达相机 EEVEE 帧 | 检查 `live_frame.png` 是否在更新、内容是否是相机视角 | Blender 端确认渲染相机设置；不要用视口截图 |
| 4 | 小地图无点 | 图案未识别 / 定位失败 | 主画面有没有 `R3/B1` 标签？终端 `slots M` 是否 >0 | 先 §10.5 `--mode conf` 看图案；再查标定 |
| 5 | PnP 残差 >12 px | 点错地标 / 坐标表与实物不符 | 对照 §6.3 表逐点核对 | 重标；换年先改 `landmarks.json` |
| 6 | 单应点位整体偏 | 点了建筑顶部（单应要地面落点） | 检查点击是否落在建筑底座 | 改点地面落点重标 |
| 7 | `need at least 4 clicks` | 回车前没点满 | 点击计数 `N/M` | 点满 4 个（PnP 为 6 个）再回车 |
| 8 | 相机源起不来 | URL/设备号不对、分辨率不符 | `run_radar hik camera …` 打印了什么 | 纯数字=设备号；其余=RTSP；断流会自动重连 |
| 9 | 权重找不到 | `weights/` 缺文件 | 看 §2.2 查找顺序 | 放入 `car_best.*` / `armor_best.*` / `pattern_best.*`；不要提交 git |
| 10 | 真机小地图镜像/旋转 | +y 方向与官方俯视图不符 | 已知坐标的车位置是否左右颠倒 | 联调后改 `referee/commands.py::_OUR_PLUS_Y_IS_REFEREE_PLUS_Y`；标定侧重查点击目标 |
| 11 | 相机刚碰过、点全偏 | 外参失效 | 重标前先看是否相机移位 | 重跑 §6（PnP）或 §7（单应） |
| 12 | Blender 帧率极低 | EEVEE 实时渲染 4096×3000 | `live_frame.png` 更新间隔 | 正常现象（~0.4 fps）；基准一律走 §5 的 10 Hz 回放 |
| 13 | 车模型换了之后装甲/图案全崩 | 级联 ROI 域偏移 | `score_stage_funnel.py` 看断在哪级 | 车模型不能单独换入——见[训练指南 §3](training_guide.md) 全链重训顺序 |
| 14 | `.engine` 在别的机器/容器上不能用 | TensorRT engine 与 GPU 绑定 | 跨机器运行报 engine 反序列化错误 | 拷 `.pt` 过去，目标机首次启动自动重编 engine（§2.2，`RADAR_AUTO_ENGINE` 默认开；Docker 内同理） |
| 15 | 启动即报 `preflight FAIL: CUDA 不可用` | 驱动没装/版本不对，但 `weights/` 里已有 `.engine` | `nvidia-smi` 能否正常列出 GPU | 修驱动；急用可把 `.engine` 改名移走，回退 `.pt` 跑（CPU，慢） |
| 16 | 真机跑一半视觉没了 | `run_radar.py` 崩了（CUDA/驱动抽风等） | 终端是否还活着；`watch_radar` 是否在重启 | 用 `scripts\watch_radar.cmd`（Linux 用 `.sh`）做守护：崩溃 3 秒自动重启，`q` 正常退出不重启 |

---

## 附录 A：命令速查

```bash
# ── 运行 ─────────────────────────────────────────────
scripts\start_sim.cmd                                          # Windows 仿真双窗口
RADAR_SOURCE=images RADAR_HEADLESS=1 python scripts/run_radar.py        # 回放跑批
RADAR_SOURCE=hik RADAR_HIK_URL=rtsp://... python scripts/run_radar.py   # 真相机
scripts\watch_radar.cmd                                        # 真机守护：崩溃自动重启（先设好 RADAR_* 再启动）
scripts/watch_radar.sh                                         # 同上（Linux/Docker）

# ── 标定 ─────────────────────────────────────────────
python scripts/validate_pnp.py                                 # PnP 精度预期（无需图像）
python scripts/calibrate_pnp.py frame.jpg --intrinsics K.json  # 真机 PnP（推荐）
python scripts/calibrate_landmarks.py                          # 单应（备选）
python scripts/calibrate_minimap.py                            # 俯视图四角

# ── 打分 / 诊断 ──────────────────────────────────────
python scripts/eval_foot_ray.py --weights best.pt               # 车模型验收门禁
python scripts/score_identity_log.py                           # 身份召回
python scripts/score_false_reports.py                          # 虚报审计（发布门，按身份，§10.7）
python scripts/score_stage_funnel.py                            # 分级漏斗
python scripts/diagnose.py --mode conf                         # 诊断（conf/lamp/confusion/digits）

# ── stub 裁判基准 ────────────────────────────────────
RADAR_REFEREE_STUB=1 RADAR_SEND_LOST=1 python scripts/run_radar.py
python scripts/score_referee.py

# ── 训练 / 导出（详见训练指南）─────────────────────────
python -m models.train car --data models/configs/data_car.yaml
python -m models.train armor --data models/configs/data_armor.yaml
python -m models.pattern.train --data dataset/pattern_balanced
python tools/pt_to_engine.py            # 车（装甲用 YOLO_IMGSZ=192 YOLO_BATCH=8 覆盖）
python tools/pattern_to_engine.py

# ── 测试 ─────────────────────────────────────────────
pytest
```

---

*发现手册与代码不一致？以代码为准，并立刻改这里——本手册的价值在于永远和仓库对得上。*
