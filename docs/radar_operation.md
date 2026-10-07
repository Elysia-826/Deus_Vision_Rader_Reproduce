# 雷达视觉操作手册

> 版本：v1.0 | 仓库：Deus Vision Radar Reproduce | 最后更新：2026-09

---

## 目录

1. [概述](#1-概述)
2. [准备](#2-准备)
3. [仿真联调](#3-仿真联调)
4. [上场四点标定](#4-上场四点标定)
5. [接真实摄像头](#5-接真实摄像头)
6. [故障排查](#6-故障排查)

---

## 1. 概述

检测和小地图不绑 Blender。图源换成录像或海康后，后面同一条链：

```
图像 → 车 YOLO → 装甲 YOLO → 图案车号 → 像素到底边
      → 定位（射线 或 画面地标单应）→ 场地 (x, y) 米 → 俯视图
```

| 模式 | 开关 | 何时用 |
|------|------|--------|
| 仿真射线 | `LOCATE_MODE = "ray"` | Blender 已有 K/R/t |
| 上场单应 | `LOCATE_MODE = "homography"` | 真机，点前哨/能量机关/基地 |

单位：米、像素。场地 28 m × 15 m，原点在中心，**+x 蓝侧**（蓝雷达约 x = 13 m）。

---

## 2. 准备

- Python 环境已能 `import cv2`、`ultralytics`、TensorRT（有 `.engine` 时）
- 权重放 `weights/`：`car_best.engine`、`armor_best.engine`、`pattern_efficientnet_b0.engine`（不进 git；main 分支图案仍是 `pattern_best.engine`）
- 仿真另需本机 Blender：`D:\Program Files\blender\blender.exe`
- 仓库根目录执行下面所有命令

Windows 若禁止跑 `.ps1`：

```powershell
powershell -ExecutionPolicy Bypass -File scripts\start_sim.ps1
```

或：

```bat
scripts\start_sim.cmd
```

小地图在 OpenCV 窗口 **radar** 右下角，不在 Blender 里。检测终端不要关。`q` 退检测。

---

## 3. 仿真联调

1. 根目录启动：`scripts\start_sim.cmd`
2. 等终端刷完 `loading detector`，出现 **radar** 窗口
3. 主画面应是偏暗的雷达相机图（不是灰视口），应有车框、装甲 ROI、车号
4. 小地图只标图案车号（`R3` / `B1` / `S`），没有图案的车不打点

`scripts/run_radar.py` 里仿真默认：

```python
SOURCE = "blender"
LOCATE_MODE = "ray"
```

---

## 4. 上场四点标定

赛场雷达位姿每年每场都会变，不要用仿真的 `Camera_Radar_Blue.json`。做法和 `sports` 雷达相同：在**相机画面**里点 ≥4 个不动建筑的**地面落点**，求单应。

### 4.1 地标表

`locate/calib/landmarks.json` 默认点击顺序：

| 顺序 | id | 中文 | 默认 (x, y) m |
|------|-----|------|----------------|
| 1 | `outpost_blue` | 蓝前哨站 | (5.5, 0) |
| 2 | `outpost_red` | 红前哨站 | (-5.5, 0) |
| 3 | `energy` | 能量机关 | (0, 0) |
| 4 | `base_blue` | 蓝方基地 | (12.5, 0) |

换年规则图时改这份 json 的 `x/y`，与官方俯视图对齐。可选地标还有 `base_red`。

只有「两前哨 + 能量机关」是 3 个点，不够。第四点用己方基地。点尽量铺开。

### 4.2 操作

1. 雷达架稳，画面不再动。
2. 抓一帧（仿真可用 `scene/locate_out/live_frame.jpg`）：

```powershell
python scripts/calibrate_landmarks.py
python scripts/calibrate_landmarks.py path\to\frame.jpg
```

3. 按窗口提示依次点 **建筑在地面上的底座中心**，不要点装甲板中心。
4. `z` 撤销，回车保存。写出 `locate/calib/match_homography.json`（本地文件，不进 git）。
5. `scripts/run_radar.py`：

```python
LOCATE_MODE = "homography"
```

6. 再开检测。车底边像素走单应到场地，再画到俯视图。

相机被碰过就重新点一次。

### 4.3 验收

- 图像中心投到场地应接近能量机关附近（蓝侧雷达时 x 为正）
- 已知停在前哨旁的车，小地图应落在对应前哨附近
- 残差明显：重新点地面落点，或把 `landmarks.json` 改成该年官方尺寸

---

## 5. 接真实摄像头

`SOURCE = "hik"` 尚未接线。过渡：

1. 先 `SOURCE = "video"`，用赛场录像把检测和单应跑通
2. 标定 `match_homography.json`
3. 海康 RTSP 接上后，用 `cv2.VideoCapture(rtsp_url)` 替换图源，检测/小地图不用改
4. 分辨率必须和标定时那一帧一致。选定机是海康 MV-CH120-60UC，4096×3000，像元 3.45 μm；镜头 1.1 英寸靶面 8 mm f/1.4。仿真内参按这个写，上场单应仍要在实拍画面上重标

俯视图四角 `locate/calib/minimap.json` 在场地还是 28×15 时不用重做。

---

## 6. 故障排查

| 现象 | 处理 |
|------|------|
| 禁止运行脚本 | `scripts\start_sim.cmd` 或 Bypass 跑 ps1 |
| 只有 Blender、没有小地图 | 看检测终端；窗口名是 radar |
| `cars 0`、灰画面 | 出图必须是雷达相机 EEVEE，不是视口截图 |
| 小地图无点 | 图案没认出则不标；先看主画面有没有 `R3/B1` |
| 单应点位整体偏 | 点了建筑顶部；改点底座 |
| `need at least 4 clicks` | 回车前要点满 4 个 |
| `source hik is not wired yet` | 尚未接海康，用 blender 或 video |
| 权重找不到 | 把 engine/pt 放到 `weights/`，不要提交 git |
