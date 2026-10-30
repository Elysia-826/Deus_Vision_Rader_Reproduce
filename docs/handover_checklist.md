# 接手与赛日清单

> 版本：v1.0 | 最后更新：2026-10
> 用途：**新队员接手的第一份操作卡** + **赛日现场检查单**。
> 照着勾，勾完就对系统有最基本的掌握。深入原理看[架构深讲](architecture.md)。

---

## 目录

1. [交接物清单（前任 → 你）](#1-交接物清单前任--你)
2. [接手第一小时：环境冒烟](#2-接手第一小时环境冒烟)
3. [接手第一天：跑通全链路](#3-接手第一天跑通全链路)
4. [接手第一周：复现基准](#4-接手第一周复现基准)
5. [赛日上场清单](#5-赛日上场清单)
6. [基准验收表](#6-基准验收表)
7. [已知坑（前人踩过的雷）](#7-已知坑前人踩过的雷)

---

## 1. 交接物清单（前任 → 你）

交接时逐项核对，缺什么当场要：

- [ ] **权重** `weights/`：`car_best.engine` + `armor_best.engine` + `pattern_best.engine`（以及对应 `.pt`/`.pth` 源文件）——**不进 git，必须单独拷**；`weights/archive/` 里是历代备份，不是当前验收版
- [ ] **测试视频** `scripts/RM_TestVideo.mp4`：video 源的默认输入——`*.mp4` 不进 git，同样必须单独拷
- [ ] **训练 run 目录** `runs/`（可选，但新车 checkpoint 在里面）
- [ ] **数据集**（不进 git）：`dataset/car_yolo` / `dataset/pattern_balanced` 等在什么机器、什么路径
- [ ] **仿真产物**：`scene/locate_out/` 基准序列（`videoseq_dwell` 等）在什么机器
- [ ] **官方留档** `refs/`：2027 规则手册 + 通信协议 PDF（不进 git）
- [ ] **Blender 场景**：`RMUC2027_V1.0.0_radar_fixed.blend` 能打开（CAD/贴图不进 git）
- [ ] **硬件信息**：相机型号（当前 MV-CH120-60UC + 8mm）、雷达平台高度（2027 规则 ~2.4m）、串口接线
- [ ] **账号/设备**：训练机（5070 台式机）在哪、远程访问方式
- [ ] **文档已读**：[README](../README.md) → [操作手册](radar_operation.md) → 本文 → [架构深讲](architecture.md)

---

## 2. 接手第一小时：环境冒烟

目标：确认代码和环境是好的，与算法无关。

```bash
# 1. 克隆 + 进入仓库根目录
git clone <repo> && cd DEUS_VISION_RADER_TEST_reproduce

# 2. 依赖能 import
python -c "import cv2, ultralytics, torch; print('ok')"

# 3. 全量测试（无 GPU 也能跑，2-5 分钟）
pytest
```

- [ ] `pytest` 全绿（testpaths=tests，按 detect/locate/models/referee/track/dataset 分组）
- [ ] 权重已就位（§1 第一项），`python scripts/infer_yolo_trt.py` 可选做推理冒烟
- [ ] Linux/无本机 GPU：改用 `docker compose run --rm radar pytest`（见 [Docker 指南](../docker/README.md)）

> 任一步失败 → 先看[操作手册 §13 故障排查表](radar_operation.md)，再看 §7 已知坑。

---

## 3. 接手第一天：跑通全链路

目标：亲眼看系统动起来，并理解每个窗口/文件是什么。

### 3.1 仿真（需要本机 Blender）

```bat
scripts\start_sim.cmd
```

- [ ] 出现两个东西：Blender 窗口（出图）+ OpenCV 窗口 **radar**（检测+右下角小地图）
- [ ] 主画面有车框、装甲 ROI、车号；小地图有 `R3/B1/S` 点
- [ ] 终端逐帧打印 `[frame] cars N slots M`；按 `q` 退出
- [ ] 理解点型：实心=看见、空心+`=LOST 外推（[操作手册 §1.6](radar_operation.md)）

### 3.2 回放（不需要 Blender）

```bash
RADAR_SOURCE=images RADAR_IMAGES_DIR=scene/locate_out/videoseq_dwell RADAR_HEADLESS=1 python scripts/run_radar.py
```

- [ ] 结束无报错，`scene/locate_out/identity_log.jsonl` 有内容
- [ ] 能看懂日志格式（[架构深讲 §8](architecture.md)）

### 3.3 打分

```bash
python scripts/score_identity_log.py
```

- [ ] 理解输出：可见 GT / 配对率 / 误差分布 / 野点

---

## 4. 接手第一周：复现基准

目标：把主基准指标跑出来，与历史值对照——**能复现 = 接手成功**。

### 4.1 主基准（stub 裁判 union vuln）

```bash
# 1. 回放 + stub
RADAR_SOURCE=images RADAR_IMAGES_DIR=scene/locate_out/videoseq_dwell \
RADAR_HEADLESS=1 RADAR_REFEREE_STUB=1 RADAR_SEND_LOST=1 python scripts/run_radar.py

# 2. 报告
python scripts/score_referee.py
```

### 4.2 对照历史（[架构深讲 §9](architecture.md) 有完整表）

| 场地 | 配置 | union vuln | 准确率 | 你的复现值 |
|---|---|---|---|---|
| 2026 dwell | v3 + 发布门 + 停顿 | 17.2 s | 50.9% | _____ |
| 2027 | 旧模型直跑 | 9.6 s | 44.3% | _____ |

> 2027 行的序列与 2026 不同，复现命令（其余开关与 §4.1 一致）：
>
> ```bash
> RADAR_SOURCE=images RADAR_IMAGES_DIR=scene/locate_out/videoseq_2027_dwell \
> RADAR_HEADLESS=1 RADAR_REFEREE_STUB=1 RADAR_SEND_LOST=1 python scripts/run_radar.py
> python scripts/score_referee.py
> ```

- [ ] 复现值与历史值量级一致（±10% 内；差异大 → 先查权重版本，再查序列是否同一份）
- [ ] 理解为什么 `RADAR_SEND_LOST=1` 是基准配置（+67% 易伤秒，[操作手册 §11.4](radar_operation.md)）

### 4.3 诊断四件套过一遍

```bash
python scripts/score_stage_funnel.py      # 链路漏斗：死在哪一级
python scripts/diagnose.py --mode conf    # 图案精度
python scripts/diagnose.py --mode lamp   # 装甲颜色误读
python scripts/diagnose.py --mode confusion  # 槽位吃车关系
```

- [ ] 每个工具都能跑、都能说出一个从输出里读出的结论

### 4.4 标定演练（仿真内）

```bash
python scripts/validate_pnp.py     # 精度预期表
python scripts/calibrate_pnp.py scene/locate_out/live_frame.jpg --intrinsics locate/calib/Camera_Radar_Blue.json
```

- [ ] 在仿真帧上完成一次 6 点 PnP 标定，残差 < 12 px
- [ ] 理解 PnP 点**建筑顶**、单应点**地面落点**的区别（[操作手册 §6.3](radar_operation.md)）

---

## 5. 赛日上场清单

比赛现场按此顺序执行，全程约 30-40 分钟。

### 5.1 赛前夜（驻地）

- [ ] 权重三件套就位且是最新验收版本（`eval_foot_ray.py` 过线的那个）
- [ ] 内参 json 就位（分辨率与相机实出一致：4096×3000）
- [ ] `landmarks.json` 坐标与当年官方俯视图核对过
- [ ] 备好：一台能跑检测的机器、相机线、USB/串口线、延长线、备用镜头纸
- [ ] Docker 机器（如用）：镜像已 build（现场 build TensorRT 轮子要 10 分钟）

### 5.2 现场：架机 → 标定 → 验证

1. - [ ] 雷达上架、锁死、走线，**画面不再晃**（再晃一切作废）
2. - [ ] 抓一帧标定帧（相机原生分辨率，不截图工具压缩）
3. - [ ] PnP 标定（[操作手册 §6](radar_operation.md)）：

```bash
python scripts/calibrate_pnp.py frame.jpg --intrinsics <内参.json>
```

4. - [ ] 残差 < 12 px；`Camera_Field.json` 写出
5. - [ ] `run_radar.py` 指向 `Camera_Field.json`，`LOCATE_MODE = "ray"`
6. - [ ] 接相机源走比赛档（不要手开 stub / 录像 / 猜点）：

```bat
scripts\match_radar.cmd Blue
scripts\match_radar.cmd Red
```

Linux：`scripts/match_radar.sh Blue` 或 `Red`。脚本锁 `RADAR_SOURCE=hik`、全屏、相位不压缩，并拒绝考核开关。USB 默认第 0 路；RTSP 先设 `RADAR_HIK_URL` 再启动。

7. - [ ] **实地验收**：
   - 找一台车停在前哨旁 → 小地图点应落在该前哨附近（误差 < 半个车身）
   - 车开到高地/斜坡 → 点不飞（单应会飞，PnP 不会）
   - 图案车号（R1/B3…）与实际车号一致
8. - [ ] 记录：本场的 `Camera_Field.json` 备份 + 标定帧照片（赛后归档，别只存机器上）

### 5.3 赛后

- [ ] 归档本场的标定产物与日志（identity_log / referee_log）
- [ ] 记录一个问题清单：哪台车跟丢、哪个地标难点、光照如何 → 反哺训练数据

---

## 6. 基准验收表

任何"改了代码/换了权重"之后填一次这张表（**阈值表是唯一权威**——改流程先改这里）：

| 环节 | 指标 | 阈值/历史 | 实测 | 通过 |
|---|---|---|---|---|
| 车模型验收 | eval_foot_ray | 无漏检/无场外/无超量 | _____ | ☐ |
| 定位精度 | PnP 仿真端到端 P90 | < 0.3 m（10px 噪声 0.24 m） | _____ m | ☐ |
| 现场标定 | PnP 残差 | < 12 px | _____ px | ☐ |
| 身份召回 | 配对率（可见 GT） | 与上次基准持平或更好 | _____ % | ☐ |
| 图案分类 | diagnose conf 准确率 | 与上次持平或更好 | _____ % | ☐ |
| **主基准** | **union vuln（dwell 序列）** | **≥ 17.2 s（2026 配置）** | _____ s | ☐ |
| 回归 | pytest | 全绿 | _____ | ☐ |

> 任一未达 → 按[操作手册 §13](radar_operation.md) 排查；换车模型必须走全链流程（[训练指南 §3](training_guide.md)），禁止单独换入。

---

## 7. 已知坑（前人踩过的雷）

按"第一次接手最可能撞上"排序：

1. **车模型不能单独换**。新 checkpoint 直接替换 `car_best.engine` → 装甲 ROI 域偏移 → 下游全崩。
   必须走全链重训顺序（[训练指南 §3](training_guide.md)，血泪教训写进了流程）。

2. **TensorRT engine 与 GPU 绑定**。Windows 上编的 `.engine` 在 Linux 容器/别的卡上不能加载。
   拷 `.pt`/`.onnx`，目标机首次推理自动重编。

3. **PnP 与单应的点击目标不同**。PnP 点建筑**顶**（z 在表里），单应点**地面落点**（z 被忽略）。
   用同一张 `landmarks.json` 但点错目标 → 系统性偏差，肉眼难察觉。

4. **仿真实时源只有 ~0.4 fps**。这不是 bug——EEVEE 渲染 4096×3000 就这速度。
   所有基准结论必须走 10 Hz images 回放，实时源只看链路通不通。

5. **CUDNN v9.22 与 PyTorch 自带 9.2 冲突**（Windows 训练机）。症状：训练启动即崩或极慢。
   修复：`tools/pt_to_engine.py` 与 `scripts/run_car_train.py` 开头已自动从 PATH 剥离 CUDNN；
   自己写脚本时记得 `os.environ["PATH"]` 过滤 `"CUDNN"`。

6. **`RADAR_IMAGES_DIR` 默认是 `videoseq` 不是 `videoseq_dwell`**。跑基准必须显式指向
   `videoseq_dwell`，否则序列不对、数字不可比。

7. **Blender 实时窗口 ≠ 渲染帧**。视口截图没有 EEVEE 的曝光/变换，检测会瘸。
   检测端读的是 `live_frame.png`，确认这个文件在更新。

8. **灯条修复链勿重跑**。`scene/armor_tools/archive/` 里的脚本已在当前 blend 上应用过，
   再跑会重复修改（README 里写了重建顺序，仅在全新 blend 上用）。

9. **两台雷达共用一张裁判图，上传不做红蓝镜像**。坐标系细节见
   [协议笔记 §4](referee_protocol_notes.md)；+y 方向联调时用一个布尔翻转，别到处改号。

10. **`armor dead` 单帧不可信**（28% 误闪）。任何基于 dead 的逻辑至少要连续 5 帧
    （`DEAD_FRAMES = 5`），别图省事用单帧。

11. **2027 的遗留依赖**（详见[迁移计划](migration_2027.md)）：点阵屏样式未发布 → 图案分类器
    无法从零重训。通信协议 V1.0.0 线格式已对齐 `referee/`（0x0305 仍 48B，槽 1/4 不发）。
    加固策略和无线解调还没做。
