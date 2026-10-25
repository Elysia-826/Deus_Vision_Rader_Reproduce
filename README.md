# Deus Vision Radar

港科大 [RM2025-Radar-Algorithm](https://github.com/hkustenterprize/RM2025-Radar-Algorithm) 起步的单目雷达，现已面向 RM2027 独立演进。骨干 YOLO26，图案 MobileNetV3-Small @ 64。场景已切换到 RMUC2027。

## 文档导航

| 想做什么 | 看哪份 |
|---|---|
| 跑起来 / 标定 / 打分 / 排错 | [操作手册](docs/radar_operation.md) |
| 改代码前理解各模块设计 | [架构深讲](docs/architecture.md) |
| 新人接手 / 赛日现场检查 | [接手与赛日清单](docs/handover_checklist.md) |
| 训练 / 收割数据 / 导出换入 | [训练与导出指南](docs/training_guide.md) |
| RM2027 场地与规则迁移 | [迁移计划](docs/migration_2027.md) |
| 裁判串口协议细节 | [协议笔记](docs/referee_protocol_notes.md) |
| Linux GPU 容器 | [Docker 指南](docker/README.md) |

## 当前能跑

车辆 YOLO → 装甲 YOLO → 图案分类；仿真射线定位或上场 PnP 标定（推荐）；相机画面内嵌小地图（实心=看见、空圈=外推）；固定兵种槽级联跟踪（身份历史投票 + 发布门 + 转弯自适应过程噪声）；裁判 stub 基准（P 公式离线复刻 + 比赛阶段 + 0x0301 哨兵引导）。

## 分支

- `main`：图案 MobileNetV3-Small @ 64，权重 `weights/pattern_best.engine`。
- `pattern-efficientnet-b0`：图案对照实验，EfficientNet-B0 @ 96。其余与 main 相同。

## 目录

```
detect/            车 → 装甲推理（BoT-SORT 在 detect/trackers/）
models/            YOLO26 + MobileNetV3 训练
locate/            像素 → 场地米制 → 俯视图（射线 / 地标单应 / PnP 标定）
referee/           裁判串口协议 + stub 裁判 + 0x0301 哨兵引导 + 双倍易伤策略
track/             槽位级联跟踪 + 战略猜点 + 身份发布门
scripts/           运行、标定、打分、诊断（diagnose.py）
tools/             pt → onnx / TensorRT
tests/             pytest
docs/              操作手册、架构深讲、接手清单、训练指南、协议笔记、2027 迁移计划
docker/            Linux GPU 容器（Dockerfile + compose）
dataset/           收割 + 划分脚本（图像不进 git）
weights/           本地权重（不进 git）
scene/             RMUC2027 Blender 仿真（CAD/贴图不进 git）
refs/              官方协议/规则手册本地留档（不进 git）
zaowu_audit/       场地建模审计工作区（在用，不进 git）
```

## 和参考方案的差异

| | 港科大 2025 | 本仓 |
|---|---|---|
| 车检测 | YOLOv12-s @ 1280 | YOLO26s @ 1280 |
| 装甲检测 | YOLOv12-n @ 192 | YOLO26n @ 192 |
| 图案 | MobileNet-V2，带颜色 | MobileNetV3-Small @ 64，6 类 `(1,2,3,4,S,Q)` |
| 定位 | 像素射线打场地 mesh | 仿真射线 + 真机 PnP 6 点（推荐）/ 四点单应（备选） |
| 跟踪 | 匈牙利 + 卡尔曼 | 级联匹配 + 身份投票 + 发布门 + 自适应 Q |
| 裁判系统 | 有（C++） | Python 骨架 + stub 裁判 + 离线基准 |
| 哨兵引导 | — | 0x0301 机器人间通信 |

## 命令（仓库根目录）

```bash
# 训练
python -m models.train car --data models/configs/data_car.yaml
python -m models.train armor --data models/configs/data_armor.yaml
python -m models.pattern --data dataset/pattern_balanced --epochs 30

# 运行（仿真 / 回放 / 真相机）
python scripts/run_radar.py                                        # Blender 仿真
RADAR_SOURCE=images RADAR_HEADLESS=1 python scripts/run_radar.py  # 回放
RADAR_SOURCE=hik RADAR_HIK_URL=0 python scripts/run_radar.py      # USB 相机
RADAR_SOURCE=hik RADAR_HIK_URL=rtsp://... python scripts/run_radar.py  # RTSP
scripts\watch_radar.cmd                                            # 真机守护：崩溃自动重启（先设好环境变量再启动）
scripts\match_radar.cmd Red                                       # 比赛档：锁 hik + 全屏，拒绝考核开关，再交给守护

# 标定
python scripts/calibrate_pnp.py frame.jpg --intrinsics <K.json>  # 真机 PnP（推荐，点击后有 WASD 放大镜微调）
python scripts/validate_pnp.py                                    # 仿真精度验证
python scripts/calibrate_landmarks.py                             # 单应（备选）

# 测试
pytest
```

## 打分与诊断

```bash
python scripts/eval_foot_ray.py --weights <best.pt>   # 底边射线验收
python scripts/score_identity_log.py [log.jsonl]        # 召回/误差/野点
python scripts/score_stage_funnel.py                    # GT→车→装甲→图案漏斗
python scripts/score_referee.py [log.jsonl]             # stub 裁判：易伤/准确率/P
python scripts/diagnose.py --mode conf                  # 图案分类精度
python scripts/diagnose.py --mode lamp                   # 装甲颜色误读率
python scripts/diagnose.py --mode confusion             # 槽→GT 混淆表
python scripts/diagnose.py --mode digits                # 1↔3 数字混淆
```

## stub 裁判基准

规则手册 P 计分公式的离线复刻（含比赛阶段 + 双倍易伤策略）：

```bash
RADAR_REFEREE_STUB=1 RADAR_SEND_LOST=1 python scripts/run_radar.py
python scripts/score_referee.py
```

环境变量：
- `RADAR_SEND_LOST=1` — LOST 外推也发 0x0305（实测 +67% 易伤秒）
- `RADAR_GUESS=1` — LOST ≥0.5s（中断线）后坐标换战略猜点；幽灵发布窗内真实可触发，近似点位表下净值中性，默认关
- `RADAR_PHASE_SCALE=<float>` — 比赛阶段压缩（0.05 → 准备 9s，0.0001 → 准备 18ms）
- `RADAR_SENTRY_GUIDE=1` — 启用 0x0301 哨兵引导

## 环境变量总表

| 变量 | 默认 | 说明 |
|---|---|---|
| `RADAR_MATCH` | `0` | 1 = 比赛档。必须 hik、全屏、相机名 Blue/Red；拒绝 stub、录像、猜点、外推发布、相位压缩。上场用 `scripts/match_radar.*`，不要手设 |
| `RADAR_SOURCE` | `blender` | 图源：blender / video / images / hik |
| `RADAR_HEADLESS` | `0` | 1 = 不弹窗口 |
| `RADAR_IMAGES_DIR` | `scene/locate_out/videoseq` | images 源目录（跑基准用 `videoseq_dwell`） |
| `RADAR_HIK_URL` | `0` | USB 设备号或 RTSP 地址 |
| `RADAR_REFEREE_STUB` | `0` | 1 = 启用 stub 裁判 |
| `RADAR_REFEREE_LOG` | `scene/locate_out/referee_log.jsonl` | stub 裁判日志路径 |
| `RADAR_SEND_LOST` | `0` | 1 = LOST 槽也发标记 |
| `RADAR_GUESS` | `0` | 1 = LOST ≥0.5s 换战略猜点（时长门 + σ 副门） |
| `RADAR_PHASE_SCALE` | `0.05` | 比赛阶段时间压缩比 |
| `RADAR_SENTRY_GUIDE` | `0` | 1 = 0x0301 哨兵引导 |
| `RADAR_CAMERA_NAME` | `Camera_Radar_Blue` | 真机相机名（红方上场必须设 Red，决定外参 + 颜色先验） |
| `RADAR_RECORD` | `0` | 1 = 录 1920×1080 官方合成（相机画面 + 自划区域图 + 字幕/数据表）+ 同步 CSV 数据记录表（`同名.csv`，帧号与视频右下角一致，考核交表用） |
| `RADAR_RECORD_PATH` | `scene/locate_out/official_record.mp4` | 录像输出路径 |
| `RADAR_RECORD_FPS` | `10` | 录像帧率（仿真 10Hz；真机相机源设 25/30） |
| `RADAR_FULLSCREEN` | `0` | 1 = 显示窗口全屏（上场机接基座 HDMI 屏） |
| `RADAR_FONT_PATH` | 自动探测 | 考核字幕中文字体路径；找不到时退 ASCII 并打警告 |
| `RADAR_IDENTITY_LOG` | `.../identity_log.jsonl` | identity 日志路径 |
| `RADAR_AUTO_ENGINE` | `1` | 0 = 关闭"engine 缺失自动编译"（pt→engine 子进程重编，失败回退 pt） |

> 完整口径与用法见[操作手册 §12](docs/radar_operation.md)；`LOCATE_MODE` 等代码内开关不在环境变量里。

## 基准历史

| 场地 | 配置 | union vuln | 准确率 |
|---|---|---|---|
| 2026 + 旧模型 | 修复前 | 7.1s | 26% |
| 2026 + v2 模型 | 修复后 | 15.2s | 47% |
| 2026 + v3 + 发布门 | + 停顿行为 | **17.2s** | **50.9%** |
| 2027 + 旧模型 | 新场地直跑 | 9.6s | 44.3% |
| 2027 + 双寿命跟踪 + 多装甲投票 | 幽灵发布窗 + 时长门猜点 + 离线优先接线 | 9.6s | 48.0% |

> 2027 场地用 2026 训练的模型直跑得 9.6s——d_med 0.02-0.54m 说明新场地更利于检测，
> 但模型还是旧域的，全链重训后预期显著提升。

### 为什么只换地图，检测和定位会变好

模型和脚本没换。变好的是检出框打到地上之后的落点（d_med 0.02–0.54 m），不是比赛分：易伤秒 17.2s → 9.6s，准确率 50.9% → 44.3%。两套序列不同，总分不能横比。

雷达帧对照（`videoseq_dwell` vs `videoseq_2027_dwell`，同号帧）：

1. **2026 近场是半透明叠层。** 墙、坡、标线互相穿插，假边缘盖住真车。2027 是实心单层，轮廓干净。
2. **对比度。** 2026 灰压灰；2027 是浅灰地砖、黑地面、亮蓝/红线，车是一块暗斑，旧检测器更好认。
3. **灯条颜色。** 自发光约 40 把辐射度推到约 100，AgX 把高光去饱和成白，装甲 YOLO 把蓝灯条 71% 读成红。2027 blend 走了强度下调 + Standard 视图变换，灯条保住饱和色。这是场景属性，不是权重改动。
4. **相机和外参一起换了。** 蓝雷达从 `(13.17, 0, 3.70)` 到 `(14.61, 0, 3.80)`，稍高、稍远。定位用新导出的外参打新 mesh，框一旦找对，落点就能到几厘米到半米。

总分更差，是因为图案分类器仍认 2026 印刷数字，2027 的隧道、机库、制高点也对不上旧猜点。身份错了就不发标记，易伤秒就少。全链重训之前，比赛分不会跟着涨。

## 2027 遗留项

1. **点阵屏样式未发布** — 图案分类器从零重训被阻塞（最大依赖）
2. **2027 通信协议未发布** — 0x0305 槽位格式、加固命令字、哨兵消息格式
3. **全链重训** — 新场地渲染 → 收割 → 车 → 装甲 → 图案，预计 6-8 小时（需 5070 台式机）
4. **自适应装甲加固** — 2027 全新机制，雷达决策加固中心点（见迁移计划 §6.1）
5. **哨兵→雷达接收** — 缺接收方向的数据链（见迁移计划 §6.2）
6. **车模型级联依赖** — 新车模型 checkpoint 不能单独换入（下游 ROI 域偏移会崩链）

## Windows 仿真

```bat
scripts\start_sim.cmd
```

小地图在 OpenCV 窗口 `radar` 右下角（实心=看见，空圈+?=外推）。

权重从 `weights/` 读，不要提交。