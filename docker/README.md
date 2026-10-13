# Docker 环境搭建指南

## 前提（在 Linux 机器上执行一次）

```bash
# 1. 装 Docker
curl -fsSL https://get.docker.com | sh
sudo usermod -aG docker $USER  # 免 sudo（重新登录生效）

# 2. 装 NVIDIA Container Toolkit（容器访问 GPU 的桥）
sudo apt-get install -y nvidia-container-toolkit
sudo nvidia-ctk runtime configure --runtime=docker
sudo systemctl restart docker

# 3. 验证 GPU 透通
docker run --rm --gpus all nvidia/cuda:12.8-base-ubuntu22.04 nvidia-smi
# ↑ 能打出显卡信息 = GPU 透通畅了
```

## 构建

```bash
cd /path/to/DEUS_VISION_RADER_TEST_reproduce
docker compose build
```

首次构建约 5-10 分钟（TensorRT pip 轮子 ~600MB）。

## 使用

```bash
# 全量测试
docker compose run --rm radar pytest

# 跑回放基准（当前最优：dwell 序列 17.2s union vuln）
docker compose run --rm radar python scripts/run_radar.py

# 训练车 YOLO（20 epochs，5070 Ti 预计 3-5 小时）
docker compose run --rm radar python -m models.train car \
    --data models/configs/data_car.yaml --imgsz 1280 --batch 8 --epochs 20

# 训练装甲 YOLO（60 epochs，约 30 分钟）
docker compose run --rm radar python -m models.train armor \
    --data models/configs/data_armor.yaml --imgsz 192 --batch 64 --epochs 60

# 训练图案分类器（30 epochs，约 15 分钟）
docker compose run --rm radar python -m models.pattern \
    --data dataset/pattern_balanced --epochs 30 --name mobilenetv3-docker

# stub 裁判基准报告
docker compose run --rm radar python scripts/score_referee.py
```

## 注意事项

1. **TensorRT engine 是 GPU 绑定的**：Windows 上编的 `.engine` 在容器内不能用。
   容器内首次推理时会自动从 `.pt`/`.onnx` 重编译。
2. **权重不进 Docker 镜像**：通过 volume 挂载项目根目录，`weights/` 随代码一起挂进去。
3. **相机/串口**：`docker-compose.yml` 里的 `devices:` 段默认注释掉了，
   真机联调时取消注释。
4. **Ubuntu 26.04 兼容性**：Docker base 镜像（Ubuntu 22.04）独立于宿主机版本，
   只依赖内核 NVIDIA 驱动。只要宿主机 `nvidia-smi` 正常，容器就能用 GPU。

## 已知限制

- 车 YOLO 重训在笔记本上太慢（1h/epoch），需要在这台 Linux 机器的 Docker 里跑
- E3 步兵的易伤依赖巡逻停顿行为（`DWELL_S=3.0`），这是仿真特性不是模型能力
