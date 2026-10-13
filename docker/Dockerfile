# ──────────────────────────────────────────────────────────────
# Deus Vision Radar — Linux GPU 推理/训练容器
# 目标机：Ubuntu 26.04 + RTX 5070 Ti (Blackwell sm_120)
# 基础层：nvidia/cuda:12.8 runtime（Blackwell 最低 CUDA 版本）
# ──────────────────────────────────────────────────────────────
#
# 构建（首次约 5-10 分钟，TensorRT 下载较大）：
#     docker build -t deus-radar .
#
# 推理（跑回放基准）：
#     docker run --rm --gpus all \
#       -v $(pwd):/workspace \
#       -e RADAR_SOURCE=images \
#       -e RADAR_HEADLESS=1 \
#       deus-radar python scripts/run_radar.py
#
# 训练（车 YOLO，需要更长 GPU 时间）：
#     docker run --rm --gpus all \
#       -v $(pwd):/workspace \
#       deus-radar python -m models.train car --data models/configs/data_car.yaml
#
# 测试：
#     docker run --rm --gpus all -v $(pwd):/workspace deus-radar pytest
# ──────────────────────────────────────────────────────────────

FROM nvidia/cuda:12.8.0-runtime-ubuntu22.04

# 基础工具 + Python 3.11（Ubuntu 22.04 默认源）
RUN apt-get update && \
    apt-get install -y --no-install-recommends \
        python3.11 python3.11-venv python3-pip \
        libgl1-mesa-glx libglib2.0-0 \
        usbutils \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /workspace

# Python 依赖（TensorRT pip 轮子自带 CUDA 绑定）
COPY docker/requirements.txt /tmp/requirements.txt
RUN pip3 install --no-cache-dir -r /tmp/requirements.txt && \
    rm /tmp/requirements.txt

# 项目代码通过 volume 挂载，不 COPY 进镜像（代码频繁改动，镜像保持干净）
# 实际使用时：
#   docker run -v /path/to/repo:/workspace ...

# 预热提示：首次推理时 TensorRT 需要在容器内重新编译 .engine
# （GPU 绑定：不同 GPU 架构的 engine 不通用）
ENV RADAR_HEADLESS=1
ENV PYTHONUNBUFFERED=1

CMD ["python3", "--version"]
