#!/usr/bin/env bash
# 雷达进程守护：崩溃自动重启；q 正常退出 / Ctrl+C 不重启。
# 用法：先 export RADAR_* 环境变量，再启动（继承当前环境）：
#     scripts/watch_radar.sh
set -u
cd "$(dirname "$0")/.." || exit 1

while true; do
  python scripts/run_radar.py
  code=$?
  if [ "$code" -eq 0 ]; then
    echo "run_radar 正常退出，守护结束。"
    exit 0
  fi
  echo "run_radar 崩溃（退出码 $code），3 秒后重启。Ctrl+C 可停守护。"
  sleep 3
done
