#!/usr/bin/env bash
# 比赛档：锁死上场开关，再交给守护进程。考核/打分不要用这个脚本。
# 用法：scripts/match_radar.sh Blue
#       scripts/match_radar.sh Red
# USB 第 0 路是默认。RTSP 先 export RADAR_HIK_URL 再启动。
set -u
cd "$(dirname "$0")/.." || exit 1

side="${1:-}"
case "$side" in
  Blue|blue) export RADAR_CAMERA_NAME=Camera_Radar_Blue ;;
  Red|red) export RADAR_CAMERA_NAME=Camera_Radar_Red ;;
  *)
    echo "用法: scripts/match_radar.sh Blue|Red" >&2
    exit 1
    ;;
esac

export RADAR_MATCH=1
export RADAR_SOURCE=hik
export RADAR_FULLSCREEN=1
export RADAR_HEADLESS=0
export RADAR_REFEREE_STUB=0
export RADAR_RECORD=0
export RADAR_GUESS=0
export RADAR_SEND_LOST=0
export RADAR_PHASE_SCALE=1

exec scripts/watch_radar.sh
