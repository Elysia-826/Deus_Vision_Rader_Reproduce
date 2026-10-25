@echo off
rem 比赛档：锁死上场开关，再交给守护进程。考核/打分不要用这个脚本。
rem 用法：scripts\match_radar.cmd Blue
rem       scripts\match_radar.cmd Red
rem USB 第 0 路是默认。RTSP 先设 RADAR_HIK_URL 再启动。
setlocal
cd /d "%~dp0.."

if /I "%~1"=="Blue" (
    set RADAR_CAMERA_NAME=Camera_Radar_Blue
) else if /I "%~1"=="Red" (
    set RADAR_CAMERA_NAME=Camera_Radar_Red
) else (
    echo 用法: scripts\match_radar.cmd Blue^|Red
    exit /b 1
)

set RADAR_MATCH=1
set RADAR_SOURCE=hik
set RADAR_FULLSCREEN=1
set RADAR_HEADLESS=0
set RADAR_REFEREE_STUB=0
set RADAR_RECORD=0
set RADAR_GUESS=0
set RADAR_SEND_LOST=0
set RADAR_PHASE_SCALE=1

call scripts\watch_radar.cmd
exit /b %errorlevel%
