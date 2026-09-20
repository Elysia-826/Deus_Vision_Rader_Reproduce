@echo off
setlocal
cd /d "%~dp0.."
set "BLENDER=D:\Program Files\blender\blender.exe"
if not exist "%BLENDER%" (
    echo blender not found: %BLENDER%
    exit /b 1
)
set "MOTION=patrol"
if /I "%~1"=="off" set "MOTION=off"
start "blender" "%BLENDER%" scene\RMUC2026_V2.0.0_radar.blend --python scene\armor_assets\radar_sim_loop.py -- --motion %MOTION%
echo Blender started (motion=%MOTION%).
echo Loading detector in this terminal. Minimap is the OpenCV window named radar (not inside Blender). q quits detector.
python scripts\run_radar.py
endlocal
