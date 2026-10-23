@echo off
rem 雷达进程守护：崩溃自动重启；q 正常退出 / Ctrl+C 不重启。
rem 用法：先在当前终端设好 RADAR_* 环境变量，再启动（继承当前环境）：
rem     scripts\watch_radar.cmd
setlocal
cd /d "%~dp0.."

:loop
python scripts\run_radar.py
if %errorlevel%==0 (
    echo run_radar 正常退出，守护结束。
    exit /b 0
)
if %errorlevel%==-1073741510 (
    echo 收到 Ctrl+C，守护退出。
    exit /b 0
)
echo run_radar 崩溃（退出码 %errorlevel%），3 秒后重启。Ctrl+C 可停守护。
timeout /t 3 >nul
goto loop
