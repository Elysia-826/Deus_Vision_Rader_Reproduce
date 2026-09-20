# 同时开 Blender 仿真（车在跑）和检测+小地图窗口。
# 仓库根目录执行：
#     powershell -File scripts/start_sim.ps1
#     powershell -File scripts/start_sim.ps1 -Motion off

param(
    [ValidateSet("off", "patrol")]
    [string]$Motion = "patrol"
)

$Root = Split-Path -Parent $PSScriptRoot
$Blender = "D:\Program Files\blender\blender.exe"
$Blend = Join-Path $Root "scene\RMUC2026_V2.0.0_radar.blend"
$Sim = Join-Path $Root "scene\armor_assets\radar_sim_loop.py"
$Radar = Join-Path $Root "scripts\run_radar.py"

if (-not (Test-Path -LiteralPath $Blender)) {
    throw "blender not found: $Blender"
}

Set-Location -LiteralPath $Root

Start-Process -FilePath $Blender -ArgumentList @(
    $Blend,
    "--python", $Sim,
    "--",
    "--motion", $Motion
)

Write-Host "Blender started (motion=$Motion)."
Write-Host "Loading detector in this terminal. Minimap is the OpenCV window named radar (not inside Blender). q quits detector."
python $Radar
