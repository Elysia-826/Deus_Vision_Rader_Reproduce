# 已应用的一次性修复脚本

这些脚本已经跑过、blend 已修改。不要在当前 .blend 上重复执行
（会重复修改参数或产生意外效果）。保留在这里作为修复过程的文档。

如需在新 .blend 上从头重建修复链，按以下顺序执行：
  1. fix_lamp_strength.py   （自发光强度 40 → 6）
  2. fix_view_transform.py  （AgX → Standard）
  3. fix_lamp_color.py      （纯净发射色 + 强度微调）

其余一次性工具（同样勿在当前 blend 重跑，仅重建/排查时参考）：

  - set_radar_lens_12mm.py     雷达镜头切 12mm 的视场调整
  - fix_visibility_overlay.py  可见度平面：视口可开关 + 场上可见（2026-10 自 armor_assets 移入）
  - restore_robots.py          取消所有机器人隐藏并保存，场景恢复工具（2026-10 自 armor_assets 移入）
  - _build_official.py         RM2026 官方贴图 → 装甲面 albedo 构建器（2026-10 自 armor_assets 移入）
  - _build_textures.py         RM2026 附录贴图 + LED 条 → YOLO 装甲贴图构建器（2026-10 自 armor_assets 移入）
