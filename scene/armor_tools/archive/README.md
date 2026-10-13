# 已应用的一次性修复脚本

这些脚本已经跑过、blend 已修改。不要在当前 .blend 上重复执行
（会重复修改参数或产生意外效果）。保留在这里作为修复过程的文档。

如需在新 .blend 上从头重建修复链，按以下顺序执行：
  1. fix_lamp_strength.py   （自发光强度 40 → 6）
  2. fix_view_transform.py  （AgX → Standard）
  3. fix_lamp_color.py      （纯净发射色 + 强度微调）
