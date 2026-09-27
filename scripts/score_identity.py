# ─── How to run ───
#     python scripts/score_identity.py
#     python scripts/score_identity.py scene/locate_out/identity_log.jsonl
# ──────────────────

"""用仿真 JSONL 数每个真值 name 的身份切换。不看真实录像。"""

from __future__ import annotations

import sys
from pathlib import Path

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

from track.log import load_score_frames  # noqa: E402
from track.score import score_frames  # noqa: E402

DEFAULT = _ROOT / "scene" / "locate_out" / "identity_log.jsonl"


def main() -> None:
    path = Path(sys.argv[1]) if len(sys.argv) > 1 else DEFAULT
    report = score_frames(load_score_frames(path))
    if not report.by_name:
        print(f"no named robots in {path}")
        return
    for item in report.by_name:
        print(f"{item.name}  switches={item.switches}  misses={item.misses}")


if __name__ == "__main__":
    main()
