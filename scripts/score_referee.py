# ─────────────────────────────────────────────────────────────────
# How to run
#     RADAR_REFEREE_STUB=1 python scripts/run_radar.py   （回放 + stub 打分）
#     python scripts/score_referee.py                    （读 referee_log 出报告）
# ─────────────────────────────────────────────────────────────────

"""读 referee_log.jsonl：把 stub 裁判的逐帧判定汇总成易伤基准报告。

指标口径（规则手册 5.6.6）：
  d：雷达所发坐标与机器人实际位置的直线距离。准确 <0.8m，半准确 <1.6m，错误 ≥1.6m。
  P：连击计分。敌方 P≥100 记易伤；多台同时易伤不叠加（工会时间取并集）。
  中断：累计 0.5s 没收到该机器人坐标 → 一次错误判定。
"""

from __future__ import annotations

import json
import sys
from pathlib import Path
from typing import Final

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT))

LOG = Path(sys.argv[1]) if len(sys.argv) > 1 else _ROOT / "scene" / "locate_out" / "referee_log.jsonl"

ENEMY_SLOTS: Final[tuple[str, ...]] = ("E1", "E2", "E3", "E4", "E6", "E7")
JUDGES: Final[tuple[str, ...]] = ("accurate", "semi", "error", "interrupt", "silent")


def main() -> None:
    records = [json.loads(line) for line in LOG.read_text(encoding="utf-8").splitlines() if line.strip()]
    if not records:
        raise SystemExit(f"empty log: {LOG}")
    duration_s = records[-1]["t"] - records[0]["t"]

    names = sorted({name for record in records for name in record["slots"]})
    counts = {name: {j: 0 for j in JUDGES} for name in names}
    vuln = {name: 0.0 for name in names}
    union_s = 0.0
    p_final = {name: 0.0 for name in names}
    d_values: dict[str, list[float]] = {name: [] for name in names}
    sent_frames = 0
    any_sent = 0

    previous_t: float | None = None
    for record in records:
        t = record["t"]
        dt = 0.0 if previous_t is None else max(0.0, t - previous_t)
        previous_t = t
        frame_sent = False
        enemy_vuln = False
        for name, slot in record["slots"].items():
            judge = slot["judge"]
            if judge in counts[name]:
                counts[name][judge] += 1
            if slot["vuln"]:
                vuln[name] += dt
                if name in ENEMY_SLOTS:
                    enemy_vuln = True
            if slot["sent"]:
                frame_sent = True
            if slot["d"] is not None:
                d_values[name].append(slot["d"])
        if frame_sent:
            sent_frames += 1
        if enemy_vuln:
            union_s += dt
        for name, slot in record["slots"].items():
            p_final[name] = slot["p"]

    active = [name for name in names if any(counts[name][j] for j in JUDGES if j != "silent")]
    print(f"log          {LOG}")
    print(f"frames       {len(records)}  ({duration_s:.1f}s @ {len(records) / max(duration_s, 1e-9):.1f}Hz)")
    print(f"slots seen   {', '.join(active)}")
    print()
    header = f"{'slot':<5} {'P':>7} {'vuln_s':>7} {'acc':>5} {'semi':>5} {'err':>5} {'intr':>5} {'d_med':>7}"
    print(header)
    for name in active:
        c = counts[name]
        sent = c["accurate"] + c["semi"] + c["error"]
        ds = sorted(d_values[name])
        d_med = f"{ds[len(ds) // 2]:.2f}" if ds else "-"
        print(
            f"{name:<5} {p_final[name]:>7.1f} {vuln[name]:>7.2f} "
            f"{c['accurate']:>5} {c['semi']:>5} {c['error']:>5} {c['interrupt']:>5} {d_med:>7}"
            + (f"  (sent {sent})" if sent else "")
        )
    total = {j: sum(counts[name][j] for name in names) for j in JUDGES}
    sent_total = total["accurate"] + total["semi"] + total["error"]
    if sent_total:
        print()
        print(
            f"mark frames  {sent_frames}/{len(records)}"
            f"   accurate {total['accurate'] / sent_total:.1%}"
            f"   semi {total['semi'] / sent_total:.1%}"
            f"   error {total['error'] / sent_total:.1%}"
        )
    print(f"interrupts   {total['interrupt']}")
    print(f"union vuln   {union_s:.2f}s of {duration_s:.1f}s   double chances {int(union_s // 60.0)}")


if __name__ == "__main__":
    main()
