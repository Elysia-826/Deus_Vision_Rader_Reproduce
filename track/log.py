"""每帧 JSONL：真值带 name，预测带槽名。打分读这个，不看窗口。"""

from __future__ import annotations

import json
from pathlib import Path

from track.score import ScoreFrame
from track.truth import JsonScalar, parse_sim_robots
from track.types import PublishedTrack, SimRobot, Team, TrackPhase

_PRED_KEYS = ("slot", "x", "y", "state")


def append_frame(path: Path, frame: int, truth: tuple[SimRobot, ...], pred: tuple[PublishedTrack, ...]) -> None:
    record = {
        "frame": frame,
        "gt": [
            {"name": robot.name, "team": robot.team.value, "x": robot.x, "y": robot.y, "z": robot.z}
            for robot in truth
        ],
        "pred": [
            {"slot": item.label, "x": item.x, "y": item.y, "state": item.phase.value}
            for item in pred
        ],
    }
    with path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def load_score_frames(path: Path) -> tuple[ScoreFrame, ...]:
    frames: list[ScoreFrame] = []
    text = path.read_text(encoding="utf-8")
    for line in text.splitlines():
        if line.strip() == "":
            continue
        raw = json.loads(line)
        gt_items = raw["gt"]
        if not isinstance(gt_items, list):
            raise TypeError(f"gt is not a list in frame {raw.get('frame')}")  # noqa: GENERIC_ERR_OK
        mappings: list[dict[str, JsonScalar]] = []
        for item in gt_items:
            if not isinstance(item, dict):
                raise TypeError("gt item is not an object")  # noqa: GENERIC_ERR_OK
            mappings.append(item)
        frames.append(ScoreFrame(gt=parse_sim_robots(mappings), pred=_pred(raw["pred"])))
    return tuple(frames)


def _pred(raw: list[dict[str, str | int | float | None]]) -> tuple[PublishedTrack, ...]:
    parsed: list[PublishedTrack] = []
    for item in raw:
        label = str(item["slot"])
        team = Team.RED if label.startswith("R") else Team.BLUE
        parsed.append(
            PublishedTrack(
                label=label,
                team=team,
                x=float(item["x"]),
                y=float(item["y"]),
                phase=TrackPhase(str(item["state"])),
                bot_id=None,
            )
        )
    return tuple(parsed)
