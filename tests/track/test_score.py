from __future__ import annotations

from pathlib import Path

from track.log import append_frame, load_score_frames
from track.score import ScoreFrame, score_frames
from track.types import PublishedTrack, SimRobot, Team, TrackPhase


def _robot(name: str, team: Team, x: float, label: str) -> SimRobot:
    return SimRobot(name=name, team=team, x=x, y=0.0, z=0.2, label=label)


def _point(label: str, team: Team, x: float) -> PublishedTrack:
    return PublishedTrack(label=label, team=team, x=x, y=0.0, phase=TrackPhase.CONFIRMED, bot_id=1)


def test_slot_name_change_is_counted_per_robot() -> None:
    red = _robot("Robot_3_Red", Team.RED, 0.0, "R3")
    blue = _robot("Robot_1_Blue", Team.BLUE, 4.0, "B1")
    stuck = ScoreFrame(
        gt=(red, blue),
        pred=(_point("R3", Team.RED, 0.1), _point("B1", Team.BLUE, 4.1)),
    )
    swapped = ScoreFrame(
        gt=(red, blue),
        pred=(_point("R3", Team.RED, 4.0), _point("B1", Team.BLUE, 0.1)),
    )
    report = score_frames((stuck, swapped))
    assert report.switches("Robot_3_Red") == 1
    assert report.switches("Robot_1_Blue") == 1


def test_stable_binding_has_zero_switches(tmp_path: Path) -> None:
    red = _robot("Robot_3_Red", Team.RED, 1.0, "R3")
    pred = _point("R3", Team.RED, 1.1)
    path = tmp_path / "identity_log.jsonl"
    append_frame(path, 1, (red,), (pred,))
    append_frame(path, 2, (red,), (pred,))
    report = score_frames(load_score_frames(path))
    assert report.switches("Robot_3_Red") == 0
    assert report.misses("Robot_3_Red") == 0
