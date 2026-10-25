"""比赛档只拒绝考核开关，不改默认开发路径。"""

from __future__ import annotations

import sys
from pathlib import Path

import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "scripts"))

from run_radar import match_profile_errors  # noqa: E402


def _ok(
    *,
    match: bool = True,
    source: str = "hik",
    camera_name: str = "Camera_Radar_Red",
    fullscreen: bool = True,
    headless: bool = False,
    stub: bool = False,
    record: bool = False,
    guess: bool = False,
    send_lost: bool = False,
    phase_scale: float = 1.0,
) -> tuple[str, ...]:
    return match_profile_errors(
        match=match,
        source=source,
        camera_name=camera_name,
        fullscreen=fullscreen,
        headless=headless,
        stub=stub,
        record=record,
        guess=guess,
        send_lost=send_lost,
        phase_scale=phase_scale,
    )


def test_match_off_ignores_assessment_flags() -> None:
    assert _ok(match=False, source="blender", stub=True, record=True, phase_scale=0.05) == ()


def test_locked_match_profile_is_clean() -> None:
    assert _ok() == ()
    assert _ok(camera_name="Camera_Radar_Blue") == ()


@pytest.mark.parametrize(
    ("errors", "needle"),
    [
        (_ok(source="blender"), "RADAR_SOURCE"),
        (_ok(camera_name=""), "RADAR_CAMERA_NAME"),
        (_ok(fullscreen=False), "RADAR_FULLSCREEN"),
        (_ok(headless=True), "RADAR_HEADLESS"),
        (_ok(stub=True), "RADAR_REFEREE_STUB"),
        (_ok(record=True), "RADAR_RECORD"),
        (_ok(guess=True), "RADAR_GUESS"),
        (_ok(send_lost=True), "RADAR_SEND_LOST"),
        (_ok(phase_scale=0.05), "RADAR_PHASE_SCALE"),
    ],
)
def test_match_refuses_each_assessment_flag(errors: tuple[str, ...], needle: str) -> None:
    assert len(errors) == 1
    assert needle in errors[0]
