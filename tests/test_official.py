from __future__ import annotations

import sys
from pathlib import Path

import cv2
import numpy as np
import pytest

_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(_ROOT / "scripts"))

import visualize_official as vo  # noqa: E402
from track.types import PublishedTrack, SimRobot, Team, TrackPhase  # noqa: E402


def _publish(label: str, team: Team, x: float, y: float, *, phase: TrackPhase = TrackPhase.CONFIRMED) -> PublishedTrack:
    return PublishedTrack(label=label, team=team, x=x, y=y, phase=phase, bot_id=1)


def _gt(label: str, team: Team, x: float, y: float) -> SimRobot:
    return SimRobot(name=label, team=team, x=x, y=y, z=0.0, label=label)


# ---------- 兵种叫法 ----------


def test_role_name_uses_2027_chinese_roles() -> None:
    assert vo.role_name("R1", Team.RED) == "红方重装"
    assert vo.role_name("B3", Team.BLUE) == "蓝方步兵"
    assert vo.role_name("R4", Team.RED) == "红方步兵"
    assert vo.role_name("BS", Team.BLUE) == "蓝方哨兵"
    assert vo.role_name("RQ", Team.RED) == "红方前哨"
    assert vo.role_name("RX", Team.RED) == "RX"  # 认不了的槽号原样返回


def test_role_name_ascii_fallback_without_cjk() -> None:
    assert vo.role_name("R1", Team.RED, cjk=False) == "REDHVY"
    assert vo.role_name("BS", Team.BLUE, cjk=False) == "BLUSNT"


# ---------- 误差与字幕 ----------


def test_error_vs_truth_matches_by_label() -> None:
    track = _publish("R3", Team.RED, 2.0, 1.0)
    truth = (_gt("R3", Team.RED, 2.3, 1.4), _gt("B1", Team.BLUE, 0.0, 0.0))
    assert vo.error_vs_truth(track, truth) == pytest.approx(0.5)


def test_error_vs_truth_returns_none_without_truth() -> None:
    track = _publish("R3", Team.RED, 2.0, 1.0)
    assert vo.error_vs_truth(track, ()) is None
    assert vo.error_vs_truth(track, (_gt("B1", Team.BLUE, 0.0, 0.0),)) is None


def test_caption_text_carries_accuracy_marks() -> None:
    accurate = _publish("R3", Team.RED, 2.0, 1.0)
    text = vo.caption_text(accurate, 0.32)
    assert "红方步兵" in text
    assert "(   2.00,   1.00 )" in text
    assert "0.32m" in text and " T" in text  # 准确线内 → T（大写，纯 ASCII 不挑字体）

    wrong = vo.caption_text(_publish("R3", Team.RED, 2.0, 1.0), 1.4)
    assert " F" in wrong  # 超过 0.8m 准确线 → F

    unknown = vo.caption_text(_publish("R3", Team.RED, 2.0, 1.0), None)
    assert "误差 --" in unknown  # 真机没真值：留位不臆造

    coasting = vo.caption_text(_publish("R3", Team.RED, 2.0, 1.0, phase=TrackPhase.LOST), 0.2)
    assert "〔外推〕" in coasting  # 盲推坐标要看得出来


def test_caption_rows_sorts_enemy_first() -> None:
    publishes = (
        _publish("B1", Team.BLUE, 1.0, 1.0),
        _publish("R3", Team.RED, 2.0, 0.0),
    )
    rows = vo.caption_rows(publishes, (), own_team=Team.BLUE)
    assert rows[0][0].label == "R3"  # 敌方在前：裁判只按敌方记分


def test_band_lines_wraps_and_truncates() -> None:
    publishes = tuple(_publish(f"R{i}", Team.RED, float(i), 0.0) for i in range(1, 8))
    rows = vo.caption_rows(publishes, (), cjk=True)
    lines = vo._band_lines(rows)
    assert len(lines) == 3  # 3 行 × 2 条 = 6 条上屏
    assert "(+1)" in lines[-1]  # 第 7 条被截断计数


# ---------- 1080p 合成 ----------


def _fake_camera(w: int = 4096, h: int = 3000) -> vo.ImageU8:
    gradient = np.linspace(30, 200, w, dtype=np.uint8)
    return np.repeat(gradient[np.newaxis, :, np.newaxis], h, axis=0) * np.ones((h, w, 3), dtype=np.uint8)


def _fake_board() -> vo.ImageU8:
    board = np.full((400, 700, 3), 40, dtype=np.uint8)
    cv2.rectangle(board, (0, 0), (699, 399), (200, 200, 200), 3)
    return board


def test_official_frame_is_fixed_1080p() -> None:
    publishes = (
        _publish("R1", Team.RED, 2.31, 1.05),
        _publish("B3", Team.BLUE, -3.0, 0.5, phase=TrackPhase.LOST),
    )
    truth = (_gt("R1", Team.RED, 2.0, 1.0),)
    frame = vo.build_official_frame(
        _fake_camera(), _fake_board(), publishes, truth, frame_no=42, fps=9.8, own_team=Team.BLUE
    )
    assert frame.shape == (vo.OFFICIAL_H, vo.OFFICIAL_W, 3)
    # 左侧相机区不是全黑（画面真的铺进去了）
    assert frame[:200, :200].mean() > 10
    # 右侧面板顶部贴了区域图（非黑）
    panel_x = vo.OFFICIAL_W - vo._PANEL_W + vo._MARGIN
    assert frame[60:100, panel_x : panel_x + 100].mean() > 5


def test_official_frame_ascii_fallback_without_font(monkeypatch: pytest.MonkeyPatch) -> None:
    """缺中文字体也不能崩：字幕退英文缩写，形状不变。"""
    monkeypatch.setattr(vo, "_font_path_found", None)
    monkeypatch.setattr(vo, "_font_searched", True)
    assert vo.cjk_available() is False
    frame = vo.build_official_frame(
        _fake_camera(), _fake_board(), (_publish("R3", Team.RED, 1.0, 1.0),), (), frame_no=1, fps=10.0
    )
    assert frame.shape == (vo.OFFICIAL_H, vo.OFFICIAL_W, 3)


def test_official_frame_with_empty_publishes() -> None:
    frame = vo.build_official_frame(_fake_camera(), _fake_board(), (), (), frame_no=0, fps=10.0)
    assert frame.shape == (vo.OFFICIAL_H, vo.OFFICIAL_W, 3)


# ---------- 录像出口 ----------


def test_recorder_writes_playable_mp4(tmp_path: Path) -> None:
    recorder = vo.OfficialRecorder(tmp_path / "out.mp4", fps=10.0)
    for i in range(5):
        frame = vo.build_official_frame(
            _fake_camera(),
            _fake_board(),
            (_publish("R3", Team.RED, float(i), 0.0),),
            (),
            frame_no=i,
            fps=10.0,
        )
        recorder.write(frame)
    recorder.release()
    out = tmp_path / "out.mp4"
    if not out.is_file():
        out = tmp_path / "out.avi"  # mp4v 缺编码器时的回退
    assert out.is_file() and out.stat().st_size > 10_000
    cap = cv2.VideoCapture(str(out))
    assert cap.isOpened()
    ok, first = cap.read()
    cap.release()
    assert ok and first.shape[:2] == (vo.OFFICIAL_H, vo.OFFICIAL_W)


def test_recorder_rejects_wrong_size(tmp_path: Path) -> None:
    recorder = vo.OfficialRecorder(tmp_path / "out.mp4", fps=10.0)
    with pytest.raises(ValueError):
        recorder.write(np.zeros((720, 1280, 3), dtype=np.uint8))
    recorder.release()


# ---------- 字形完整性（考核画面的字不许是空框） ----------


def test_caption_glyphs_are_not_tofu() -> None:
    """字幕/表用到的每个非 ASCII 字符必须在字体里真实存在。

    背景实测：✓(U+2713)/✗(U+2717) 在微软雅黑里是 notdef 空框——
    误差标记因此改用 T/F。本测试守住「不会再放进任何缺字」：
    对每个用到的非 ASCII 字符渲染墨迹，与 notdef 基准比对。
    """
    if not vo.cjk_available():
        pytest.skip("本机没有中文字体，CJK 路径不会被走到")
    from PIL import Image, ImageDraw, ImageFont

    import numpy as np_img  # noqa: F401 — numpy 已在顶部导入

    font = ImageFont.truetype(str(vo.find_font_path()), 22)

    def ink(ch: str) -> bytes:
        img = Image.new("L", (40, 40), 0)
        ImageDraw.Draw(img).text((2, 2), ch, fill=255, font=font)
        return np.asarray(img).tobytes()

    # 收齐所有会渲染的非 ASCII 字符：兵种表、阵营表、固定面板文案、
    # 误差/外推标记，以及全部兵种 × 阵营的实际 caption 串。
    corpus = "".join(vo._ROLE_CN.values()) + "".join(vo._TEAM_CN.values())
    corpus += "自划区域图数据表（与字幕一致）误差准超外推无发布槽位画"
    corpus += "〔〕（），"
    for label in ("R1", "R2", "R3", "R4", "RS", "RQ"):
        for team in (Team.RED, Team.BLUE):
            for phase in (TrackPhase.CONFIRMED, TrackPhase.LOST):
                corpus += vo.caption_text(
                    _publish(label, team, 1.0, 2.0, phase=phase), 0.5, cjk=True
                )
    non_ascii = {ch for ch in corpus if ord(ch) > 0x7F}
    tofu_reference = ink("\ue000")  # PUA 确定缺字 → notdef 基准
    for ch in sorted(non_ascii):
        assert ink(ch) != tofu_reference, f"U+{ord(ch):04X} 在 {vo.find_font_path()} 里是 notdef 空框"
