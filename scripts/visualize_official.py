"""1920×1080 官方合成输出：基座 HDMI 屏（规则手册 §4.2.5）+ 6 分考核录像。

官方基座只收一根 HDMI 公头线，屏是 1920×1080——相机 4096×3000 的
原生 overlay 从这根线上不过去。本文件把任意分辨率的画面压成固定
1080p 三层合成：

  ┌────────────────────┬─────────────┐
  │ 相机画面（等比缩放）   │  自划区域图   │
  │ 4096×3000→1474×1080 │  面板顶部     │
  │                    ├─────────────┤
  │                    │  数据表       │
  ├────────────────────┤  兵种/坐标/误差│
  │ 字幕条（与表同文）    │              │
  └────────────────────┴─────────────┘

6 分考核三条硬要求对到这里：
- 自划区域图 + 位置标在图上 → 右侧面板画的是我们自己的场地图，
  每个发布槽一个点 + 轨迹 + 外推空圈（overlay 产出的 board）。
- 字幕和数据表一致 → 底部字幕条与面板数据表渲染**同一份字符串**
  （caption_text 的返回值，两处不各写各的）。
- 坐标标米制 + 误差 → 每行 `( x, y )m  误差 d m T/F`；误差对真值
  （仿真/回放有 GT）算欧氏距离，T = 0.8m 准确线内 / F = 超线。
  真机没真值显示 `--`，考核现场拿卷尺位置对着表核对。

兵种名用 2027 叫法（重装/步兵/哨兵），不用 R3/B1 槽号——录像层
先把人话标上，槽位结构等 2027 协议发布再收。工程/前哨也给了名字
防旧场景漏网。

中文字体：RADAR_FONT_PATH 优先，其次常见 Windows/Linux 字体；
找不到退 ASCII（红方重装 → RED-HVY），调用方启动时警告一次。
"""

from __future__ import annotations

import os
from pathlib import Path
from typing import Final

import cv2
import numpy as np
from numpy import uint8
from numpy.typing import NDArray
from PIL import Image, ImageDraw, ImageFont

from track.types import PublishedTrack, SimRobot, Team, TrackPhase

ImageU8 = NDArray[uint8]

OFFICIAL_W: Final = 1920
OFFICIAL_H: Final = 1080
_ACCURATE_M: Final = 0.8  # 裁判准确线：误差 ✓/✗ 的分界

_PANEL_W: Final = 424  # 右侧面板宽（区域图 + 数据表）
_MARGIN: Final = 8
_BAND_H: Final = 78  # 底部字幕条高
_BAND_PER_LINE: Final = 2  # 字幕条每行塞几条目标
_BAND_MAX_LINES: Final = 3

_ROLE_CN: Final[dict[str, str]] = {
    "1": "重装",
    "2": "工程",
    "3": "步兵",
    "4": "步兵",
    "S": "哨兵",
    "Q": "前哨",
}
_TEAM_CN: Final[dict[Team, str]] = {Team.RED: "红方", Team.BLUE: "蓝方"}
# 无中文字体时的 ASCII 兜底——考核字幕要人话，但缺字体不能崩。
_ROLE_ASCII: Final[dict[str, str]] = {
    "1": "HVY",
    "2": "ENG",
    "3": "INF",
    "4": "INF",
    "S": "SNT",
    "Q": "OUT",
}
_TEAM_ASCII: Final[dict[Team, str]] = {Team.RED: "RED", Team.BLUE: "BLU"}

# 字幕/表共用行的形状：发布槽 + 共用文本
CaptionRow = tuple[PublishedTrack, str]


def role_name(label: str, team: Team, *, cjk: bool = True) -> str:
    """槽名 → 2027 兵种叫法（红方重装 / 蓝方哨兵）。识别不了的槽号原样返回。"""
    role = (_ROLE_CN if cjk else _ROLE_ASCII).get(label[-1])
    if role is None:
        return label
    team_name = (_TEAM_CN if cjk else _TEAM_ASCII).get(team, "")
    return f"{team_name}{role}"


def error_vs_truth(track: PublishedTrack, truth: tuple[SimRobot, ...]) -> float | None:
    """发布坐标对同名真值的欧氏距离（米）。真值里没这台车 → None。"""
    for gt in truth:
        if gt.label == track.label:
            return float(np.hypot(track.x - gt.x, track.y - gt.y))
    return None


def caption_text(track: PublishedTrack, err: float | None, *, cjk: bool = True) -> str:
    """字幕条与数据表共用的同一行文本——「一致」由共用本函数保证。

    例：`红方重装  (  2.31,  1.05 )m  误差 0.32m ✓ 〔外推〕`
    外推标注：LOST（卡尔曼盲推）期间坐标可信度不同，表和字幕都看得见。
    """
    if err is None:
        err_text = "误差 --" if cjk else "err --"
    else:
        # T/F 而不是 ✓/✗：U+2713/2717 在雅黑里是 notdef 空框（实测），
        # T=准确线内 F=超线。纯 ASCII 在任何字体（含 Linux 上场机缺
        # CJK 字体时）都零 tofu 风险，中英文模式共用同一套标记。
        mark = "T" if err < _ACCURATE_M else "F"
        prefix = "误差" if cjk else "err"
        err_text = f"{prefix} {err:.2f}m {mark}"
    coast = "〔外推〕" if (cjk and track.phase is TrackPhase.LOST) else ("" if cjk else " [coast]")
    return f"{role_name(track.label, track.team, cjk=cjk)}  ( {track.x:6.2f}, {track.y:6.2f} )m  {err_text}{coast}"


def caption_rows(
    publishes: tuple[PublishedTrack, ...],
    truth: tuple[SimRobot, ...],
    *,
    own_team: Team | None = None,
    cjk: bool = True,
) -> list[CaptionRow]:
    """(槽, 文本) 列表。own_team 给了敌方排前——裁判只按敌方记分。"""
    rows: list[CaptionRow] = [
        (item, caption_text(item, error_vs_truth(item, truth), cjk=cjk)) for item in publishes
    ]
    if own_team is not None:
        rows.sort(key=lambda pair: 0 if pair[0].team is not own_team else 1)
    return rows


# ---------- 中文字体 ----------

_FONT_CANDIDATES: Final = (
    "C:/Windows/Fonts/msyh.ttc",
    "C:/Windows/Fonts/simhei.ttf",
    "C:/Windows/Fonts/simsun.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-zenhei.ttc",
    "/usr/share/fonts/truetype/wqy/wqy-microhei.ttc",
    "/usr/share/fonts/opentype/noto/NotoSansCJK-Regular.ttc",
    "/usr/share/fonts/noto-cjk/NotoSansCJK-Regular.ttc",
)

_font_cache: dict[str, ImageFont.FreeTypeFont] = {}
_font_path_found: str | None = None
_font_searched: bool = False
_cjk_warned: bool = False


def find_font_path() -> str | None:
    """RADAR_FONT_PATH → 平台常见字体 → None。结果缓存。"""
    global _font_path_found, _font_searched
    if not _font_searched:
        env = os.environ.get("RADAR_FONT_PATH", "")
        candidates: list[str] = ([env] if env else []) + list(_FONT_CANDIDATES)
        _font_path_found = next((p for p in candidates if p and Path(p).is_file()), None)
        _font_searched = True
    return _font_path_found


def _font(size: int) -> ImageFont.FreeTypeFont | None:
    path = find_font_path()
    if path is None:
        return None
    key = f"{path}:{size}"
    if key not in _font_cache:
        _font_cache[key] = ImageFont.truetype(path, size)
    return _font_cache[key]


def cjk_available() -> bool:
    return _font(24) is not None


def warn_once_if_no_cjk() -> None:
    """启动检查用：没找到中文字体时打一次明确警告（不影响运行）。"""
    global _cjk_warned
    if _cjk_warned or cjk_available():
        return
    _cjk_warned = True
    print(
        "run_radar WARN: 未找到中文字体（RADAR_FONT_PATH 未设且常见路径都没有），"
        "考核字幕退英文缩写。Linux 上场机装 fonts-wqy-zenhei 或设 RADAR_FONT_PATH。",
        flush=True,
    )


# ---------- 合成 ----------


def _fit(image: ImageU8, box_w: int, box_h: int) -> ImageU8:
    """等比缩放进盒子（不裁切）。"""
    h, w = image.shape[:2]
    scale = min(box_w / w, box_h / h)
    return cv2.resize(image, (max(int(w * scale), 1), max(int(h * scale), 1)))


def _team_rgb(team: Team) -> tuple[int, int, int]:
    """PIL 画布是 RGB。"""
    match team:
        case Team.RED:
            return (255, 96, 96)
        case Team.BLUE:
            return (96, 160, 255)


def build_official_frame(
    camera: ImageU8,
    board: ImageU8,
    publishes: tuple[PublishedTrack, ...],
    truth: tuple[SimRobot, ...],
    *,
    frame_no: int,
    fps: float,
    own_team: Team | None = None,
) -> ImageU8:
    """画一帧 1920×1080。imshow 与 VideoWriter 共用本函数，像素级一致。

    camera：任意分辨率的相机画面（含检测框，4096×3000 也行）。
    board：本帧的自划区域图（含轨迹点/外推圈，run_radar 的 overlay 产出）。
    own_team：给了就把敌方槽排前面（裁判只按敌方记分）。
    """
    cjk = cjk_available()
    canvas = np.zeros((OFFICIAL_H, OFFICIAL_W, 3), dtype=np.uint8)

    # ── 左：相机画面，等比铺满 1080 高（4.2.5 那块屏）──
    cam_box_w = OFFICIAL_W - _PANEL_W - _MARGIN * 3
    fitted = _fit(camera, cam_box_w, OFFICIAL_H)
    ch, cw = fitted.shape[:2]
    canvas[0:ch, 0:cw] = fitted

    # ── 右：面板（区域图 + 数据表）──
    panel_x0 = OFFICIAL_W - _PANEL_W + _MARGIN
    panel_w = _PANEL_W - _MARGIN * 2
    rows = caption_rows(publishes, truth, own_team=own_team, cjk=cjk)

    map_img = _fit(board, panel_w, 340)
    mh, mw = map_img.shape[:2]
    map_y = _MARGIN + 44
    canvas[map_y : map_y + mh, panel_x0 : panel_x0 + mw] = map_img
    cv2.rectangle(canvas, (panel_x0, map_y), (panel_x0 + mw, map_y + mh), (160, 160, 160), 1)

    table_y = map_y + mh + 34

    # ── 底部字幕条：半透明黑底压在相机画面下沿 ──
    band_lines = _band_lines(rows)
    if band_lines:
        band_y0 = OFFICIAL_H - _BAND_H
        roi = canvas[band_y0 : band_y0 + _BAND_H, 0:ch]
        canvas[band_y0 : band_y0 + _BAND_H, 0:ch] = (roi * 0.45).astype(np.uint8)

    # ── 文本统一收集，末尾一次性渲染（有字体走 PIL，没字体 cv2 ASCII）──
    texts: list[tuple[int, int, str, tuple[int, int, int], int]] = []

    texts.append((panel_x0, _MARGIN + 4, "自划区域图" if cjk else "region map", (255, 255, 255), 26))
    texts.append((panel_x0, table_y, "数据表（与字幕一致）" if cjk else "table (same as captions)", (255, 255, 255), 24))
    legend = "误差T=0.8m内准确  F=超线" if cjk else "err T=within 0.8m  F=beyond"
    texts.append((panel_x0, table_y + 28, legend, (150, 150, 150), 18))

    y = table_y + 58
    if not rows:
        texts.append((panel_x0, y, "（无发布槽位）" if cjk else "(no published slots)", (120, 120, 120), 22))
        y += 30
    for track, text in rows:
        texts.append((panel_x0, y, text, _team_rgb(track.team), 22))
        y += 30

    texts.append(
        (panel_x0, OFFICIAL_H - 36, f"frame {frame_no}   {fps:.1f} fps   1920x1080", (170, 170, 170), 20)
    )

    line_y0 = OFFICIAL_H - _BAND_H + 8
    for li, line in enumerate(band_lines):
        texts.append((12, line_y0 + li * 26, line, (240, 240, 240), 22))

    if cjk:
        pil = Image.fromarray(cv2.cvtColor(canvas, cv2.COLOR_BGR2RGB))
        draw = ImageDraw.Draw(pil)
        for x, ty, text, color, size in texts:
            font = _font(size)
            if font is not None:
                draw.text((x, ty), text, fill=color, font=font)
        return cv2.cvtColor(np.asarray(pil), cv2.COLOR_RGB2BGR)

    for x, ty, text, color, _size in texts:
        bgr = (color[2], color[1], color[0])
        cv2.putText(canvas, text, (x, ty + 18), cv2.FONT_HERSHEY_SIMPLEX, 0.55, bgr, 1)
    return canvas


def _band_lines(rows: list[CaptionRow]) -> list[str]:
    """字幕条：每行 2 条目标，至多 3 行；超出显示 +N。与数据表同文。"""
    if not rows:
        return []
    texts = [text for _track, text in rows]
    shown = texts[: _BAND_PER_LINE * _BAND_MAX_LINES]
    lines = ["    ".join(shown[i : i + _BAND_PER_LINE]) for i in range(0, len(shown), _BAND_PER_LINE)]
    rest = len(texts) - len(shown)
    if rest > 0:
        lines[-1] = lines[-1] + f"  (+{rest})"
    return lines


class OfficialRecorder:
    """cv2.VideoWriter 包装：mp4v 优先，打不开退 XVID avi。固定 1920×1080。"""

    def __init__(self, path: Path, fps: float) -> None:
        self._path = Path(path)
        self._path.parent.mkdir(parents=True, exist_ok=True)
        self._frames = 0
        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
        self._writer = cv2.VideoWriter(str(self._path), fourcc, fps, (OFFICIAL_W, OFFICIAL_H))
        if not self._writer.isOpened():
            fallback = self._path.with_suffix(".avi")
            print(f"run_radar record: mp4 打不开（缺编码器？），退 {fallback.name}", flush=True)
            self._writer = cv2.VideoWriter(
                str(fallback), cv2.VideoWriter_fourcc(*"XVID"), fps, (OFFICIAL_W, OFFICIAL_H)
            )
            if not self._writer.isOpened():
                raise SystemExit(f"record FAIL: VideoWriter 打不开 {self._path}（也无 .avi 回退）")
            self._path = fallback

    def write(self, frame: ImageU8) -> None:
        if frame.shape[0] != OFFICIAL_H or frame.shape[1] != OFFICIAL_W:
            raise ValueError(
                f"official frame must be {OFFICIAL_W}x{OFFICIAL_H}, got {frame.shape[1]}x{frame.shape[0]}"
            )
        self._writer.write(frame)
        self._frames += 1

    def release(self) -> None:
        self._writer.release()
        size_mb = self._path.stat().st_size / 1e6 if self._path.is_file() else 0.0
        print(f"record done: {self._path}  frames {self._frames}  {size_mb:.1f} MB", flush=True)
