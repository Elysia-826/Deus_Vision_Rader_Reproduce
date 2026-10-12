"""雷达相关命令字的载荷编解码。依据《RM2026 通信协议 V1.3.0》，全小端。

上行（雷达 → 服务器）：
  0x0305 小地图敌方+己方坐标（48B，≤5Hz，(0,0) 表示该机器人不发送）
  0x0301 机器人交互数据外壳；雷达自主决策作为其子内容 0x0121（8B）
下行（服务器 → 雷达）：
  0x0001 比赛状态  0x0003 己方血量  0x020C 标记进度掩码  0x020E 雷达决策信息

坐标系（0x020B 注）：原点在红方补给站附近场角，+X 沿长边向蓝方，+Y 沿短边
向红方停机坪，单位 cm。本仓场地系原点在中心、+x 蓝侧，与裁判 +X 同向；
y 轴是否同向待真机联调核对（见 _OUR_PLUS_Y_IS_REFEREE_PLUS_Y）。
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import StrEnum
from typing import Final

from locate.field import FIELD_LENGTH_M, FIELD_WIDTH_M
from referee.errors import CommandDecodeError, FrameFormatError, PasswordFormatError
from track.types import PublishedTrack, Team, TrackPhase

CMD_GAME_STATUS: Final[int] = 0x0001
CMD_ROBOT_HP: Final[int] = 0x0003
CMD_ROBOT_INTERACTION: Final[int] = 0x0301
CMD_MAP_POSITIONS: Final[int] = 0x0305
CMD_MARK_PROGRESS: Final[int] = 0x020C
CMD_RADAR_INFO: Final[int] = 0x020E
SUB_RADAR_DECISION: Final[int] = 0x0121

SERVER_ID: Final[int] = 0x8080  # 哨兵/雷达自主决策指令的接收者
RED_RADAR_ID: Final[int] = 9
BLUE_RADAR_ID: Final[int] = 109

_MAP_DATA_LEN: Final[int] = 48  # 24 × uint16 cm
_HP_DATA_LEN: Final[int] = 16  # 8 × uint16
_GAME_STATUS_LEN: Final[int] = 11
_MARK_PROGRESS_LEN: Final[int] = 2
_RADAR_INFO_LEN: Final[int] = 1
_RADAR_DECISION_LEN: Final[int] = 8  # radar_cmd(1) + password_cmd(1) + key(6)

_PASSWORD_ACTION_NONE: Final[int] = 0
_PASSWORD_ACTION_UPDATE_OWN: Final[int] = 1
_PASSWORD_ACTION_VERIFY_ENEMY: Final[int] = 2

# 我方 +y 是否就是裁判 +Y（红方停机坪方向）。联调时若发现红方停机坪在
# 俯视图下方（即我们 -y 侧），把这里翻成 False 即可，其余代码不动。
_OUR_PLUS_Y_IS_REFEREE_PLUS_Y: Final[bool] = True


class PasswordAction(StrEnum):
    """0x0121 里 byte1 的密钥指令类型。"""

    NONE = "none"
    UPDATE_OWN = "update_own"  # 开局/被提权后改己方密钥
    VERIFY_ENEMY = "verify_enemy"  # 提交破解的对方密钥（10s 冷却）


@dataclass(frozen=True, slots=True)
class MapMark:
    """一个 0x0305 槽位。裁判图坐标，cm。"""

    x_cm: int
    y_cm: int


def mark_from_field(x_m: float, y_m: float) -> MapMark:
    """场地米制（中心原点）→ 裁判图 cm，越界钳到边界。"""
    half_len = FIELD_LENGTH_M / 2.0
    half_width = FIELD_WIDTH_M / 2.0
    if _OUR_PLUS_Y_IS_REFEREE_PLUS_Y:
        ref_y = y_m + half_width
    else:
        ref_y = half_width - y_m
    x_cm = int(round((x_m + half_len) * 100.0))
    y_cm = int(round(ref_y * 100.0))
    x_cm = max(0, min(int(FIELD_LENGTH_M * 100.0), x_cm))
    y_cm = max(0, min(int(FIELD_WIDTH_M * 100.0), y_cm))
    return MapMark(x_cm=x_cm, y_cm=y_cm)


def encode_map_positions(marks: tuple[MapMark | None, ...]) -> bytes:
    """12 槽 → 48B。槽序：敌方 hero/engineer/inf3/inf4/aerial/sentry，己方同序。

    None = 该机器人不发（(0,0)）。空中机器人当前视觉不标，传 None。
    """
    if len(marks) != 12:
        raise FrameFormatError(f"0x0305 需要 12 个槽，收到 {len(marks)}")
    out = bytearray()
    for mark in marks:
        if mark is None:
            out += b"\x00\x00\x00\x00"
            continue
        if not 0 <= mark.x_cm <= 0xFFFF or not 0 <= mark.y_cm <= 0xFFFF:
            raise FrameFormatError(f"坐标超 uint16: {mark}")
        out += mark.x_cm.to_bytes(2, "little")
        out += mark.y_cm.to_bytes(2, "little")
    if len(out) != _MAP_DATA_LEN:
        raise FrameFormatError(f"0x0305 应为 {_MAP_DATA_LEN}B，得到 {len(out)}B")
    return bytes(out)


_ROLE_TO_INDEX: Final[dict[str, int]] = {"1": 0, "2": 1, "3": 2, "4": 3, "S": 5}
"""槽内序：hero=0 engineer=1 inf3=2 inf4=3 aerial=4(视觉不占) sentry=5。"""


def marks_from_publishes(
    own_team: Team,
    publishes: tuple[PublishedTrack, ...],
    *,
    include_lost: bool = False,
) -> tuple[MapMark | None, ...]:
    """跟踪发布 → 12 个 0x0305 槽。质量门：默认只发 CONFIRMED。

    include_lost=True 时 LOST（卡尔曼外推中）也发——只在基准证明外推误差
    中位数 < 0.8m 准确线时才考虑打开；错发比不发扣得更快（x 连击清零）。
    TENTATIVE 一律不发：两帧确认前的身份不可信。
    """
    marks: list[MapMark | None] = [None] * 12
    for publish in publishes:
        if publish.phase is TrackPhase.TENTATIVE:
            continue
        if publish.phase is TrackPhase.LOST and not include_lost:
            continue
        index = _ROLE_TO_INDEX.get(publish.label[-1])
        if index is None:
            continue  # Q（前哨）不进 0x0305
        slot = index if publish.team is not own_team else index + 6
        marks[slot] = mark_from_field(publish.x, publish.y)
    return tuple(marks)


def _check_password(password: str) -> bytes:
    if len(password) != 6:
        raise PasswordFormatError(f"密钥要 6 位，收到 {len(password)} 位")
    if not password.isascii():
        raise PasswordFormatError(f"密钥只能是 ASCII 字母数字: {password!r}")
    data = password.encode("ascii")
    if not all(0x30 <= b <= 0x39 or 0x41 <= b <= 0x5A or 0x61 <= b <= 0x7A for b in data):
        raise PasswordFormatError(f"密钥只能是 ASCII 字母数字: {password!r}")
    return data


def encode_radar_decision_content(
    double_counter: int,
    action: PasswordAction,
    password: str = "000000",
) -> bytes:
    """0x0121 内容段 8B：双倍易伤触发计数 + 密钥指令。

    double_counter 必须单调 +1（服务器校验），由 RefereeLink 管理，不在这校验。
    """
    if not 0 <= double_counter <= 0xFF:
        raise FrameFormatError(f"双倍计数超界: {double_counter}")
    match action:
        case PasswordAction.NONE:
            cmd = _PASSWORD_ACTION_NONE
        case PasswordAction.UPDATE_OWN:
            cmd = _PASSWORD_ACTION_UPDATE_OWN
        case PasswordAction.VERIFY_ENEMY:
            cmd = _PASSWORD_ACTION_VERIFY_ENEMY
        case unreachable:
            raise FrameFormatError(f"未知密钥指令: {unreachable}")
    return bytes((double_counter, cmd)) + _check_password(password)


def encode_robot_interaction(
    sub_cmd: int,
    sender_id: int,
    receiver_id: int,
    content: bytes,
) -> bytes:
    """0x0301 外壳：子内容ID + 发送者 + 接收者 + 内容段（V1.3.0 表 1-25）。"""
    if len(content) > 112:
        raise FrameFormatError(f"0x0301 内容段超 112B: {len(content)}")
    head = sub_cmd.to_bytes(2, "little") + sender_id.to_bytes(2, "little")
    head += receiver_id.to_bytes(2, "little")
    return head + content


@dataclass(frozen=True, slots=True)
class GameStatus:
    """0x0001。game_progress：0 未开始 1 准备 2 自检 3 倒计时 4 比赛中 5 结算。"""

    game_type: int
    game_progress: int
    stage_remain_time_s: int
    unix_time: int


def parse_game_status(payload: bytes) -> GameStatus:
    if len(payload) != _GAME_STATUS_LEN:
        raise CommandDecodeError(f"0x0001 应为 {_GAME_STATUS_LEN}B，得到 {len(payload)}B")
    return GameStatus(
        game_type=payload[0] & 0x0F,
        game_progress=(payload[0] >> 4) & 0x0F,
        stage_remain_time_s=int.from_bytes(payload[1:3], "little"),
        unix_time=int.from_bytes(payload[3:11], "little"),
    )


@dataclass(frozen=True, slots=True)
class RobotHp:
    """0x0003 己方血量。哨兵/前哨/基地齐全；保留位丢弃。"""

    hero: int
    engineer: int
    infantry3: int
    infantry4: int
    sentry: int
    outpost: int
    base: int


def parse_robot_hp(payload: bytes) -> RobotHp:
    if len(payload) != _HP_DATA_LEN:
        raise CommandDecodeError(f"0x0003 应为 {_HP_DATA_LEN}B，得到 {len(payload)}B")

    def hp(offset: int) -> int:
        return int.from_bytes(payload[offset : offset + 2], "little")

    return RobotHp(
        hero=hp(0),
        engineer=hp(2),
        infantry3=hp(4),
        infantry4=hp(6),
        sentry=hp(10),
        outpost=hp(12),
        base=hp(14),
    )


@dataclass(frozen=True, slots=True)
class MarkProgress:
    """0x020C 掩码。对方易伤 = P≥100；己方特殊标识 = P≥50。

    敌方槽序同 0x0305：hero/工程师/步3/步4/空中/哨兵 → bit 0..5；
    己方同序 → bit 6..11。
    """

    mask: int

    def enemy_vulnerable(self, index: int) -> bool:
        return bool(self.mask & (1 << index))

    def ally_marked(self, index: int) -> bool:
        return bool(self.mask & (1 << (index + 6)))


def parse_mark_progress(payload: bytes) -> MarkProgress:
    if len(payload) != _MARK_PROGRESS_LEN:
        raise CommandDecodeError(f"0x020C 应为 {_MARK_PROGRESS_LEN}B，得到 {len(payload)}B")
    return MarkProgress(mask=int.from_bytes(payload, "little"))


@dataclass(frozen=True, slots=True)
class RadarInfo:
    """0x020E：双倍易伤机会数 / 对方触发中 / 己方加密等级 / 可否改密钥。"""

    double_chances: int  # 0~2
    enemy_double_active: bool
    own_encryption_level: int  # 1~3，即对方干扰波难度
    can_change_password: bool


def parse_radar_info(payload: bytes) -> RadarInfo:
    if len(payload) != _RADAR_INFO_LEN:
        raise CommandDecodeError(f"0x020E 应为 {_RADAR_INFO_LEN}B，得到 {len(payload)}B")
    raw = payload[0]
    return RadarInfo(
        double_chances=raw & 0x03,
        enemy_double_active=bool(raw & 0x04),
        own_encryption_level=(raw >> 3) & 0x03,
        can_change_password=bool(raw & 0x20),
    )
