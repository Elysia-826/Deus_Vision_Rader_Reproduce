"""雷达相关命令字的载荷编解码。依据《RM2027 通信协议 V1.0.0（20261029）》，全小端。

上行（雷达 → 服务器）：
  0x0305 小地图敌方+己方坐标（48B，≤5Hz，(0,0) 表示该机器人不发送）
  0x0301 机器人交互数据外壳；雷达自主决策作为其子内容 0x0121（13B）
下行（服务器 → 雷达）：
  0x0001 比赛状态  0x0003 机器人血量  0x020C 标记进度掩码  0x020E 雷达决策信息

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

_MAP_DATA_LEN: Final[int] = 48  # 12 槽 × 2 × uint16 cm
_HP_DATA_LEN: Final[int] = 24  # 12 × uint16；offset 8 为 int16 伤害差
_GAME_STATUS_LEN: Final[int] = 11
_MARK_PROGRESS_LEN: Final[int] = 2
_RADAR_INFO_LEN: Final[int] = 8  # byte0 双倍/密钥；byte1 加固；2/4/6 圆心 mm
_RADAR_DECISION_LEN: Final[int] = 13  # 双倍(1)+密钥指令(1)+key(6)+加固计数(1)+xy mm(4)

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
    """12 槽 → 48B。槽序：敌方 重装/保留/步3/步4/保留/哨兵，己方同序。

    None = 该机器人不发（(0,0)）。下标 1、4 无角色映射，保持 (0,0)。
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


def decode_map_positions(payload: bytes) -> tuple[MapMark | None, ...]:
    """48B → 12 槽。全零槽 → None。stub 裁判用这读雷达发来的坐标。"""
    if len(payload) != _MAP_DATA_LEN:
        raise CommandDecodeError(f"0x0305 应为 {_MAP_DATA_LEN}B，得到 {len(payload)}B")
    marks: list[MapMark | None] = []
    for offset in range(0, _MAP_DATA_LEN, 4):
        x_cm = int.from_bytes(payload[offset : offset + 2], "little")
        y_cm = int.from_bytes(payload[offset + 2 : offset + 4], "little")
        marks.append(None if x_cm == 0 and y_cm == 0 else MapMark(x_cm=x_cm, y_cm=y_cm))
    return tuple(marks)


def field_from_mark(mark: MapMark) -> tuple[float, float]:
    """裁判图 cm → 场地米制（mark_from_field 的逆变换，供 stub 算误差 d 用）。"""
    x_m = mark.x_cm / 100.0 - FIELD_LENGTH_M / 2.0
    if _OUR_PLUS_Y_IS_REFEREE_PLUS_Y:
        y_m = mark.y_cm / 100.0 - FIELD_WIDTH_M / 2.0
    else:
        y_m = FIELD_WIDTH_M / 2.0 - mark.y_cm / 100.0
    return x_m, y_m


ROLE_TO_INDEX: Final[dict[str, int]] = {"1": 0, "3": 2, "4": 3, "S": 5}
"""槽内序：重装=0 保留=1 步3=2 步4=3 保留=4 哨兵=5。标签 2 不进 0x0305。"""

_ROBOT_IDS: Final[tuple[int, ...]] = (1, 2, 3, 4, 6, 7)
SLOT_NAMES: Final[tuple[str, ...]] = tuple(
    f"{'E' if i < 6 else 'A'}{_ROBOT_IDS[i % 6]}" for i in range(12)
)
"""12 槽的报告名。下标 1/4 对应保留位（旧工程/空中 ID），无角色映射。"""


def slot_index_of(own_team: Team, team: Team, label: str) -> int | None:
    """真值/发布里带阵营和槽名（如 R2/B1/BS）→ 0x0305 槽下标。识别不了返回 None。"""
    index = ROLE_TO_INDEX.get(label[-1])
    if index is None:
        return None
    return index if team is not own_team else index + 6


def marks_from_publishes(
    own_team: Team,
    publishes: tuple[PublishedTrack, ...],
    *,
    include_lost: bool = False,
    priority_slots: frozenset[int] = frozenset(),
) -> tuple[MapMark | None, ...]:
    """跟踪发布 → 12 个 0x0305 槽。质量门：默认只发 CONFIRMED。

    include_lost=True 时 LOST（卡尔曼外推）也发——只在基准证明外推误差
    中位数 < 0.8m 准确线时才考虑打开；错发比不发扣得更快（x 连击清零）。
    TENTATIVE 一律不发：两帧确认前的身份不可信。

    priority_slots：规则 §5.1.5.4——敌方定位模块离线时，我方发的任何
    坐标都判「准确」（免费 P）。检测到离线目标的槽加入此集合后，
    即使 LOST/低置信也发坐标（反正怎么发都算准）。
    """
    marks: list[MapMark | None] = [None] * 12
    for publish in publishes:
        slot = slot_index_of(own_team, publish.team, publish.label)
        if slot is None:
            continue  # Q（前哨）不进 0x0305
        if publish.phase is TrackPhase.TENTATIVE:
            continue
        is_lost = publish.phase is TrackPhase.LOST
        if is_lost and not include_lost and slot not in priority_slots:
            continue  # LOST 但不在优先集合 → 不发
        marks[slot] = mark_from_field(publish.x, publish.y)
    return tuple(marks)


STALE_MARK_LOST_S: Final = 0.5
"""生外推标记的年龄上限（秒）= 裁判中断线。

P 经济学：错发一发 −0.8 且清连击（5Hz 下 −4/s），静默只吃中断
（−0.8/0.5s = −1.6/s）。0.5s 后的外推大概率 d>1.6m → 生外推
从这时刻起是负资产。实测 COAST 8 帧 @10Hz 全发会把基准从
9.6s 打到 5.6s。年龄 ≥0.5s 的 LOST 只有两种坐标值得发：
猜点换过的（GUESS_LOST 已改写 x/y）和离线槽的（怎么发都判准）。
"""


def drop_stale_lost(
    own_team: Team,
    publishes: tuple[PublishedTrack, ...],
    priority_slots: frozenset[int],
) -> tuple[PublishedTrack, ...]:
    """从标记路径里剔除 ≥0.5s 的生外推 LOST（小地图发布不受影响）。

    猜点开启时由调用方先跑 apply_guess_to_lost 换坐标再调本函数——
    换过战略点的 LOST 不算「生外推」，照发。离线优先槽照发（免费 P）。
    """
    out: list[PublishedTrack] = []
    for item in publishes:
        if item.phase is TrackPhase.LOST and item.lost_s >= STALE_MARK_LOST_S:
            slot = slot_index_of(own_team, item.team, item.label)
            if slot is not None and slot not in priority_slots:
                continue
        out.append(item)
    return tuple(out)


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
    *,
    fortify_counter: int = 0,
    fortify_x_mm: int = 0,
    fortify_y_mm: int = 0,
) -> bytes:
    """0x0121 内容段 13B：双倍计数 + 密钥指令 + 加固（默认后 5 字节为 0）。

    double_counter 必须单调 +1（服务器校验），由 RefereeLink 管理，不在这校验。
    fortify_counter 独立单调，trigger_double / 密钥更新不得递增。
    加固坐标是裁判图 mm，不走 mark_from_field（那是 cm）。
    """
    if not 0 <= double_counter <= 0xFF:
        raise FrameFormatError(f"双倍计数超界: {double_counter}")
    if not 0 <= fortify_counter <= 0xFF:
        raise FrameFormatError(f"加固计数超界: {fortify_counter}")
    if not 0 <= fortify_x_mm <= 0xFFFF or not 0 <= fortify_y_mm <= 0xFFFF:
        raise FrameFormatError(f"加固坐标超 uint16 mm: {fortify_x_mm},{fortify_y_mm}")
    match action:
        case PasswordAction.NONE:
            cmd = _PASSWORD_ACTION_NONE
        case PasswordAction.UPDATE_OWN:
            cmd = _PASSWORD_ACTION_UPDATE_OWN
        case PasswordAction.VERIFY_ENEMY:
            cmd = _PASSWORD_ACTION_VERIFY_ENEMY
        case unreachable:
            raise FrameFormatError(f"未知密钥指令: {unreachable}")
    return (
        bytes((double_counter, cmd))
        + _check_password(password)
        + bytes((fortify_counter,))
        + fortify_x_mm.to_bytes(2, "little")
        + fortify_y_mm.to_bytes(2, "little")
    )


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


# ---------- 雷达→哨兵引导（0x0301 子内容 0x0200 机器人间通信） ----------

SUB_ROBOT_TO_ROBOT: Final[int] = 0x0200  # 0x0200~0x02FF：机器人间通信
BLUE_SENTRY_ID: Final[int] = 107
RED_SENTRY_ID: Final[int] = 7


def encode_sentry_guidance(
    marks: tuple[MapMark | None, ...],
) -> bytes:
    """打包敌方位置发给己方哨兵（≤30Hz，与 0x0305 共享带宽）。

    内容格式（自定协议，哨兵侧需匹配解码）：
      uint8 数量 N
      N × (uint16 x_cm, uint16 y_cm)  裁判图坐标 cm
    最大 1 + 6×4 = 25B，远小于 112B 上限。
    """
    enemy_marks = marks[:6]  # 只发敌方（前 6 槽）
    content = bytearray()
    valid = [m for m in enemy_marks if m is not None]
    content.append(len(valid))
    for mark in valid:
        content += mark.x_cm.to_bytes(2, "little")
        content += mark.y_cm.to_bytes(2, "little")
    return bytes(content)


def decode_sentry_guidance(content: bytes) -> tuple[MapMark, ...]:
    """哨兵侧解包雷达引导。stub/测试用。"""
    if not content:
        return ()
    n = content[0]
    marks: list[MapMark] = []
    for i in range(n):
        off = 1 + i * 4
        if off + 4 > len(content):
            break
        x_cm = int.from_bytes(content[off : off + 2], "little")
        y_cm = int.from_bytes(content[off + 2 : off + 4], "little")
        marks.append(MapMark(x_cm=x_cm, y_cm=y_cm))
    return tuple(marks)


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
    """0x0003 血量 24B。未上场/罚下 HP=0。不驱动双倍或标记门。"""

    heavy: int
    infantry3: int
    infantry4: int
    damage_diff: int  # int16，本队相对伤害差
    sentry: int
    red_outpost: int
    own_base: int
    blue_outpost: int
    enemy_base: int
    own_control: int
    enemy_control: int


def parse_robot_hp(payload: bytes) -> RobotHp:
    if len(payload) != _HP_DATA_LEN:
        raise CommandDecodeError(f"0x0003 应为 {_HP_DATA_LEN}B，得到 {len(payload)}B")

    def u16(offset: int) -> int:
        return int.from_bytes(payload[offset : offset + 2], "little")

    def i16(offset: int) -> int:
        return int.from_bytes(payload[offset : offset + 2], "little", signed=True)

    return RobotHp(
        heavy=u16(0),
        infantry3=u16(4),
        infantry4=u16(6),
        damage_diff=i16(8),
        sentry=u16(10),
        red_outpost=u16(12),
        own_base=u16(14),
        blue_outpost=u16(16),
        enemy_base=u16(18),
        own_control=u16(20),
        enemy_control=u16(22),
    )


@dataclass(frozen=True, slots=True)
class MarkProgress:
    """0x020C 掩码 2B。对方易伤 = P≥100；己方特殊标识 = P≥50。

    敌方槽序同 0x0305：重装/保留/步3/步4/保留/哨兵 → bit 0..5；
    己方同序 → bit 6..11。bit1、bit4、己方 bit7、bit10 为保留位。
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
    """0x020E 8B：双倍/密钥（byte0）+ 加固状态与圆心 mm（byte1 及 offset 2/4/6）。"""

    double_chances: int  # 0~2
    enemy_double_active: bool
    own_encryption_level: int  # 1~3，即对方干扰波难度
    can_change_password: bool
    armor_fortify_active: bool = False
    fortify_chances: int = 0
    fortify_x_mm: int = 0
    fortify_y_mm: int = 0
    fortify_radius_mm: int = 0


def parse_radar_info(payload: bytes) -> RadarInfo:
    if len(payload) != _RADAR_INFO_LEN:
        raise CommandDecodeError(f"0x020E 应为 {_RADAR_INFO_LEN}B，得到 {len(payload)}B")
    raw = payload[0]
    fortify = payload[1]
    return RadarInfo(
        double_chances=raw & 0x03,
        enemy_double_active=bool(raw & 0x04),
        own_encryption_level=(raw >> 3) & 0x03,
        can_change_password=bool(raw & 0x20),
        armor_fortify_active=bool(fortify & 0x01),
        fortify_chances=(fortify >> 1) & 0x07,
        fortify_x_mm=int.from_bytes(payload[2:4], "little"),
        fortify_y_mm=int.from_bytes(payload[4:6], "little"),
        fortify_radius_mm=int.from_bytes(payload[6:8], "little"),
    )
