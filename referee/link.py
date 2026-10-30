"""RefereeLink：雷达侧裁判链路。发 0x0305（5Hz 节流）和 0x0121 决策，收下行事件。

用法（骨架阶段）：
    pair = LoopbackPair()
    link = RefereeLink(pair.radar, own_team=Team.RED)
    link.send_map(marks, now_s=...)        # 0.2s 内重复调用会被节流掉
    link.verify_enemy_password("A1B2C3")   # 无线电组解出 0x0A06 后经这提交
    for event in link.poll(now_s=...):     # 0x0001/0x0003/0x020C/0x020E
        ...
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from referee.commands import (
    CMD_GAME_STATUS,
    CMD_MAP_POSITIONS,
    CMD_MARK_PROGRESS,
    CMD_RADAR_INFO,
    CMD_ROBOT_HP,
    CMD_ROBOT_INTERACTION,
    SUB_RADAR_DECISION,
    SUB_ROBOT_TO_ROBOT,
    BLUE_RADAR_ID,
    BLUE_SENTRY_ID,
    RED_RADAR_ID,
    RED_SENTRY_ID,
    SERVER_ID,
    GameStatus,
    MapMark,
    MarkProgress,
    PasswordAction,
    RadarInfo,
    RobotHp,
    encode_map_positions,
    encode_radar_decision_content,
    encode_robot_interaction,
    encode_sentry_guidance,
    parse_game_status,
    parse_mark_progress,
    parse_radar_info,
    parse_robot_hp,
)
from referee.errors import CommandDecodeError
from referee.frame import FrameParser, pack_frame
from referee.transport import RefereeTransport
from track.types import Team

MIN_SEND_INTERVAL_S: Final[float] = 0.2  # 0x0305 上限 5Hz
_SEND_EPS_S: Final[float] = 1e-6  # 0.6-0.4=0.19999… 这类浮点误差不算超频


@dataclass(frozen=True, slots=True)
class RefereeEvent:
    """下行帧解析成强类型后，四个命令之一。未知命令在 poll 里静默丢弃。"""

    game_status: GameStatus | None = None
    robot_hp: RobotHp | None = None
    mark_progress: MarkProgress | None = None
    radar_info: RadarInfo | None = None


class RefereeLink:
    """一个串口会话。seq 自增回绕；0x0305 节流；0x0121 计数由本类保证单调 +1。"""

    def __init__(self, transport: RefereeTransport, *, own_team: Team) -> None:
        self._transport = transport
        self._parser = FrameParser()
        self._seq = 0
        self._last_map_sent_s: float | None = None
        self._double_counter = 0
        self._fortify_counter = 0
        self._own_team = own_team
        self._self_id = RED_RADAR_ID if own_team is Team.RED else BLUE_RADAR_ID

    # ---------- 上行 ----------

    def send_map(self, marks: tuple[MapMark | None, ...], *, now_s: float) -> bool:
        """发一帧 0x0305。距上次不足 MIN_SEND_INTERVAL_S 时丢弃并返回 False。

        marks 是 12 槽（commands.marks_from_publishes 的输出）。服务器按
        连击公式给 P 计分：宁可 (0,0) 不发，也别把没把握的坐标发出去。
        """
        if (
            self._last_map_sent_s is not None
            and now_s - self._last_map_sent_s < MIN_SEND_INTERVAL_S - _SEND_EPS_S
        ):
            return False
        self._send(CMD_MAP_POSITIONS, encode_map_positions(marks))
        self._last_map_sent_s = now_s
        return True

    def trigger_double(self) -> None:
        """消耗一次双倍易伤机会。计数单调 +1，uint8 饱和；有没有机会看 0x020E 反馈。"""
        self._double_counter = min(self._double_counter + 1, 0xFF)
        self._send_radar_decision(PasswordAction.NONE, "000000")

    def send_sentry_guidance(self, marks: tuple[MapMark | None, ...]) -> None:
        """把敌方位置通过 0x0301 机器人间通信发给己方哨兵。

        与 0x0305（小地图标记）共享 0x0301 命令的 30Hz 带宽。
        marks 是 12 槽（marks_from_publishes 输出），只取敌方前 6 槽。
        哨兵侧按 encode_sentry_guidance 的自定格式解码。
        """
        sentry_id = RED_SENTRY_ID if self._own_team is Team.RED else BLUE_SENTRY_ID
        content = encode_sentry_guidance(marks)
        payload = encode_robot_interaction(SUB_ROBOT_TO_ROBOT, self._self_id, sentry_id, content)
        self._send(CMD_ROBOT_INTERACTION, payload)

    def update_own_password(self, password: str) -> None:
        """改己方密钥。仅开局和对方破解成功后有效（0x020E bit5 可改时）。"""
        self._send_radar_decision(PasswordAction.UPDATE_OWN, password)

    def verify_enemy_password(self, password: str) -> None:
        """提交无线电组破解的对方密钥。服务器验证后 10s 内再发无效。"""
        self._send_radar_decision(PasswordAction.VERIFY_ENEMY, password)

    # ---------- 下行 ----------

    def poll(self, *, timeout_s: float = 0.0) -> tuple[RefereeEvent, ...]:
        """读传输层、解帧、转成事件。坏帧由 FrameParser 重同步；坏长度帧跳过。"""
        chunk = self._transport.read(timeout_s)
        events: list[RefereeEvent] = []
        for frame in self._parser.feed(chunk):
            event = self._to_event(frame.cmd_id, frame.payload)
            if event is not None:
                events.append(event)
        return tuple(events)

    # ---------- 内部 ----------

    def _send(self, cmd_id: int, payload: bytes) -> None:
        self._transport.write(pack_frame(self._seq, cmd_id, payload))
        self._seq = (self._seq + 1) & 0xFF

    def _send_radar_decision(self, action: PasswordAction, password: str) -> None:
        content = encode_radar_decision_content(
            self._double_counter,
            action,
            password,
            fortify_counter=self._fortify_counter,
        )
        payload = encode_robot_interaction(SUB_RADAR_DECISION, self._self_id, SERVER_ID, content)
        self._send(CMD_ROBOT_INTERACTION, payload)

    def _to_event(self, cmd_id: int, payload: bytes) -> RefereeEvent | None:
        try:
            if cmd_id == CMD_GAME_STATUS:
                return RefereeEvent(game_status=parse_game_status(payload))
            if cmd_id == CMD_ROBOT_HP:
                return RefereeEvent(robot_hp=parse_robot_hp(payload))
            if cmd_id == CMD_MARK_PROGRESS:
                return RefereeEvent(mark_progress=parse_mark_progress(payload))
            if cmd_id == CMD_RADAR_INFO:
                return RefereeEvent(radar_info=parse_radar_info(payload))
            return None
        except CommandDecodeError:
            return None  # 协议版本差异导致的长度不符：丢帧不崩，联调再对表
