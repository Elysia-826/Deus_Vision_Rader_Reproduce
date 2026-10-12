"""串口帧的封装与流式解析。全小端。

帧 = SOF(1) | data_length(2) | seq(1) | CRC8(1) | cmd_id(2) | data(n) | CRC16(2)。
CRC8 只管帧头前 4 字节；CRC16 覆盖 帧头+cmd_id+data。
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Final

from referee.crc import crc8, crc16
from referee.errors import FrameFormatError

SOF: Final[int] = 0xA5
_HEADER_LEN: Final[int] = 5  # SOF + len(2) + seq + crc8
_CMD_LEN: Final[int] = 2
_TAIL_LEN: Final[int] = 2
_MIN_FRAME_LEN: Final[int] = _HEADER_LEN + _CMD_LEN + _TAIL_LEN  # 空载荷 9 字节
_MAX_DATA_LEN: Final[int] = 112  # 0x0301 内容段上限，防异常长度撑爆缓冲


@dataclass(frozen=True, slots=True)
class Frame:
    """一条完整串口帧。payload 不含帧头、cmd_id、CRC。"""

    seq: int
    cmd_id: int
    payload: bytes


def pack_frame(seq: int, cmd_id: int, payload: bytes) -> bytes:
    """按协议拼一条帧。seq 由调用方（RefereeLink）管理。"""
    if not 0 <= seq <= 0xFF:
        raise FrameFormatError(f"seq 超界: {seq}")
    if not 0 <= cmd_id <= 0xFFFF:
        raise FrameFormatError(f"cmd_id 超界: {cmd_id:x}")
    if len(payload) > _MAX_DATA_LEN:
        raise FrameFormatError(f"payload {len(payload)}B 超上限 {_MAX_DATA_LEN}B")
    header = bytes((SOF,)) + len(payload).to_bytes(2, "little") + bytes((seq,))
    header += bytes((crc8(header),))
    body = header + cmd_id.to_bytes(2, "little") + payload
    return body + crc16(body).to_bytes(2, "little")


class FrameParser:
    """字节流 → Frame。容忍垃圾前缀、坏 CRC 自动重同步（丢 1 字节继续扫）。"""

    def __init__(self) -> None:
        self._buf = bytearray()

    def feed(self, chunk: bytes) -> tuple[Frame, ...]:
        """喂入任意长度字节，返回本次能解出的所有完整帧。"""
        self._buf += chunk
        frames: list[Frame] = []
        while True:
            frame = self._parse_one()
            if frame is None:
                break
            frames.append(frame)
        return tuple(frames)

    def _parse_one(self) -> Frame | None:
        """None = 缓冲不够或已清空；坏 CRC 丢 1 字节继续扫，不中断本批。"""
        while True:
            start = self._buf.find(bytes((SOF,)))
            if start < 0:
                self._buf.clear()  # 没有 SOF：整段垃圾
                return None
            del self._buf[:start]
            if len(self._buf) < _MIN_FRAME_LEN:
                return None
            if crc8(bytes(self._buf[:4])) != self._buf[4]:
                del self._buf[:1]  # 假 SOF
                continue
            data_len = int.from_bytes(self._buf[1:3], "little")
            if data_len > _MAX_DATA_LEN:
                del self._buf[:1]
                continue
            total = _HEADER_LEN + _CMD_LEN + data_len + _TAIL_LEN
            if len(self._buf) < total:
                return None  # 半截帧，等下次 feed
            body = bytes(self._buf[: total - _TAIL_LEN])
            tail = int.from_bytes(self._buf[total - _TAIL_LEN : total], "little")
            if crc16(body) != tail:
                del self._buf[:1]
                continue
            del self._buf[:total]
            return Frame(
                seq=body[3],
                cmd_id=int.from_bytes(body[5:7], "little"),
                payload=body[7:],
            )
