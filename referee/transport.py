"""传输层抽象：真实串口（惰性 pyserial）+ 内存回环（测试与无硬件联调）。

常规链路物理形态：裁判系统电源管理模块的 User 串口，115200 8N1。
"""

from __future__ import annotations

from typing import Final, Protocol

from referee.errors import SerialUnavailableError

BAUD: Final[int] = 115200


class RefereeTransport(Protocol):
    """RefereeLink 只依赖这两个方法。"""

    def write(self, data: bytes) -> None: ...

    def read(self, timeout_s: float) -> bytes:
        """最多等 timeout_s，返回当前能读到的字节（可为空）。"""
        ...


class SerialTransport:
    """pyserial 直连。真机用；没装 pyserial 时给出可执行的提示。"""

    def __init__(self, port: str, baud: int = BAUD) -> None:
        try:
            import serial  # noqa: PLC0415 —— 惰性导入，测试环境不需要
        except ImportError as exc:
            raise SerialUnavailableError(
                "需要 pyserial：pip install pyserial（测试可用 LoopbackPair，无需安装）"
            ) from exc
        try:
            self._serial = serial.Serial(port=port, baudrate=baud, timeout=0.05)
        except Exception as exc:  # serial.SerialException 等硬件错误统一转
            raise SerialUnavailableError(f"串口 {port} 打不开: {exc}") from exc

    def write(self, data: bytes) -> None:
        self._serial.write(data)

    def read(self, timeout_s: float) -> bytes:
        self._serial.timeout = timeout_s
        size = self._serial.in_waiting or 1
        return bytes(self._serial.read(size))


class _LoopbackEnd:
    """回环的一端。对端写 → 本端读缓冲。线程不安全，单线程测试用。"""

    def __init__(self) -> None:
        self._incoming = bytearray()
        self._peer: _LoopbackEnd | None = None

    def _bind(self, peer: _LoopbackEnd) -> None:
        self._peer = peer

    def write(self, data: bytes) -> None:
        if self._peer is None:
            raise SerialUnavailableError("回环未配对")
        self._peer._incoming += data

    def read(self, timeout_s: float) -> bytes:
        del timeout_s  # 内存回环无阻塞语义
        data = bytes(self._incoming)
        self._incoming.clear()
        return data


class LoopbackPair:
    """一对互通的内存传输端。radar 端 ↔ stub 裁判端，测试链路全程不碰硬件。"""

    def __init__(self) -> None:
        self.radar = _LoopbackEnd()
        self.peer = _LoopbackEnd()
        self.radar._bind(self.peer)
        self.peer._bind(self.radar)
