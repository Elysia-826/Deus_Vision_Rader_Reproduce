"""referee 包的异常。"""

from __future__ import annotations


class RefereeError(Exception):
    """裁判链路错误基类。"""


class FrameFormatError(RefereeError):
    """帧/载荷不合法：seq 越界、长度超限、字段编码失败等。"""


class CommandDecodeError(RefereeError):
    """下行帧长度和协议不符（比如 0x0305 不是 48 字节）。"""


class PasswordFormatError(FrameFormatError):
    """密钥不是 6 位 ASCII 字母数字。"""


class SerialUnavailableError(RefereeError):
    """pyserial 未安装或串口打不开。骨架阶段只影响真实串口，回环不受影响。"""
