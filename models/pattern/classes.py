"""港科大第三级标签：公开集是 B0–B5/BS + R0–R5/RS 共 14 个文件夹。

本仓丢掉颜色（红蓝交给装甲 YOLO），丢掉 5 号（港科大自己也不拿 5 号当主类），
合成 6 类：1 2 3 4 S(哨兵) Q(前哨，文件夹 0)。
推理侧 Detection.label 仍是 red/blue/dead，图案只写 Detection.pattern.name。
"""

from __future__ import annotations

from enum import IntEnum, StrEnum
from typing import Final, assert_never


class PatternId(IntEnum):
    """训练/推理用的图案 id，必须和 CLASS_NAMES 下标对齐。"""

    DIGIT_1 = 0
    DIGIT_2 = 1
    DIGIT_3 = 2
    DIGIT_4 = 3
    SENTRY = 4
    OUTPOST = 5


class ArmorFolder(StrEnum):
    """公开集文件夹名。5 号有目录但不进训练集。"""

    B0 = "B0"
    B1 = "B1"
    B2 = "B2"
    B3 = "B3"
    B4 = "B4"
    B5 = "B5"
    BS = "BS"
    R0 = "R0"
    R1 = "R1"
    R2 = "R2"
    R3 = "R3"
    R4 = "R4"
    R5 = "R5"
    RS = "RS"


CLASS_NAMES: Final[tuple[str, ...]] = ("1", "2", "3", "4", "S", "Q")


def pattern_for_folder(folder: ArmorFolder) -> PatternId | None:
    """颜色丢掉，只留兵种。None = 港科大明确不训的 5 号。"""
    match folder:
        case ArmorFolder.B1 | ArmorFolder.R1:
            return PatternId.DIGIT_1
        case ArmorFolder.B2 | ArmorFolder.R2:
            return PatternId.DIGIT_2
        case ArmorFolder.B3 | ArmorFolder.R3:
            return PatternId.DIGIT_3
        case ArmorFolder.B4 | ArmorFolder.R4:
            return PatternId.DIGIT_4
        case ArmorFolder.BS | ArmorFolder.RS:
            return PatternId.SENTRY
        case ArmorFolder.B0 | ArmorFolder.R0:
            return PatternId.OUTPOST
        case ArmorFolder.B5 | ArmorFolder.R5:
            return None
        case unreachable:
            assert_never(unreachable)


def parse_armor_folder(name: str) -> ArmorFolder | None:
    """磁盘目录名 → 枚举。未知名字不是错误，直接忽略。"""
    try:
        return ArmorFolder(name)
    except ValueError:
        return None
