"""从 Blender 物体名解析期望槽位。解析不出就失败，不猜。"""

from __future__ import annotations

import re
from typing import Final

from track.errors import UnmappedRobotName
from track.types import Role

_TOKEN = re.compile(r"[A-Za-z0-9]+")
_ROLE_BY_TOKEN: Final[dict[str, Role]] = {
    "1": Role.ONE,
    "2": Role.TWO,
    "3": Role.THREE,
    "4": Role.FOUR,
    "hero": Role.ONE,
    "engineer": Role.TWO,
    "infantry": Role.THREE,
    "sentry": Role.SENTRY,
    "s": Role.SENTRY,
}


def role_from_name(name: str) -> Role:
    """名字里恰好一个兵种记号才接受。`Robot_3_Red` → 3，`Robot_Red_Alpha` 拒绝。"""
    found: set[Role] = set()
    for token in _TOKEN.findall(name):
        role = _ROLE_BY_TOKEN.get(token.lower())
        if role is not None:
            found.add(role)
    if len(found) != 1:
        raise UnmappedRobotName(name)
    return next(iter(found))
