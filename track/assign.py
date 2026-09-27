"""槽 × 观测的最小代价分配。池子只有 10 个，拒绝代价的配对事后丢掉。"""

from __future__ import annotations

import numpy as np
from scipy.optimize import linear_sum_assignment

REJECT = 1.0e6


def min_cost_pairs(cost: list[list[float]]) -> tuple[tuple[int, int], ...]:
    """返回 (slot_index, observation_index)。空观测或被拒绝的配对不返回。"""
    if not cost or not cost[0]:
        return ()
    matrix = np.asarray(cost, dtype=np.float64)
    rows, cols = linear_sum_assignment(matrix)
    kept: list[tuple[int, int]] = []
    for row, col in zip(rows, cols, strict=True):
        if float(matrix[row, col]) < REJECT:
            kept.append((int(row), int(col)))
    return tuple(kept)
