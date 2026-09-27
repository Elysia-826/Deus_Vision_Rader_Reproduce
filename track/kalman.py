"""场地平面恒速卡尔曼。状态是 x, y, vx, vy，单位米和米/秒。"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

FloatVec = NDArray[np.float64]
FloatMat = NDArray[np.float64]

_MEAS_VAR = 0.04
_ACCEL = 3.0


class PlanarKalman:
    """跨帧滤波器。可变是因为它要原地预测和校正。"""

    def __init__(self) -> None:
        self._state = np.zeros(4, dtype=np.float64)
        self._cov = np.eye(4, dtype=np.float64)
        self.ready = False

    def reset(self, x: float, y: float) -> None:
        self._state = np.array([x, y, 0.0, 0.0], dtype=np.float64)
        self._cov = np.diag([0.05, 0.05, 4.0, 4.0]).astype(np.float64)
        self.ready = True

    def predict(self, dt_s: float) -> None:
        if not self.ready or dt_s <= 0.0:
            return
        motion = _motion(dt_s)
        self._state = motion @ self._state
        self._cov = motion @ self._cov @ motion.T + _process_noise(dt_s)

    def update(self, x: float, y: float, vx: float | None = None, vy: float | None = None) -> None:
        """位置必校正。有上一帧测量时，把差分速度也当观测，外推才跟得上。"""
        if not self.ready:
            self.reset(x, y)
            return
        if vx is None or vy is None:
            observe = np.array([[1.0, 0.0, 0.0, 0.0], [0.0, 1.0, 0.0, 0.0]], dtype=np.float64)
            noise = np.eye(2, dtype=np.float64) * _MEAS_VAR
            residual = np.array([x, y], dtype=np.float64) - observe @ self._state
        else:
            observe = np.eye(4, dtype=np.float64)
            noise = np.diag([_MEAS_VAR, _MEAS_VAR, 0.05, 0.05]).astype(np.float64)
            residual = np.array([x, y, vx, vy], dtype=np.float64) - self._state
        innov = observe @ self._cov @ observe.T + noise
        gain = self._cov @ observe.T @ np.linalg.inv(innov)
        self._state = self._state + gain @ residual
        self._cov = (np.eye(4, dtype=np.float64) - gain @ observe) @ self._cov

    @property
    def x(self) -> float:
        return float(self._state[0])

    @property
    def y(self) -> float:
        return float(self._state[1])

    @property
    def vx(self) -> float:
        return float(self._state[2])

    @property
    def vy(self) -> float:
        return float(self._state[3])


def _motion(dt_s: float) -> FloatMat:
    return np.array(
        [
            [1.0, 0.0, dt_s, 0.0],
            [0.0, 1.0, 0.0, dt_s],
            [0.0, 0.0, 1.0, 0.0],
            [0.0, 0.0, 0.0, 1.0],
        ],
        dtype=np.float64,
    )


def _process_noise(dt_s: float) -> FloatMat:
    dt2 = dt_s * dt_s
    dt3 = dt2 * dt_s
    dt4 = dt2 * dt2
    scale = _ACCEL * _ACCEL
    return scale * np.array(
        [
            [dt4 / 4.0, 0.0, dt3 / 2.0, 0.0],
            [0.0, dt4 / 4.0, 0.0, dt3 / 2.0],
            [dt3 / 2.0, 0.0, dt2, 0.0],
            [0.0, dt3 / 2.0, 0.0, dt2],
        ],
        dtype=np.float64,
    )
