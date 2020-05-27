"""Simple Kalman Filter implementation."""
from abc import ABC, abstractmethod

import numpy as np

from systems.base import DynamicSystem, MeasurementRelation


class KalmanFilter(ABC):
    def __init__(self, system: DynamicSystem, x0: np.ndarray, P0: np.ndarray):
        self._system = system
        self._x = x0
        self._P = P0
        self._t = 0
        self._dim = len(x0)

    @property
    def x(self) -> np.ndarray:
        """Current state estimate."""
        return self._x
    
    @property
    def P(self) -> np.ndarray:
        """Current covariance of estimate."""
        return self._P

    @abstractmethod
    def propagate(self, u: np.ndarray, t: float, Q: np.ndarray) -> None:
        """Propagate state in time."""
        pass

    @abstractmethod
    def update(
        self, measurement: np.ndarray, relation: MeasurementRelation,
        R: np.ndarray) -> None:
        """Update estimate based on measurement received."""
        pass