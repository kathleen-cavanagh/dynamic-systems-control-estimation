"""Define dynamic policy and system"""
from abc import ABC, abstractmethod

import numpy as np


class DynamicSystem(ABC):

    """Define a dynamic system such as a pendulum, cart pole, acrobot."""

    @abstractmethod
    def derivative(
        self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate system derivative at ``t`` given ``state`` and ``u``."""
        pass

    @abstractmethod
    def jacobian(
        self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate jacobian of system with respect to state."""
        pass

    @abstractmethod
    def linearization(
        self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Linearize system."""
        pass

    def validate_state(self, state: np.ndarray):
        """Validate or modify state value for any constraints."""
        pass

class MeasurementRelation(ABC):

    """Define a relationship between a measurement and the system."""

    @abstractmethod
    def jacobian(
        self, t: float, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate jacobian of measurement with respect to state."""
        pass

    @abstractmethod
    def calculate(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """Calculate measurement given state and input."""
        pass