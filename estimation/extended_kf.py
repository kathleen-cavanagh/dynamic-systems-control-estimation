"""Simple Kalman Filter implementation."""
import numpy as np
from scipy import integrate

from systems.base import DynamicSystem, MeasurementRelation
from estimation.kalman_filter import KalmanFilter

class ExtendedKF(KalmanFilter):

    """Implementation of an Extended Kalman Filter on a dynamic system."""

    def __init__(
        self, system: DynamicSystem, x0: np.ndarray, P0: np.ndarray) -> None:
        """
        Initialize an extended kalman filer to observe a given dynamic system.

        Arguments:
            system: Dynamic system to observe
            x0: Initial state estimate
            P0: Initial state covariance matrix
        """
        super().__init__(system, x0, P0)

        # Instantiate integrator for propagation
        self.integrator = integrate.ode(system.derivative, system.jacobian)

    def propagate(self, u: np.ndarray, t: float, Q: np.ndarray) -> None:
        """
        Propagate state in time.

        Note that the system input is treated as a deterministic quantity.

        Arguments:
            u: system input
            t: time to which to propagate estimate
            Q: positive semidefinite process noise
        """
        # Propagate state in time through integration
        self.integrator.set_f_params(u).set_jac_params(u)
        self.integrator.set_initial_value(self._x, self._t)
        self._x = self.integrator.integrate(t)

        # Validate state.
        self._system.validate_state(self._x)

        # Store current time and input
        self._u = u
        self._t = t

        # Calculate the jacobian of the dynamic system to propagate covariance
        jac_prop = self._system.jacobian(t, self._x, u)
        self._P = jac_prop @ self._P @ jac_prop.T + Q 

    def update(
        self, measurement: np.ndarray,
        measurement_relation: MeasurementRelation, R: np.ndarray) -> None:
        """
        Use measurement to update state estimation.

        Arguments:
            measurement: observation of system
            measurement_relation: Description of how state relates to
                measurement
            R: Positive semi definite matrix of measurement noise
        """
        # Calculate expected measurement based on current state
        pred_measurement = measurement_relation.calculate(self._x, self._u)

        # Calculate jacobian of nonlinear measurement relation
        jac_meas = measurement_relation.jacobian(self._x, self._u)

        # Calculate Kalman Gain
        gain = self._P @ jac_meas.T @ np.linalg.inv(
            jac_meas @ self._P @ jac_meas.T + R)

        # Update and validate state and covariance based on Kalman Gain
        self._x = self._x + gain @ (measurement - pred_measurement)
        self._system.validate_state(self._x)

        # Update covariance
        self._P = (np.eye(self._dim) - gain @ jac_meas) @ self._P
