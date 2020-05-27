"""Simple Kalman Filter implementation."""
import numpy as np
from scipy import integrate

from DynamicSystems.base import DynamicSystem, MeasurementRelation
from StateEstimation.kalman_filter import KalmanFilter


class UnscentedKF(KalmanFilter):
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

    def propagate(
        self, u: np.ndarray, t: float, Q: np.ndarray, k: float) -> None:
        """
        Propagate state in time.

        Arguments:
            u: system input
            t: time to which to propagate estimate
            Q: positive semidefinite process noise
            k: parameter to specify sigma points        
        """
        # Calculate sigma points
        sigma_pts = self._calculate_sigma_points(k)
        n_sigma_pts = len(sigma_pts)
        weights = sigma_pts['weight']

        # Propagate sigma points
        propagated_pts = np.zeros((n_sigma_pts, self._dim))
        self.integrator.set_f_params(u).set_jac_params(u)
        for idx, pt in enumerate(sigma_pts['pt']):
            self.integrator.set_initial_value(pt, self._t)
            propagated_pts[idx, :] = self.integrator.integrate(t)
        self._x = np.sum(weights * propagated_pts.T, 1)

        # Store current time and state
        self._t = t
        self._u = u

        # Calculate covariance of propagated estimate
        propagated_pts_error = (
            propagated_pts - np.tile(self._x, (n_sigma_pts, 1)))
        updated_P = np.zeros((self._dim, self._dim))
        for i in range(n_sigma_pts):
            updated_P += (weights[i] * np.outer(
                propagated_pts_error[i, :], propagated_pts_error[i, :]))
        self._P = updated_P

    def update(
        self, measurement: np.ndarray,
        measurement_relation: MeasurementRelation, R: np.ndarray,
        k: float) -> None:
        """
        Update estimate based on measurement received.

        Arguments:
            measurement: observation of system
            measurement_relation: Description of how state relates to
                measurement
            R: Positive semi definite matrix of measurement noise
            k: parameter to specify sigma points
        """
        # Calculate sigma points again based on new distribution
        sigma_pts = self._calculate_sigma_points(k)
        weights = sigma_pts['weight']

        # Calculate expected measurement based on sigma points
        meas_dim = len(measurement)
        meas_sigma_pts = np.asarray([
            measurement_relation.calculate(pt, self._u)
            for pt in sigma_pts['pt']])
        predicted_meas = np.sum(
            sigma_pts['weight'] * meas_sigma_pts.T, 1)

        # Calculate the innovation and cross covariance matrices
        innovation_cov = R
        cross_cov = np.zeros((self._dim, meas_dim))
        for i in range(len(sigma_pts)):
            meas_diff = meas_sigma_pts[i, :] - predicted_meas
            innovation_cov += weights[i] * np.outer(meas_diff, meas_diff)
            cross_cov += weights[i] * np.outer(
                (sigma_pts['pt'][i, :] - self._x), meas_diff)

        # Calculate kalman gain and update state
        gain = cross_cov @ np.linalg.inv(innovation_cov)
        self._x = self._x + gain @ (measurement - predicted_meas)
        self._system.validate_state(self._x)

        # Update covariance based on gain
        self._P = self._P - gain @ innovation_cov @ gain.T

    def _calculate_sigma_points(self, k: float) -> np.ndarray:
        """
        Calculate sigma points and weights for unscented transform.

        Arguments:
            k: parameter to specify sigma points

        Returns:
            2 * dim +1 structured array with fields "pt" and "weight"
        """
        # Define sigma point structured array
        n_pts = 2 * self._dim + 1
        sigma_pts = np.zeros(n_pts, dtype = [
            ('pt', float, self._dim), ('weight', float)])
        sigma_pts['pt'] = np.full((n_pts, self._dim), self._x)

        # Calculate sigma points
        evals, evecs = np.linalg.eig(self._P)
        sigma_pt_mods = np.asarray([
            np.sqrt(eval_) * evec for (eval_, evec) in zip(evals, evecs.T)])
        sigma_pts['pt'][1:self._dim+1, :] += (
            np.sqrt(self._dim + k) * sigma_pt_mods)
        sigma_pts['pt'][self._dim+1:, :] -= (
            np.sqrt(self._dim + k) * sigma_pt_mods)

        # Validate sigma points
        for i in range(n_pts):
            self._system.validate_state(sigma_pts['pt'][i, :])

        # Calculate weights
        sigma_pts['weight'][0] = k / (self._dim + k) 
        sigma_pts['weight'][1:] = 1 / (2 * (self._dim + k))

        return sigma_pts
