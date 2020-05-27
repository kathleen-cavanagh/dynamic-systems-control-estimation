"""Define a Cart Pole system."""
import numpy as np

from typing import Any, Dict, Optional, Tuple

from pydrake.math import sin, cos
from pydrake.symbolic import inv

from systems.base import DynamicSystem, MeasurementRelation
from utils import wrap_angles


class CartPole(DynamicSystem):

    """Implementation of the dynamics of a Cart-Pole system."""

    def __init__(self, mass_cart: float = 1, mass_pendulum:float = 1,
                 pendulum_length: float = 1, gravity: float = 9.8) -> None:
        """
        Initialize CartPole system with physical parameters.

        Arguments:
            mass_cart: mass of cart (kg)
            mass_pendulum: mass of pendulum (kg)
            pendulum_length: length of pendulum (m)
            gravity: acceleration due to gravity (m/s/s)
        """
        self._mc = mass_cart
        self._mp = mass_pendulum
        self._l = pendulum_length
        self._g = gravity

        # Define state and input dimensions
        self.n_state = 4
        self.n_u = 1

    def parameters(self) -> Dict:
        """Return system parameters"""
        return {'l': self._l, 'mass_cart': self._mc,
                'mass_pendulum': self._mp, 'gravity': self._g}

    def validate_state(self, state: np.ndarray):
        """
        Validate value of state by wrapping angle theta.

        Arguments:
            state: current state value
        """
        state[1] = wrap_angles(state[1])
    
    def _coriolis_matrix(
            self, state: np.ndarray, dtype: Any = float) -> np.ndarray:
        """Helper method to calculate the Coriolis matrix."""
        C = np.zeros((2, 2), dtype = dtype)
        C[0, 1] = -self._mp * self._l * state[3] * sin(state[1])
        return C
    
    def _inertia_matrix(
            self, state: np.ndarray, dtype: Any = float) -> np.ndarray:
        """Helper method to calculate the inertia matrix."""
        M12 = self._mp * self._l * cos(state[1])
        return np.array([[self._mc + self._mp, M12],
                         [M12, self._mp * self._l ** 2]], dtype = dtype)

    def _inertia_matrix_inv(
            self, state: np.ndarray, dtype: Any = float) -> np.ndarray:
        """
        Helper method to calculate the inverse of the inertia matrix.

        This method is primarily useful for symbolic computation where the
        inverse functionality of numpy nd arrays cannot be used.
        """
        inertia_matrix = self._inertia_matrix(state, dtype = dtype)
        M11, M12, M22 = (
            inertia_matrix[0, 0], inertia_matrix[0, 1], inertia_matrix[1, 1])
        det = M11 * M12 - M12 * M12
        return (1.0 / det) * np.array(
            [[M22, -M12], [-M12, M11]], dtype = dtype)
    
    def _gravity_vector(self, state: np.ndarray, dtype = float) -> np.ndarray:
        """Calculate gravitation influence on equations of motion."""
        return np.array(
            [0, -self._mp * self._g * self._l * sin(state[1])], dtype = dtype)

    def derivative(
            self, t: float, state: np.ndarray, u: float,
            dtype: Optional[Any] = float
        ) -> np.ndarray:
        """
        Calculate the derivative of the system given current state and input.

        Arguments:
            t: time (s)
            state: state at time ``t``
            u: force control signal (N)
            dtype: dtype of output

        Returns:
            state derivative at time ``t`` evaluated at ``state`` and ``u``
        """
        derivs = np.zeros(self.n_state, dtype = dtype)
        derivs[0:2] = state[2:]
        
        input_vec = np.array([u, 0.0], dtype = dtype)
        temp = (-self._coriolis_matrix(state, dtype=dtype) @ state[2:]
                + self._gravity_vector(state, dtype=dtype) + input_vec)
        
        qdd = self._inertia_matrix_inv(state, dtype=dtype) @ temp
        derivs[2] = qdd[0]
        derivs[3] = qdd[1]
        
        return derivs

    def jacobian(self, t: float, state: np.ndarray, u: np.float) -> np.ndarray:
        c_theta = cos(state[1])
        s_theta = sin(state[1])
        theta = state[1]
        omega = state[3]

        # d(xdd)/dx, d(xdd/dxd), d(thetadd)/dx, d(theta_dd)/d_xd are all zero
        jac = np.zeros((4,4))

        jac[0][2] = 1.0  # d/dxdot (xdot)
        jac[1][3] = 1.0  # d/dthetadot (theta_dot)

        #d(xdd)/dtheta
        term1 = 1.0 / (self._mc + self._mp * s_theta ** 2)
        dterm1dtheta = -(2.0 * self._mp * s_theta * c_theta) * (term1 ** 2)
        term2 = (u + self._mp * s_theta
                 * (self._l * (omega ** 2) + self._g * c_theta))
        dterm2dtheta = (self._mp * c_theta * self._l * (omega ** 2)
                        - self._g * s_theta)
        jac[2][1] = term1 * dterm2dtheta + term2 * dterm1dtheta

        #d(xdd)/dthetad
        jac[2][3] = term1 * 2.0 * self._mp * s_theta * self._l * state[3]

        #d(thetadd)/dtheta
        term3 = 1.0 / self._l * term1
        dterm3dtheta = 1.0 / self._l * dterm1dtheta
        term4 = (-u * c_theta - self._mp * self._l * (state[3]**2) * c_theta
                 * s_theta - (self._mc + self._mp) * self._g * s_theta)
        dterm4dtheta = (
            u * s_theta - self._mp * self._l * state[3] * cos(2*state[1])
            - (self._mc + self._mp) * self._g * c_theta)
        jac[3][1] = term3 * dterm4dtheta + term4 * dterm3dtheta

        #d(thetadd)/dthetad
        jac[3][3] = term3 * (-self._mp * self._l * state[3] * sin(2 * theta))

        return jac

    def linearization(
            self, t: float, state: np.ndarray, u: float
        ) -> Tuple[np.ndarray, np.ndarray]:
        """
        Calculate linearization about arbitrary state.

        Arguments:
            t: time (s)
            state: state at time ``t``
            u: force control signal in N

        Returns:
            linearized "A" and "B" matrices according to xdot = Ax + Bu
        """
        B = np.zeros(4)
        B[2] = 1.0 / (self._mc + self._mp * sin(state[1]) ** 2)
        B[3] = 1/self._l * B[2] * (1.0 / (self._l * (
            self._mc + self._mp * sin(state[1]) ** 2))) * (-cos(state[1]))

        return (self.jacobian(t, state, u), B)


class PendulumTipPosition(MeasurementRelation):

    """
    Helper for generating x-y pendulum tip position measurement.
    """

    def __init__(self, system: CartPole):
        """Initialize relation with parameters from specific Cart Pole system"""
        self._l = system.parameters()['l']
        self._dim = 2
    
    @property
    def dim(self) -> int:
        """Dimension of measurement."""
        return self._dim
    
    def calculate(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Define measurement expected from ``state`` and input ``u``.

        Arguments:
            state: current system state
            u: current system input

        Returns:
            (2,) x, y position measurement
        """
        return np.array([
            state[0] + self._l * np.sin(state[1]),
            self._l * np.cos(state[1])])

    def jacobian(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Define measurement jacobian.

        Arguments:
            state: system state value
            u: system input value

        Returns:
            (2, 4) Jacobian of the measurement-state relationship
        """
        return np.array([
            [1, self._l* np.cos(state[1]), 0, 0], 
            [0, -self._l * np.sin(state[1]), 0, 0]])


class PendulumTipVelocity(MeasurementRelation):

    """
    Helper for generating x-y pendulum tip velocity measurement for Cart Pole.
    """

    def __init__(self, system: CartPole):
        """Initialize relation with parameters from specific Cart Pole system"""
        self._l = system.parameters()['l']
        self._dim = 2

    @property
    def dim(self) -> int:
        """Dimension of measurement."""
        return self._dim

    def calculate(self, state: np.ndarray, u: np.ndarray) -> np.ndarray:
        """
        Define measurement expected from ``state`` and input ``u``.

        Arguments:
            state: current system state
            u: current system input

        Returns:
            (2,) x, y velocity measurement
        """
        theta = state[1]

        # Calculate velocity of tip due to rotation.
        tip_position = self._l * np.array([np.sin(theta),-np.cos(theta), 0])
        tip_velocity = np.cross(np.array([0, 0, state[3]]), tip_position)

        # Combine velocity due to rotation with the cart velocity
        return np.array([state[2], 0]) + tip_velocity[:-1]

    def jacobian(self, state, u):
        """
        Define measurement jacobian.

        Arguments:
            state: system state value
            u: system input value

        Returns:
            (2, 4) Jacobian of the measurement-state relationship
        """
        omega = state[3]
        theta = state[1]
        return np.array(
            [[0, self._l * omega * -np.sin(theta), 1,
              self._l * np.cos(theta)],
             [0, self._l * np.cos(theta) * omega, 0,
              self._l * np.sin(theta)]])
