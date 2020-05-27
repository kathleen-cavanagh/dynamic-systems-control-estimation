"""Create utilities for dynamic systems, state estimation etc."""
import copy
from typing import Any, Callable, Union

import numpy as np
from scipy import integrate

import pydrake.solvers.mathematicalprogram as math_prog
from pydrake.systems.controllers import LinearQuadraticRegulator

from DynamicSystems.trajectory import trajectory_dtype


def wrap_angles(theta: float) -> float:
    """
    Wrap angles between -pi and pi

    Arguments:
        theta: angle to be adjusted

    Returns:
        ``theta`` adjust to be between -pi and pi
    """
    return np.arctan2(np.sin(theta), np.cos(theta))


def add_gaussian_noise(
    raw: Union[float, np.ndarray], mean: Union[float, np.ndarray] = 0,
    std: Union[float, np.ndarray] = 1) -> Union[float, np.ndarray]:
    """
    Add Gaussian noise to the input.

    Arguments:
        raw: input signal
        mean: mean of additive Gaussian noise
        std: standard deviation of additive Gaussian noise

    Returns:
        Signal adjusted to have gaussian noise.
    """
    raw_shape = np.shape(raw)
    return raw + mean * np.ones(raw_shape) + std*np.random.randn(*raw_shape)


def saturate(
    raw: Union[float, np.ndarray], lower_bound: Union[float, np.ndarray],
    upper_bound: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Apply saturation limits. to input signal ``raw``.

    Arguments:
        raw: input value
        lower_bound: lower saturation bound
        upper_bound: upper saturation bound

    Returns:
        Saturated signal

    Raises:
        ValueError: if any element of the ```upper_bound``` is less than the
        ```lower_bound```
    """
    if np.any(upper_bound <= lower_bound):
        raise ValueError('Upper bound must be greater than lower bound')

    output = copy.copy(raw)
    output[raw > upper_bound] = upper_bound[raw > upper_bound]
    output[raw < lower_bound] = lower_bound[raw < lower_bound] 

    return output


def simulate_system(
    system: Any, s0: np.ndarray, policy: Callable, t_final: float,
    approximate: bool = False, n_steps: int = 100) -> np.ndarray:
    """
    Simulate a dynamic system.

    Arguments:
        system: Dynamic system
        s0: initial state
        policy: input policy
        t_final: duration of simulation
        approximate: describes if approximation or full integration is used
        n_steps: number of intermediate point in simulation.

    Returns:
        Structured array of state values over time
    """
    system_dtype = trajectory_dtype(system.n_state, system.n_u)
    dt = t_final / n_steps
    
    states = np.zeros(n_steps + 1, dtype = system_dtype)
    states['state'][0, :] = s0

    if not approximate:
        # Initialize integrator
        r = integrate.ode(system.derivative, system.jacobian)
    
    prev_state = s0

    for t in range(1, n_steps + 1):
        this_t = dt * t
        states['t'][t] = this_t

        this_u = policy(this_t, prev_state)
        
        if approximate:
            derivs = system.derivative(
                prev_state, t_steps[t-1], this_u)
            next_state = states[t-1,:] + dt * derivs
        else:
            r.set_f_params(this_u).set_jac_params(this_u)
            r.set_initial_value(prev_state, states['t'][t-1])
            next_state = r.integrate(this_t)

        system.validate_state(next_state)
        states['state'][t, :] = next_state
        states['input'][t] = this_u
        prev_state = next_state

    return states


def step(system, s0, policy, t_final, approximate = False) -> np.ndarray:
    """
    Wrapper for simulate_system for simulating system over one time step.

    Arguments:
        See description under simulate_system

    Returns:
        state array after step propagation
    """
    step_result = simulate_system(
        system, s0, policy, t_final, approximate = approximate, n_steps = 1)
    return step_result['state']


def time_varying_lqr(
    trajectory_optimization, Q = None, R = None) -> np.ndarray:
    """
    Policy for implementing TVLQR on an existing trajectory

    Arguments:
        trajectory_optimization: Trajectory with trajectory dtype
        Q: Q matrix in LQR (must match dimension of system state)
        R: R matrix in LQR (must match dimension of system input)

    Returns:
        States and inputs followed over time
    """
    # Extract desired trajectory information
    trajectory = trajectory_optimization.solution
    system = trajectory_optimization.system

    traj_state = trajectory['state']
    traj_input = trajectory['input']
    traj_t = trajectory['t']
    n_steps = len(traj_t)

    # Initialize data structure to hold output data
    path = np.zeros(n_steps, dtype = trajectory_dtype(
        system.n_state, system.n_u))
    path['t'] = traj_t
    path['state'][0] = x_initial

    if Q is None:
        Q = np.eye((system.n_state, system.n_state))
    if R is None:
        R = np.eye((system.n_u, system.n_u))

    # Simulate system with policy
    for i in range(0, n_steps - 1):
        current_time = traj_t[i]
        xbar = path[i,:] - x_traj[i]; 
        (A_lin, B_lin) = system.linearized_dynamics(
            current_time, traj_state[i,:], traj_input[i])
        K, S = LinearQuadraticRegulator(Alin, Blin, Q, R)
        ubar = -np.dot(K, xbar)
        u = ubar + u_traj[ii]
        path['input'][i] = u
        path['state'][i+1,:] = step(path[i], u, traj_t[i+1] - current_time)

    return path

