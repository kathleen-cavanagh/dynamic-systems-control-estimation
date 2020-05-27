"""Test EKF implementation with CartPole"""

import numpy as np
import matplotlib.pyplot as plt
import random

from systems.cart_pole import (
    CartPole, PendulumTipPosition, PendulumTipVelocity)
from estimation.extended_kf import ExtendedKF
from utils import add_gaussian_noise, simulate_system, wrap_angles

# Simulate system to extract simulated measurements and ground truth
cp = CartPole()
t_final = 15
s0 = np.zeros(cp.n_state)

class BangBang:
    def __init__(self, u_max: float, t_final: float):
        self._u_max = u_max
        self._switch_pt = t_final/2 

    def __call__(self, t, state):
        if t < self._switch_pt:
            return self._u_max
        else:
           return -self._u_max


policy = BangBang(1, t_final)

simulated_sys = simulate_system(cp, s0, policy, t_final, n_steps = 150)

# define x0 and P0
x0 = np.array([.2, 0, 0, 0])
P0 = np.diag([1, .01, .5, .01])
Q = .1 * np.eye(cp.n_state)
R = np.eye(2)

ekf = ExtendedKF(cp, x0, P0)

state_estimate = []

measurement_types = [PendulumTipPosition(cp), PendulumTipVelocity(cp)]

for sys_info in simulated_sys[1:]:

    ekf.propagate(policy(sys_info['t'], ekf.x), sys_info['t'], Q)
    measurement_method = random.choice(measurement_types)
    raw_meas = measurement_method.calculate(sys_info['state'], sys_info['input'])
    meas = add_gaussian_noise(raw_meas, std = np.array([.05, .05]))
    ekf.update(
        meas, measurement_method, np.diag([.05, .05]))

    state_estimate.append(ekf.x)

