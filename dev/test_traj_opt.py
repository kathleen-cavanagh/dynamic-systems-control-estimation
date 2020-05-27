import numpy as np
import matplotlib.pyplot as plt

from systems.cart_pole import CartPole
from systems.trajectory import SystemTrajectoryOptimization

cp = CartPole()
s0 = np.array([0, 0, 0, 0])
sf = np.array([0, np.pi, 0, 0])

constraints = {
    'u': 20, 'state': np.full(cp.n_state, np.nan)
}

traj_opt = SystemTrajectoryOptimization(cp, 50, knot_points = 500, slack = .01)

traj_opt.add_direct_collocation_constraints(s0, sf)

traj_opt.solve_program()

sol = traj_opt.solution
