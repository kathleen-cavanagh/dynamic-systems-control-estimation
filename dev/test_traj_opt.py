import numpy as np
import matplotlib.pyplot as plt

from systems.cart_pole import CartPole
from systems.trajectory import SystemTrajectoryOptimization

cp = CartPole()
s0 = np.array([0, 0, 0, 0])
sf = np.array([0, np.pi, 0, 0])

# Solve with direct collocation
traj_opt_collocation = SystemTrajectoryOptimization(cp, 50, knot_points = 500, slack = .01)
traj_opt_collocation.add_direct_collocation_constraints(s0, sf)
traj_opt_collocation.solve_program()
sol_collocation = traj_opt_collocation.solution

# Solve with direct transcription
traj_opt_transcription = SystemTrajectoryOptimization(cp, 50, knot_points = 500, slack = .01)
traj_opt_transcription.add_direct_transcription_constraints(s0, sf)
traj_opt_transcription.solve_program()
sol_transcription = traj_opt_transcription.solution
