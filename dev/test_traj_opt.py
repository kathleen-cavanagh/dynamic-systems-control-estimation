import numpy as np
import matplotlib.pyplot as plt

from systems.cart_pole import CartPole
from utils import direct_collocation

parameters = {
    'mass_cart': 1,
    'mass_pendulum': 1,
    'pendulum_length': 1,
    'gravity': 9.8
}

cp = CartPole(parameters)
s0 = np.array([0, 0, 0, 0])
sf = np.array([np.nan, np.pi, 0, 0])

constraints = {
    'u': 20, 'state': np.full(cp.n_state, np.nan)
}

sol = direct_collocation(
    cp, s0, sf, 50, constraints, n_knot = 500, slack = .001)


plt.plot(sol['t'], sol['state'][:, 0], label = 'x')
plt.plot(sol['t'], sol['state'][:, 1], label = 'theta')
plt.plot(sol['t'], sol['state'][:, 2], label = 'v')
plt.plot(sol['t'], sol['state'][:, 3], label = 'omega')
plt.legend()
plt.show()

plt.plot(sol['t'], sol['input'])
plt.show()
