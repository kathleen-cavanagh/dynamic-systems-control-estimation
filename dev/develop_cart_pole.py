import numpy as np
import matplotlib.pyplot as plt

from DynamicSystems.cart_pole import CartPole
from utils import simulate_system

parameters = {
	'mass_cart': 1,
	'mass_pendulum': 1,
	'pendulum_length': 1,
	'gravity': 9.8
}

cp = CartPole(parameters)

test_state = np.array([0, np.pi, 0, -.01])

deriv = cp.derivative(0, test_state, 0)
print(deriv)

jac = cp.jacobian(0, test_state, 0)
print(jac)

A, B = cp.linearized_dynamics(0, test_state, 0)
print(A)
print(B)

def passive(state):
	return 0

output = simulate_system(cp, test_state, passive, 5)

plt.plot(output['t'], output['state'])
plt.show()
