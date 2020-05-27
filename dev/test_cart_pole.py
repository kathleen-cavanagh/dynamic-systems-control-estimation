import numpy as np
import matplotlib.pyplot as plt

from systems.cart_pole import CartPole
from utils import simulate_system


cp = CartPole()

test_state = np.array([1, 0, 0, .1])

deriv = cp.derivative(0, test_state, 0)
print(deriv)

jac = cp.jacobian(0, test_state, 0)
print(jac)

A, B = cp.linearization(0, test_state, 0)
print(A)
print(B)

def passive(t, state):
	return 0

output = simulate_system(cp, test_state, passive, 5)

plt.plot(output['t'], output['state'])
plt.show()
