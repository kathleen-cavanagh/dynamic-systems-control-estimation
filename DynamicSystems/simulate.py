"""Simulate systems"""

def simulate(system, initial_state, u, tvec):

    states_over_time = np.zeros((np.size(tvec), len(initial_state)))
    states_over_time[0,:] = initial_state
    for tt in range(1, len(tvec)):
        dt = tvec[tt] - tvec[tt-1]
        derivs = system.get_derivs(states_over_time[tt-1, :], tvec[tt-1], uvec[tt-1])
        next_state = states_over_time[tt-1,:]+dt*derivs
        # states_over_time[tt,:] = self.add_estimation_noise(next_state)
    
    return states_over_time

