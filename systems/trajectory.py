"""Trajectory optimization and generation capabilities"""
from typing import Any, Union

import numpy as np

# Create a standard dtype to store trajectory information
def trajectory_dtype(state_dim, input_dim):
    return np.dtype([
        ('t', float),
        ('state', float, (state_dim,)),
        ('input', float, (input_dim,))])


class SystemTrajectoryOptimization():

    """Wrapper around pydrake methods for trajectory optimization."""

    def __init__(
        self, system: Any, t_final: float, knot_points: int = 250,
        slack: float = .001) -> None:
        """
        Define trajectory optimization parameters and variables.

        Args:
            system: dynamic system to optimize
            t_final: duration of optimization (s)
            knot_points: points evaluated for constriants
            slack: amount of error allowed in optimization
        """
        self.system = system
        self.t = np.linspace(0, t_final, knot_points)
        self.slack = slack

        self._solution = None
        self.solver_name = None

        self.program = math_prog.MathematicalProgram()

        self.state_decision_var = []
        self.input_decision_var = []

        # Define state decision variables: x1 -> x_{n_knot}
        for t in range(1, n_knot):
            self.state_var.append(
                self.program.NewContinuousVariables(
                    self.system.n_state, ("x_%d" % t)))

        # Define input decision variables: u0 -> u_{n_knot - 1}
        for t in range(0, n_knot-1):
            self.input_var.append(
                program.NewContinuousVariables(
                    self.system.n_u, ("u_%d" % t)))

    @property
    def solution(self) -> np.ndarray:
        """Return solution to optimization if one has been found."""
        if self._solution is None:
            raise ValueError('Optimization problem has not yet been solved.')

        return self._solution

    def add_dynamic_constraint(
        var: Union[np.ndarray, object], des: Union[np.ndarray, object]
    ) -> None:
        """
        Add a constraint corresponding to the dynamics of the system.

        Arguments:
            var: decision variable being constrained
            des: desired value of the decision variable
        """
        for i in range(len(var)):
            self.program.AddConstraint(var[i] <= des[i] + self.slack)
            self.program.AddConstraint(var[i] >= des[i] - self.slack)

    def add_bounding_box(
            var: Union[np.ndarray, object], lb: float, ub: float) -> None:
        """Convenience wrapper around pydrake method AddBoundingBoxConstraint"""
        self.program.AddBoundingBoxConstraint(lb, ub, var)

    def add_direct_transcription_constraints(self, s0: np.ndarray) -> None:
        """
        Constrain dynamics according to the direct conscription method.

        Arguments:
            s0: initial state
        """
        derivs_init = system.derivative(
            self.t[0], s0.astype(object), self.u_over_time[0][0], dtype = object)
        x_expected_init = s0.astype(object) + dt * derivs_init
        self.add_dynamic_constraint(x_over_time[0], x_expected_init)

        #during trajectory
        for t in range(1, n_knot-1):
            deriv = system.derivative(
                self.t[t], self.state_var[t-1], self.input_var[t][0],
                dtype = object)
            x_expected = self.state_var[t-1] + dt * deriv
            self.add_dynamic_constraint(self.state_var[t], x_expected)  

    def add_direct_collocation_constraints(self, s0: np.ndarray) -> None:
        """
        Constrain dynamics according to the direct collocation method.

        Arguments:
            s0: initial state
        """
        # Initialize algorithm with initial states
        past_state = s0.as_type(object)
        past_input = self.input_var[0]
        past_deriv = self.system.derivative(selt.t[0], past_state, past_input)

        for t in range(1, n_knot - 1):
            # Define time delta and collocation time
            delta = self.t[t] - self.t[t-1]
            t_collocation = 0.5 * (self.t[t-1] + self[t])

            # Define variables at next knot point
            next_input = self.input_var[t]
            next_state = self.state_var[t]
            next_deriv = self.system.derivative(self.t[t], next_state, next_input)

            # Calculate state, input and derivative at collocation point
            u_collocation = 0.5 * (next_input + past_input)
            x_collocation = (
                0.5 * (next_state + previous_state)
                + delta / 8 * (next_deriv + previous_deriv))
            collocation_deriv_expected = (
                - 1.5 / delta * (past_state - next_state)
                - .25 * (next_deriv + past_deriv))

            # Get symbolic derivative from dynamics and add constraint
            collocation_deriv = self.system.derivative(
                t_collocation, x_collocation, u_collocation)
            self.add_dynamic_constraint(
                collocation_deriv, collocation_deriv_expected)

            # Reassign "next" variables to "past" for next iteration
            past_state, past_input, past_deriv = (
                next_state, next_input, next_deriv)

    def solve_program(self) -> None:
        """Solve program created with class methods."""
        # Find best solver and store name for reference.
        solver_id = math_prog.ChooseBestSolver(program)
        solver = math_prog.MakeSolver(solver_id)
        self.solver_name = math_prog.SolverName(solver)

        # Solve the mathematical program
        sol = solver.Solve(self.program) 

        # Confirm solution has been found.
        if not sol.is_success():
            raise ValueError('Program could not find solution')

         # Output as structured array
        self.solution = np.zeros(n_knot, dtype = trajectory_dtype(
            self.system.n_state, self.system.n_i))
        self.solution['t'] = time_vec
        self.solution['state'] = np.vstack((s0, sol.GetSolution(x_over_time)))
        self.solution['input'] = np.reshape(
            np.r_[sol.GetSolution(u_over_time), 0], (n_knot, system.n_u))

