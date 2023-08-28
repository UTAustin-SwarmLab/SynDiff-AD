import math
from dataclasses import dataclass

import cvxpy
import torch
import numpy as np
from cvxpylayers.torch import CvxpyLayer
from IPython import embed


@dataclass
class MPCCost:
    total: float
    tracking: float
    control: float
    goal: float


class DynamicUnicycleModel:
    def __init__(self):
        self.n_s = 4
        self.n_u = 2

    def discrete_dynamics(self, s, u, dt):
        """Dynamically-extended unicycle discrete dynamics. Dynamics are
        discretized assuming a zero-order hold on the controls.

        Inputs
        s - current state [x (m), y (m), theta (rad), v (m/s)]
        u - control [omega (rad/s), a (m/s^2)]
        dt - time step (s)
        Outputs
        s_next - state at next time step
        """

        s_next = torch.zeros(self.n_s)
        if abs(u[0]) > 1e-3:
            Ds = (torch.sin(s[2] + u[0] * dt) - torch.sin(s[2])) / u[0]
            Dc = (torch.cos(s[2] + u[0] * dt) - torch.cos(s[2])) / u[0]

            s_next[0] = s[0] + s[3] * Ds + (u[1] * torch.sin(s[2] + u[0] * dt) * dt) / u[0] + (u[1] / u[0]) * Dc
            s_next[1] = s[1] - s[3] * Dc - (u[1] * torch.cos(s[2] + u[0] * dt) * dt) / u[0] + (u[1] / u[0]) * Ds
            s_next[2] = s[2] + u[0] * dt
            s_next[3] = s[3] + u[1] * dt
        else:
            s_next[0] = s[0] + s[3] * dt * torch.cos(s[2]) + 0.5 * u[1] * dt ** 2 * torch.cos(s[2])
            s_next[1] = s[1] + s[3] * dt * torch.sin(s[2]) + 0.5 * u[1] * dt ** 2 * torch.sin(s[2])
            s_next[2] = s[2] + u[0] * dt
            s_next[3] = s[3] + u[1] * dt

        return s_next

    def linearized_dynamics(self, s0, u0, dt):
        if s0.requires_grad:
            create_graph = True
        else:
            create_graph = False
        dt = torch.tensor(dt)
        Js, Ju = autof.jacobian(
            lambda s, u: self.discrete_dynamics(s, u, dt),
            (s0, u0),
            create_graph=create_graph,
        )
        c = self.discrete_dynamics(s0, u0, dt) - Js @ s0 - Ju @ u0

        return Js, Ju, c


class TrackingProblem:
    def __init__(self, dt, N, setpoint, penalty_weights, control_lims):
        # problem settings
        self.model = DynamicUnicycleModel()
        self.dt = dt
        self.N = N

        self.P, self.q = self.objective_matrices(setpoint, penalty_weights)
        A_dyn, l_and_u_dyn = self.dynamics_constraints()
        A_control, l_control, u_control = self.control_constraints(control_lims)
        self.A = torch.cat((A_dyn, A_control))
        self.l = torch.cat((l_and_u_dyn, l_control))
        self.u = torch.cat((l_and_u_dyn, u_control))

    def objective_matrices(self, setpoint, penalty_weights):
        P_tracking, q_tracking = self.setpoint_tracking(setpoint)
        P_slew, q_slew = self.slew_rate_penalty(penalty_weights)
        P = 2 * torch.block_diag(P_tracking, P_slew)
        q = torch.cat((q_tracking, q_slew))

        return P, q

    def setpoint_tracking(self, setpoint):
        """Computes objective terms for tracking objective.

        Inputs
        setpoint - state tuple (x (m), y (m), theta (rad), v (m/s))
        Outputs
        P_tracking - quadratic objective term
        q_tracking - linear objective term
        """
        assert len(setpoint) == self.model.n_s, "State dimension and setpoint dimension mismatch."

        P_data = torch.zeros(self.model.n_s, self.N)
        q_data = torch.zeros(self.model.n_s, self.N)

        for (i, component_ref) in enumerate(setpoint):
            if component_ref is None:
                P_data[i, :] = torch.zeros(self.N)
                q_data[i, :] = torch.zeros(self.N)
            else:
                P_data[i, :] = torch.ones(self.N)
                q_data[i, :] = -2 * component_ref * torch.ones(self.N)

        P_tracking = torch.diag(P_data.T.flatten())
        q_tracking = q_data.T.flatten()

        return P_tracking, q_tracking

    def slew_rate_penalty(self, penalty_weights):
        """Computes objective terms for rate of change of controls.

        Inputs
        penalty weights - tuple of weights (r_om , r_a)
        Outputs
        P_slew - quadratic objective term
        q_slew - linear objective term
        """
        assert len(penalty_weights) == self.model.n_u, "State dimension and setpoint dimension mismatch."

        diag_entries = torch.cat(
            (
                torch.tensor(penalty_weights),
                2 * torch.tensor(penalty_weights).repeat(self.N - 2),
                torch.tensor(penalty_weights),
            )
        )
        offdiag_entries = torch.tensor(penalty_weights).repeat(self.N - 1)

        P_slew = (
            torch.diag(diag_entries)
            - torch.diag(offdiag_entries, self.model.n_u)
            - torch.diag(offdiag_entries, -self.model.n_u)
        )
        q_slew = torch.zeros(self.N * self.model.n_u)

        return P_slew, q_slew

    def dynamics_constraints(self):
        n_s = self.model.n_s
        n_u = self.model.n_u
        dt = self.dt
        N = self.N

        Js_list = []
        Ju_list = []
        v_list = []
        s0 = torch.zeros(self.model.n_s)
        ref_state = torch.zeros(self.model.n_s)
        ref_state[2] = 0.0
        for i in range(self.N):
            if i == 0:
                Js, Ju, c = self.model.linearized_dynamics(s0, torch.zeros(n_u), dt)
                Ju_list.append(Ju)
                v_list.append(c + Js @ s0)
            else:
                ### HARDCODED ###
                ref_state[0] = ref_state[0] + 5 * self.dt
                ref_state[3] = 5
                #################
                Js, Ju, c = self.model.linearized_dynamics(ref_state, torch.zeros(n_u), dt)
                Js_list.append(Js)
                Ju_list.append(Ju)
                v_list.append(c)

        A_left = torch.eye(N * n_s)
        A_left[n_s:, : n_s * (N - 1)] -= torch.block_diag(*Js_list)
        A_right = -torch.block_diag(*Ju_list)
        A_dyn = torch.hstack((A_left, A_right))
        l_and_u_dyn = torch.cat(v_list)

        return A_dyn, l_and_u_dyn

    def control_constraints(self, control_lims):
        n_s = self.model.n_s
        n_u = self.model.n_u
        dt = self.dt
        N = self.N

        A_control = torch.zeros((N * n_u, N * (n_s + n_u)))
        A_control[:, N * n_s :] = torch.eye(N * n_u)
        repeated_control_lims = torch.tensor(control_lims).repeat(N, 1)
        l_control = repeated_control_lims[:, 0]
        u_control = repeated_control_lims[:, 1]

        return A_control, l_control, u_control

    def initial_state_constraint(self, s0):
        n_s = self.model.n_s
        n_u = self.model.n_u
        dt = self.dt
        N = self.N

        Js, Ju, c = self.model.linearized_dynamics(s0, torch.zeros(n_u, dtype=s0.dtype), dt)

        A = self.A.clone().detach()
        l = self.l.clone().detach()
        u = self.u.clone().detach()

        A[:n_s, N * n_s : N * n_s + n_u] = -Ju
        l[:n_s] = c + Js @ s0
        u[:n_s] = c + Js @ s0

        return A, l, u

    def get_problem_data(self, cte_and_he, v_init):
        n_s = self.model.n_s
        n_u = self.model.n_u
        N = self.N
        try:
            device = cte_and_he.device
        except:
            device = "cpu"

        if cte_and_he.ndimension() == 1:
            cte, he = cte_and_he
            s0 = torch.zeros(n_s, dtype=cte.dtype)
            s0[1] = cte
            s0[2] = he * math.pi / 180.0
            s0[3] = v_init

            A, l, u = self.initial_state_constraint(s0)

            return (
                self.P.clone().detach().to(device).type(cte.dtype),
                self.q.clone().detach().to(device).type(cte.dtype),
                A.to(device).type(cte.dtype),
                l.to(device).type(cte.dtype),
                u.to(device).type(cte.dtype),
            )
        else:
            batch_size = cte_and_he.shape[0]
            cte, he = cte_and_he[:, 0], cte_and_he[:, 1]
            s0 = torch.zeros(batch_size, n_s, dtype=cte.dtype)
            s0[:, 1] = cte
            s0[:, 2] = he * math.pi / 180.0
            s0[:, 3] = v_init

            A_batch = torch.zeros(batch_size, N * (n_s + n_u), N * (n_s + n_u), dtype=cte.dtype)
            l_batch = torch.zeros(batch_size, N * (n_s + n_u), dtype=cte.dtype)
            u_batch = torch.zeros(batch_size, N * (n_s + n_u), dtype=cte.dtype)
            for i in range(batch_size):
                A_batch[i], l_batch[i], u_batch[i] = self.initial_state_constraint(s0[i])

            return (
                self.P.tile(batch_size, 1, 1).clone().detach().to(device).type(cte.dtype),
                self.q.repeat(batch_size, 1).clone().detach().to(device).type(cte.dtype),
                A_batch.to(device),
                l_batch.to(device),
                u_batch.to(device),
            )


def solve_mpc_tracking(target_trac):

    x = cvxpy.Variable((NX, T))
    u = cvxpy.Variable((NU, T))
    slack = cvxpy.Variable((4, T))
    XREF = cvxpy.Parameter((NX, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(XREF[:, t] - x[:, t], Q)

        if t < (T - 1):
            A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    cost += cvxpy.quad_form(XREF[:, -1] - x[:, -1], QF_GAIN * Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)

    # assert prob.is_dgp(dpp=True), "Problem is not DGP"
    cvxpylayer = CvxpyLayer(
        prob,
        parameters=[XREF],
        variables=[
            x,
            u,
            slack,
        ],
    )

    opt_x, opt_u, opt_slack = cvxpylayer(xref)

    cost = 0.0
    tracking_cost = 0.0
    control_cost = 0.0
    goal_cost = 0.0

    for t in range(T):
        control_cost += opt_u[:, t].T @ torch.Tensor(R) @ opt_u[:, t]

        if t != 0:
            tracking_cost += (xref[:, t] - opt_x[:, t]).T @ torch.Tensor(Q) @ (xref[:, t] - opt_x[:, t])

        if t < (T - 1):
            control_cost += (opt_u[:, t + 1] - opt_u[:, t]).T @ torch.Tensor(Rd) @ (opt_u[:, t + 1] - opt_u[:, t])

    goal_cost = (xref[:, -1] - opt_x[:, -1]).T @ (QF_GAIN * torch.Tensor(Qf)) @ (xref[:, -1] - opt_x[:, -1])

    cost += control_cost
    cost += tracking_cost
    cost += goal_cost

    cost.backward()

    mpc_costs = MPCCost(total=cost, tracking=tracking_cost, control=control_cost, goal=goal_cost)

    return mpc_costs, opt_x, opt_u
