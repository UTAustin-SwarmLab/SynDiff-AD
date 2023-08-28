import math
import os
import sys

import cvxpy
import diffcp
import matplotlib.pyplot as plt
import numpy as np
import torch
from cvxpylayers.torch import CvxpyLayer
from IPython import embed
from matplotlib.patches import Ellipse

from adversarial.planners.utils import MPCCost

OKBLUE = "\033[94m"
ENDC = "\033[0m"

NX = 4  # x = x, y, v, yaw
NU = 2  # a = [accel, steer]
T = 20  # horizon length

# mpc parameters
R = np.diag([0.01, 0.01])  # input cost matrix
Rd = np.diag([0.01, 1.0])  # input difference cost matrix
Q = np.diag([1.0, 1.0, 0.0, 0.0])  # state cost matrix
Qf = Q  # state final matrix
GOAL_DIS = 1.5  # goal distance
STOP_SPEED = 0.5 / 3.6  # stop speed
MAX_TIME = 500.0  # max simulation time

# iterative paramter
MAX_ITER = 3  # Max iteration
DU_TH = 0.1  # iteration finish param

TARGET_SPEED = 60.0 / 3.6  # [m/s] target speed
N_IND_SEARCH = 10  # Search index number

DT = 0.2  # [s] time tick

# Vehicle parameters
LENGTH = 4.5  # [m]
WIDTH = 2.0  # [m]
BACKTOWHEEL = 1.0  # [m]
WHEEL_LEN = 0.3  # [m]
WHEEL_WIDTH = 0.2  # [m]
TREAD = 0.7  # [m]
WB = 2.5  # [m]

MAX_STEER = np.deg2rad(60.0)  # maximum steering angle [rad]
MAX_DSTEER = np.deg2rad(45.0)  # maximum steering speed [rad/s]
MAX_SPEED = 80.0 / 3.6  # maximum speed [m/s]
MIN_SPEED = 0.0 / 3.6  # minimum speed [m/s]
MAX_ACCEL = 6.0  # maximum accel [m/ss]

OBSTACLE_GAIN = 2000.0  # Cost for hiting an obstacle
QF_GAIN = 200.0  # Gain ending at the goal location.
ELLIP_CONSTR_RELAX = 1.5

show_animation = True


class State:
    """
    vehicle state class
    """

    def __init__(self, x=0.0, y=0.0, yaw=0.0, v=0.0):
        self.x = x
        self.y = y
        self.yaw = yaw
        self.v = v
        self.predelta = None


def pi_2_pi(angle):
    while angle > math.pi:
        angle = angle - 2.0 * math.pi

    while angle < -math.pi:
        angle = angle + 2.0 * math.pi

    return angle


def get_linear_model_matrix(v, phi, delta):

    A = np.zeros((NX, NX))
    A[0, 0] = 1.0
    A[1, 1] = 1.0
    A[2, 2] = 1.0
    A[3, 3] = 1.0
    A[0, 2] = DT * math.cos(phi)
    A[0, 3] = -DT * v * math.sin(phi)
    A[1, 2] = DT * math.sin(phi)
    A[1, 3] = DT * v * math.cos(phi)
    A[3, 2] = DT * math.tan(delta) / WB

    B = np.zeros((NX, NU))
    B[2, 0] = DT
    B[3, 1] = DT * v / (WB * math.cos(delta) ** 2)

    C = np.zeros(NX)
    C[0] = DT * v * math.sin(phi) * phi
    C[1] = -DT * v * math.cos(phi) * phi
    C[3] = -DT * v * delta / (WB * math.cos(delta) ** 2)

    return A, B, C


def update_state(state, a, delta):

    # input check
    if delta >= MAX_STEER:
        delta = MAX_STEER
    elif delta <= -MAX_STEER:
        delta = -MAX_STEER

    state.x = state.x + state.v * math.cos(state.yaw) * DT
    state.y = state.y + state.v * math.sin(state.yaw) * DT
    state.yaw = state.yaw + state.v / WB * math.tan(delta) * DT
    state.v = state.v + a * DT

    if state.v > MAX_SPEED:
        state.v = MAX_SPEED
    elif state.v < MIN_SPEED:
        state.v = MIN_SPEED

    return state


def get_nparray_from_matrix(x):
    return np.array(x).flatten()


def calc_nearest_index(state, cx, cy, cyaw, pind):

    dx = [state.x - icx for icx in cx[pind : (pind + N_IND_SEARCH)]]
    dy = [state.y - icy for icy in cy[pind : (pind + N_IND_SEARCH)]]

    d = [idx ** 2 + idy ** 2 for (idx, idy) in zip(dx, dy)]

    mind = min(d)

    ind = d.index(mind) + pind

    mind = math.sqrt(mind)

    dxl = cx[ind] - state.x
    dyl = cy[ind] - state.y

    angle = pi_2_pi(cyaw[ind] - math.atan2(dyl, dxl))
    if angle < 0:
        mind *= -1

    return ind, mind


def predict_motion(x0, oa, od, xref):
    xbar = xref * 0.0
    for i, _ in enumerate(x0):
        xbar[i, 0] = x0[i]

    state = State(x=x0[0], y=x0[1], yaw=x0[3], v=x0[2])
    for (ai, di, i) in zip(oa, od, range(1, T)):
        state = update_state(state, ai, di)
        xbar[0, i] = state.x
        xbar[1, i] = state.y
        xbar[2, i] = state.v
        xbar[3, i] = state.yaw

    return xbar


def iterative_linear_mpc_control(xref, x0, dref, oa, od, obstacles=None, horizon=T, orig_waypoint=None):
    """
    MPC contorl with updating operational point iteraitvely
    """

    if oa is None or od is None:
        oa = [0.0] * T
        od = [0.0] * T

    # for i in range(MAX_ITER):
    xbar = predict_motion(x0, oa, od, xref)
    poa, pod = oa[:], od[:]
    # cost, opt_x, opt_u = linear_mpc_control(xref, xbar, x0, dref, obstacles=obstacles, horizon=horizon)
    if obstacles is not None:

        try:
            cost, opt_x, opt_u = linear_mpc_obstacles_control(
                xref, xbar, x0, dref, obstacles, orig_waypoint=orig_waypoint, allow_slack=False
            )
        except diffcp.cone_program.SolverError:
            print(OKBLUE + "Allowing Collisions" + ENDC)
            cost, opt_x, opt_u = linear_mpc_obstacles_control(
                xref, xbar, x0, dref, obstacles, orig_waypoint=orig_waypoint, allow_slack=True
            )

    else:
        cost, opt_x, opt_u = linear_mpc_control(xref, xbar, x0, dref, obstacles=obstacles)
    #     du = sum(abs(oa - poa)) + sum(abs(od - pod))  # calc u change value
    #     if du <= DU_TH:
    #         break
    # else:
    #     print("Iterative is max iter")

    return cost, opt_x, opt_u


def line_seg_point_dist(x1, y1, x2, y2, point_x, point_y):  # point_x,point_y is the point
    px = x2 - x1
    py = y2 - y1

    norm = px * px + py * py

    u = ((point_x - x1) * px + (point_y - y1) * py) / float(norm)

    if u > 1:
        u = 1
    elif u < 0:
        u = 0

    x = x1 + u * px
    y = y1 + u * py

    dx = x - point_x
    dy = y - point_y

    # Note: If the actual distance does not matter,
    # if you only want to compare what this function
    # returns to other results of this function, you
    # can just return the squared distance instead
    # (i.e. remove the sqrt) to gain a little performance

    dist = dx * dx + dy * dy

    return dist


def dist_p2p(p_x, p_y, x, y):
    return (p_x - x) ** 2 + (p_y - y) ** 2


def linear_mpc_control(xref, xbar, x0, dref, obstacles=None):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
    # x = cvxpy.Variable((NX, T + 1))
    x = cvxpy.Variable((NX, T))
    u = cvxpy.Variable((NU, T))
    # XREF = cvxpy.Parameter((NX, T + 1))
    XREF = cvxpy.Parameter((NX, T))
    # OBS_UB = cvxpy.Parameter((obstacles.shape))

    cost = 0.0
    constraints = []

    for t in range(T):
        cost += cvxpy.quad_form(u[:, t], R)
        # cost += cvxpy.log(cnstr_max[:, t] - XREF[:, t])
        # cost += cvxpy.log(XREF[:, t] - cnstr_min[:, t])

        if t != 0:
            cost += cvxpy.quad_form(XREF[:, t] - x[:, t], Q)

        if t < (T - 1):
            A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * DT]

    cost += cvxpy.quad_form(XREF[:, -1] - x[:, -1], Qf)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)
    assert prob.is_dpp()

    # cvxpylayer = CvxpyLayer(prob, parameters=[XREF, OBS_UB], variables=[x, u])
    cvxpylayer = CvxpyLayer(prob, parameters=[XREF], variables=[x, u])

    opt_x, opt_u = cvxpylayer(xref)

    cost = 0.0
    control_cost = 0.0
    tracking_cost = 0.0

    for t in range(T):
        control_cost += opt_u[:, t].T @ torch.Tensor(R) @ opt_u[:, t]
        # cost += cvxpy.log(cnstr_max[:, t] - opt_x[:, t])
        # cost += cvxpy.log(opt_x[:, t] - cnstr_min[:, t])

        if t != 0:
            # tracking_cost += opt_x[:, t].T @ torch.Tensor(Q) @ opt_x[:, t]
            tracking_cost += (xref[:, t] - opt_x[:, t]).T @ torch.Tensor(Q) @ (xref[:, t] - opt_x[:, t])

        if t < (T - 1):
            control_cost += (opt_u[:, t + 1] - opt_u[:, t]).T @ torch.Tensor(Rd) @ (opt_u[:, t + 1] - opt_u[:, t])

    goal_cost = (xref[:, -1] - opt_x[:, -1]).T @ torch.Tensor(Qf) @ (xref[:, -1] - opt_x[:, -1])

    cost += control_cost
    cost += tracking_cost
    cost += goal_cost

    cost.backward()

    mpc_costs = MPCCost(total=cost, tracking=tracking_cost, control=control_cost, goal=goal_cost)

    return mpc_costs, opt_x, opt_u


def obs_func(ob, ego):
    xy, H = ob
    return cvxpy.quad_form(xy - ego, H)


def estimate_distance(x, y, rx, ry, x0=0, y0=0):
    """Given a point (x, y), and an ellipse with major - minor axis (rx, ry),
    its center at (x0, y0), and with a counter clockwise rotation of
    `angle` degrees, will return the distance between the ellipse and the
    closest point on the ellipses boundary.
    """
    from scipy import optimize

    def f(theta):
        return (rx ** 2 - ry ** 2) * np.cos(theta) * np.sin(theta) - x * rx * np.sin(theta) + y * ry * np.cos(theta)
        # x0 * rx * np.sin(theta) - y0 * ry * np.cos(theta)

    def f_prime(theta):
        return (
            (rx ** 2 - ry ** 2) * (np.cos(theta) ** 2 - np.sin(theta) ** 2)
            - x * rx * np.cos(theta)
            - y * ry * np.sin(theta)
        )
        # x0 * rx * np.cos(theta) + y0 * ry * np.sin(theta)

    init_z = np.arctan2(rx * y, ry * x)
    opt_theta = optimize.newton(f, init_z, fprime=f_prime, rtol=1.48e-03, maxiter=100)

    return rx * np.cos(opt_theta), ry * np.sin(opt_theta)


def find_closest_point_on_ellipsoid(xref, cx, cy, ob_xy):
    best_cx = cx
    best_cy = cy

    min_dist = np.inf

    def dist(x, y):
        return np.sqrt((np.square(x[0] - y[0]) + np.square(x[1] - y[1])))

    comp_dist = dist(xref, [ob_xy[0] + cx, ob_xy[1] + cy])
    if comp_dist < min_dist:
        min_dist = comp_dist
        best_cx, best_cy = ob_xy[0] + cx, ob_xy[1] + cy

    comp_dist = dist(xref, [ob_xy[0] + cx, ob_xy[1] - cy])
    if comp_dist < min_dist:
        min_dist = comp_dist
        best_cx, best_cy = ob_xy[0] + cx, ob_xy[1] - cy

    comp_dist = dist(xref, [ob_xy[0] - cx, ob_xy[1] + cy])
    if comp_dist < min_dist:
        min_dist = comp_dist
        best_cx, best_cy = ob_xy[0] - cx, ob_xy[1] + cy

    comp_dist = dist(xref, [ob_xy[0] - cx, ob_xy[1] - cy])
    if comp_dist < min_dist:
        min_dist = comp_dist
        best_cx, best_cy = ob_xy[0] - cx, ob_xy[1] - cy

    return best_cx, best_cy


def plot(x, xref, obstacles, hyperplanes):

    colors = [
        "tab:red",
        "tab:purple",
        "tab:orange",
        "tab:pink",
        "tab:olive",
        "tab:brown",
        "tab:cyan",
        "tab:gray",
        "tab:blue",
        "lime",
        "maroon",
    ]

    def abline(ax, slope, intercept, color):
        """Plot a line from slope and intercept"""
        axes = plt.gca()
        x_vals = np.array(axes.get_xlim())
        y_vals = intercept + slope * x_vals
        ax.plot(x_vals, y_vals, "--", c=color)
        return x_vals, y_vals

    fig, ax = plt.subplots(subplot_kw={"aspect": "equal"})
    ax.plot(x[0, :], x[1, :], "o", c="k", label="mpc")
    ax.plot(xref[0, :], xref[1, :], "o", c="b", label="ref")

    for i in range(x.shape[1] - 1):
        ells = [Ellipse(xy=(ox, oy), width=2 * ow, height=2 * oh) for (ox, oy, ow, oh) in obstacles]
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        for e in ells:
            ax.add_artist(e)
            e.set(alpha=0.05, facecolor="green")

        ax.plot(x[0, i], x[1, i], "-o", c=colors[i])
        for hp in hyperplanes[i]:
            xvals, yvals = abline(ax, hp[0], hp[1], colors[i])
            ax.plot(hp[2], hp[3], "xr")
            A = np.array([-hp[0], 1])
            # ax.fill_between(xvals, yvals, where=A @ xvals <= hp[1], alpha=0.1, facecolor=colors[i], interpolate=True)

    plt.legend()

    plt.show()
    plt.close("all")


def linear_mpc_obstacles_control(xref, xbar, x0, dref, obstacles, orig_waypoint, allow_slack=False):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
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

        ob_hyp = []
        for obdx, ob in enumerate(obstacles):
            xy, H = ob
            xref_np = xref[:2, t].detach().numpy()
            cx, cy = estimate_distance(x=xref_np[0], y=xref_np[1], rx=H[0, 0], ry=H[1, 1], x0=xy[0], y0=xy[1])
            cx, cy = find_closest_point_on_ellipsoid(xref_np, cx, cy, xy)

            slope = None
            if xref[0, t] < xy[0] and xref[1, t] < xy[1]:
                slope = -((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] <= b + slack[obdx, t]]

            if xref[0, t] > xy[0] and xref[1, t] < xy[1]:
                slope = ((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] <= b + slack[obdx, t]]

            if xref[0, t] < xy[0] and xref[1, t] > xy[1]:
                slope = ((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] + slack[obdx, t] >= b]

            if xref[0, t] > xy[0] and xref[1, t] > xy[1]:
                slope = -((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] + slack[obdx, t] >= b]

            if allow_slack:
                constraints += [slack[obdx, t] >= 0.0]
            else:
                constraints += [slack[obdx, t] == 0.0]

            assert slope is not None
            ob_hyp.append([slope, b, cx, cy])

        # hyperplanes.append(ob_hyp)

    cost += cvxpy.quad_form(XREF[:, -1] - x[:, -1], QF_GAIN * Qf)
    cost += OBSTACLE_GAIN * cvxpy.sum(slack)

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

    opt_x, opt_u, opt_slack = cvxpylayer(xref, solver_args={"solve_method": "ECOS"})

    cost_state = opt_x

    if orig_waypoint is not None:
        new_x = torch.zeros((NX, T))
        new_x[:2, 0] = orig_waypoint[0, 0]
        new_x[2, 0] = x0[2]
        new_x[3, 0] = x0[3]

        for t in range(T - 1):
            A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            A_torch = torch.from_numpy(A).type(torch.FloatTensor)
            B_torch = torch.from_numpy(B).type(torch.FloatTensor)
            C_torch = torch.from_numpy(C).type(torch.FloatTensor)
            new_x[:, t + 1] = A_torch @ new_x[:, t] + B_torch @ opt_u[:, t] + C_torch

        cost_state = new_x

    cost = 0.0
    tracking_cost = 0.0
    control_cost = 0.0
    goal_cost = 0.0

    for t in range(T):
        control_cost += opt_u[:, t].T @ torch.Tensor(R) @ opt_u[:, t]

        if t != 0:
            tracking_cost += (xref[:, t] - cost_state[:, t]).T @ torch.Tensor(Q) @ (xref[:, t] - cost_state[:, t])

        if t < (T - 1):
            control_cost += (opt_u[:, t + 1] - opt_u[:, t]).T @ torch.Tensor(Rd) @ (opt_u[:, t + 1] - opt_u[:, t])

    goal_cost = (xref[:, -1] - cost_state[:, -1]).T @ (QF_GAIN * torch.Tensor(Qf)) @ (xref[:, -1] - cost_state[:, -1])

    cost += control_cost
    cost += tracking_cost
    cost += goal_cost

    if torch.sum(opt_slack) > 1.0:
        cost += OBSTACLE_GAIN * torch.sum(opt_slack)

    mpc_costs = MPCCost(total=cost, tracking=tracking_cost, control=control_cost, goal=goal_cost)

    return mpc_costs, cost_state, opt_u


def compute_ellipsoid_method_traj(opt_x, xref, xbar, x0, dref, obstacles, allow_slack=False):
    x_ellip = cvxpy.Variable((NX, T))
    u_ellip = cvxpy.Variable((NU, T))
    slack_ellip = cvxpy.Variable((4, T))

    OPT_X = cvxpy.Parameter((NX, T))

    cost_ellip = 0.0
    constraints_ellip = []

    for t in range(T):

        if t < (T - 1):
            A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints_ellip += [x_ellip[:, t + 1] == A @ x_ellip[:, t] + B @ u_ellip[:, t] + C]

            constraints_ellip += [cvxpy.abs(u_ellip[1, t + 1] - u_ellip[1, t]) <= MAX_DSTEER * DT]

        ob_hyp = []
        for obdx, ob in enumerate(obstacles):
            xy, H = ob
            xref_np = xref[:2, t].detach().numpy()
            cx, cy = estimate_distance(x=xref_np[0], y=xref_np[1], rx=H[0, 0], ry=H[1, 1], x0=xy[0], y0=xy[1])
            cx, cy = find_closest_point_on_ellipsoid(xref_np, cx, cy, xy)

            slope = None
            if xref[0, t] < xy[0] and xref[1, t] < xy[1]:
                slope = -((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints_ellip += [A @ x_ellip[:2, t] <= b + slack_ellip[obdx, t]]

            if xref[0, t] > xy[0] and xref[1, t] < xy[1]:
                slope = ((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints_ellip += [A @ x_ellip[:2, t] <= b + slack_ellip[obdx, t]]

            if xref[0, t] < xy[0] and xref[1, t] > xy[1]:
                slope = ((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints_ellip += [A @ x_ellip[:2, t] + slack_ellip[obdx, t] >= b]

            if xref[0, t] > xy[0] and xref[1, t] > xy[1]:
                slope = -((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints_ellip += [A @ x_ellip[:2, t] + slack_ellip[obdx, t] >= b]

            if allow_slack:
                constraints_ellip += [slack_ellip[obdx, t] >= 0.0]
            else:
                constraints_ellip += [slack_ellip[obdx, t] == 0.0]

            assert slope is not None
            ob_hyp.append([slope, b, cx, cy])

    cost_ellip += cvxpy.sum((OPT_X - x_ellip) ** 2)
    cost_ellip += OBSTACLE_GAIN * cvxpy.sum(slack_ellip)

    constraints_ellip += [x_ellip[:, 0] == x0]
    constraints_ellip += [x_ellip[2, :] <= MAX_SPEED]
    constraints_ellip += [x_ellip[2, :] >= MIN_SPEED]
    constraints_ellip += [cvxpy.abs(u_ellip[0, :]) <= MAX_ACCEL]
    constraints_ellip += [cvxpy.abs(u_ellip[1, :]) <= MAX_STEER]

    prob = cvxpy.Problem(cvxpy.Minimize(cost_ellip), constraints_ellip)

    assert prob.is_dpp(), "Problem is not DPP"
    cvxpylayer = CvxpyLayer(prob, parameters=[OPT_X], variables=[x_ellip, u_ellip, slack_ellip])

    opt_x_ellip, opt_u_ellip, opt_slack_ellip = cvxpylayer(opt_x)

    return opt_x_ellip, opt_u_ellip, opt_slack_ellip


def linear_mpc_obstacles_ellipsoid(xref, xbar, x0, dref, obstacles, allow_slack=False):
    """
    linear mpc control
    xref: reference point
    xbar: operational point
    x0: initial state
    dref: reference steer angle
    """
    x = cvxpy.Variable((NX, T))
    u = cvxpy.Variable((NU, T))
    slack = cvxpy.Variable((4, T))
    XREF = cvxpy.Parameter((NX, T))

    cost = 0.0
    constraints = []

    for t in range(T):
        # cost += cvxpy.quad_form(u[:, t], R)

        if t != 0:
            cost += cvxpy.quad_form(XREF[:, t] - x[:, t], Q)

        if t < (T - 1):
            A, B, C = get_linear_model_matrix(xbar[2, t], xbar[3, t], dref[0, t])
            constraints += [x[:, t + 1] == A @ x[:, t] + B @ u[:, t] + C]

            # cost += cvxpy.quad_form(u[:, t + 1] - u[:, t], Rd)
            constraints += [cvxpy.abs(u[1, t + 1] - u[1, t]) <= MAX_DSTEER * ELLIP_CONSTR_RELAX * DT]

        ob_hyp = []
        for obdx, ob in enumerate(obstacles):
            xy, H = ob
            xref_np = xref[:2, t].detach().numpy()
            cx, cy = estimate_distance(x=xref_np[0], y=xref_np[1], rx=H[0, 0], ry=H[1, 1], x0=xy[0], y0=xy[1])
            cx, cy = find_closest_point_on_ellipsoid(xref_np, cx, cy, xy)

            slope = None
            if xref[0, t] < xy[0] and xref[1, t] < xy[1]:
                slope = -((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] <= b + slack[obdx, t]]

            if xref[0, t] > xy[0] and xref[1, t] < xy[1]:
                slope = ((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] <= b + slack[obdx, t]]

            if xref[0, t] < xy[0] and xref[1, t] > xy[1]:
                slope = ((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] + slack[obdx, t] >= b]

            if xref[0, t] > xy[0] and xref[1, t] > xy[1]:
                slope = -((H[1, 1]) ** 2 * cx) / (H[0, 0] ** 2 * cy)
                A = np.array([-slope, 1])
                b = -slope * cx + cy
                constraints += [A @ x[:2, t] + slack[obdx, t] >= b]

            if allow_slack:
                constraints += [slack[obdx, t] >= 0.0]
            else:
                constraints += [slack[obdx, t] == 0.0]

            assert slope is not None
            ob_hyp.append([slope, b, cx, cy])

    cost += cvxpy.quad_form(XREF[:, -1] - x[:, -1], QF_GAIN * Qf)
    cost += OBSTACLE_GAIN * cvxpy.sum(slack)

    constraints += [x[:, 0] == x0]
    constraints += [x[2, :] <= MAX_SPEED]
    constraints += [x[2, :] >= MIN_SPEED]
    constraints += [cvxpy.abs(u[0, :]) <= MAX_ACCEL * ELLIP_CONSTR_RELAX]
    constraints += [cvxpy.abs(u[1, :]) <= MAX_STEER * ELLIP_CONSTR_RELAX]

    prob = cvxpy.Problem(cvxpy.Minimize(cost), constraints)

    assert prob.is_dpp(), "Problem is not DPP"
    cvxpylayer = CvxpyLayer(prob, parameters=[XREF], variables=[x, u, slack])

    opt_x, opt_u, opt_slack = cvxpylayer(xref)

    cost = torch.zeros((1))
    tracking_cost = torch.zeros((1))
    control_cost = torch.zeros((1))
    goal_cost = torch.zeros((1))

    for t in range(T):
        # control_cost += opt_u[:, t].T @ torch.Tensor(R) @ opt_u[:, t]

        if t != 0:
            tracking_cost += (xref[:, t] - opt_x[:, t]).T @ torch.Tensor(Q) @ (xref[:, t] - opt_x[:, t])

        # if t < (T - 1):
        #     control_cost += (
        #         (opt_u[:, t + 1] - opt_u[:, t]).T
        #         @ torch.Tensor(Rd)
        #         @ (opt_u[:, t + 1] - opt_u[:, t])
        #     )

    goal_cost = (xref[:, -1] - opt_x[:, -1]).T @ (QF_GAIN * torch.Tensor(Qf)) @ (xref[:, -1] - opt_x[:, -1])

    # cost += control_cost
    # cost += goal_cost
    cost += tracking_cost

    if torch.sum(opt_slack) > 1.0:
        cost += OBSTACLE_GAIN * torch.sum(opt_slack)

    cost.backward()

    mpc_costs = MPCCost(total=cost, tracking=tracking_cost, control=control_cost, goal=goal_cost)

    return mpc_costs, opt_x, opt_u


def calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, pind):
    xref = torch.zeros((NX, T + 1))
    dref = np.zeros((1, T + 1))
    ncourse = len(cx)

    ind, _ = calc_nearest_index(state, cx, cy, cyaw, pind)

    if pind >= ind:
        ind = pind

    xref[0, 0] = cx[ind]
    xref[1, 0] = cy[ind]
    xref[2, 0] = sp[ind]
    xref[3, 0] = cyaw[ind]
    dref[0, 0] = 0.0  # steer operational point should be 0

    travel = 0.0

    for i in range(T + 1):
        travel += abs(state.v) * DT
        dind = int(round(travel / dl))

        if (ind + dind) < ncourse:
            xref[0, i] = cx[ind + dind]
            xref[1, i] = cy[ind + dind]
            xref[2, i] = sp[ind + dind]
            xref[3, i] = cyaw[ind + dind]
            dref[0, i] = 0.0
        else:
            xref[0, i] = cx[ncourse - 1]
            xref[1, i] = cy[ncourse - 1]
            xref[2, i] = sp[ncourse - 1]
            xref[3, i] = cyaw[ncourse - 1]
            dref[0, i] = 0.0

    return xref, ind, dref


def check_goal(state, goal, tind, nind):

    # check goal
    dx = state.x - goal[0]
    dy = state.y - goal[1]
    d = math.hypot(dx, dy)

    isgoal = d <= GOAL_DIS

    if abs(tind - nind) >= 5:
        isgoal = False

    isstop = abs(state.v) <= STOP_SPEED

    if isgoal and isstop:
        return True

    return False


def track_traj(cx, cy, cyaw, ck, obstacles=None, horizon=T, init_v=3.0, orig_waypoint=None):
    init_state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=init_v)
    target_ind, _ = calc_nearest_index(init_state, cx, cy, cyaw, 0)
    sp = calc_speed_profile(cx, cy, cyaw, TARGET_SPEED)
    sp[-1] = sp[-2]

    # xref, target_ind, dref = calc_ref_trajectory(init_state, cx, cy, cyaw, ck, sp, 1.0, target_ind)
    dref = torch.zeros((1, T))

    if init_state.yaw - cyaw[0] >= torch.Tensor([math.pi]):
        init_state.yaw -= torch.Tensor([math.pi]) * 2.0
    elif init_state.yaw - cyaw[0] <= -torch.Tensor([math.pi]):
        init_state.yaw += torch.Tensor([math.pi]) * 2.0

    x0 = [init_state.x, init_state.y, init_state.v, init_state.yaw]  # current state
    xref = torch.zeros((4, T))
    xref[0, :] = cx
    xref[1, :] = cy
    xref[2, :] = torch.from_numpy(np.array(sp))
    xref[3, :] = cyaw
    xref[:, 0] = torch.Tensor(x0)

    cost, opt_x, opt_u = iterative_linear_mpc_control(
        xref, x0, dref, oa=None, od=None, obstacles=obstacles, horizon=horizon, orig_waypoint=orig_waypoint
    )

    return cost, opt_x, opt_u


def do_simulation(cx, cy, cyaw, ck, sp, dl, initial_state):
    """
    Simulation
    cx: course x position list
    cy: course y position list
    cy: course yaw position list
    ck: course curvature list
    sp: speed profile
    dl: course tick [m]
    """

    goal = [cx[-1], cy[-1]]

    state = State(x=cx[0], y=cy[0], yaw=cyaw[0], v=0.0)

    # initial yaw compensation
    if state.yaw - cyaw[0] >= math.pi:
        state.yaw -= math.pi * 2.0
    elif state.yaw - cyaw[0] <= -math.pi:
        state.yaw += math.pi * 2.0

    time = 0.0
    x = [state.x]
    y = [state.y]
    yaw = [state.yaw]
    v = [state.v]
    t = [0.0]
    d = [0.0]
    a = [0.0]
    target_ind, _ = calc_nearest_index(state, cx, cy, cyaw, 0)

    odelta, oa = None, None

    cyaw = smooth_yaw(cyaw)

    while MAX_TIME >= time:
        xref, target_ind, dref = calc_ref_trajectory(state, cx, cy, cyaw, ck, sp, dl, target_ind)

        x0 = [state.x, state.y, state.v, state.yaw]  # current state

        oa, odelta, ox, oy, oyaw, ov = iterative_linear_mpc_control(xref, x0, dref, oa, odelta)

        if odelta is not None:
            di, ai = odelta[0], oa[0]

        state = update_state(state, ai, di)
        time = time + DT

        x.append(state.x)
        y.append(state.y)
        yaw.append(state.yaw)
        v.append(state.v)
        t.append(time)
        d.append(di)
        a.append(ai)

        if check_goal(state, goal, target_ind, len(cx)):
            print("Goal")
            break

    return t, x, y, yaw, v, d, a


def calc_speed_profile(cx, cy, cyaw, target_speed):

    speed_profile = [target_speed] * len(cx)
    direction = 1.0  # forward

    # Set stop point
    for i in range(len(cx) - 1):
        dx = cx[i + 1] - cx[i]
        dy = cy[i + 1] - cy[i]

        move_direction = math.atan2(dy, dx)

        if dx != 0.0 and dy != 0.0:
            dangle = abs(pi_2_pi(move_direction - cyaw[i]))
            if dangle >= math.pi / 4.0:
                direction = -1.0
            else:
                direction = 1.0

        if direction != 1.0:
            speed_profile[i] = -target_speed
        else:
            speed_profile[i] = target_speed

    speed_profile[-1] = 0.0

    return speed_profile


def smooth_yaw(yaw):

    for i in range(len(yaw) - 1):
        dyaw = yaw[i + 1] - yaw[i]

        while dyaw >= math.pi / 2.0:
            yaw[i + 1] -= math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

        while dyaw <= -math.pi / 2.0:
            yaw[i + 1] += math.pi * 2.0
            dyaw = yaw[i + 1] - yaw[i]

    return yaw
