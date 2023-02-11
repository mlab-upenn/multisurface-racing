# MIT License

# Copyright (c) Tomas Nagy, Ahmad Amine, Hongrui Zheng, Johannes Betz

# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:

# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.

# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

"""
Dynamic Single Track MPC waypoint tracker

Author: Tomas Nagy, Hongrui Zheng, Johannes Betz, Ahmad Amine
Last Modified: 8/1/22
"""
import time
from dataclasses import dataclass, field
import cvxpy
import numpy as np
from scipy.linalg import block_diag
from scipy.sparse import block_diag, csc_matrix, diags
from numba import njit
import copy


@njit(cache=True)
def nearest_point(point, trajectory):
    """
    Return the nearest point along the given piecewise linear trajectory.
    Args:
        point (numpy.ndarray, (2, )): (x, y) of current pose
        trajectory (numpy.ndarray, (N, 2)): array of (x, y) trajectory waypoints
            NOTE: points in trajectory must be unique. If they are not unique, a divide by 0 error will destroy the world
    Returns:
        nearest_point (numpy.ndarray, (2, )): nearest point on the trajectory to the point
        nearest_dist (float): distance to the nearest point
        t (float): nearest point's location as a segment between 0 and 1 on the vector formed by the closest two points on the trajectory. (p_i---*-------p_i+1)
        i (int): index of nearest point in the array of trajectory waypoints
    """
    diffs = trajectory[1:, :] - trajectory[:-1, :]
    l2s = diffs[:, 0] ** 2 + diffs[:, 1] ** 2
    dots = np.empty((trajectory.shape[0] - 1,))
    for i in range(dots.shape[0]):
        dots[i] = np.dot((point - trajectory[i, :]), diffs[i, :])
    t = dots / l2s
    t[t < 0.0] = 0.0
    t[t > 1.0] = 1.0
    projections = trajectory[:-1, :] + (t * diffs.T).T
    dists = np.empty((projections.shape[0],))
    for i in range(dists.shape[0]):
        temp = point - projections[i]
        dists[i] = np.sqrt(np.sum(temp * temp))
    min_dist_segment = np.argmin(dists)
    dist_from_segment_start = np.linalg.norm(diffs[min_dist_segment] * t[min_dist_segment])
    return projections[min_dist_segment], dist_from_segment_start, dists[min_dist_segment], t[
        min_dist_segment], min_dist_segment


class STMPCPlanner:
    """
    Dynamic Single Track MPC Controller, uses the ST model from Common Road

    All vehicle pose used by the planner should be in the map frame.

    Args:
        waypoints (numpy.ndarray [N x 4], optional): static waypoints to track
        mass, l_f, l_r, h_CoG, c_f, c_r, Iz, mu

    Attributes:
        waypoints (numpy.ndarray [N x 4]): static list of waypoints, columns are [x, y, velocity, heading]
    """

    def __init__(
            self,
            model,
            config,
            waypoints=None,
            track=None,
    ):
        self.waypoints = waypoints
        self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        self.model = model
        self.config = config
        self.track = track
        self.target_ind = 0
        self.input_o = np.zeros(self.config.NU) * np.NAN
        self.states_output = np.ones((self.config.NXK, self.config.TK)) * np.NaN  # (7, 16)
        self.odelta_v = np.NAN
        self.oa = np.NAN
        self.origin_switch = 1
        self.init_flag = 0
        self.mpc_prob_init()
        self.solve_time = time.time()

        self.track_length = 100.0

        self.trajectry_interpolation = 0  # 0 - linear, 1 - constant curvature

        # safe set
        self.SS_frenet = []  # safe set (states) Going to be used to calculate safe set every iteration
        self.SS_cartesian = []  # safe set (states) Going to be used to recalculate self.SS_frenet and costs in case of change of the reference traj
        self.uSS = []  # safe set (inputs)
        self.Qfinal_speed = []  # final cost approximation for speed
        self.Qfinal_tracking = []  # final cost approximation
        self.it = 0  # LMPC iteration
        self.LapTimes = []
        self.zt = np.array([0.0, 0.0, 5.0, 0.0, 0.0, 0.0, 0.0])
        self.time_step = 0
        self.tracking_accumulated_cost = 0

        self.mpc_tracking_weight = 0
        self.mpc_speed_opt_weight = 0

    def plan(self, states, waypoints=None):
        """
        Planner plan function overload for Pure Pursuit, returns acutation based on current state

        Args:
            pose_x (float): current vehicle x position
            pose_y (float): current vehicle y position
            pose_theta (float): current vehicle heading angle
            lookahead_distance (float): lookahead distance to find next waypoint to track
            waypoints (numpy.ndarray [N x 4], optional): list of dynamic waypoints to track, columns are [x, y, velocity, heading]

        Returns:
            speed (float): commanded vehicle longitudinal velocity
            steering_angle (float):  commanded vehicle steering angle

        TODO: implement switching between different controllers here
        """
        if waypoints is not None:
            if waypoints.shape[1] < 3 or len(waypoints.shape) != 2:
                raise ValueError("Waypoints needs to be a (Nxm), m >= 3, numpy array!")
            self.waypoints = waypoints
            self.waypoints_distances = np.linalg.norm(self.waypoints[1:, (1, 2)] - self.waypoints[:-1, (1, 2)], axis=1)
        else:
            if self.waypoints is None:
                raise ValueError(
                    "Please set waypoints to track during planner instantiation or when calling plan()"
                )

        (
            u,
            mpc_ref_path_x,
            mpc_ref_path_y,
            mpc_pred_x,
            mpc_pred_y,
            mpc_ox,
            mpc_oy,
        ) = self.MPC_Control(states, self.waypoints)

        return u, mpc_ref_path_x, mpc_ref_path_y, mpc_pred_x, mpc_pred_y, mpc_ox, mpc_oy

    def get_reference_trajectory(self, predicted_speeds, dist_from_segment_start, idx, waypoints):
        s_relative = np.zeros((self.config.TK + 1,))
        s_relative[0] = dist_from_segment_start
        s_relative[1:] = predicted_speeds * self.config.DTK
        s_relative = np.cumsum(s_relative)

        waypoints_distances_relative = np.cumsum(np.roll(self.waypoints_distances, -idx))

        index_relative = np.int_(np.ones((self.config.TK + 1,)))
        for i in range(self.config.TK + 1):
            index_relative[i] = (waypoints_distances_relative <= s_relative[i]).sum()
        index_absolute = np.mod(idx + index_relative, waypoints.shape[0] - 1)

        segment_part = s_relative - (
                waypoints_distances_relative[index_relative] - self.waypoints_distances[index_absolute])

        t = (segment_part / self.waypoints_distances[index_absolute])
        # print(np.all(np.logical_and((t < 1.0), (t > 0.0))))

        position_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, (1, 2)] -
                          waypoints[index_absolute][:, (1, 2)])

        orientation_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 3] -
                             waypoints[index_absolute][:, 3])

        speed_diffs = (waypoints[np.mod(index_absolute + 1, waypoints.shape[0] - 1)][:, 5] -
                       waypoints[index_absolute][:, 5])

        interpolated_positions = waypoints[index_absolute][:, (1, 2)] + (t * position_diffs.T).T

        interpolated_orientations = waypoints[index_absolute][:, 3] + (t * orientation_diffs)
        interpolated_orientations = (interpolated_orientations + np.pi) % (2 * np.pi) - np.pi

        interpolated_speeds = waypoints[index_absolute][:, 5] + (t * speed_diffs)

        reference = self.model.sort_reference_trajectory(interpolated_positions,
                                                         interpolated_orientations,
                                                         interpolated_speeds)

        return reference

    def calc_ref_trajectory(self, position, orientation, speed, path):
        """
        calc referent trajectory ref_traj in T steps: [x, y, v, yaw]
        using the current velocity, calc the T points along the reference path
        :param cx: Course X-Position
        :param cy: Course y-Position
        :param cyaw: Course Heading
        :param sp: speed profile
        :dl: distance step
        :pind: Setpoint Index
        :return: reference trajectory ref_traj, reference steering angle
        """

        speeds = np.ones(self.config.TK) * speed  # method 1

        # speeds = np.take(path[:, 5], range(ind, ind + self.config.TK), mode='wrap')  # method 2

        # speeds_ref = np.take(path[:, 5], range(ind, ind + self.config.TK), mode='wrap')  # method 3
        # speeds = (speeds_ref + speed) / 2.0

        if self.trajectry_interpolation == 0:
            # TODO curently only for cartesian frame
            # Find nearest index/setpoint from where the trajectories are calculated
            _, dist, _, _, ind = nearest_point(np.array([position[0], position[1]]), path[:, (1, 2)])
            # path[:, 5]

            reference = self.get_reference_trajectory(speeds, dist, ind, path)

            reference[3, :][reference[3, :] - orientation > 5] = np.abs(
                reference[3, :][reference[3, :] - orientation > 5] - (2 * np.pi))
            reference[3, :][reference[3, :] - orientation < -5] = np.abs(
                reference[3, :][reference[3, :] - orientation < -5] + (2 * np.pi))

            reference[2] = np.where(reference[2] - speed > 5.0, speed + 5.0, reference[2])
        elif self.trajectry_interpolation == 1:
            # TODO curently only for frenet frame
            # TODO calculate ref speed in better way -- speed_ref[i + 1] = speed is not accurate approximation for linearization point

            position_ref = np.zeros((self.config.TK + 1, 2))
            speed_ref = np.zeros(self.config.TK + 1)
            position_ref[0, 0] = position[0]
            speed_ref[0] = speed
            for i in range(self.config.TK):
                position_ref[i + 1, 0] = position_ref[i, 0] + speeds[i] * self.config.DTK
                speed_ref[i + 1] = speed

            reference = self.model.sort_reference_trajectory(position_ref, np.zeros(self.config.TK + 1), speed_ref)
        else:
            print("ERROR: unknown trajectory interpolation method")

        return reference, 0

    def compute_speed_cost(self, x, u):
        # Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
        for i in range(0, len(x)):
            if i == 0:
                cost = [0]
            else:
                cost.append(cost[-1] + 1)

        # Finally flip the cost to have correct order
        return np.flip(cost).tolist()

    def compute_tracking_cost(self, x, u):
        # Compute the cost in a DP like strategy: start from the last point x[len(x)-1] and move backwards
        for i in range(0, len(x)):
            idx = len(x) - 1 - i
            if i == 0:
                cost = [np.dot(np.dot(x[idx], self.config.Qk), x[idx])]  # Last element (finish line)
            elif idx == 0:
                cost.append(np.dot(np.dot(x[idx], self.config.Qk), x[idx]) + np.dot(np.dot(u[idx], self.config.Rk), u[idx]) + cost[-1])
            else:
                #                                  tracking error                             input cost                                      input difference                               prev cost
                cost.append(np.dot(np.dot(x[idx], self.config.Qk), x[idx]) + np.dot(np.dot(u[idx], self.config.Rk), u[idx]) + np.dot(
                    np.dot(u[idx] - u[idx - 1], self.config.Rdk), u[idx] - u[idx - 1]) + cost[-1])

        # Finally flip the cost to have correct order
        return np.flip(cost).tolist()

    def add_safe_trajectory(self, x, u):

        self.LapTimes.append(len(x))
        # Add the feasible trajectory x and the associated input sequence u to the safe set
        self.SS_cartesian.append(copy.copy(x))

        x_frenet = []
        for state in x:
            pose_frenet = self.track.cartesian_to_frenet(np.array([state[0], state[1], state[3]]))
            x_frenet.append(np.array([pose_frenet[0], pose_frenet[1], state[2], pose_frenet[2], state[4], state[5], state[6]]))

        self.SS_frenet.append(copy.copy(x_frenet))
        self.uSS.append(copy.copy(u))

        # Compute and store the cost associated with the feasible trajectory
        cost = self.compute_speed_cost(x, u)
        self.Qfinal_speed.append(cost)  # final cost approximation for speed

        cost = self.compute_tracking_cost(x, u)
        self.Qfinal_tracking.append(cost)  # final cost approximation for tracking

        # Initialize zVector
        # self.zt = np.array(x[self.ftocp.N])

        # Augment iteration counter and print the cost of the trajectories stored in the safe set
        self.it = self.it + 1
        self.time_step = 0
        self.tracking_accumulated_cost = 0
        print("Trajectory added to the Safe Set. Current Iteration: ", self.it)
        print("Performance stored trajectories: \n", [self.Qfinal_speed[i][0] for i in range(0, self.it)])

    def calc_safe_set_components(self, x0, xPred, uPred):
        # Update zt and xLin is they have crossed the finish line. We want s \in [0, TrackLength]
        if (self.zt[0] - x0[0] > self.track_length / 2):
            self.zt[0] = np.max([self.zt[0] - self.track_length, 0])
            self.xLin[0, -1] = self.xLin[0, -1] - self.track_length  # TODO change xLin to correct variable

        # sort trajectories by time to construct safe set only from the best possible laps
        sortedLapTime = np.argsort(np.array(self.LapTimes))

        # Select Points from historical data. These points will be used to construct the terminal cost function and constraint set
        # TODO change for our problem
        # SS_PointSelectedTot = np.empty((self.n, 0))
        # Succ_SS_PointSelectedTot = np.empty((self.n, 0))
        # Succ_uSS_PointSelectedTot = np.empty((self.d, 0))
        # Qfun_SelectedTot = np.empty((0))

        self.numSS_it = 4
        self.numSS_Points = 12 * self.numSS_it

        for traj_idx in sortedLapTime[0:self.numSS_it]:  # from zero to N trajectories to compute SS with
            SS_PointSelected, uSS_PointSelected, Qfun_speed_Selected, Qfun_track_Selected = self.select_points(traj_idx, self.zt,
                                                                                                               self.numSS_Points / self.numSS_it + 1, xPred, uPred)

    def select_points(self, traj_idx, zt, num_points, xPred, uPred):
        '''
        Selects num_points samples from the traj_idx trajectory based on the num_points closest points to zt
        :param traj_idx: index of trajectory to select from
        :param zt: Expected end state
        :param num_points: Number of points to approximate safe set with
        :param xPred: The open-loop states along the control horizon from the MPC solution
        :param uPred: The open-loop control inputs from the MPC solution

        :return: SS_Points, the states of the chosen points of the safe set
                 SSu_Points, the control inputs of the chosen points of the safe set
                 Sel_Qfun_speed, the speed (minimum time) cost of the chosen points of the safe set
                 Sel_Qfun_track, the tracking cost of the chosen points of the safe set
        '''

        # DONE FOR NOW, DEBUG FOREVER
        x = self.SS_frenet[traj_idx]
        u = self.uSS[traj_idx]

        # Find the closest state between state zt and a last few runs
        MinNorm = np.argmin(np.linalg.norm(x - zt, 1, axis=1))

        if (MinNorm - num_points / 2 >= 0):
            indexSSandQfun = np.arange(-int(num_points / 2) + MinNorm, int(num_points / 2) + MinNorm + 1)
        else:
            indexSSandQfun = np.arange(MinNorm, MinNorm + int(num_points))

        SS_Points = x[indexSSandQfun, :].T
        SSu_Points = u[indexSSandQfun, :].T

        # Modify the cost if the predicion has crossed the finisch line
        if xPred == [] or (np.all((xPred[:, 0] > self.track_length) == False)):
            Sel_Qfun_speed = self.Qfinal_speed[traj_idx][indexSSandQfun]
            Sel_Qfun_track = self.Qfinal_tracking[traj_idx][indexSSandQfun]
        elif traj_idx < self.it - 1:  # Going through the finish line
            Sel_Qfun_speed = self.Qfinal_speed[traj_idx][indexSSandQfun] + self.Qfinal_speed[traj_idx][0]
            Sel_Qfun_track = self.Qfinal_tracking[traj_idx][indexSSandQfun] + self.Qfinal_tracking[traj_idx][0]
        else:  # Going through the finish line and did not finish the second lap
            sPred = xPred[:, 0]
            predCurrLap = self.config.TK - sum(sPred > self.track_length)
            currLapTime = self.time_step
            curr_tracking_cost = self.tracking_accumulated_cost
            pred_tracking_cost = self.compute_tracking_cost(xPred[sPred <= self.track_length, :], uPred[sPred <= self.track_length])[0] # SHOULD BE SUM TO N OF XQX + URU
            # Sel_Qfun = self.Qfun[traj_idx][indexSSandQfun] + currLapTime + predCurrLap
            Sel_Qfun_speed = self.Qfinal_speed[traj_idx][indexSSandQfun] + currLapTime + predCurrLap  # this is not good enough approximation
            Sel_Qfun_track = self.Qfinal_tracking[traj_idx][indexSSandQfun] + curr_tracking_cost + pred_tracking_cost  # this is not good enough approximation
            #       X      =                      h(x)(it)                  +         H(x)       +      (N-OVER)

        return SS_Points, SSu_Points, Sel_Qfun_speed, Sel_Qfun_track

    def add_point(self, x, u):
        """at iteration j add the current point to SS, uSS and Qfun of the previous iteration
        Arguments:
            x: current state
            u: current input
        """
        self.SS_cartesian[self.it - 1] = np.append(self.SS_frenet[self.it - 1], np.array([x]), axis=0)
        pose_frenet = self.track.cartesian_to_frenet(np.array([x[0], x[1], x[3]]))
        x_frenet = np.array([pose_frenet[0], pose_frenet[1], x[2], pose_frenet[2], x[4], x[5], x[6]])
        self.SS_frenet[self.it - 1] = np.append(self.SS_frenet[self.it - 1], np.array([x_frenet + np.array([self.track_length, 0, 0, 0, 0, 0, 0])]),
                                                axis=0)
        self.uSS[self.it - 1] = np.append(self.uSS[self.it - 1], np.array([u]), axis=0)
        self.Qfinal_speed[self.it - 1] = np.append(self.Qfinal_speed[self.it - 1], self.Qfinal_speed[self.it - 1][-1] - 1)
        # self.SS_frenet[self.it - 1][-1] == x
        # self.uSS[self.it - 1][-1] == u
        u_diff = self.uSS[self.it - 1][-1] - self.uSS[self.it - 1][-2]  # u_now - u_prev
        current_track_cost = np.dot(np.dot(x, self.config.Qk), x) + \
                             np.dot(np.dot(u, self.config.Rk), u) + \
                             np.dot(np.dot(u_diff, self.config.Rdk), u_diff)

        self.Qfinal_tracking[self.it - 1] = np.append(self.Qfinal_tracking[self.it - 1],
                                                      self.Qfinal_tracking[self.it - 1][-1] - current_track_cost)
        # The above two lines are needed as the once the predicted trajectory has crossed the finish line the goal is
        # to reach the end of the lap which is about to start

    def mpc_prob_init(self):
        """
        Create MPC quadratic optimization problem using cvxpy, solver: OSQP
        Will be solved every iteration for control.
        More MPC problem information here: https://osqp.org/docs/examples/mpc.html

        xref: reference trajectory (desired trajectory: [x, y, v, yaw])
        path_predict: predicted states in T steps
        x0: initial state
        dref: reference steer angle
        :return: optimal acceleration and steering strateg
        """
        # Initialize and create vectors for the optimization problem
        self.xk = cvxpy.Variable(
            (self.config.NXK, self.config.TK + 1)
        )  # Vehicle State Vector
        self.uk = cvxpy.Variable(
            (self.config.NU, self.config.TK)
        )  # Control Input vector
        objective = 0.0  # Objective value of the optimization problem, set to zero
        constraints = []  # Create constraints array

        # Initialize reference vectors
        self.x0k = cvxpy.Parameter((self.config.NXK,))
        self.x0k.value = np.zeros((self.config.NXK,))

        # Initialize reference trajectory parameter
        self.ref_traj_k = cvxpy.Parameter((self.config.NXK, self.config.TK + 1))
        self.ref_traj_k.value = np.zeros((self.config.NXK, self.config.TK + 1))

        # Initializes block diagonal form of R = [R, R, ..., R] (NU*T, NU*T)
        R_block = block_diag(tuple([self.config.Rk] * self.config.TK))

        # Initializes block diagonal form of Rd = [Rd, ..., Rd] (NU*(T-1), NU*(T-1))
        Rd_block = block_diag(tuple([self.config.Rdk] * (self.config.TK - 1)))

        # Initializes block diagonal form of Q = [Q, Q, ..., Qf] (NX*T, NX*T)
        Q_block = [self.config.Qk] * (self.config.TK)
        Q_block.append(self.config.Qfk)
        Q_block = block_diag(tuple(Q_block))

        # Formulate and create the finite-horizon optimal control problem (objective function)
        # The FTOCP has the horizon of T timesteps

        # Goal 1: Follow trajectory MPC

        # Objective 1: Influence of the control inputs: Inputs u multiplied by the penalty R
        objective += cvxpy.quad_form(cvxpy.vec(self.uk), R_block)

        # Objective 2: Deviation of the vehicle from the reference trajectory weighted by Q, including final Timestep T weighted by Qf
        objective += cvxpy.quad_form(cvxpy.vec(self.xk - self.ref_traj_k), Q_block)

        # Objective 3: Difference from one control input to the next control input weighted by Rd
        objective += cvxpy.quad_form(cvxpy.vec(cvxpy.diff(self.uk, axis=1)), Rd_block)

        # Goal 2: Go as fast as possible

        # Constraints 1: Calculate the future vehicle behavior/states based on the vehicle dynamics model matrices
        # Evaluate vehicle Dynamics for next T timesteps
        A_block = []
        B_block = []
        C_block = []
        # init path to zeros
        path_predict = np.zeros((self.config.NXK, self.config.TK + 1))
        input_predict = np.zeros((self.config.NU, self.config.TK + 1))
        for t in range(self.config.TK):
            A, B, C = self.model.get_model_matrix(path_predict[:, t], input_predict[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.extend(C)

        A_block = block_diag(tuple(A_block))
        B_block = block_diag(tuple(B_block))
        C_block = np.array(C_block)

        # [AA] Sparse matrix to CVX parameter for proper stuffing
        # Reference: https://github.com/cvxpy/cvxpy/issues/1159#issuecomment-718925710
        m, n = A_block.shape
        self.Annz_k = cvxpy.Parameter(A_block.nnz)
        data = np.ones(self.Annz_k.size)
        rows = A_block.row * n + A_block.col
        cols = np.arange(self.Annz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Annz_k.size))

        # Setting sparse matrix data
        self.Annz_k.value = A_block.data

        # Now we use this sparse version instead of the old A_ block matrix
        self.Ak_ = cvxpy.reshape(Indexer @ self.Annz_k, (m, n), order="C")

        # Same as A
        m, n = B_block.shape
        self.Bnnz_k = cvxpy.Parameter(B_block.nnz)
        data = np.ones(self.Bnnz_k.size)
        rows = B_block.row * n + B_block.col
        cols = np.arange(self.Bnnz_k.size)
        Indexer = csc_matrix((data, (rows, cols)), shape=(m * n, self.Bnnz_k.size))
        self.Bk_ = cvxpy.reshape(Indexer @ self.Bnnz_k, (m, n), order="C")
        self.Bnnz_k.value = B_block.data

        # No need for sparse matrices for C as most values are parameters
        self.Ck_ = cvxpy.Parameter(C_block.shape)
        self.Ck_.value = C_block

        # Add dynamics constraints to the optimization problem
        constraints += [
            cvxpy.vec(self.xk[:, 1:])
            == self.Ak_ @ cvxpy.vec(self.xk[:, :-1])
            + self.Bk_ @ cvxpy.vec(self.uk)
            + (self.Ck_)
        ]

        # Set x[k=0] as x0
        constraints += [self.xk[:, 0] == self.x0k]

        # Create the constraints (upper and lower bounds of states and inputs) for the optimization problem
        state_constraints, input_constraints, input_diff_constraints = self.model.get_model_constraints()

        for i in range(self.config.NXK):
            constraints += [state_constraints[0, i] <= self.xk[i, :], self.xk[i, :] <= state_constraints[1, i]]

        for i in range(self.config.NU):
            constraints += [input_constraints[0, i] <= self.uk[i, :], self.uk[i, :] <= input_constraints[1, i]]
            constraints += [input_diff_constraints[0, i] <= cvxpy.diff(self.uk[i, :]),
                            cvxpy.diff(self.uk[i, :]) <= input_diff_constraints[1, i]]

        # Create the optimization problem in CVXPY and setup the workspace
        # Optimization goal: minimize the objective function
        self.MPC_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)

    def mpc_prob_solve(self, ref_traj, path_predict, x0, input_predict):

        self.x0k.value = x0
        A_batch, B_batch, C_batch = self.model.batch_get_model_matrix(path_predict[:, :self.config.TK], input_predict[:, :self.config.TK])
        A_block = block_diag(A_batch)
        B_block = block_diag(B_batch)
        C_block = np.array(C_batch.flatten())

        self.Annz_k.value = A_block.data
        self.Bnnz_k.value = B_block.data
        self.Ck_.value = C_block

        self.ref_traj_k.value = ref_traj

        # time_start = time.time()
        # Solve the optimization problem in CVXPY
        # Solver selections: cvxpy.OSQP; cvxpy.GUROBI
        # self.MPC_prob.solve(solver=cvxpy.OSQP, verbose=False, warm_start=True)
        self.MPC_prob.solve(solver=cvxpy.OSQP, polish=True, adaptive_rho=True, rho=0.01, eps_abs=0.0005, eps_rel=0.0005, verbose=False,
                            warm_start=True)

        # time_end = time.time()
        # print(f'Solving time = {time_end - time_start}')
        # self.solve_time = time_end - time_start

        if self.MPC_prob.status == cvxpy.OPTIMAL or self.MPC_prob.status == cvxpy.OPTIMAL_INACCURATE:
            o_states = self.xk.value
            ou = self.uk.value

        else:
            print("Error: Cannot solve KS mpc... Status : ", self.MPC_prob.status)
            ou, o_states = np.zeros(self.config.NU) * np.NAN, np.zeros(self.config.NXK) * np.NAN

        self.time_step += 1
        self.tracking_accumulated_cost += self.compute_tracking_cost(x0, ou[0])

        return ou, o_states

    def linear_mpc_control(self, ref_path, x0, ref_control_input):
        """
        MPC contorl with updating operational point iteraitvely
        :param ref_path: reference trajectory in T steps
        :param x0: initial state vector
        :param a_old: acceleration of T steps of last time
        :param delta_old: delta of T steps of last time
        :return: acceleration and delta strategy based on current information
        """

        if np.isnan(ref_control_input).any():
            ref_control_input = np.zeros((self.config.NU, self.config.TK))
        else:
            ref_control_input = self.input_o

            # Call the Motion Prediction function: Predict the vehicle motion for x-steps

        if not np.any(np.isnan(self.states_output)):  # and False:
            state_prediction = self.model.predict_kin_from_dyn(self.states_output, x0)
            input_prediction = np.zeros((self.config.NU, self.config.TK))  # self.input_o
            # input_prediction = self.input_o  # self.input_o
        else:
            state_prediction, input_prediction = self.model.predict_motion(x0, ref_control_input, self.config.DTK)

        # Run the MPC optimization: Create and solve the optimization problem
        mpc_input_output, mpc_states_output = self.mpc_prob_solve(ref_path, state_prediction, x0, input_prediction)

        return mpc_input_output, mpc_states_output, state_prediction

    def MPC_Control(self, x0, path):
        # Calculate the next reference trajectory for the next T steps

        speed, orientation, position = self.model.get_general_states(x0)

        ref_path, self.target_ind = self.calc_ref_trajectory(position, orientation, speed, path)
        # Solve the Linear MPC Control problem
        (
            input_o,
            states_output,
            state_predict,
        ) = self.linear_mpc_control(ref_path, x0, self.input_o)

        if not np.any(np.isnan(states_output)):
            self.states_output = states_output
            self.input_o = input_o

        # Steering Output: First entry of the MPC steering angle output vector in degree
        u = self.input_o[:, 0]
        ox = states_output[0]
        oy = states_output[1]
        # o_input[:, 0]
        return u, ref_path[0], ref_path[1], state_predict[0], state_predict[1], ox, oy
