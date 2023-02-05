import numpy as np
import torch
import gpytorch
from torch.autograd.functional import jacobian
import os
import time
import cvxpy
from models.GP_model_ensembling_extended import GPEnsembleModel_v2

np.set_printoptions(precision=4, suppress=True)


class GPEnsembleModels2GPsExtended:
    """
    states - [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    inputs - [drive force (proportional to acceleration), steering velocity]
    reference point - center of rear axle ? Need to check BayesRace paper
    """

    def __init__(self, config):
        self.config = config

        self.gp_model1 = GPEnsembleModel_v2(config)
        self.gp_model2 = GPEnsembleModel_v2(config)

        self.w1 = np.array(0.5)
        self.w2 = np.array(0.5)
        self.w_opt_prob_intiialized = False

    def clip_input(self, u):
        # u matrix Nx2
        u = np.clip(u, [self.config.MAX_DECEL * self.config.MASS, -self.config.MAX_STEER_V],
                    [self.config.MAX_ACCEL * self.config.MASS, self.config.MAX_STEER_V])
        return u

    def clip_output(self, state):
        # state matrix Nx7
        state[2] = np.clip(state[2], self.config.MIN_SPEED, self.config.MAX_SPEED)
        state[6] = np.clip(state[6], self.config.MIN_STEER, self.config.MAX_STEER)
        return state

    def get_model_constraints(self):
        state_constraints = np.array(
            [[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf, -np.inf, -np.inf, self.config.MIN_STEER, -np.inf, -np.inf],
             [np.inf, np.inf, self.config.MAX_SPEED, np.inf, np.inf, np.inf, self.config.MAX_STEER, np.inf, np.inf]])

        input_constraints = np.array([[self.config.MAX_DECEL * self.config.MASS, -self.config.MAX_STEER_V],
                                      [self.config.MAX_ACCEL * self.config.MASS, self.config.MAX_STEER_V]])

        input_diff_constraints = np.array([[-np.inf, -np.inf],
                                           [np.inf, np.inf]])
        return state_constraints, input_constraints, input_diff_constraints

    @staticmethod
    def sort_reference_trajectory(position_ref, yaw_ref, speed_ref):
        reference = np.array([
            # Sort reference trajectory so the order of reference match the order of the states
            position_ref[:, 0],
            position_ref[:, 1],
            speed_ref,
            yaw_ref,
            # Fill zeros to the rest so number of references mathc number of states (x[k] - ref[k])
            np.zeros(len(speed_ref)),
            np.zeros(len(speed_ref)),
            np.zeros(len(speed_ref)),
            np.zeros(len(speed_ref)),
            np.zeros(len(speed_ref))
        ])
        return reference

    @staticmethod
    def get_general_states(state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]
        return speed, orientation, position

    def get_f(self, state, control_input):
        f1 = self.gp_model1.get_f(state, control_input)
        f2 = self.gp_model2.get_f(state, control_input)
        f = self.w1 * f1 + self.w2 * f2
        return f

    def batch_get_model_matrix(self, state_vec, control_vec):
        A1, B1, C1 = self.gp_model1.batch_get_model_matrix(state_vec, control_vec)
        A2, B2, C2 = self.gp_model2.batch_get_model_matrix(state_vec, control_vec)
        A = self.w1 * A1 + self.w2 * A2
        B = self.w1 * B1 + self.w2 * B2
        C = self.w1 * C1 + self.w2 * C2
        return A, B, C
    def get_model_matrix(self, state, control_input):
        A1, B1, C1 = self.gp_model1.get_model_matrix(state, control_input)
        A2, B2, C2 = self.gp_model2.get_model_matrix(state, control_input)
        A = self.w1 * A1 + self.w2 * A2
        B = self.w1 * B1 + self.w2 * B2
        C = self.w1 * C1 + self.w2 * C2
        return A, B, C

    def predict_kin_from_dyn(self, states, x0):
        # states - [x, y, vx, yaw angle, vy, yaw rate, steering angle]
        states[:, 0][0] = x0[0]  # x
        states[:, 0][1] = x0[1]  # y
        states[:, 0][3] = x0[3]  # yaw

        for i in range(1, states.shape[1]):
            states[:, i][0] = states[:, i - 1][0] + (
                    states[:, i - 1][2] * np.cos(states[:, i - 1][3]) - states[:, i - 1][4] * np.sin(
                states[:, i - 1][3])) * self.config.DTK
            states[:, i][1] = states[:, i - 1][1] + (
                    states[:, i - 1][2] * np.sin(states[:, i - 1][3]) + states[:, i - 1][4] * np.cos(
                states[:, i - 1][3])) * self.config.DTK
            states[:, i][3] = states[:, i - 1][3] + (states[:, i - 1][5]) * self.config.DTK
        return states

    def predict_motion(self, x0, control_input, dt):
        """
        :param x0: np.array [x, y, vx, yaw angle, vy, yaw rate, steering angle]
        :param control_input: [drive force (proportional to acceleration), steering velocity]
        :param dt:
        :return:
        """

        predicted_states = np.zeros((x0.size, control_input.shape[1] + 1))
        predicted_states[:, 0] = x0
        state = x0

        for i in range(1, control_input.shape[1] + 1):
            state = state + self.get_f(state, control_input[:, i - 1]) * dt
            state = self.clip_output(state)
            predicted_states[:, i] = state

        input_prediction = np.zeros((2, control_input.shape[1] + 1))
        return predicted_states, input_prediction

    def scale_and_predict_model_step(self, state, control_input):
        scaled_mean1, scaled_lower1, scaled_upper1 = self.gp_model1.scale_and_predict_model_step(state, control_input)
        scaled_mean2, scaled_lower2, scaled_upper2 = self.gp_model2.scale_and_predict_model_step(state, control_input)
        scaled_mean = self.w1 * scaled_mean1 + self.w2 * scaled_mean2
        scaled_lower = self.w1 * scaled_lower1 + self.w2 * scaled_lower2
        scaled_upper = self.w1 * scaled_upper1 + self.w2 * scaled_upper2
        return scaled_mean, scaled_lower, scaled_upper, scaled_mean1, scaled_mean2

    def init_w_opt_prob(self, Y_reals, means, prev_w, eps):
        self.w_opt_prob_intiialized = True
        F = means

        # Create problem
        self.prev_w_var = cvxpy.Parameter((means.shape[1], 1))  # (m, 1)
        self.w_var = cvxpy.Variable((means.shape[1], 1))  # (m, 1)
        self.F_par = cvxpy.Parameter(F.shape)  # (n*3, m)
        self.Y_par = cvxpy.Parameter((Y_reals.size, 1))  # (n*3, 1)

        self.F_par.value = F
        self.Y_par.value = Y_reals
        self.prev_w_var.value = prev_w

        objective = cvxpy.sum_squares(self.Y_par - self.F_par @ self.w_var)
        # objective += eps * cvxpy.sum_squares(self.w_var - self.prev_w_var)

        constraints = [self.w_var >= 0.0, self.w_var <= 1.0, cvxpy.sum(self.w_var) == 1.0]
        self.w_prob = cvxpy.Problem(cvxpy.Minimize(objective), constraints)
        self.w_prob.solve(solver=cvxpy.OSQP, polish=True, adaptive_rho=True, rho=0.0001, eps_abs=0.001, eps_rel=0.001, verbose=False, warm_start=False)
        # print('W1: %f    W2: %f' % (w.value[0], w.value[1]))
        self.w1 = self.w_var.value[0]
        self.w2 = self.w_var.value[1]
        # self.w1 = 1.0
        # self.w2 = 0.0

    def compute_w(self, Y_reals, means, prev_w, eps, u):
        if self.w_opt_prob_intiialized:
            F = means

            self.prev_w_var.value = prev_w  # (m, 1)
            self.F_par.value = F  # (3*n, m)
            self.Y_par.value = Y_reals  # (3*n, 1)
            self.w_prob.solve(solver=cvxpy.OSQP, polish=True, adaptive_rho=True, rho=0.0001, eps_abs=0.001, eps_rel=0.001, verbose=False, warm_start=False)
            self.w1 = self.w_var.value[0]
            self.w2 = self.w_var.value[1]
            # self.w1 = 1.0
            # self.w2 = 0.0
            # print('W1: %f    W2: %f' % (w.value[0], w.value[1]))
        else:
            return self.init_w_opt_prob(Y_reals, means, prev_w, eps)

