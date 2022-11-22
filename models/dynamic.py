import numpy as np
from torch.autograd.functional import jacobian
import torch


class DynamicBicycleModel:
    """
    states - [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    inputs - [drive force (proportional to acceleration), steering velocity]
    reference point - CoG
    """

    def __init__(self, config):
        self.config = config

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
            [[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf, -np.inf, -np.inf, self.config.MIN_STEER],
             [np.inf, np.inf, self.config.MAX_SPEED, np.inf, np.inf, np.inf, self.config.MAX_STEER]])

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
            np.zeros(len(speed_ref))
        ])
        return reference

    @staticmethod
    def get_general_states(state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]
        return speed, orientation, position

    def get_f(self, state, control_input):  # used for forward model propagation
        # state =
        # input = drive force (proportional to acceleration), steering velocity
        # input check
        control_input = self.clip_input(control_input)

        # control inputs
        Fxr, delta_v = control_input

        # states x[k]
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state

        # tire model from: AMZ Driverless: The Full Autonomous Racing System, A simplified Pacejka tire model
        alfa_f = steering_angle - np.arctan((yaw_rate * self.config.LF + vy) / vx)
        alfa_r = np.arctan((yaw_rate * self.config.LR - vy) / vx)

        Ffy = self.config.DF * np.sin(self.config.CF * np.arctan(self.config.BF * alfa_f))
        Fry = self.config.DR * np.sin(self.config.CR * np.arctan(self.config.BR * alfa_r))

        Fx = self.config.CM * Fxr - self.config.CR0 - self.config.CR2 * vx ** 2.0  # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
        Frx = Fx * (1.0 - self.config.TORQUE_SPLIT)
        Ffx = Fx * self.config.TORQUE_SPLIT

        # vehicle model from: AMZ Driverless: The Full Autonomous Racing System
        f = np.zeros(7)
        f[0] = vx * np.cos(yaw) - vy * np.sin(yaw)
        f[1] = vx * np.sin(yaw) + vy * np.cos(yaw)
        f[2] = 1.0 / self.config.MASS * (Frx - Ffy * np.sin(steering_angle) + Ffx * np.cos(steering_angle) + vy * yaw_rate * self.config.MASS)
        f[3] = yaw_rate
        f[4] = 1.0 / self.config.MASS * (Fry + Ffy * np.cos(steering_angle) + Ffx * np.sin(steering_angle) - vx * yaw_rate * self.config.MASS)
        f[5] = 1.0 / self.config.I_Z * (Ffy * self.config.LF * np.cos(steering_angle) - Fry * self.config.LR)
        f[6] = delta_v

        return f

    def get_f_torch_batch(self, state, control_input):  # batch torch implementation used for autograd of dynamics
        # state =
        # input = drive force (proportional to acceleration), steering velocity
        # input check

        # control inputs
        Fxr = control_input[:, 0]
        delta_v = control_input[:, 1]

        # states x[k]
        x = state[:, 0]
        y = state[:, 1]
        vx = state[:, 2]
        yaw = state[:, 3]
        vy = state[:, 4]
        yaw_rate = state[:, 5]
        steering_angle = state[:, 6]

        # tire model from: AMZ Driverless: The Full Autonomous Racing System, A simplified Pacejka tire model
        alfa_f = steering_angle - torch.arctan((yaw_rate * self.config.LF + vy) / vx)
        alfa_r = torch.arctan((yaw_rate * self.config.LR - vy) / vx)

        Ffy = self.config.DF * torch.sin(self.config.CF * torch.arctan(self.config.BF * alfa_f))
        Fry = self.config.DR * torch.sin(self.config.CR * torch.arctan(self.config.BR * alfa_r))

        Fx = self.config.CM * Fxr - self.config.CR0 - self.config.CR2 * vx ** 2.0  # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
        Frx = Fx * (1.0 - self.config.TORQUE_SPLIT)
        Ffx = Fx * self.config.TORQUE_SPLIT

        # vehicle model from: AMZ Driverless: The Full Autonomous Racing System
        f = torch.zeros(state.shape)
        f[:, 0] = vx * torch.cos(yaw) - vy * torch.sin(yaw)
        f[:, 1] = vx * torch.sin(yaw) + vy * torch.cos(yaw)
        f[:, 2] = 1.0 / self.config.MASS * (Frx - Ffy * torch.sin(steering_angle) + Ffx * torch.cos(steering_angle) + vy * yaw_rate * self.config.MASS)
        f[:, 3] = yaw_rate
        f[:, 4] = 1.0 / self.config.MASS * (Fry + Ffy * torch.cos(steering_angle) + Ffx * torch.sin(steering_angle) - vx * yaw_rate * self.config.MASS)
        f[:, 5] = 1.0 / self.config.I_Z * (Ffy * self.config.LF * torch.cos(steering_angle) - Fry * self.config.LR)
        f[:, 6] = delta_v

        return f

    def get_model_matrix(self, state, control_input):
        # x, y, vx, yaw, vy, yaw_rate, steering_angle = state
        # Fxr, delta_v = control_input
        A_, B_ = jacobian(self.get_f_torch_batch, (torch.from_numpy(state).reshape(1, 7), torch.from_numpy(control_input).reshape(1, 2)))
        A_ = A_.squeeze().numpy()
        A_ = np.nan_to_num(A_)
        B_ = B_.squeeze().numpy()

        # State (or system) matrix A, 7x7
        A = np.diag(np.ones(7)) + A_ * self.config.DTK

        # Input Matrix B; 7x2
        B = B_ * self.config.DTK

        C = (np.nan_to_num(self.get_f(state, control_input).reshape(7, 1)) - A_ @ state.reshape(7, 1) - B_ @ control_input.reshape(2, 1)) * self.config.DTK
        C = C.reshape(7)

        return A, B, C

    def batch_get_model_matrix(self, state_vec, control_vec):
        C_block = []

        state_vec_ = np.moveaxis(state_vec, 0, 1)
        control_vec_ = np.moveaxis(control_vec, 0, 1)

        def batch_jacobian(f, x):
            f_sum = lambda x1, x2: torch.sum(f(x1, x2), axis=0)
            return jacobian(f_sum, x)

        A_, B_ = batch_jacobian(self.get_f_torch_batch, (torch.from_numpy(state_vec_), torch.from_numpy(control_vec_)))
        A_ = np.nan_to_num(A_.numpy())
        A_ = np.moveaxis(A_, 1, 0)
        B_ = B_.numpy()
        B_ = np.moveaxis(B_, 1, 0)

        # State (or system) matrix A, 7x7
        A = np.repeat(np.diag(np.ones(7))[:, :, np.newaxis], state_vec_.shape[0], axis=2)
        A = np.moveaxis(A, -1, 0) + A_ * self.config.DTK
        B = B_ * self.config.DTK

        for t in range(state_vec.shape[1]):
            C = (np.nan_to_num(self.get_f(state_vec_[t], control_vec_[t]).reshape(7, 1)) - A_[t] @ state_vec_[t].reshape(7, 1) - B_[t] @ control_vec_[
                t].reshape(2, 1)) * self.config.DTK
            C = C.reshape(7)
            C_block.append(C)

        return list(A), list(B), np.array(C_block)

    def predict_motion(self, x0, control_input, dt):
        predicted_states = np.zeros((self.config.NXK, self.config.TK + 1))
        predicted_states[:, 0] = x0
        state = x0
        for i in range(1, self.config.TK + 1):
            state = state + self.get_f(state, control_input[:, i - 1]) * dt
            state = self.clip_output(state)
            predicted_states[:, i] = state
        input_prediction = np.zeros((2, self.config.TK + 1))
        return predicted_states, input_prediction

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
