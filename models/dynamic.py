import numpy as np
from torch.autograd.functional import jacobian
import torch
from scipy import integrate
import time
from dataclasses import dataclass, field
from numba import njit, prange


@njit(cache=True)
def get_model_matrix_cf_batch_jit(state, control_input, TK, MASS, I_Z, LF, LR, TORQUE_SPLIT, BR, DR, BF, CF, DF, CM, CR, CR0, CR2, A, B):
    x, y, vx, yaw, vy, yaw_rate, steering_angle = state
    Fxr, delta_v = control_input

    # A = np.zeros((TK, 7, 7))
    A[:, 0, 2] = np.cos(yaw)
    A[:, 0, 3] = -vx * np.sin(yaw) - vy * np.cos(yaw)
    A[:, 0, 4] = -np.sin(yaw)

    A[:, 1, 2] = np.sin(yaw)
    A[:, 1, 3] = vx * np.cos(yaw) - vy * np.sin(yaw)
    A[:, 1, 4] = np.cos(yaw)

    A[:, 2, 2] = 1.0 * (-2.0 * vx ** 1.0 * CR2 * (1.0 - TORQUE_SPLIT) - 2.0 * vx ** 1.0 * CR2 * np.cos(
        steering_angle) * TORQUE_SPLIT - DF * CF * BF * np.sin(
        steering_angle) * (vy + LF * yaw_rate) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx ** 2 * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / MASS
    A[:, 2, 4] = 1.0 * (MASS * yaw_rate + DF * CF * BF * np.sin(steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / MASS
    A[:, 2, 5] = 1.0 * (vy * MASS + LF * DF * CF * BF * np.sin(steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / MASS
    A[:, 2, 6] = 1.0 * (-DF * np.cos(steering_angle) * np.sin(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) - np.sin(
        steering_angle) * (-CR0 + CM * Fxr - vx ** 2.0 * CR2) * TORQUE_SPLIT - DF * CF * BF * np.sin(
        steering_angle) * np.cos(CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                1 + BF ** 2 * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2)) / MASS

    A[:, 3, 5] = np.ones(TK)

    A[:, 4, 2] = 1.0 * (-MASS * yaw_rate - 2.0 * vx ** 1.0 * CR2 * np.sin(steering_angle) * TORQUE_SPLIT - DR * CR * BR * (
            -vy + LR * yaw_rate) * np.cos(CR * np.arctan(BR * np.arctan((-vy + LR * yaw_rate) / vx))) / (
                                vx ** 2 * (1 + (-vy + LR * yaw_rate) ** 2 / vx ** 2) * (1 + BR ** 2 * np.arctan((
                                                                                                                        -vy + LR * yaw_rate) / vx) ** 2)) + DF * CF * BF * np.cos(
        steering_angle) * (vy + LF * yaw_rate) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx ** 2 * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / MASS
    A[:, 4, 4] = 1.0 * (-DR * CR * BR * np.cos(
        CR * np.arctan(BR * np.arctan((-vy + LR * yaw_rate) / vx))) / (vx * (1 + (-vy + LR * yaw_rate) ** 2 / vx ** 2) * (1 + BR ** 2 * np.arctan(
        (-vy + LR * yaw_rate) / vx) ** 2)) - DF * CF * BF * np.cos(steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / MASS
    A[:, 4, 5] = 1.0 * (-vx * MASS + LR * DR * CR * BR * np.cos(CR * np.arctan(BR * np.arctan((-vy + LR * yaw_rate) / vx))) / (
            vx * (1 + (-vy + LR * yaw_rate) ** 2 / vx ** 2) * (1 + BR ** 2 * np.arctan((
                                                                                               -vy + LR * yaw_rate) / vx) ** 2)) - LF * DF * CF * BF * np.cos(
        steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / MASS
    A[:, 4, 6] = 1.0 * (-DF * np.sin(steering_angle) * np.sin(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) + np.cos(
        steering_angle) * (
                                -CR0 + CM * Fxr - vx ** 2.0 * CR2) * TORQUE_SPLIT + DF * CF * BF * np.cos(
        steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2)) / MASS

    A[:, 5, 2] = 1.0 * (LR * DR * CR * BR * (-vy + LR * yaw_rate) * np.cos(
        CR * np.arctan(BR * np.arctan((-vy + LR * yaw_rate) / vx))) / (
                                vx ** 2 * (1 + (-vy + LR * yaw_rate) ** 2 / vx ** 2) * (
                                1 + BR ** 2 * np.arctan((
                                                                -vy + LR * yaw_rate) / vx) ** 2)) + LF * DF * CF * BF * np.cos(
        steering_angle) * (vy + LF * yaw_rate) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx ** 2 * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / I_Z
    A[:, 5, 4] = 1.0 * (LR * DR * CR * BR * np.cos(
        CR * np.arctan(BR * np.arctan((-vy + LR * yaw_rate) / vx))) / (
                                vx * (1 + (-vy + LR * yaw_rate) ** 2 / vx ** 2) * (1 + BR ** 2 * np.arctan((
                                                                                                                   -vy + LR * yaw_rate) / vx) ** 2)) - LF * DF * CF * BF * np.cos(
        steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / I_Z
    A[:, 5, 5] = 1.0 * (-LR ** 2 * DR * CR * BR * np.cos(CR * np.arctan(BR * np.arctan((-vy + LR * yaw_rate) / vx))) / (
            vx * (1 + (-vy + LR * yaw_rate) ** 2 / vx ** 2) * (1 + BR ** 2 * np.arctan((
                                                                                               -vy + LR * yaw_rate) / vx) ** 2)) - LF ** 2 * DF * CF * BF * np.cos(
        steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                vx * (1 + (vy + LF * yaw_rate) ** 2 / vx ** 2) * (1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2))) / I_Z
    A[:, 5, 6] = 1.0 * (-LF * DF * np.sin(steering_angle) * np.sin(CF * np.arctan(BF * (
            steering_angle - np.arctan(
        (vy + LF * yaw_rate) / vx)))) + LF * DF * CF * BF * np.cos(
        steering_angle) * np.cos(
        CF * np.arctan(BF * (steering_angle - np.arctan((vy + LF * yaw_rate) / vx)))) / (
                                1 + BF ** 2 * (
                                steering_angle - np.arctan((vy + LF * yaw_rate) / vx)) ** 2)) / I_Z

    B[:, 2, 0] = 1.0 * (CM * (1.0 - TORQUE_SPLIT) + CM * np.cos(steering_angle) * TORQUE_SPLIT) / MASS
    B[:, 4, 0] = 1.0 * CM * np.sin(steering_angle) * TORQUE_SPLIT / MASS

    # f (20, 7),  A (20, 7, 7),  B (20, 7, 2), state (7, 20),  control_input (2, 20)
    f = get_f_batch_jit(state, control_input, MASS, I_Z, LF, LR, TORQUE_SPLIT, BR, DR, BF, CF, DF, CM, CR, CR0, CR2)

    C = np.zeros((TK, 7))

    for i in prange(TK):
        C[i] = f[i] - A[i] @ state.T[i] - B[i] @ control_input.T[i]
    # C = f - (A @ state.T[:, :, None]).squeeze(-1) - (B @ control_input.T[:, :, None]).squeeze(-1)

    return A, B, C


@njit(cache=True)
def get_f_batch_jit(state, control_input, MASS, I_Z, LF, LR, TORQUE_SPLIT, BR, DR, BF, CF, DF, CM, CR, CR0, CR2):
    # state =
    # input = drive force (proportional to acceleration), steering velocity

    x, y, vx, yaw, vy, yaw_rate, steering_angle = state
    Fxr, delta_v = control_input

    # tire model from: AMZ Driverless: The Full Autonomous Racing System, A simplified Pacejka tire model
    alfa_f = steering_angle - np.arctan((yaw_rate * LF + vy) / vx)
    alfa_r = np.arctan((yaw_rate * LR - vy) / vx)

    Ffy = DF * np.sin(CF * np.arctan(BF * alfa_f))
    Fry = DR * np.sin(CR * np.arctan(BR * alfa_r))

    Fx = CM * Fxr - CR0 - CR2 * vx ** 2.0  # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
    Frx = Fx * (1.0 - TORQUE_SPLIT)
    Ffx = Fx * TORQUE_SPLIT

    # vehicle model from: AMZ Driverless: The Full Autonomous Racing System
    f = np.zeros((state.shape[1], state.shape[0]))
    f[:, 0] = vx * np.cos(yaw) - vy * np.sin(yaw)
    f[:, 1] = vx * np.sin(yaw) + vy * np.cos(yaw)
    f[:, 2] = 1.0 / MASS * (
            Frx - Ffy * np.sin(steering_angle) + Ffx * np.cos(steering_angle) + vy * yaw_rate * MASS)
    f[:, 3] = yaw_rate
    f[:, 4] = 1.0 / MASS * (
            Fry + Ffy * np.cos(steering_angle) + Ffx * np.sin(steering_angle) - vx * yaw_rate * MASS)
    f[:, 5] = 1.0 / I_Z * (Ffy * LF * np.cos(steering_angle) - Fry * LR)
    f[:, 6] = delta_v

    return f


class DynamicBicycleModel:
    """
    states - [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    inputs - [drive force (proportional to acceleration), steering velocity]
    reference point - CoG
    """

    def __init__(self, config):
        self.config = config

        # init of matrices A, B
        self.A = np.zeros((self.config.TK, 7, 7))
        self.B = np.zeros((self.config.TK, 7, 2))
        self.B[:, 6, 1] = np.ones(self.config.TK)

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

    def reference_global_to_local(self, reference, x0):
        # TODO change reference frame for kinematics from Global to Local ->>  x0 -> [0.0, 0.0, -, 0.0, -, -, -]
        pass

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
        f[:, 2] = 1.0 / self.config.MASS * (
                Frx - Ffy * torch.sin(steering_angle) + Ffx * torch.cos(steering_angle) + vy * yaw_rate * self.config.MASS)
        f[:, 3] = yaw_rate
        f[:, 4] = 1.0 / self.config.MASS * (
                Fry + Ffy * torch.cos(steering_angle) + Ffx * torch.sin(steering_angle) - vx * yaw_rate * self.config.MASS)
        f[:, 5] = 1.0 / self.config.I_Z * (Ffy * self.config.LF * torch.cos(steering_angle) - Fry * self.config.LR)
        f[:, 6] = delta_v

        return f

    def get_model_matrix_cf_batch(self, state, control_input):
        # state (7, Tk)
        # control_input (2, TK)

        x, y, vx, yaw, vy, yaw_rate, steering_angle = state
        Fxr, delta_v = control_input

        A = np.array([[np.zeros(self.config.TK), np.zeros(self.config.TK), np.cos(yaw), -vx * np.sin(yaw) - vy * np.cos(yaw), -np.sin(yaw),
                       np.zeros(self.config.TK), np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK), np.sin(yaw), vx * np.cos(yaw) - vy * np.sin(yaw), np.cos(yaw),
                       np.zeros(self.config.TK), np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK), 1.0 * (
                              -2.0 * vx ** 1.0 * self.config.CR2 * (1.0 - self.config.TORQUE_SPLIT) - 2.0 * vx ** 1.0 * self.config.CR2 * np.cos(
                          steering_angle) * self.config.TORQUE_SPLIT - self.config.DF * self.config.CF * self.config.BF * np.sin(
                          steering_angle) * (vy + self.config.LF * yaw_rate) * np.cos(
                          self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      vx ** 2 * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.MASS,
                       np.zeros(self.config.TK),
                       1.0 * (self.config.MASS * yaw_rate + self.config.DF * self.config.CF * self.config.BF * np.sin(steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      vx * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.MASS, 1.0 * (
                               vy * self.config.MASS + self.config.LF * self.config.DF * self.config.CF * self.config.BF * np.sin(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                       vx * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                       steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.MASS,
                       1.0 * (-self.config.DF * np.cos(steering_angle) * np.sin(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) - np.sin(
                           steering_angle) * (
                                      -self.config.CR0 + self.config.CM * Fxr - vx ** 2.0 * self.config.CR2) * self.config.TORQUE_SPLIT - self.config.DF * self.config.CF * self.config.BF * np.sin(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2)) / self.config.MASS],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK), np.zeros(self.config.TK), np.zeros(self.config.TK),
                       np.zeros(self.config.TK), np.ones(self.config.TK), np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK),
                       1.0 * (-self.config.MASS * yaw_rate - 2.0 * vx ** 1.0 * self.config.CR2 * np.sin(
                           steering_angle) * self.config.TORQUE_SPLIT - self.config.DR * self.config.CR * self.config.BR * (
                                      -vy + self.config.LR * yaw_rate) * np.cos(
                           self.config.CR * np.arctan(self.config.BR * np.arctan((-vy + self.config.LR * yaw_rate) / vx))) / (
                                      vx ** 2 * (1 + (-vy + self.config.LR * yaw_rate) ** 2 / vx ** 2) * (
                                      1 + self.config.BR ** 2 * np.arctan((
                                                                                  -vy + self.config.LR * yaw_rate) / vx) ** 2)) + self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * (vy + self.config.LF * yaw_rate) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      vx ** 2 * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.MASS,
                       np.zeros(self.config.TK),
                       1.0 * (-self.config.DR * self.config.CR * self.config.BR * np.cos(
                           self.config.CR * np.arctan(self.config.BR * np.arctan((-vy + self.config.LR * yaw_rate) / vx))) / (
                                      vx * (1 + (-vy + self.config.LR * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BR ** 2 * np.arctan(
                                  (-vy + self.config.LR * yaw_rate) / vx) ** 2)) - self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      vx * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.MASS, 1.0 * (
                               -vx * self.config.MASS + self.config.LR * self.config.DR * self.config.CR * self.config.BR * np.cos(
                           self.config.CR * np.arctan(self.config.BR * np.arctan((-vy + self.config.LR * yaw_rate) / vx))) / (
                                       vx * (1 + (-vy + self.config.LR * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BR ** 2 * np.arctan((
                                                                                                                                                  -vy + self.config.LR * yaw_rate) / vx) ** 2)) - self.config.LF * self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                       vx * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                       steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.MASS,
                       1.0 * (-self.config.DF * np.sin(steering_angle) * np.sin(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) + np.cos(
                           steering_angle) * (
                                      -self.config.CR0 + self.config.CM * Fxr - vx ** 2.0 * self.config.CR2) * self.config.TORQUE_SPLIT + self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2)) / self.config.MASS],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK),
                       1.0 * (self.config.LR * self.config.DR * self.config.CR * self.config.BR * (-vy + self.config.LR * yaw_rate) * np.cos(
                           self.config.CR * np.arctan(self.config.BR * np.arctan((-vy + self.config.LR * yaw_rate) / vx))) / (
                                      vx ** 2 * (1 + (-vy + self.config.LR * yaw_rate) ** 2 / vx ** 2) * (
                                      1 + self.config.BR ** 2 * np.arctan((
                                                                                  -vy + self.config.LR * yaw_rate) / vx) ** 2)) + self.config.LF * self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * (vy + self.config.LF * yaw_rate) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      vx ** 2 * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.I_Z,
                       np.zeros(self.config.TK),
                       1.0 * (self.config.LR * self.config.DR * self.config.CR * self.config.BR * np.cos(
                           self.config.CR * np.arctan(self.config.BR * np.arctan((-vy + self.config.LR * yaw_rate) / vx))) / (
                                      vx * (1 + (-vy + self.config.LR * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BR ** 2 * np.arctan((
                                                                                                                                                 -vy + self.config.LR * yaw_rate) / vx) ** 2)) - self.config.LF * self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      vx * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.I_Z, 1.0 * (
                               -self.config.LR ** 2 * self.config.DR * self.config.CR * self.config.BR * np.cos(
                           self.config.CR * np.arctan(self.config.BR * np.arctan((-vy + self.config.LR * yaw_rate) / vx))) / (
                                       vx * (1 + (-vy + self.config.LR * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BR ** 2 * np.arctan((
                                                                                                                                                  -vy + self.config.LR * yaw_rate) / vx) ** 2)) - self.config.LF ** 2 * self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                       vx * (1 + (vy + self.config.LF * yaw_rate) ** 2 / vx ** 2) * (1 + self.config.BF ** 2 * (
                                       steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2))) / self.config.I_Z,
                       1.0 * (-self.config.LF * self.config.DF * np.sin(steering_angle) * np.sin(self.config.CF * np.arctan(self.config.BF * (
                               steering_angle - np.arctan(
                           (vy + self.config.LF * yaw_rate) / vx)))) + self.config.LF * self.config.DF * self.config.CF * self.config.BF * np.cos(
                           steering_angle) * np.cos(
                           self.config.CF * np.arctan(self.config.BF * (steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)))) / (
                                      1 + self.config.BF ** 2 * (
                                      steering_angle - np.arctan((vy + self.config.LF * yaw_rate) / vx)) ** 2)) / self.config.I_Z],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK), np.zeros(self.config.TK), np.zeros(self.config.TK),
                       np.zeros(self.config.TK), np.zeros(self.config.TK), np.zeros(self.config.TK)]])

        B = np.array([[np.zeros(self.config.TK), np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK)],
                      [1.0 * (self.config.CM * (1.0 - self.config.TORQUE_SPLIT) + self.config.CM * np.cos(
                          steering_angle) * self.config.TORQUE_SPLIT) / self.config.MASS, np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK)],
                      [1.0 * self.config.CM * np.sin(steering_angle) * self.config.TORQUE_SPLIT / self.config.MASS, np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.zeros(self.config.TK)],
                      [np.zeros(self.config.TK), np.ones(self.config.TK)]])

        A = np.moveaxis(A, 2, 0)
        B = np.moveaxis(B, 2, 0)

        return A, B

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

        C = (np.nan_to_num(self.get_f(state, control_input).reshape(7, 1)) - A_ @ state.reshape(7, 1) - B_ @ control_input.reshape(2,
                                                                                                                                   1)) * self.config.DTK
        C = C.reshape(7)

        return A, B, C

    def batch_get_model_matrix(self, state_vec, control_vec, mode=1):  # mode (0-cl, 1-cl_jit, 2-autograd)
        C = []

        state_vec_ = np.moveaxis(state_vec, 0, 1)
        control_vec_ = np.moveaxis(control_vec, 0, 1)

        if mode == 0:
            A_, B_ = self.get_model_matrix_cf_batch(state_vec, control_vec)
        elif mode == 1:
            A_, B_, C_ = get_model_matrix_cf_batch_jit(state_vec, control_vec, TK=self.config.TK, MASS=self.config.MASS, I_Z=self.config.I_Z,
                                                   LF=self.config.LF, LR=self.config.LR, TORQUE_SPLIT=self.config.TORQUE_SPLIT, BR=self.config.BR,
                                                   DR=self.config.DR, BF=self.config.BF, CF=self.config.CF, DF=self.config.DF, CM=self.config.CM,
                                                   CR=self.config.CR, CR0=self.config.CR0, CR2=self.config.CR2, A=self.A, B=self.B)
            C = C_ * self.config.DTK
        elif mode == 2:
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

        if mode == 0 or mode == 2:
            for t in range(state_vec.shape[1]):
                C_ = (np.nan_to_num(self.get_f(state_vec_[t], control_vec_[t]).reshape(7, 1)) - A_[t] @ state_vec_[t].reshape(7, 1) - B_[t] @ control_vec_[
                    t].reshape(2, 1)) * self.config.DTK
                C_ = C_.reshape(7)
                C.append(C_)
            C = np.array(C)

        return list(A), list(B), C

    def predict_motion(self, x0, control_input, dt):
        # TODO change reference frame for kinematics from Global to Local ->>  x0 -> [0.0, 0.0, -, 0.0, -, -, -]
        def get_f_wraper(x, t, u):
            return self.get_f(x, u)

        predicted_states = np.zeros((self.config.NXK, self.config.TK + 1))
        predicted_states[:, 0] = x0
        state = x0
        for i in range(1, self.config.TK + 1):
            # x_left = integrate.odeint(get_f_wraper, state,
            #                           np.array([0.0, dt]),
            #                           args=(control_input[:, i - 1],),
            #                           mxstep=10000, full_output=1)
            # state = x_left[0][1]

            state = state + self.get_f(state, control_input[:, i - 1]) * dt

            state = self.clip_output(state)
            predicted_states[:, i] = state
        input_prediction = control_input
        return predicted_states, input_prediction

    def predict_kin_from_dyn(self, states, x0):

        # TODO change reference frame for kinematics from Global to Local ->>  x0 -> [0.0, 0.0, -, 0.0, -, -, -]
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


@dataclass
class MPCConfigDYNTest:
    NXK: int = 7  # length of kinematic state vector: z = [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    NU: int = 2  # length of input vector: u = = [acceleration, steering speed]
    TK: int = 100  # finite time horizon length kinematic
    DTK: float = 0.1  # time step [s] kinematic
    dlk: float = 3.0  # dist step [m] kinematic
    LENGTH: float = 4.298  # Length of the vehicle [m]
    WIDTH: float = 1.674  # Width of the vehicle [m]
    LR: float = 1.50876
    LF: float = 0.88392
    WB: float = 0.88392 + 1.50876  # Wheelbase [m]
    MIN_STEER: float = -0.4189  # maximum steering angle [rad]
    MAX_STEER: float = 0.4189  # maximum steering angle [rad]
    MAX_STEER_V: float = 3.2  # maximum steering speed [rad/s]
    MAX_SPEED: float = 45.0  # maximum speed [m/s]
    MIN_SPEED: float = 0.0  # minimum backward speed [m/s]
    MAX_ACCEL: float = 11.5  # maximum acceleration [m/ss]
    MAX_DECEL: float = -45.0  # maximum acceleration [m/ss]

    # model parameters
    MASS: float = 1225.887  # Vehicle mass
    I_Z: float = 1560.3729  # Vehicle inertia
    TORQUE_SPLIT: float = 0.0  # Torque distribution

    BR: float = 15.9504  # Pacejka tire model parameter B - rear tire
    CR: float = 1.3754  # Pacejka tire model parameter C - rear tire
    DR: float = 4500.9280  # Pacejka tire model parameter D - rear tire

    BF: float = 9.4246  # Pacejka tire model parameter B - front tire
    CF: float = 5.9139  # Pacejka tire model parameter C - front tire
    DF: float = 4500.8218  # Pacejka tire model parameter D - front tire

    # https://arxiv.org/pdf/1905.05150.pdf - equation (7)
    CM: float = 0.9459
    CR0: float = 2.3451
    CR2: float = -0.0095


if __name__ == "__main__":
    config = MPCConfigDYNTest()

    # Test 1:
    print("Test 1, prediction horizon 20")

    config.TK = 20
    model = DynamicBicycleModel(config)
    state_vec = np.random.random((7, 20))
    control_vec = np.random.random((2, 20))

    start_3 = time.time()
    model.batch_get_model_matrix(state_vec, control_vec, mode=1)
    model.batch_get_model_matrix(state_vec, control_vec, mode=1)
    model.batch_get_model_matrix(state_vec, control_vec, mode=1)
    model.batch_get_model_matrix(state_vec, control_vec, mode=1)

    print("Are solutions from cl and autograd the same:")
    start_1 = time.time()
    A_1, B_1, C_1 = model.batch_get_model_matrix(state_vec, control_vec, mode=0)
    end_1 = time.time()

    start_2 = time.time()
    A_2, B_2, C_2 = model.batch_get_model_matrix(state_vec, control_vec, mode=2)
    end_2 = time.time()

    start_3 = time.time()
    A_3, B_3, C_3 = model.batch_get_model_matrix(state_vec, control_vec, mode=1)
    end_3 = time.time()

    print(f"Matrix A: {np.all(np.isclose(np.array(A_1), np.array(A_2)))}, {np.all(np.isclose(np.array(A_1), np.array(A_3)))}")
    print(f"Matrix B: {np.all(np.isclose(np.array(B_1), np.array(B_2)))}, {np.all(np.isclose(np.array(B_1), np.array(B_3)))}")
    print(f"Matrix C: {np.all(np.isclose(np.array(C_1), np.array(C_2)))}, {np.all(np.isclose(np.array(C_1), np.array(C_3)))}")
    print("Timing:")
    print(f"Close form solution time: {end_1 - start_1}")
    print(f"Close form solution jit time: {end_3 - start_3}")
    print(f"Autograd solution time: {end_2 - start_2}")
    print("----------------------------")

    # Test 2:
    print("Test 2, prediction horizon 100")
    config2 = MPCConfigDYNTest()
    config.TK = 100
    model2 = DynamicBicycleModel(config2)
    state_vec = np.random.random((7, 100))
    control_vec = np.random.random((2, 100))

    print("Are solutions from cl and autograd the same:")
    start_1 = time.time()
    A_1, B_1, C_1 = model2.batch_get_model_matrix(state_vec, control_vec, mode=0)
    end_1 = time.time()

    start_2 = time.time()
    A_2, B_2, C_2 = model2.batch_get_model_matrix(state_vec, control_vec, mode=2)
    end_2 = time.time()

    start_3 = time.time()
    A_3, B_3, C_3 = model2.batch_get_model_matrix(state_vec, control_vec, mode=1)
    end_3 = time.time()

    print(f"Matrix A: {np.all(np.isclose(np.array(A_1), np.array(A_2)))}, {np.all(np.isclose(np.array(A_1), np.array(A_3)))}")
    print(f"Matrix B: {np.all(np.isclose(np.array(B_1), np.array(B_2)))}, {np.all(np.isclose(np.array(B_1), np.array(B_3)))}")
    print(f"Matrix C: {np.all(np.isclose(np.array(C_1), np.array(C_2)))}, {np.all(np.isclose(np.array(C_1), np.array(C_3)))}")
    print("Timing:")
    print(f"Close form solution time: {end_1 - start_1}")
    print(f"Close form solution jit time: {end_3 - start_3}")
    print(f"Autograd solution time: {end_2 - start_2}")
    print("----------------------------")
