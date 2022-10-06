import numpy as np


class ExtendedKinematicModel:
    """
    states - [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    inputs - [drive force (proportional to acceleration), steering velocity]
    reference point - center of rear axle ? Need to check BayesRace paper
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

    def sort_reference_trajectory(self, position_ref, yaw_ref, speed_ref):
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

    def get_general_states(self, state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]
        return speed, orientation, position

    def get_f(self, state, control_input):
        # state = x, y, vx, yaw angle, vy, yaw rate, steering angle
        # input = drive force (proportional to acceleration), steering velocity
        # input check
        control_input = self.clip_input(control_input)

        # control inputs
        Fxr, delta_v = control_input

        # states x[k]
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state

        f = np.zeros(7)
        f[0] = vx * np.cos(yaw) - vy * np.sin(yaw)
        f[1] = vx * np.sin(yaw) + vy * np.cos(yaw)
        f[2] = Fxr / self.config.MASS
        f[3] = yaw_rate
        f[4] = (self.config.LR / self.config.WB) * (delta_v * vx + steering_angle * (Fxr / self.config.MASS))
        f[5] = (1.0 / self.config.WB) * (delta_v * vx + steering_angle * (Fxr / self.config.MASS))
        f[6] = delta_v

        return f

    def batch_get_model_matrix(self, state_vec, control_vec):
        A_block = []
        B_block = []
        C_block = []
        for t in range(state_vec.shape[1]):
            A, B, C = self.get_model_matrix(state_vec[:, t], control_vec[:, t])
            A_block.append(A)
            B_block.append(B)
            C_block.append(C)

        return np.array(A_block), np.array(B_block), np.array(C_block)

    def get_model_matrix(self, state, control_input):
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state
        Fxr, delta_v = control_input

        # State (or system) matrix A, 7x7
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[4, 4] = 1.0
        A[5, 5] = 1.0
        A[6, 6] = 1.0

        A[0, 2] = self.config.DTK * np.cos(yaw)
        A[0, 3] = self.config.DTK * (- vx * np.sin(yaw) - vy * np.cos(yaw))
        A[0, 4] = - self.config.DTK * np.sin(yaw)

        A[1, 2] = self.config.DTK * np.sin(yaw)
        A[1, 3] = self.config.DTK * (vx * np.cos(yaw) - vy * np.sin(yaw))
        A[1, 4] = self.config.DTK * np.cos(yaw)

        A[3, 5] = self.config.DTK * 1.0

        A[4, 2] = self.config.DTK * self.config.LR / self.config.WB * delta_v
        A[4, 6] = self.config.DTK * self.config.LR / (self.config.WB * self.config.MASS) * Fxr

        A[5, 2] = self.config.DTK * 1.0 / self.config.WB * delta_v
        A[5, 6] = self.config.DTK * 1.0 / (self.config.WB * self.config.MASS) * Fxr

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK * 1.0 / self.config.MASS
        B[4, 0] = self.config.DTK * self.config.LR / (self.config.WB * self.config.MASS) * steering_angle
        B[4, 1] = self.config.DTK * self.config.LR / self.config.WB * vx
        B[5, 0] = self.config.DTK * 1.0 / (self.config.WB * self.config.MASS) * steering_angle
        B[5, 1] = self.config.DTK * 1.0 / self.config.WB * vx
        B[6, 1] = self.config.DTK * 1.0

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * (yaw * vx * np.sin(yaw) + yaw * vy * np.cos(yaw))
        C[1] = self.config.DTK * (- yaw * vx * np.cos(yaw) + yaw * vy * np.sin(yaw))
        C[4] = self.config.DTK * (- self.config.LR / (self.config.WB * self.config.MASS) * steering_angle * Fxr + self.config.LR / self.config.WB * vx * delta_v)
        C[5] = self.config.DTK * (- 1.0 / (self.config.WB * self.config.MASS) * steering_angle * Fxr + 1.0 / self.config.WB * vx * delta_v)

        # print(A)
        # print(B)
        # print(C)

        return A, B, C

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
