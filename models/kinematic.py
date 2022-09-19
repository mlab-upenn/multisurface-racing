import numpy as np


class KinematicModel:
    """
    states - [x, y, v, yaw]
    inputs - [acceleration, steering angle]
    reference point - center of rear axle
    """

    def __init__(self, config):
        self.config = config

    def clip_input(self, u):
        # u matrix Nx2
        u = np.clip(u, [self.config.MAX_DECEL, self.config.MIN_STEER], [self.config.MAX_ACCEL, self.config.MAX_STEER])
        return u

    def clip_output(self, state):
        # state matrix Nx4
        state[2] = np.clip(state[2], self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state

    def get_model_constraints(self):
        state_constraints = np.array([[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf],
                                      [np.inf, np.inf, self.config.MAX_SPEED, np.inf]])

        input_constraints = np.array([[self.config.MAX_DECEL, self.config.MIN_STEER],
                                      [self.config.MAX_ACCEL, self.config.MAX_STEER]])

        input_diff_constraints = np.array([[-np.inf, -self.config.MAX_STEER_V * self.config.DTK],
                                           [np.inf, self.config.MAX_STEER_V * self.config.DTK]])
        return state_constraints, input_constraints, input_diff_constraints

    def sort_reference_trajectory(self, position_ref, yaw_ref, speed_ref):
        reference = np.array([
            position_ref[:, 0],
            position_ref[:, 1],
            speed_ref,
            yaw_ref,
        ])
        return reference

    def get_general_states(self, state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]
        return speed, orientation, position

    def get_f(self, state, control_input):
        # state = x, y, v, yaw
        # input check
        control_input = self.clip_input(control_input)
        delta = control_input[1]
        a = control_input[0]

        f = np.zeros(4)
        f[0] = state[2] * np.cos(state[3])
        f[1] = state[2] * np.sin(state[3])
        f[3] = state[2] / self.config.WB * np.tan(delta)
        f[2] = a

        return f

    def get_model_matrix(self, state, u):
        """
        Calc linear and discrete time dynamic model-> Explicit discrete time-invariant
        Linear System: Xdot = Ax + Bu + C
        State vector: x=[x, y, v, yaw]
        :param v: speed
        :param phi: heading angle of the vehicle
        :param delta: steering angle: delta_bar
        :return: A, B, C
        """
        v = state[2]
        phi = state[3]

        delta = u[1]

        # State (or system) matrix A, 4x4
        A = np.zeros((self.config.NXK, self.config.NXK))
        A[0, 0] = 1.0
        A[1, 1] = 1.0
        A[2, 2] = 1.0
        A[3, 3] = 1.0
        A[0, 2] = self.config.DTK * np.cos(phi)
        A[0, 3] = -self.config.DTK * v * np.sin(phi)
        A[1, 2] = self.config.DTK * np.sin(phi)
        A[1, 3] = self.config.DTK * v * np.cos(phi)
        A[3, 2] = self.config.DTK * np.tan(delta) / self.config.WB

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = self.config.DTK
        B[3, 1] = self.config.DTK * v / (self.config.WB * np.cos(delta) ** 2)

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * v * np.sin(phi) * phi
        C[1] = -self.config.DTK * v * np.cos(phi) * phi
        C[3] = -self.config.DTK * v * delta / (self.config.WB * np.cos(delta) ** 2)

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
