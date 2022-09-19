import numpy as np


class LocalRegressionModel:
    """
    This model is based on work by Ugo Rosolia
    states - [x, y, vx, yaw, vy, yaw rate]
    inputs - [acceleration, steering angle]
    reference point - centre of mass
    """

    def __init__(self, config):
        self.config = config

    def clip_input(self, u):
        # u matrix Nx2
        u = np.clip(u, [self.config.MAX_DECEL, self.config.MIN_STEER],
                    [self.config.MAX_ACCEL, self.config.MAX_STEER])
        return u

    def clip_output(self, state):
        # state matrix Nx6
        state[2] = np.clip(state[2], self.config.MIN_SPEED, self.config.MAX_SPEED)
        return state

    def get_model_constraints(self):
        state_constraints = np.array(
            [[-np.inf, -np.inf, self.config.MIN_SPEED, -np.inf, -np.inf, -np.inf],
             [np.inf, np.inf, self.config.MAX_SPEED, np.inf, np.inf, np.inf]])

        input_constraints = np.array([[self.config.MAX_DECEL, self.config.MIN_STEER],
                                      [self.config.MAX_ACCEL, self.config.MAX_STEER]])

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
            np.zeros(len(speed_ref))
        ])
        return reference

    def get_general_states(self, state):
        speed = state[2]
        orientation = state[3]
        position = state[[0, 1]]
        return speed, orientation, position

    def get_f(self, state, control_input):
        # state = x, y, vx, yaw angle, vy, yaw rate
        # input = acceleration, steering angle
        # input check
        control_input = self.clip_input(control_input)

        # control inputs
        acceleration, steering_angle = control_input

        # states x[k]
        x, y, vx, yaw, vy, yaw_rate = state

        f = np.zeros(7)

        return f
