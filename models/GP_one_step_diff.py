import numpy as np
import torch
import gpytorch
from pytorch_forecasting.data.encoders import TorchNormalizer
import os
import time


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3])),
            batch_shape=torch.Size([3])
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultitaskMultivariateNormal.from_batch_mvn(
            gpytorch.distributions.MultivariateNormal(mean_x, covar_x)
        )


class GPEnsembleModel:
    """
    states - [x, y, vx, yaw angle, vy, yaw rate, steering angle]
    inputs - [drive force (proportional to acceleration), steering velocity]
    reference point - center of mass
    """

    def __init__(self, config):
        self.config = config

        # X data for GP training
        self.vx = []
        self.vy = []
        self.yaw_rate = []
        self.steering_angle = []
        self.u_drive = []
        self.u_steering_velocity = []

        # Y data for GP training
        self.vx_error = []
        self.vy_error = []
        self.yaw_rate_error = []

        self.scaler_x = [TorchNormalizer(), TorchNormalizer(), TorchNormalizer(),
                         TorchNormalizer(), TorchNormalizer(), TorchNormalizer()]
        self.scaler_y = [TorchNormalizer(), TorchNormalizer(), TorchNormalizer()]

        self.gp_trained = False

        self.gp_likelihood = None
        self.gp_model = None

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

    #  Done until here

    def get_f(self, state, control_input):
        # state = x, y, vx, yaw angle, vy, yaw rate, steering angle
        # input = drive force (proportional to acceleration), steering velocity
        # input check
        control_input = self.clip_input(control_input)

        # control inputs
        Fxr, delta_v = control_input

        # states x[k]
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state

        point = torch.tensor([float(self.scaler_x[0].transform(vx)),
                              float(self.scaler_x[1].transform(vy)),
                              float(self.scaler_x[2].transform(yaw_rate)),
                              float(self.scaler_x[3].transform(steering_angle)),
                              float(self.scaler_x[4].transform(Fxr)),
                              float(self.scaler_x[5].transform(delta_v)),
                              ]).reshape((1, 6)).cuda()

        mean, lower, upper = self.predict_model_step(point)

        scaled_mean = np.array([self.scaler_y[0].inverse_transform(mean[:, 0]).numpy(),
                                self.scaler_y[1].inverse_transform(mean[:, 1]).numpy(),
                                self.scaler_y[2].inverse_transform(mean[:, 2]).numpy(), ])

        scaled_lower = np.array([self.scaler_y[0].inverse_transform(lower[0]).numpy(),
                                 self.scaler_y[1].inverse_transform(lower[1]).numpy(),
                                 self.scaler_y[2].inverse_transform(lower[2]).numpy(), ])

        scaled_upper = np.array([self.scaler_y[0].inverse_transform(upper[0]).numpy(),
                                 self.scaler_y[1].inverse_transform(upper[1]).numpy(),
                                 self.scaler_y[2].inverse_transform(upper[2]).numpy(), ])

        f = np.zeros(7)
        f[0] = vx * np.cos(yaw) - vy * np.sin(yaw)
        f[1] = vx * np.sin(yaw) + vy * np.cos(yaw)
        f[2] = scaled_mean[0] / self.config.DTK
        f[3] = yaw_rate
        f[4] = scaled_mean[1] / self.config.DTK
        f[5] = scaled_mean[2] / self.config.DTK
        f[6] = delta_v

        return f

    def get_model_matrix(self, state, control_input):
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state
        Fxr, delta_v = control_input

        # State (or system) matrix A, 7x7
        A = np.zeros((self.config.NXK, self.config.NXK))

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))

        C = np.zeros(self.config.NXK)

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

    def predict_motion_w_gp(self, x0, control_input, dt):
        """
        :param x0: np.array [x, y, vx, yaw angle, vy, yaw rate, steering angle]
        :param control_input: [drive force (proportional to acceleration), steering velocity]
        :param dt:
        :return:
        """
        predicted_states = np.zeros((x0.size, control_input.shape[1] + 1))
        predicted_states[:, 0] = x0
        state = x0
        dxdt = []
        dxdt2 = []

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            for i in range(1, control_input.shape[1] + 1):

                # control_input[0] = control_input[0] * self.config.MASS
                state = state + self.get_f(state, control_input[:, i - 1]) * dt
                dxdt2.append(np.array([self.get_f(state, control_input[:, i - 1])[0] * dt,
                                       self.get_f(state, control_input[:, i - 1])[1] * dt,
                                       self.get_f(state, control_input[:, i - 1])[2] * dt,
                                       self.get_f(state, control_input[:, i - 1])[3] * dt,
                                       self.get_f(state, control_input[:, i - 1])[4] * dt,
                                       self.get_f(state, control_input[:, i - 1])[5] * dt,
                                       self.get_f(state, control_input[:, i - 1])[6] * dt]))

                scaled_lower = [0, 0, 0]
                scaled_upper = [0, 0, 0]

                if self.gp_model is not None and self.gp_likelihood is not None:
                    # X_sample = np.array(
                    #     [state[2], state[4], state[5], state[6], control_input[0, i - 1], control_input[1, i - 1]])

                    start = time.time()
                    point = torch.tensor([float(self.scaler_x[0].transform(state[2])),
                                          float(self.scaler_x[1].transform(state[4])),
                                          float(self.scaler_x[2].transform(state[5])),
                                          float(self.scaler_x[3].transform(state[6])),
                                          float(self.scaler_x[4].transform(control_input[0, i - 1])),
                                          float(self.scaler_x[5].transform(control_input[1, i - 1])),
                                          ]).reshape((1, 6)).cuda()
                    # error_prediction = self.predict_model_error(torch.from_numpy(X_sample))

                    mean, lower, upper = self.predict_model_step(point)
                    end = time.time()
                    print(end - start)
                    # np.array [x, y, vx, yaw angle, vy, yaw rate, steering angle]

                    # state = state + self.get_f(state, control_input[:, i - 1]) * dt + np.array(
                    #     [0.0, 0.0, mean[:, 0].numpy(), 0.0, mean[:, 1].numpy(), mean[:, 2].numpy(), 0.0])

                    scaled_mean = np.array([self.scaler_y[0].inverse_transform(mean[:, 0]).numpy(),
                                            self.scaler_y[1].inverse_transform(mean[:, 1]).numpy(),
                                            self.scaler_y[2].inverse_transform(mean[:, 2]).numpy(), ])

                    scaled_lower = np.array([self.scaler_y[0].inverse_transform(lower[0]).numpy(),
                                             self.scaler_y[1].inverse_transform(lower[1]).numpy(),
                                             self.scaler_y[2].inverse_transform(lower[2]).numpy(), ])

                    scaled_upper = np.array([self.scaler_y[0].inverse_transform(upper[0]).numpy(),
                                             self.scaler_y[1].inverse_transform(upper[1]).numpy(),
                                             self.scaler_y[2].inverse_transform(upper[2]).numpy(), ])

                    dxdt.append(np.array([self.get_f(state, control_input[:, i - 1])[0] * dt,
                                          self.get_f(state, control_input[:, i - 1])[1] * dt,
                                          scaled_mean[0],
                                          self.get_f(state, control_input[:, i - 1])[3] * dt,
                                          scaled_mean[1],
                                          scaled_mean[2],
                                          self.get_f(state, control_input[:, i - 1])[6] * dt]))
                else:
                    dxdt.append(np.zeros(7))

                state = self.clip_output(state)
                predicted_states[:, i] = state

            input_prediction = np.zeros((2, control_input.shape[1] + 1))
            return predicted_states, input_prediction, dxdt, dxdt2, scaled_lower, scaled_upper

    def predict_model_step(self, X_sample):
        """
        :param X_sample: [vx, vy, yaw_rate, steering_angle, u_drive, u_steering_velocity]
        :return: model error prediction
        """
        predictions = self.gp_likelihood(self.gp_model(X_sample))
        confidence = predictions.confidence_region()  # (tensor([[ 5.8466, -2.3680, -2.0273]]), tensor([[ 6.2180, -0.3759, -0.9362]]))
        return predictions.mean.cpu(), torch.squeeze(confidence[0].cpu()), torch.squeeze(
            confidence[1].cpu())  # mean, lower, upper

    def add_new_datapoint(self, X_sample, Y_sample):
        """
        :param X_sample: (np.array) state and input vector:
                        [vx, vy, yaw_rate, steering_angle, u_drive, u_steering_velocity]
        :param Y_sample: (np.array) error vector: [vx_error, vy_error, yaw_rate_error]
        :return:
        """
        X_sample = X_sample.tolist()
        Y_sample = Y_sample.tolist()

        self.vx.append(X_sample[0])
        self.vy.append(X_sample[1])
        self.yaw_rate.append(X_sample[2])
        self.steering_angle.append(X_sample[3])
        self.u_drive.append(X_sample[4])
        self.u_steering_velocity.append(X_sample[5])

        self.vx_error.append(Y_sample[0])
        self.vy_error.append(Y_sample[1])
        self.yaw_rate_error.append(Y_sample[2])

    def train_gp(self):
        train_x = torch.tensor([self.vx, self.vy, self.yaw_rate,
                                self.steering_angle, self.u_drive,
                                self.u_steering_velocity])

        train_y = torch.tensor([self.vx_error, self.vy_error, self.yaw_rate_error])
        train_y = train_y.contiguous()

        train_x_scaled = torch.transpose(torch.vstack((self.scaler_x[0].fit_transform(train_x[0]),
                                                       self.scaler_x[1].fit_transform(train_x[1]),
                                                       self.scaler_x[2].fit_transform(train_x[2]),
                                                       self.scaler_x[3].fit_transform(train_x[3]),
                                                       self.scaler_x[4].fit_transform(train_x[4]),
                                                       self.scaler_x[5].fit_transform(train_x[5]),)), 0, 1).cuda()

        train_y_scaled = torch.transpose(torch.vstack((self.scaler_y[0].fit_transform(train_y[0]),
                                                       self.scaler_y[1].fit_transform(train_y[1]),
                                                       self.scaler_y[2].fit_transform(train_y[2]),)), 0, 1).cuda()

        self.gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3)
        self.gp_model = BatchIndependentMultitaskGPModel(train_x_scaled, train_y_scaled, self.gp_likelihood)

        self.gp_model = self.gp_model.cuda()
        self.gp_likelihood = self.gp_likelihood.cuda()

        training_iterations = 200

        # Find optimal model hyper-parameters
        self.gp_model.train()
        self.gp_likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

        for i in range(training_iterations):
            optimizer.zero_grad()
            output = self.gp_likelihood(self.gp_model(train_x_scaled))
            loss = -mll(output, train_y_scaled)
            loss.backward()
            print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        # Set into eval mode
        self.gp_model.eval()
        self.gp_likelihood.eval()
