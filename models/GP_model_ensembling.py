import numpy
import numpy as np
import torch
import gpytorch
from torch.autograd.functional import jacobian
import os
import time
import gc

np.set_printoptions(precision=4, suppress=True)


class TorchNormalizer:
    def __init__(self):
        self.mean = 0.0
        self.std = 0.0

    def fit(self, x):
        self.mean = x.mean().item()
        self.std = x.std().item()

    def transform(self, x):
        x = (x - self.mean) / self.std
        return x

    def inverse_transform(self, x):
        x = x * self.std + self.mean
        return x

    def fit_transform(self, x):
        self.fit(x)
        x = self.transform(x)
        return x


class BatchIndependentMultitaskGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean(batch_shape=torch.Size([3]))
        self.covar_module = gpytorch.kernels.ScaleKernel(
            gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3]), ard_num_dims=6),
            # gpytorch.kernels.RQKernel(batch_shape=torch.Size([3])),
            # gpytorch.kernels.MaternKernel(nu=1.5, batch_shape=torch.Size([3])),
            batch_shape=torch.Size([3])
        )
        # self.covar_module = gpytorch.kernels.ProductKernel(
        #     gpytorch.kernels.RBFKernel(batch_shape=torch.Size([3])),
        #     gpytorch.kernels.PeriodicKernel(batch_shape=torch.Size([3])),
        # )

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
    reference point - center of rear axle ? Need to check BayesRace paper
    """

    def __init__(self, config):
        self.config = config

        # gpytorch.settings.cg_tolerance(0.1)

        self.x_measurements = None
        self.y_measurements = None

        self.x_samples = None
        self.y_samples = None

        self.train_x_scaled = None
        self.train_y_scaled = None

        self.scaler_x = [TorchNormalizer(), TorchNormalizer(), TorchNormalizer(),
                         TorchNormalizer(), TorchNormalizer(), TorchNormalizer()]
        self.scaler_y = [TorchNormalizer(), TorchNormalizer(), TorchNormalizer()]

        self.trained = False

        self.gp_likelihood = None
        self.gp_model = None

        self.means = None
        self.std = None

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

    def get_f(self, state, control_input):
        # state = x, y, vx, yaw angle, vy, yaw rate, steering angle
        # input = drive force (proportional to acceleration), steering velocity
        # input check
        control_input = self.clip_input(control_input)

        # control inputs
        Fxr, delta_v = control_input

        # states x[k]
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state

        with torch.no_grad(), gpytorch.settings.fast_pred_var():
            mean, lower, upper = self.scale_and_predict_model_step(state, control_input)

        f = np.zeros(7)
        f[0] = vx * np.cos(yaw) - vy * np.sin(yaw)
        f[1] = vx * np.sin(yaw) + vy * np.cos(yaw)
        f[2] = mean[0] / self.config.DTK
        f[3] = yaw_rate
        f[4] = mean[1] / self.config.DTK
        f[5] = mean[2] / self.config.DTK
        f[6] = delta_v

        return f

    def batch_get_model_matrix(self, state_vec, control_vec):
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state_vec
        Fxr, delta_v = control_vec

        A = B = C = []

        def fun(x):
            mean = self.for_jacobian_comp(x)
            return mean.reshape((-1, 3))

        h1 = np.zeros((x.shape[0], 6))
        h2 = np.zeros((x.shape[0], 6))
        h3 = np.zeros((x.shape[0], 6))

        mean = np.zeros((x.shape[0], 3))
        lower = np.zeros((x.shape[0], 3))
        upper = np.zeros((x.shape[0], 3))

        gp_states = torch.tensor(np.vstack((state_vec[[2, 4, 5, 6], :], control_vec)).T, dtype=torch.float, device=torch.device('cuda'))
        if self.trained:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                def batch_jacobian(f, x):
                    f_sum = lambda x: torch.sum(f(x), axis=0)
                    return jacobian(f_sum, x)

                jac = batch_jacobian(fun, gp_states)

                # x, y, vx, yaw angle, vy, yaw rate, steering angle
                # start = time.time()
                mean, lower, upper = self.scale_and_predict_model_step(state_vec, control_vec)
                # end = time.time()
                # print(end - start)

                h1 = jac[0].cpu().numpy()
                h2 = jac[1].cpu().numpy()
                h3 = jac[2].cpu().numpy()

                # mean1_test, _, _ = self.scale_and_predict_model_step(state_vec[:, 0], control_vec[:, 0])
                # mean1_test = mean1_test.squeeze()
                # mean2_test = self.for_jacobian_comp(gp_states[0]).squeeze().numpy()
                #
                # print(np.isclose(mean2_test[0], mean1_test[0], rtol=1.e-3, atol=1.e-5))
                # print(np.isclose(mean2_test[1], mean1_test[1], rtol=1.e-3, atol=1.e-5))
                # print(np.isclose(mean2_test[2], mean1_test[2], rtol=1.e-3, atol=1.e-5))

        # State (or system) matrix A, 7x7
        A = np.zeros((x.shape[0], self.config.NXK, self.config.NXK))
        A[:, 0, 0] = 1.0
        A[:, 1, 1] = 1.0
        A[:, 2, 2] = 1.0
        A[:, 3, 3] = 1.0
        A[:, 4, 4] = 1.0
        A[:, 5, 5] = 1.0
        A[:, 6, 6] = 1.0

        A[:, 0, 2] = self.config.DTK * np.cos(yaw)
        A[:, 0, 3] = self.config.DTK * (- vx * np.sin(yaw) - vy * np.cos(yaw))
        A[:, 0, 4] = - self.config.DTK * np.sin(yaw)

        A[:, 1, 2] = self.config.DTK * np.sin(yaw)
        A[:, 1, 3] = self.config.DTK * (vx * np.cos(yaw) - vy * np.sin(yaw))
        A[:, 1, 4] = self.config.DTK * np.cos(yaw)

        A[range(x.shape[0]), 2, 2] += h1[range(x.shape[0]), 0]  # dh1/dvx
        A[range(x.shape[0]), 2, 4] = h1[range(x.shape[0]), 1]  # dh1/dvy
        A[range(x.shape[0]), 2, 5] = h1[range(x.shape[0]), 2]  # dh1/d omega
        A[range(x.shape[0]), 2, 6] = h1[range(x.shape[0]), 3]  # dh1/d delta

        A[:, 3, 5] = self.config.DTK * 1.0

        A[range(x.shape[0]), 4, 2] = h2[range(x.shape[0]), 0]  # dh2/dvx
        A[range(x.shape[0]), 4, 4] += h2[range(x.shape[0]), 1]  # dh2/dvy
        A[range(x.shape[0]), 4, 5] = h2[range(x.shape[0]), 2]  # dh2/d omega
        A[range(x.shape[0]), 4, 6] = h2[range(x.shape[0]), 3]  # dh2/d delta

        A[range(x.shape[0]), 5, 2] = h3[range(x.shape[0]), 0]  # dh3/dvx
        A[range(x.shape[0]), 5, 4] = h3[range(x.shape[0]), 1]  # dh3/dvy
        A[range(x.shape[0]), 5, 5] += h3[range(x.shape[0]), 2]  # dh3/d omega
        A[range(x.shape[0]), 5, 6] = h3[range(x.shape[0]), 3]  # dh3/d delta

        # Input Matrix B; 4x2
        B = np.zeros((x.shape[0], self.config.NXK, self.config.NU))
        B[range(x.shape[0]), 2, 0] = h1[range(x.shape[0]), 4]  # dh1/dFx
        B[range(x.shape[0]), 2, 1] = h1[range(x.shape[0]), 5]  # dh1/d speed delta
        B[range(x.shape[0]), 4, 0] = h2[range(x.shape[0]), 4]  # dh2/dFx
        B[range(x.shape[0]), 4, 1] = h2[range(x.shape[0]), 5]  # dh2/d speed delta
        B[range(x.shape[0]), 5, 0] = h3[range(x.shape[0]), 4]  # dh3/dFx
        B[range(x.shape[0]), 5, 1] = h3[range(x.shape[0]), 5]  # dh3/d speed delta
        B[:, 6, 1] = self.config.DTK * 1.0

        C = np.zeros((x.shape[0], self.config.NXK))
        C[:, 0] = self.config.DTK * (yaw * vx * np.sin(yaw) + yaw * vy * np.cos(yaw))
        C[:, 1] = self.config.DTK * (- yaw * vx * np.cos(yaw) + yaw * vy * np.sin(yaw))
        C[range(x.shape[0]), 2] = mean[0, range(x.shape[0])] - h1[range(x.shape[0]), 0] * vx - h1[range(x.shape[0]), 1] * vy \
                                  - h1[range(x.shape[0]), 2] * yaw_rate - h1[range(x.shape[0]), 3] * steering_angle - h1[
                                      range(x.shape[0]), 4] * Fxr - h1[range(x.shape[0]), 5] * delta_v
        C[range(x.shape[0]), 4] = mean[1, range(x.shape[0])] - h2[range(x.shape[0]), 0] * vx - h2[range(x.shape[0]), 1] * vy \
                                  - h2[range(x.shape[0]), 2] * yaw_rate - h2[range(x.shape[0]), 3] * steering_angle - h2[
                                      range(x.shape[0]), 4] * Fxr - h2[range(x.shape[0]), 5] * delta_v
        C[range(x.shape[0]), 5] = mean[2, range(x.shape[0])] - h3[range(x.shape[0]), 0] * vx - h3[range(x.shape[0]), 1] * vy \
                                  - h3[range(x.shape[0]), 2] * yaw_rate - h3[range(x.shape[0]), 3] * steering_angle - h3[
                                      range(x.shape[0]), 4] * Fxr - h3[range(x.shape[0]), 5] * delta_v

        # print("Sigma %.6f     Mean %.6f" % (np.abs(mean[2] - lower[2]), mean[2]))
        # print(A)
        # print(B)
        # print(C)

        return A, B, C

    def get_model_matrix(self, state, control_input):
        x, y, vx, yaw, vy, yaw_rate, steering_angle = state
        Fxr, delta_v = control_input

        def fun(x):
            mean = self.for_jacobian_comp(x)
            return mean.reshape((1, 3))

        h1 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        h2 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        h3 = np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.0])
        mean = np.array([0.0, 0.0, 0.0])
        lower = np.array([0.0, 0.0, 0.0])
        upper = np.array([0.0, 0.0, 0.0])

        if self.trained:
            with torch.no_grad(), gpytorch.settings.fast_pred_var():
                jac = jacobian(fun, torch.tensor(
                    [[float(vx), float(vy), float(yaw_rate), float(steering_angle), float(Fxr),
                      float(delta_v)]]).cuda()).squeeze()

                mean, lower, upper = self.scale_and_predict_model_step(state, control_input)

                h1 = jac[0].cpu().numpy()
                h2 = jac[1].cpu().numpy()
                h3 = jac[2].cpu().numpy()

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

        A[2, 2] += h1[0]  # dh1/dvx
        A[2, 4] = h1[1]  # dh1/dvy
        A[2, 5] = h1[2]  # dh1/d omega
        A[2, 6] = h1[3]  # dh1/d delta

        A[3, 5] = self.config.DTK * 1.0

        A[4, 2] = h2[0]  # dh2/dvx
        A[4, 4] += h2[1]  # dh2/dvy
        A[4, 5] = h2[2]  # dh2/d omega
        A[4, 6] = h2[3]  # dh2/d delta

        A[5, 2] = h3[0]  # dh3/dvx
        A[5, 4] = h3[1]  # dh3/dvy
        A[5, 5] += h3[2]  # dh3/d omega
        A[5, 6] = h3[3]  # dh3/d delta

        # Input Matrix B; 4x2
        B = np.zeros((self.config.NXK, self.config.NU))
        B[2, 0] = h1[4]  # dh1/dFx
        B[2, 1] = h1[5]  # dh1/d speed delta
        B[4, 0] = h2[4]  # dh2/dFx
        B[4, 1] = h2[5]  # dh2/d speed delta
        B[5, 0] = h3[4]  # dh3/dFx
        B[5, 1] = h3[5]  # dh3/d speed delta
        B[6, 1] = self.config.DTK * 1.0

        C = np.zeros(self.config.NXK)
        C[0] = self.config.DTK * (yaw * vx * np.sin(yaw) + yaw * vy * np.cos(yaw))
        C[1] = self.config.DTK * (- yaw * vx * np.cos(yaw) + yaw * vy * np.sin(yaw))
        C[2] = mean[0] - h1[0] * vx - h1[1] * vy - h1[2] * yaw_rate - h1[3] * steering_angle - h1[4] * Fxr - h1[
            5] * delta_v
        C[4] = mean[1] - h2[0] * vx - h2[1] * vy - h2[2] * yaw_rate - h2[3] * steering_angle - h2[4] * Fxr - h2[
            5] * delta_v
        C[5] = mean[2] - h3[0] * vx - h3[1] * vy - h3[2] * yaw_rate - h3[3] * steering_angle - h3[4] * Fxr - h3[
            5] * delta_v

        # print("Sigma %.6f     Mean %.6f" % (np.abs(mean[2] - lower[2]), mean[2]))
        # print(A)
        # print(B)
        # print(C)

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

        point = torch.from_numpy(np.array([self.scaler_x[0].transform(state[2]),
                                           self.scaler_x[1].transform(state[4]),
                                           self.scaler_x[2].transform(state[5]),
                                           self.scaler_x[3].transform(state[6]),
                                           self.scaler_x[4].transform(control_input[0]),  # control_input[0, i - 1]
                                           self.scaler_x[5].transform(control_input[1]),
                                           ], dtype=numpy.float32)).reshape((-1, 6)).cuda()

        mean, lower, upper = self.predict_model_step(point)
        if len(lower.shape) == 1:
            lower = lower.reshape(1, -1)
            upper = upper.reshape(1, -1)
        scaled_mean = np.array([self.scaler_y[0].inverse_transform(mean[:, 0]).detach().numpy(),
                                self.scaler_y[1].inverse_transform(mean[:, 1]).detach().numpy(),
                                self.scaler_y[2].inverse_transform(mean[:, 2]).detach().numpy(),
                                ])

        scaled_lower = np.array([self.scaler_y[0].inverse_transform(lower[:, 0]).detach().numpy(),
                                 self.scaler_y[1].inverse_transform(lower[:, 1]).detach().numpy(),
                                 self.scaler_y[2].inverse_transform(lower[:, 2]).detach().numpy(),
                                 ])

        scaled_upper = np.array([self.scaler_y[0].inverse_transform(upper[:, 0]).detach().numpy(),
                                 self.scaler_y[1].inverse_transform(upper[:, 1]).detach().numpy(),
                                 self.scaler_y[2].inverse_transform(upper[:, 2]).detach().numpy(),
                                 ])

        return scaled_mean, scaled_lower, scaled_upper

    def for_jacobian_comp(self, x):
        x = (x - self.means) / self.std

        mean, lower, upper = self.predict_model_step(x)

        means = torch.tensor([[self.scaler_y[0].mean, self.scaler_y[1].mean, self.scaler_y[2].mean]])
        std = torch.tensor([[self.scaler_y[0].std, self.scaler_y[1].std, self.scaler_y[2].std]])

        mean = mean * std + means

        return mean

    def predict_model_step(self, X_sample):
        """
        :param X_sample: [vx, vy, yaw_rate, steering_angle, u_drive, u_steering_velocity]
        :return: model error prediction
        """
        predictions = self.gp_likelihood(self.gp_model(X_sample))
        confidence = predictions.confidence_region()
        return predictions.mean.cpu(), torch.squeeze(confidence[0].cpu()), torch.squeeze(
            confidence[1].cpu())  # mean, lower, upper

    def add_new_datapoint(self, X_sample, Y_sample):
        """
        :param X_sample: (np.array) state and input vector:
                        [vx, vy, yaw_rate, steering_angle, u_drive, u_steering_velocity]
        :param Y_sample: (np.array) error vector: [vx_error, vy_error, yaw_rate_error]
        :return:
        """
        if self.x_measurements is None:
            self.x_measurements = X_sample.copy()
        else:
            self.x_measurements = np.vstack((self.x_measurements, X_sample))

        if self.y_measurements is None:
            self.y_measurements = Y_sample.copy()
        else:
            self.y_measurements = np.vstack((self.y_measurements, Y_sample))

    def init_gp(self):
        train_x = torch.tensor([self.x_measurements[k] for k in range(6)])
        train_y = torch.tensor([self.y_measurements[k] for k in range(3)])
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

        self.means = torch.tensor([[self.scaler_x[0].mean, self.scaler_x[1].mean, self.scaler_x[2].mean,
                                    self.scaler_x[3].mean, self.scaler_x[4].mean, self.scaler_x[5].mean]], device=torch.device('cuda'))

        self.std = torch.tensor([[self.scaler_x[0].std, self.scaler_x[1].std, self.scaler_x[2].std,
                                  self.scaler_x[3].std, self.scaler_x[4].std, self.scaler_x[5].std]], device=torch.device('cuda'))

        return train_x_scaled, train_y_scaled

    def cuda(self):
        self.gp_model.cuda()
        self.gp_likelihood.cuda()

    def eval(self):
        self.trained = True
        self.gp_model.eval()
        self.gp_likelihood.eval()

    def train_gp(self, train_x_scaled, train_y_scaled, method=0):
        self.gp_model = self.gp_model.cuda()
        self.gp_likelihood = self.gp_likelihood.cuda()

        if method == 0:
            training_iterations = 300
        else:
            training_iterations = 1000

        # Find optimal model hyper-parameters
        self.gp_model.train()
        self.gp_likelihood.train()

        # Use the adam optimizer
        optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.02)  # Includes GaussianLikelihood parameters

        # "Loss" for GPs - the marginal log likelihood
        mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

        for i in range(training_iterations):
            if method == 0:
                for g in optimizer.param_groups:
                    if i < 50:
                        g['lr'] = 0.1
                    elif i < 100:
                        g['lr'] = 0.0001
                    else:
                        g['lr'] = 0.000001

            if method == 1:
                for g in optimizer.param_groups:
                    if i < 500:
                        g['lr'] = 0.01
                    elif i < 600:
                        g['lr'] = 0.001
                    elif i < 800:
                        g['lr'] = 0.0001
                    else:
                        g['lr'] = 0.00001

            optimizer.zero_grad()
            output = self.gp_likelihood(self.gp_model(train_x_scaled))
            loss = -mll(output, train_y_scaled)
            loss.backward()
            if i % 99 == 0:
                print('Iter %d/%d - Loss: %.3f' % (i + 1, training_iterations, loss.item()))
            optimizer.step()

        # Set into eval mode
        self.gp_model.eval()
        self.gp_likelihood.eval()
        self.trained = True
        print(len(self.x_samples[0]))

    def train_gp_min_variance(self, num_of_new_samples=40):
        st_model = None
        st_like = None
        for i in range(num_of_new_samples):
            torch.cuda.empty_cache()
            if not self.trained:
                for _ in range(2):
                    for j in range(6):
                        self.x_samples[j].append(self.x_measurements[j].pop(0))
                    for j in range(3):
                        self.y_samples[j].append(self.y_measurements[j].pop(0))
            else:
                errors = []

                state_vect = np.array([
                    np.zeros(len(self.x_measurements[0])),
                    np.zeros(len(self.x_measurements[0])),
                    self.x_measurements[0],
                    np.zeros(len(self.x_measurements[0])),
                    self.x_measurements[1],
                    self.x_measurements[2],
                    self.x_measurements[3],
                ])
                control_vect = np.array([self.x_measurements[4], self.x_measurements[5]])

                with torch.no_grad(), gpytorch.settings.fast_pred_var():
                    mean, lower, upper = self.scale_and_predict_model_step(state_vect, control_vect)

                if i <= num_of_new_samples / 4.0:
                    errors = abs(upper[2] - mean[2])
                elif i <= num_of_new_samples / 4.0 * 2.0:
                    errors = abs(upper[0] - mean[0])
                elif i <= num_of_new_samples / 4.0 * 3.0:
                    errors = abs(upper[1] - mean[1])
                # elif i <= num_of_new_samples / 5.0 * 4.0:
                #     errors = np.abs((upper[0] - mean[0]) / self.scaler_y[0].std) + np.abs(
                #         (upper[1] - mean[1]) / self.scaler_y[1].std) + np.abs((upper[2] - mean[2]) / self.scaler_y[2].std)
                else:
                    errors = abs(mean[1] - self.y_measurements[1])
                    # errors = 1.0 - abs(upper[1] - mean[1])

                if np.max(errors.shape) == 0:
                    break

                idx = np.argmax(errors)

                for j in range(6):
                    self.x_samples[j].append(self.x_measurements[j].pop(idx))
                for j in range(3):
                    self.y_samples[j].append(self.y_measurements[j].pop(idx))

            train_x = torch.tensor([self.x_samples[k] for k in range(6)])
            train_y = torch.tensor([self.y_samples[k] for k in range(3)])
            train_y = train_y.contiguous()

            if self.train_x_scaled is not None and self.train_y_scaled is not None:
                self.train_x_scaled.cpu()
                del self.train_x_scaled
                self.train_y_scaled.cpu()
                del self.train_y_scaled
                gc.collect()
                torch.cuda.empty_cache()

            self.train_x_scaled = torch.transpose(torch.vstack((self.scaler_x[0].fit_transform(train_x[0]),
                                                                self.scaler_x[1].fit_transform(train_x[1]),
                                                                self.scaler_x[2].fit_transform(train_x[2]),
                                                                self.scaler_x[3].fit_transform(train_x[3]),
                                                                self.scaler_x[4].fit_transform(train_x[4]),
                                                                self.scaler_x[5].fit_transform(train_x[5]),)), 0, 1).cuda()

            self.train_y_scaled = torch.transpose(torch.vstack((self.scaler_y[0].fit_transform(train_y[0]),
                                                                self.scaler_y[1].fit_transform(train_y[1]),
                                                                self.scaler_y[2].fit_transform(train_y[2]),)), 0, 1).cuda()

            if self.gp_likelihood is not None:
                self.gp_likelihood.cpu()
                del self.gp_likelihood
                gc.collect()
                torch.cuda.empty_cache()
            self.gp_likelihood = gpytorch.likelihoods.MultitaskGaussianLikelihood(num_tasks=3,
                                                                                  # noise_prior=gpytorch.priors.SmoothedBoxPrior(0.0015, 1.5, sigma=0.0005)
                                                                                  )

            if self.gp_model is not None:
                self.gp_model.cpu()
                del self.gp_model
                gc.collect()
                torch.cuda.empty_cache()
            self.gp_model = BatchIndependentMultitaskGPModel(self.train_x_scaled, self.train_y_scaled, self.gp_likelihood)

            if st_model is not None and st_like is not None:
                self.gp_model.load_state_dict(st_model)
                self.gp_likelihood.load_state_dict(st_like)

            if self.means is not None:
                self.means.cpu()
                del self.means
                gc.collect()
                torch.cuda.empty_cache()
            self.means = torch.tensor([[self.scaler_x[0].mean, self.scaler_x[1].mean, self.scaler_x[2].mean,
                                        self.scaler_x[3].mean, self.scaler_x[4].mean, self.scaler_x[5].mean]], device=torch.device('cuda'))

            if self.std is not None:
                self.std.cpu()
                del self.std
                gc.collect()
                torch.cuda.empty_cache()
            self.std = torch.tensor([[self.scaler_x[0].std, self.scaler_x[1].std, self.scaler_x[2].std,
                                      self.scaler_x[3].std, self.scaler_x[4].std, self.scaler_x[5].std]], device=torch.device('cuda'))

            self.gp_model = self.gp_model.cuda()
            self.gp_likelihood = self.gp_likelihood.cuda()

            if st_model is not None and st_like is not None:
                training_iterations = 1
            else:
                training_iterations = 500

            # Find optimal model hyper-parameters
            self.gp_model.train()
            self.gp_likelihood.train()

            # Use the adam optimizer
            optimizer = torch.optim.Adam(self.gp_model.parameters(), lr=0.0001)  # Includes GaussianLikelihood parameters
            # self.gp_model.state_dict()
            # self.gp_model.load_state_dict(st)
            # "Loss" for GPs - the marginal log likelihood
            mll = gpytorch.mlls.ExactMarginalLogLikelihood(self.gp_likelihood, self.gp_model)

            for k in range(training_iterations):
                if st_model is None or st_like is None:
                    for g in optimizer.param_groups:
                        if st_model is None or st_like is None:
                            if k < 50:
                                g['lr'] = 0.1
                            elif k < 100:
                                g['lr'] = 0.0001
                            else:
                                g['lr'] = 0.000001

                optimizer.zero_grad()
                output = self.gp_likelihood(self.gp_model(self.train_x_scaled))
                loss = -mll(output, self.train_y_scaled)
                loss.backward()
                if k % 99 == 0:
                    print('Iter %d/%d - Loss: %.3f' % (k + 1, training_iterations, loss.item()))
                optimizer.step()

            if (i + 1) % 500 == 0 or i == num_of_new_samples - 2:
                st_like = None
                st_model = None
            else:
                st_model = self.gp_model.state_dict()
                st_like = self.gp_likelihood.state_dict()

            # Set into eval mode
            self.gp_model.eval()
            self.gp_likelihood.eval()
            self.trained = True
            print(len(self.x_samples[0]))

        self.x_measurements = None
        self.y_measurements = None
