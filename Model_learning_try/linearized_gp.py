import math

import numpy as np
import torch
import gpytorch
from matplotlib import pyplot as plt
from torch.autograd.functional import jacobian, hessian
import time

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x = torch.linspace(0, 1, 1000)
# True function is sin(2*pi*x) with Gaussian noise
train_y = torch.sin(train_x * (2 * math.pi)) + torch.randn(train_x.size()) * math.sqrt(0.04)
# True function is sin(2*pi*x) with no noise
train_y_no_noise = torch.sin(train_x * (2 * math.pi))


# print(train_x)
# print(train_y)


# We will use the simplest form of GP model, exact inference
class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super(ExactGPModel, self).__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def k(x1, x2, gp_model):
    covar_x = gp_model.covar_module(x1=x1, x2=x2)
    return covar_x[0]

def k_jac(x1, x2, gp_model):
    def tmp_fun(x1, x2):
        return gp_model.covar_module(x1, x2)[0]

    jac = jacobian(tmp_fun, (x1, x2))
    return jac[0].reshape(x2.shape[0],x1.shape[0]), jac[1].reshape(x2.shape[0],x2.shape[0])


def k_hes(x1, x2, gp_model):
    def tmp_fun(x1, x2):
        return gp_model.covar_module(x1, x2)[0]

    hes = hessian(tmp_fun, (x1, x2))
    return hes[0][1].reshape(x1.shape[0],x1.shape[0])

def find_Vx(x, X, gp_model):
    kxx = k(x,x, gp_model).reshape(-1,1)
    K01_xx, K10_xx = k_jac(x,x, gp_model)
    K11_xx = k_hes(x,x,gp_model)
    first_term = torch.cat((torch.cat((kxx, K01_xx), dim=1),
                            torch.cat((K10_xx, K11_xx), dim=1)), dim=0)

    kxX = k(x, X, gp_model).reshape(-1,1)
    _, K10_xX = k_jac(x,X, gp_model)
    KXX = k(X,X, gp_model).reshape(-1,1)
    kXx = k(X,x, gp_model).reshape(-1,1)
    K01_Xx, _ = k_jac(X,x, gp_model)
    sigma_sq = 1  # TODO
    second_term = torch.cat((kxX, K10_xX), dim=0) @ np.linalg.inv(KXX + sigma_sq * np.eye(KXX.shape(0))) @ torch.cat((kXx, K01_Xx), dim=0)

    V_hatx = first_term - second_term
    return V_hatx

# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
import os

smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 50

# Find optimal model hyperparameters
model.train()
likelihood.train()

# Use the adam optimizer
optimizer = torch.optim.Adam(model.parameters(), lr=0.1)  # Includes GaussianLikelihood parameters

# "Loss" for GPs - the marginal log likelihood
mll = gpytorch.mlls.ExactMarginalLogLikelihood(likelihood, model)

for i in range(training_iter):
    # Zero gradients from previous iteration
    optimizer.zero_grad()
    # Output from model
    output = model(train_x)
    # Calc loss and backprop gradients
    loss = -mll(output, train_y)
    loss.backward()
    print('Iter %d/%d - Loss: %.3f   lengthscale: %.3f   noise: %.3f' % (
        i + 1, training_iter, loss.item(),
        model.covar_module.base_kernel.lengthscale.item(),
        model.likelihood.noise.item()
    ))
    optimizer.step()

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

def predict_mean(test_x):
    return likelihood(model(test_x)).mean

def predict_var(test_x):
    return likelihood(model(test_x)).variance

def predict_covar(test_x):
    return likelihood(model(test_x)).covariance_matrix

def predict_mean_sum(test_x):
    return likelihood(model(test_x)).mean.sum()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    def fun(x):
        mod = model(x)
        return likelihood(mod).mean


    def fun2(x):
        model_res = model(x)
        like = likelihood(model_res)
        res = like.mean + like.stddev * 2
        return res


    def batch_jacobian(f, x):
        f_sum = lambda x: torch.sum(f(x), axis=0)
        return jacobian(f_sum, x)


    point = torch.tensor([[0.5]])
    jac = batch_jacobian(fun, point)  # mean
    jac2 = jacobian(fun2, point)  # confidence

    print("Model")
    start = time.time()
    test_x = torch.linspace(-0.2, 1.2, 1000)
    # mod = model(test_x)
    end = time.time()
    print(end - start)

    # print("Likelihood")
    # start = time.time()
    # observed_pred = likelihood(model(test_x))
    # end = time.time()
    # print(end - start)

    v = torch.ones(1000)*(1/1000)
    mu = predict_mean(test_x)
    var = predict_var(test_x)
    covar = predict_covar(test_x)

    _ , dx = torch.autograd.functional.vjp(predict_mean, test_x, v)

    _, dx2 = torch.autograd.functional.vhp(predict_mean_sum, test_x, v)

    # f, ax = plt.subplots(1, 1, figsize=(8, 5))

    # plt.figure(figsize=(8, 5))
    # plt.plot(mu.detach().numpy(), label="Mean")
    # plt.plot(dx.detach().numpy(), label="Jacobian")
    # plt.plot(dx2.detach().numpy(), label="Hessian")
    # plt.legend()
    # plt.show()

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    # lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*', label='Observed Data')
    ax.plot(train_x.numpy(), train_y_no_noise.numpy(), 'k', label='Original Function')
    # Plot predictive means as blue line
    # ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b', label='Analytical Mean')
    # Shade between the lower and upper confidence bounds
    # ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5, label='Confidence')
    ax.set_ylim([-3, 3])
    # plot linear approximation
    ax.plot([float(point - 0.1), float(point + 0.1)], [float(fun(point) - 0.1 * jac), float(fun(point) + 0.1 * jac)], 'r', label='Linear Approximation of the mean')
    # ax.plot([float(point - 0.1), float(point + 0.1)], [float(fun2(point) - 0.1 * jac2), float(fun2(point) + 0.1 * jac2)], 'r')
    plt.plot(test_x, mu.detach().numpy(), label="Mean")
    # plt.plot(test_x, dx.detach().numpy(), label="Jacobian")
    # plt.plot(test_x, dx2.detach().numpy(), label="Hessian")
    ax.legend()
    # ax.legend(['Observed Data', 'Original Function', 'Mean', 'Confidence', 'Linear Approximation of the mean'])
    plt.show()
