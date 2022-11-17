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
    covar_x = gp_model.covar_module.forward(x1=x1, x2=x2)
    return covar_x


def k_jac(x1, x2, gp_model):
    jac = jacobian(gp_model.covar_module.forward, (x1, x2))
    return jac[0], jac[1]


def k_hes(x1, x2, gp_model):
    hes = hessian(gp_model.covar_module.forward, (x1, x2))
    return hes[0][1]  # d^2f/dx1dx2


def sigma_sq_n(x, gp_model, gp_likelihood):
    # TODO : USE CONSTANT FROM LIKELIHOOD FUNCTION
    model_res = gp_model(x)
    like = gp_likelihood(model_res)
    return like.stddev ** 2


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


def se_kernel(x1, x2, sigma_sq_f, delta_inv, gp_model=None):
    if gp_model is None:
        return sigma_sq_f * torch.exp(-0.5 * (x1 - x2).T @ delta_inv @ (x1 - x2)).diag()
    else:
        return gp_model.covar_module.forward(x1=x1, x2=x2)

def se_kernel_jac(x1, x2, sigma_sq_f, delta_inv, kxxprime=None):
    X_tilde = x2 - x1

    if kxxprime is None:
        k10 = delta_inv @ (X_tilde * se_kernel(x1, x2, sigma_sq_f, delta_inv))
    else:
        k10 = delta_inv @ (X_tilde * kxxprime)

    return k10, k10.T.clone()

def se_kernel_hess(x1, x2, x1_is_x2, sigma_sq_f, delta_inv, kxxprime=None):
    k11 = None
    if kxxprime is None:
        kxxprime = se_kernel(x1, x2, sigma_sq_f, delta_inv)

    if x1_is_x2 == False:
        X_tilde = x2 - x1

        el2 = X_tilde @ X_tilde.T @ delta_inv
        I   = torch.eye(el2.shape())
        k11 = delta_inv @ (I - el2) @ kxxprime
    else:
        k11 = sigma_sq_f @ delta_inv

    return k11

def find_Vx_closed(x, X, gp_model):
    sigma_sq_f = torch.tensor([[gp_model.covar_module.outputscale.item()]])
    delta_inv = torch.sqrt(gp_model.covar_module.base_kernel.lengthscale)

    print("===============")
    start = time.time()

    kxx = se_kernel(x, x, sigma_sq_f, delta_inv, gp_model)
    K10_xx, K01_xx = se_kernel_jac(x, x, sigma_sq_f, delta_inv, kxx)
    K11_xx = se_kernel_hess(x, x, True, sigma_sq_f, delta_inv, kxx)
    first_term = torch.cat((torch.cat((kxx, K01_xx), dim=1),
                            torch.cat((K10_xx, K11_xx), dim=1)), dim=0)
    end = time.time()
    print("FIRST TERM :: ", end - start)

    print("===============")
    start = time.time()

    kxX = se_kernel(x, X.reshape(-1,1), sigma_sq_f, delta_inv, gp_model)
    K10_xX, _ = se_kernel_jac(x, X, sigma_sq_f, delta_inv, kxX)
    second_term_a = torch.cat((kxX, K10_xX), dim=0)
    second_term_c = second_term_a.T.double()
    second_term = second_term_a @ second_term_b @ second_term_c

    end = time.time()
    print("SECOND TERM :: ", end - start)
    print("===============")

    V_hatx = first_term - second_term
    return V_hatx  # first_term.double()

def precompute(X, gp_model, gp_likelihood):
    sigma_squared = gp_likelihood.noise.item()  # sigma_sq_n(x, gp_model, gp_likelihood)
    KXX = k(X.reshape(-1, 1), X.reshape(-1, 1), gp_model).detach().numpy()
    second_term_b = np.linalg.inv(KXX + sigma_squared * np.eye(KXX.shape[0]))
    return second_term_b


second_term_b = precompute(train_x, model, likelihood)


def find_Vx(x, X, gp_model, gp_likelihood):
    print("===============")
    start = time.time()

    ds = X.reshape(-1, 1)
    kxx = k(x, x, gp_model)
    K10_xx, K01_xx = k_jac(x, x, gp_model)
    K01_xx = K01_xx.reshape(1, x.shape[0])
    K10_xx = K10_xx.reshape(1, x.shape[0])
    K11_xx = k_hes(x, x, gp_model)
    K11_xx = K11_xx.reshape(x.shape[0], x.shape[0])
    first_term = torch.cat((torch.cat((kxx, K01_xx), dim=1),
                            torch.cat((K10_xx, K11_xx), dim=1)), dim=0)
    end = time.time()
    print("FIRST TERM :: ", end - start)

    print("===============")

    print("===============")
    start = time.time()
    kxX = k(x, ds, gp_model)
    K10_xX, _ = k_jac(x, ds, gp_model)  # this line takes a lot of time
    K10_xX = K10_xX.reshape(ds.shape[1], -1)

    # kXx = k(ds, x, gp_model)
    # _, K01_Xx = k_jac(ds, x, gp_model)
    # K01_Xx = K01_Xx.reshape(ds.shape[0], -1)

    end = time.time()

    second_term_a = torch.cat((kxX, K10_xX), dim=0)
    second_term_c = second_term_a.T.double()
    # second_term_c = torch.cat((kXx, K01_Xx), dim=1).double()
    second_term = second_term_a @ second_term_b @ second_term_c

    print(end - start)
    print("===============")

    V_hatx = first_term - second_term
    return V_hatx  # first_term.double()


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


    def fun3(x):
        model_res = model(x)
        like = likelihood(model_res)
        res = like.mean - like.stddev * 2
        return res


    def batch_jacobian(f, x):
        f_sum = lambda x: torch.sum(f(x), axis=0)
        return jacobian(f_sum, x)


    point = torch.tensor([[-0.1]])
    jac = batch_jacobian(fun, point)  # mean
    jac2 = jacobian(fun2, point)  # confidence

    print("Model")
    start = time.time()
    test_x = torch.linspace(-0.2, 1.2, 1000)
    mod = model(test_x)
    end = time.time()
    print(end - start)

    print("Likelihood")
    start = time.time()
    observed_pred = likelihood(model(test_x))
    end = time.time()
    print(end - start)

    print('******************************')
    print("Vhatx")
    start = time.time()
    Vhx = find_Vx(point, train_x, model, likelihood)
    end = time.time()
    print(end - start)
    print('******************************')

    print("Vhatx CLOSED FORM")
    start = time.time()
    Vhx = find_Vx_closed(point, train_x.reshape(1,-1), model)
    end = time.time()
    print(end - start)

with torch.no_grad():
    # Initialize plot
    f, ax = plt.subplots(1, 1, figsize=(4, 3))

    # Get upper and lower confidence bounds
    lower, upper = observed_pred.confidence_region()
    # Plot training data as black stars
    ax.plot(train_x.numpy(), train_y.numpy(), 'k*')
    ax.plot(train_x.numpy(), train_y_no_noise.numpy(), 'k')
    # Plot predictive means as blue line
    ax.plot(test_x.numpy(), observed_pred.mean.numpy(), 'b')
    # Shade between the lower and upper confidence bounds
    ax.fill_between(test_x.numpy(), lower.numpy(), upper.numpy(), alpha=0.5)
    ax.set_ylim([-3, 3])
    # plot linear approximation
    yhat = fun(point)
    ax.plot([float(point - 0.1), float(point + 0.1)], [float(yhat - 0.1 * jac), float(yhat + 0.1 * jac)], 'g')
    plsTD = []
    mnsTD = []
    dts = torch.linspace(float(-0.2), float(1.2), 1000)
    vars = []
    for x in dts:
        deltaX = x - point[0, 0]
        xhat = torch.tensor([[1], [deltaX]], dtype=torch.float64)
        variance = xhat.T @ Vhx @ xhat

        currY = float(yhat + (deltaX) * jac)
        vars.append(variance)
        plsTD.append(currY + 2 * np.sqrt(variance))
        mnsTD.append(currY - 2 * np.sqrt(variance))

    plt.figure()
    plt.plot(dts, vars)
    plt.title("Variance")
    print(dts[np.array(vars).argmin()])
    plt.axvline(point[0].item(), color='yellow')
    ax.plot(dts, plsTD, 'm')
    ax.plot(dts, mnsTD, 'm')

    ax.legend(['Observed Data', 'Original Function', 'Mean', 'Confidence', 'Linear Approximation of the mean'])
    # ax.plot([float(point - 0.1), float(point + 0.1)], [float(fun2(point) - 0.1 * jac2), float(fun2(point) + 0.1 * jac2)], 'r')
    # ax.plot([float(point - 0.1), float(point + 0.1)], [float(fun3(point) - 0.1 * jac2), float(fun3(point) + 0.1 * jac2)], 'r')
    ax.axvline(point[0].item(), color='yellow')
    plt.show()
