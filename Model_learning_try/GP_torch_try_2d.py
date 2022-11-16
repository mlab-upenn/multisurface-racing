import math
import torch
import gpytorch
from matplotlib import pyplot as plt
from matplotlib import cm
from torch.autograd.functional import jacobian
import numpy

# %matplotlib inline
# %load_ext autoreload
# %autoreload 2

# Training data is 100 points in [0,1] inclusive regularly spaced
train_x1_s = torch.linspace(0, 1, 100)
train_x2_s = torch.linspace(0, 1, 100)
train_x1 = torch.linspace(0, 1, 10000)
train_x2 = torch.linspace(0, 1, 10000)

for i in range(len(train_x1_s)):
    for j in range(len(train_x2_s)):
        train_x1[i * 100 + j] = train_x1_s[i]
        train_x2[i * 100 + j] = train_x2_s[j]

print("Datagen done")

# train_x = torch.tensor([train_x1, train_x2])
# True function is sin(2*pi*x) with no noise
tensor_2pi = torch.tensor(2 * math.pi)
train_y_no_noise = torch.sin(train_x1 * tensor_2pi) + torch.cos(train_x2 * tensor_2pi)
# True function is sin(2*pi*x) with Gaussian noise
train_y = train_y_no_noise + torch.randn(train_x1.size()) * math.sqrt(0.04)

# print(train_x)
# print(train_y)

train_x = torch.transpose(torch.stack((train_x1, train_x2)), 0, 1)
print(train_x.size())


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


# initialize likelihood and model
likelihood = gpytorch.likelihoods.GaussianLikelihood()
model = ExactGPModel(train_x, train_y, likelihood)

# this is for running the notebook in our testing framework
import os

smoke_test = ('CI' in os.environ)
training_iter = 2 if smoke_test else 2

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

# Evaluation

# Training data is 100 points in [0,1] inclusive regularly spaced
eval_x1_s = torch.linspace(0.05, 0.95, 100)
eval_x2_s = torch.linspace(0.05, 0.95, 100)
eval_x1 = torch.linspace(0, 1, 10000)
eval_x2 = torch.linspace(0, 1, 10000)

for i in range(len(train_x1_s)):
    for j in range(len(train_x2_s)):
        eval_x1[i * 100 + j] = train_x1_s[i]
        eval_x2[i * 100 + j] = train_x2_s[j]

eval_x = torch.transpose(torch.stack((eval_x1, eval_x2)), 0, 1)

# Get into evaluation (predictive posterior) mode
model.eval()
likelihood.eval()

# Test points are regularly spaced along [0,1]
# Make predictions by feeding model through likelihood
with torch.no_grad(), gpytorch.settings.fast_pred_var():
    observed_pred = likelihood(model(eval_x))


def fun(x):
    return likelihood(model(x)).mean


point = torch.tensor([[0.2, 0.2]])
jac = jacobian(fun, point)
print('Jacobian: %s' % jac)

with torch.no_grad():
    fig = plt.figure()

    # =============
    # First subplot
    # =============
    # set up the axes for the first plot
    ax = fig.add_subplot(2, 2, 1, projection='3d')
    ax.title.set_text('Original function')

    surf = ax.scatter(train_x1.numpy(), train_x2.numpy(), train_y_no_noise.numpy())
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # ==============
    # Second subplot
    # ==============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 2, projection='3d')
    ax.title.set_text('Sampled data')
    surf = ax.scatter(train_x1.numpy(), train_x2.numpy(), train_y.numpy())
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # ==============
    # Third subplot
    # ==============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 3, projection='3d')
    ax.title.set_text('Mean of prediction')
    surf = ax.scatter(eval_x1.numpy(), eval_x2.numpy(), observed_pred.mean.numpy())
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    # ==============
    # Forth subplot
    # ==============
    # set up the axes for the second plot
    ax = fig.add_subplot(2, 2, 4, projection='3d')
    ax.title.set_text('Mean of prediction with linearization at point [0.2, 0.8]')
    surf = ax.scatter(eval_x1.numpy(), eval_x2.numpy(), observed_pred.mean.numpy())
    x = [float(point[0][0] + 0.1), float(point[0][0] + 0.1), float(point[0][0] - 0.1), float(point[0][0] - 0.1)]
    y = [float(point[0][1] + 0.1), float(point[0][1] - 0.1), float(point[0][1] + 0.1), float(point[0][1] - 0.1)]
    z = [float(fun(point) + torch.matmul(torch.tensor([0.1, 0.1]), jac.reshape(-1, 1))),
         float(fun(point) + torch.matmul(torch.tensor([0.1, -0.1]), jac.reshape(-1, 1))),
         float(fun(point) + torch.matmul(torch.tensor([-0.1, 0.1]), jac.reshape(-1, 1))),
         float(fun(point) + torch.matmul(torch.tensor([-0.1, -0.1]), jac.reshape(-1, 1)))]
    ax.plot_trisurf(x, y, torch.tensor(z).numpy(), color='r')
    ax.set_zlim(-1.01, 1.01)
    fig.colorbar(surf, shrink=0.5, aspect=10)

    plt.show()
