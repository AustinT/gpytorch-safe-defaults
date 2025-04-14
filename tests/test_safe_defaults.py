import gpytorch
import linear_operator
import torch

from actually_exact_gpytorch import exact_gpytorch


class ExactGPModel(gpytorch.models.ExactGP):
    def __init__(self, train_x, train_y, likelihood):
        super().__init__(train_x, train_y, likelihood)
        self.mean_module = gpytorch.means.ConstantMean()
        self.covar_module = gpytorch.kernels.ScaleKernel(gpytorch.kernels.RBFKernel())

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return gpytorch.distributions.MultivariateNormal(mean_x, covar_x)


def test_exact_gpytorch_context():
    # Create some toy data
    train_x = torch.randn(10, 2)
    train_y = torch.randn(10)
    likelihood = gpytorch.likelihoods.GaussianLikelihood()

    # Create model without safe defaults
    _ = ExactGPModel(train_x, train_y, likelihood)

    # Create model with safe defaults
    with exact_gpytorch:
        _ = ExactGPModel(train_x, train_y, likelihood)

    # Check that the settings are different
    assert linear_operator.settings._fast_covar_root_decomposition._default is True
    assert linear_operator.settings._fast_log_prob._default is True
    assert linear_operator.settings._fast_solves._default is True


def test_exact_gpytorch_function():
    # Set safe defaults globally
    exact_gpytorch()

    # Check that the settings are set to False
    assert linear_operator.settings._fast_covar_root_decomposition._default is False
    assert linear_operator.settings._fast_log_prob._default is False
    assert linear_operator.settings._fast_solves._default is False
