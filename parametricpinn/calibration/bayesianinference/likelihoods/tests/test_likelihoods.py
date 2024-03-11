import math

import gpytorch
import torch

from parametricpinn.ansatz.base import (
    AnsatzStrategy,
    BayesianAnsatz,
    BayesianNetworks,
    Networks,
    StandardAnsatz,
    StandardNetworks,
)
from parametricpinn.calibration.bayesianinference.likelihoods.likelihoods import (
    BayesianPPINNLikelihood,
    NoiseAndModelErrorGPsLikelihoodStrategy,
    NoiseAndModelErrorLikelihoodStrategy,
    NoiseLikelihoodStrategy,
    StandardPPINNLikelihood,
    StandardResidualCalculator,
)
from parametricpinn.calibration.data import PreprocessedCalibrationData
from parametricpinn.errors import TestConfigurationError
from parametricpinn.gps import IndependentMultiOutputGP, ZeroMeanScaledRBFKernelGP
from parametricpinn.network import BFFNN, FFNN
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import Tensor

set_default_dtype(torch.float64)

device = torch.device("cpu")


### Standard PPINNs


class FakeAnsatzStrategy(AnsatzStrategy):
    def __call__(self, input: Tensor, network: Networks) -> Tensor:
        return torch.zeros((1,))


class FakeStandardAnsatz_SingleDimension(StandardAnsatz):
    def __init__(
        self, network: StandardNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__(network, ansatz_strategy)

    def forward(self, input: Tensor) -> Tensor:
        return torch.sum(input, dim=1, keepdim=True)


class FakeStandardAnsatz_MultipleDimension(StandardAnsatz):
    def __init__(
        self, network: StandardNetworks, ansatz_strategy: AnsatzStrategy
    ) -> None:
        super().__init__(network, ansatz_strategy)

    def forward(self, input: Tensor) -> Tensor:
        return torch.concat(
            (
                torch.sum(input, dim=1, keepdim=True),
                torch.sum(input, dim=1, keepdim=True),
            ),
            dim=1,
        )


def _create_fake_standard_ansatz_single_dimension() -> StandardAnsatz:
    fake_network = FFNN(layer_sizes=[1, 1])
    fake_ansatz_strategy = FakeAnsatzStrategy()
    return FakeStandardAnsatz_SingleDimension(fake_network, fake_ansatz_strategy)


def _create_fake_standard_ansatz_multiple_dimension() -> StandardAnsatz:
    fake_network = FFNN(layer_sizes=[1, 1])
    fake_ansatz_strategy = FakeAnsatzStrategy()
    return FakeStandardAnsatz_MultipleDimension(fake_network, fake_ansatz_strategy)


# Likelihood for noise
# The standard deviation of the noise is selected so that the variance is 0.5.
# For a variance 0f 0.5, the exponent of the exponential function in the normal distribution
# is -1 times the number of flattened outputs.


def test_standard_calibration_likelihood_for_noise_single_data_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0]])
    std_noise = 1 / math.sqrt(2)
    variance_error = std_noise**2
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (1 / torch.sqrt(2 * torch.tensor(math.pi) * variance_error)) * torch.pow(
        torch.tensor(math.e), -1
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_multiple_data_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((2,), std_noise**2))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_single_data_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5]])
    parameters = torch.tensor([1.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0, 1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((2,), std_noise**2))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=2,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(covariance_error)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_multiple_data_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    parameters = torch.tensor([1.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    std_noise = 1 / math.sqrt(2)
    covariance_error = torch.diag(torch.full((4,), std_noise**2))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=2,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (
            torch.pow((2 * torch.tensor(math.pi)), 2)
            * torch.sqrt(torch.det(covariance_error))
        )
        * torch.pow(torch.tensor(math.e), -4)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


# Likelihood for noise and model error
# The variance of the noise and the model error is selected to be 0.25 so that the total variance is 0.5.
# For a variance of 0.5, the exponent of the exponential function in the normal distribution is
# # is -1 times the number of flattened outputs.


def test_standard_calibration_likelihood_for_noise_and_model_error_single_data_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    inputs = torch.tensor([[1.0]])
    outputs = torch.tensor([[1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    parameters = torch.tensor([1.0, std_model_error])
    num_model_parameters = 1
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseAndModelErrorLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (1 / torch.sqrt(2 * torch.tensor(math.pi) * total_variance)) * torch.pow(
        torch.tensor(math.e), -1
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_and_model_error_multiple_data_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    inputs = torch.tensor([[1.0], [1.0]])
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_covar_matrix = torch.diag(
        torch.full((2,), variance_noise + variance_model_error)
    )
    parameters = torch.tensor([1.0, std_model_error])
    num_model_parameters = 1
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseAndModelErrorLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(total_covar_matrix)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_and_model_error_single_data_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5]])
    outputs = torch.tensor([[1.0, 1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_covar_matrix = torch.diag(
        torch.full((2,), variance_noise + variance_model_error)
    )
    parameters = torch.tensor([1.0, std_model_error, std_model_error])
    num_model_parameters = 1
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=2,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseAndModelErrorLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(total_covar_matrix)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_and_model_error_multiple_data_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_covar_matrix = torch.diag(
        torch.full((4,), variance_noise + variance_model_error)
    )
    parameters = torch.tensor([1.0, std_model_error, std_model_error])
    num_model_parameters = 1
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=2,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    likelihood_strategy = NoiseAndModelErrorLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (
            torch.pow((2 * torch.tensor(math.pi)), 2)
            * torch.sqrt(torch.det(total_covar_matrix))
        )
        * torch.pow(torch.tensor(math.e), -4)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


# Likelihood for noise and model error GPs
# The variance of the noise and the model error is selected to be 0.25 so that the total variance is 0.5.
# For a variance of 0.5, the exponent of the exponential function in the normal distribution is
# # is -1 times the number of flattened outputs.


class FakeLazyTensor(torch.Tensor):
    def __init__(self, torch_tensor: Tensor) -> None:
        self._torch_tensor = torch_tensor

    def to_dense(self, dtype=None, *, masked_grad=True) -> Tensor:
        return self._torch_tensor


class FakeBaseKernel(gpytorch.kernels.Kernel):
    has_lengthscale = True


class FakeKernel(gpytorch.kernels.Kernel):
    def __init__(self, variance_error) -> None:
        super(FakeKernel, self).__init__()
        self._variance_error = variance_error
        self.base_kernel = FakeBaseKernel()

    def forward(self, x_1: Tensor, x_2: Tensor, **params) -> FakeLazyTensor:
        if x_1.size() != x_2.size():
            raise TestConfigurationError(
                "Fake GP kernel only defined for inputs with the same shape."
            )
        num_inputs = x_1.size()[0]
        return FakeLazyTensor(self._variance_error * torch.eye(num_inputs))


class FakeZeroMeanScaledRBFKernelGP(ZeroMeanScaledRBFKernelGP):
    def __init__(self, variance_error: float, train_x=None, train_y=None) -> None:
        likelihood = gpytorch.likelihoods.GaussianLikelihood()
        super(FakeZeroMeanScaledRBFKernelGP, self).__init__(
            train_x, train_y, likelihood
        )
        self._variance_error = variance_error
        self.kernel = FakeKernel(variance_error)
        self.num_gps = 1
        self.num_hyperparameters = 2

    def forward(self) -> None:
        pass

    def forward_kernel(self, x_1: Tensor, x_2: Tensor) -> Tensor:
        if x_1.size() != x_2.size():
            raise TestConfigurationError(
                "Fake GP only defined for inputs with the same shape."
            )
        num_inputs = x_1.size()[0]
        return self._variance_error * torch.eye(num_inputs)

    def set_covariance_parameters(self, parameters: Tensor) -> None:
        pass

    def get_uninformed_covariance_parameters_prior(self) -> None:
        pass


def test_standard_calibration_likelihood_for_noise_and_model_error_gps_single_data_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0, 0.0, 0.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    variance_model_error = 0.25
    total_variance = variance_noise + variance_model_error
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    model_error_gp = FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error)
    likelihood_strategy = NoiseAndModelErrorGPsLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (1 / torch.sqrt(2 * torch.tensor(math.pi) * total_variance)) * torch.pow(
        torch.tensor(math.e), -1
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_and_model_error_gps_multiple_data_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0, 0.0, 0.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    variance_model_error = 0.25
    total_covar_matrix = torch.diag(
        torch.full((2,), variance_noise + variance_model_error)
    )
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    model_error_gp = FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error)
    likelihood_strategy = NoiseAndModelErrorGPsLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(total_covar_matrix)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_and_model_error_gps_single_data_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5]])
    parameters = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0, 1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    variance_model_error = 0.25
    total_covar_matrix = torch.diag(
        torch.full((2,), variance_noise + variance_model_error)
    )
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=2,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    model_error_gp = IndependentMultiOutputGP(
        [
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
        ],
        device=device,
    )

    likelihood_strategy = NoiseAndModelErrorGPsLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(total_covar_matrix)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_and_model_error_gps_multiple_data_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    parameters = torch.tensor([1.0, 0.0, 0.0, 0.0, 0.0])
    num_model_parameters = 1
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    variance_model_error = 0.25
    total_covar_matrix = torch.diag(
        torch.full((4,), variance_noise + variance_model_error)
    )
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=2,
    )
    residual_calculator = StandardResidualCalculator(
        model=model, data=data, device=device
    )
    model_error_gp = IndependentMultiOutputGP(
        [
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
        ],
        device=device,
    )
    likelihood_strategy = NoiseAndModelErrorGPsLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        model_error_gp=model_error_gp,
        device=device,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (
            torch.pow((2 * torch.tensor(math.pi)), 2)
            * torch.sqrt(torch.det(total_covar_matrix))
        )
        * torch.pow(torch.tensor(math.e), -4)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


### Bayesian PPINNs
# The variance of the noise and the predicted output (by the BFFNN) is selected to be 0.25
# so that the total variance is 0.5. For a variance of 0.5, the exponent of the exponential function
# in the normal distribution is is -1 times the number of flattened outputs.


class FakeBayesianAnsatz_SingleDimension(BayesianAnsatz):
    def __init__(
        self,
        network: BayesianNetworks,
        ansatz_strategy: AnsatzStrategy,
        variance_output: float,
    ) -> None:
        super().__init__(network, ansatz_strategy)
        self._variance_output = variance_output

    def forward(self, input: Tensor) -> Tensor:
        pass

    def predict_normal_distribution(
        self, input: Tensor, parameter_samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        means = torch.sum(input, dim=1, keepdim=True)
        stddevs = torch.full_like(means, math.sqrt(self._variance_output))
        return means, stddevs


class FakeBayesianAnsatz_MultipleDimension(BayesianAnsatz):
    def __init__(
        self,
        network: BayesianNetworks,
        ansatz_strategy: AnsatzStrategy,
        variance_output: float,
    ) -> None:
        super().__init__(network, ansatz_strategy)
        self._variance_output = variance_output

    def forward(self, input: Tensor) -> Tensor:
        pass

    def predict_normal_distribution(
        self, input: Tensor, parameter_samples: Tensor
    ) -> tuple[Tensor, Tensor]:
        means = torch.concat(
            (
                torch.sum(input, dim=1, keepdim=True),
                torch.sum(input, dim=1, keepdim=True),
            ),
            dim=1,
        )
        stddevs = torch.full_like(means, math.sqrt(self._variance_output))
        return means, stddevs


def _create_fake_bayesian_ansatz_single_dimension(
    variance_output: float,
) -> BayesianAnsatz:
    fake_network = BFFNN(layer_sizes=[1, 1])
    fake_ansatz_strategy = FakeAnsatzStrategy()
    return FakeBayesianAnsatz_SingleDimension(
        fake_network, fake_ansatz_strategy, variance_output
    )


def _create_fake_bayesian_ansatz_multiple_dimension(
    variance_output: float,
) -> BayesianAnsatz:
    fake_network = BFFNN(layer_sizes=[1, 1])
    fake_ansatz_strategy = FakeAnsatzStrategy()
    return FakeBayesianAnsatz_MultipleDimension(
        fake_network, fake_ansatz_strategy, variance_output
    )


def test_bayesian_calibration_likelihood_for_noise_single_data_single_dimension():
    variance_bpinn = 0.25
    model = _create_fake_bayesian_ansatz_single_dimension(
        variance_output=variance_bpinn
    )
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_variance = variance_noise + variance_bpinn
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=1,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (1 / torch.sqrt(2 * torch.tensor(math.pi) * total_variance)) * torch.pow(
        torch.tensor(math.e), -1
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_bayesian_calibration_likelihood_for_noise_multiple_data_single_dimension():
    variance_bpinn = 0.25
    model = _create_fake_bayesian_ansatz_single_dimension(
        variance_output=variance_bpinn
    )
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0], [1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_covar_matrix = torch.diag(torch.full((2,), variance_noise + variance_bpinn))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=1,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(total_covar_matrix)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_bayesian_calibration_likelihood_for_noise_single_data_multiple_dimension():
    variance_bpinn = 0.25
    model = _create_fake_bayesian_ansatz_multiple_dimension(
        variance_output=variance_bpinn
    )
    inputs = torch.tensor([[0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_covar_matrix = torch.diag(torch.full((2,), variance_noise + variance_bpinn))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=1,
        dim_outputs=2,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (2 * torch.tensor(math.pi) * torch.sqrt(torch.det(total_covar_matrix)))
        * torch.pow(torch.tensor(math.e), -2)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_bayesian_calibration_likelihood_for_noise_multiple_data_multiple_dimension():
    variance_bpinn = 0.25
    model = _create_fake_bayesian_ansatz_multiple_dimension(
        variance_output=variance_bpinn
    )
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_covar_matrix = torch.diag(torch.full((4,), variance_noise + variance_bpinn))
    data = PreprocessedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points=2,
        dim_outputs=2,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / (
            torch.pow((2 * torch.tensor(math.pi)), 2)
            * torch.sqrt(torch.det(total_covar_matrix))
        )
        * torch.pow(torch.tensor(math.e), -4)
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)
