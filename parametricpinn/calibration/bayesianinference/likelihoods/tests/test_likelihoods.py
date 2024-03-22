import copy
import math
from typing import Callable, TypeAlias

import pytest
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
    NoiseAndErrorGPsOptimizedLikelihoodStrategy,
    NoiseAndErrorGPsSamplingLikelihoodStrategy,
    NoiseAndErrorOptimizedLikelihoodStrategy,
    NoiseAndErrorSamplingLikelihoodStrategy,
    NoiseLikelihoodStrategy,
    StandardPPINNLikelihood,
    StandardResidualCalculator,
)
from parametricpinn.calibration.data import (
    ConcatenatedCalibrationData,
    PreprocessedCalibrationData,
)
from parametricpinn.errors import TestConfigurationError
from parametricpinn.gps import GP, GaussianProcess, IndependentMultiOutputGP
from parametricpinn.gps.base import GPMultivariateNormal, NamedParameters
from parametricpinn.gps.kernels.base import Kernel
from parametricpinn.gps.means import ZeroMean
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
# 1) The inputs and parameters are selected such that the sum of them minus the true output is 1.
# 2) The standard deviation of the noise is selected such that the variance is 0.5.
# From 1 and 2 follows that the exponent of the exponential function in the normal distribution
# is -1 times the number of flattened outputs.


def test_standard_calibration_likelihood_for_noise_single_data_set_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    num_data_sets = 1
    data_points_per_set = 2
    inputs = (torch.tensor([[1.0], [1.0]]),)
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0], [1.0]]),)
    std_noise = 1 / math.sqrt(2)
    dim_outputs = 1
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), std_noise**2))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set,),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
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
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_multiple_data_sets_single_dimension():
    model = _create_fake_standard_ansatz_single_dimension()
    num_data_sets = 2
    data_points_per_set = 2
    inputs = (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [1.0]]))
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [1.0]]))
    std_noise = 1 / math.sqrt(2)
    dim_oututs = 1
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_oututs * num_total_data_points
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), std_noise**2))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set, data_points_per_set),
        num_total_data_points=num_total_data_points,
        dim_outputs=1,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
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
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_single_data_set_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    num_data_sets = 1
    data_points_per_set = 2
    inputs = (torch.tensor([[0.5, 0.5], [0.5, 0.5]]),)
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0, 1.0], [1.0, 1.0]]),)
    std_noise = 1 / math.sqrt(2)
    dim_outputs = 2
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), std_noise**2))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set,),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
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
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_standard_calibration_likelihood_for_noise_multiple_data_sets_multiple_dimension():
    model = _create_fake_standard_ansatz_multiple_dimension()
    num_data_sets = 2
    data_points_per_set = 2
    inputs = (
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    )
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    )
    std_noise = 1 / math.sqrt(2)
    dim_outputs = 2
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), std_noise**2))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set, data_points_per_set),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
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
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


# Likelihood for noise and model error
# 1) The inputs and parameters are selected such that the sum of them minus the true output is 1.
# 2) The standard deviations of the noise and the model error are selected such that the variance is 0.5.
# From 1 and 2 follows that the exponent of the exponential function in the normal distribution
# is -1 times the number of flattened outputs.


NoiseAndModelErrorLikelihoodFactoryFunc: TypeAlias = Callable[
    [StandardResidualCalculator, Tensor, PreprocessedCalibrationData, int],
    tuple[
        NoiseAndErrorSamplingLikelihoodStrategy
        | NoiseAndErrorOptimizedLikelihoodStrategy,
        Tensor,
    ],
]


def _create_noise_and_model_error_likelihood_strategy_for_sampling(
    residual_calculator: StandardResidualCalculator,
    initial_model_error_standard_deviations: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorSamplingLikelihoodStrategy, Tensor]:
    likelihood_strategy = NoiseAndErrorSamplingLikelihoodStrategy(
        residual_calculator=residual_calculator,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    parameters = torch.concat(
        (torch.tensor([1.0]), initial_model_error_standard_deviations)
    )
    return likelihood_strategy, parameters


def _create_optimized_noise_and_model_error_likelihood_strategy(
    residual_calculator: StandardResidualCalculator,
    initial_model_error_standard_deviations: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorOptimizedLikelihoodStrategy, Tensor]:
    likelihood_strategy = NoiseAndErrorOptimizedLikelihoodStrategy(
        residual_calculator=residual_calculator,
        initial_model_error_standard_deviations=(
            initial_model_error_standard_deviations,
        ),
        use_independent_model_error_standard_deviations=False,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    parameters = torch.tensor([1.0])
    return likelihood_strategy, parameters


def _create_optimized_noise_and_model_error_likelihood_strategy_independent_stddevs(
    residual_calculator: StandardResidualCalculator,
    initial_model_error_standard_deviations: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorOptimizedLikelihoodStrategy, Tensor]:
    likelihood_strategy = NoiseAndErrorOptimizedLikelihoodStrategy(
        residual_calculator=residual_calculator,
        initial_model_error_standard_deviations=tuple(
            copy.deepcopy(initial_model_error_standard_deviations)
            for _ in range(data.num_data_sets)
        ),
        use_independent_model_error_standard_deviations=True,
        data=data,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    parameters = torch.tensor([1.0])
    return likelihood_strategy, parameters


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_likelihood_strategy,
        _create_optimized_noise_and_model_error_likelihood_strategy_independent_stddevs,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_single_data_set_single_dimension(
    likelihood_factory_func: NoiseAndModelErrorLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_single_dimension()
    num_data_sets = 1
    data_points_per_set = 2
    inputs = (torch.tensor([[1.0], [1.0]]),)
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0], [1.0]]),)
    dim_outputs = 1
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    initial_model_error_standard_deviations = torch.tensor([std_model_error])
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set,),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        initial_model_error_standard_deviations,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_likelihood_strategy,
        _create_optimized_noise_and_model_error_likelihood_strategy_independent_stddevs,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_multiple_data_sets_single_dimension(
    likelihood_factory_func: NoiseAndModelErrorLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_single_dimension()
    num_data_sets = 2
    data_points_per_set = 2
    inputs = (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [1.0]]))
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [1.0]]))
    dim_outputs = 1
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    initial_model_error_standard_deviations = torch.tensor([std_model_error])
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set, data_points_per_set),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        initial_model_error_standard_deviations,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_likelihood_strategy,
        _create_optimized_noise_and_model_error_likelihood_strategy_independent_stddevs,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_single_data_set_multiple_dimension(
    likelihood_factory_func: NoiseAndModelErrorLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_multiple_dimension()
    num_data_sets = 1
    data_points_per_set = 2
    inputs = (torch.tensor([[0.5, 0.5], [0.5, 0.5]]),)
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0, 1.0], [1.0, 1.0]]),)
    dim_outputs = 2
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    initial_model_error_standard_deviations = torch.tensor(
        [std_model_error, std_model_error]
    )
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set,),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        initial_model_error_standard_deviations,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_likelihood_strategy,
        _create_optimized_noise_and_model_error_likelihood_strategy_independent_stddevs,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_multiple_data_sets_multiple_dimension(
    likelihood_factory_func: NoiseAndModelErrorLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_multiple_dimension()
    num_data_sets = 2
    data_points_per_set = 2
    inputs = (
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    )
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    )
    dim_outputs = 2
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    initial_model_error_standard_deviations = torch.tensor(
        [std_model_error, std_model_error]
    )
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set, data_points_per_set),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        initial_model_error_standard_deviations,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


# Likelihood for noise and model error GPs
# 1) The inputs and parameters are selected such that the sum of them minus the true output is 1.
# 2) The standard deviations of the noise and the model error are selected such that the variance is 0.5.
# From 1 and 2 follows that the exponent of the exponential function in the normal distribution
# is -1 times the number of flattened outputs.


NoiseAndModelErrorGPsLikelihoodFactoryFunc: TypeAlias = Callable[
    [
        StandardResidualCalculator,
        GaussianProcess,
        Tensor,
        PreprocessedCalibrationData,
        int,
    ],
    tuple[
        NoiseAndErrorGPsSamplingLikelihoodStrategy
        | NoiseAndErrorGPsOptimizedLikelihoodStrategy,
        Tensor,
    ],
]


class FakeKernel(Kernel):
    def __init__(self, variance_error) -> None:
        super().__init__(num_hyperparameters=2, device=device)
        self._variance_error = variance_error

    def forward(self, x_1: Tensor, x_2: Tensor, **params) -> Tensor:
        if x_1.size() != x_2.size():
            raise TestConfigurationError(
                "Fake GP kernel only defined for inputs with the same shape."
            )
        num_inputs = x_1.size()[0]
        return self._variance_error * torch.eye(num_inputs)

    def set_parameters(self, parameters: Tensor) -> None:
        pass

    def get_named_parameters(self) -> NamedParameters:
        return {
            "kernel_parameetr_1": torch.tensor(0.0),
            "kernel_parameetr_2": torch.tensor(0.0),
        }


class FakeZeroMeanScaledRBFKernelGP(GP):
    def __init__(self, variance_error: float) -> None:
        fake_mean_module = ZeroMean(device)
        fake_kernel_module = FakeKernel(variance_error)
        super().__init__(
            mean=fake_mean_module, kernel=fake_kernel_module, device=device
        )
        self.num_gps = 1
        self.num_hyperparameters = 2

    def set_parameters(self, parameters: Tensor) -> None:
        pass

    def get_named_parameters(self) -> NamedParameters:
        return {"parameetr": torch.tensor(0.0)}


def _create_noise_and_model_error_gps_likelihood_strategy_for_sampling(
    residual_calculator: StandardResidualCalculator,
    model_error_gp: GaussianProcess,
    initial_gp_parameters: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorGPsSamplingLikelihoodStrategy, Tensor]:
    likelihood_strategy = NoiseAndErrorGPsSamplingLikelihoodStrategy(
        model_error_gp=model_error_gp,
        data=data,
        residual_calculator=residual_calculator,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    parameters = torch.concat((torch.tensor([1.0]), initial_gp_parameters))
    return likelihood_strategy, parameters


def _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training(
    residual_calculator: StandardResidualCalculator,
    model_error_gp: GaussianProcess,
    initial_gp_parameters: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorGPsOptimizedLikelihoodStrategy, Tensor]:
    model_error_gp.set_parameters(initial_gp_parameters)
    likelihood_strategy = NoiseAndErrorGPsOptimizedLikelihoodStrategy(
        model_error_gps=(model_error_gp,),
        use_independent_model_error_gps=False,
        data=data,
        residual_calculator=residual_calculator,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    parameters = torch.tensor([1.0])
    return likelihood_strategy, parameters


def _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction(
    residual_calculator: StandardResidualCalculator,
    model_error_gp: GaussianProcess,
    initial_gp_parameters: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorGPsOptimizedLikelihoodStrategy, Tensor]:
    model_error_gp.set_parameters(initial_gp_parameters)
    likelihood_strategy = NoiseAndErrorGPsOptimizedLikelihoodStrategy(
        model_error_gps=(model_error_gp,),
        use_independent_model_error_gps=False,
        data=data,
        residual_calculator=residual_calculator,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    likelihood_strategy.prediction_mode()
    parameters = torch.tensor([1.0])
    return likelihood_strategy, parameters


def _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training_independent_gps(
    residual_calculator: StandardResidualCalculator,
    model_error_gp: GaussianProcess,
    initial_gp_parameters: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorGPsOptimizedLikelihoodStrategy, Tensor]:
    model_error_gp.set_parameters(initial_gp_parameters)
    likelihood_strategy = NoiseAndErrorGPsOptimizedLikelihoodStrategy(
        model_error_gps=tuple(
            copy.deepcopy(model_error_gp) for _ in range(data.num_data_sets)
        ),
        use_independent_model_error_gps=True,
        data=data,
        residual_calculator=residual_calculator,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    parameters = torch.tensor([1.0])
    return likelihood_strategy, parameters


def _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction_independent_gps(
    residual_calculator: StandardResidualCalculator,
    model_error_gp: GaussianProcess,
    initial_gp_parameters: Tensor,
    data: PreprocessedCalibrationData,
    num_model_parameters: int,
) -> tuple[NoiseAndErrorGPsOptimizedLikelihoodStrategy, Tensor]:
    model_error_gp.set_parameters(initial_gp_parameters)
    likelihood_strategy = NoiseAndErrorGPsOptimizedLikelihoodStrategy(
        model_error_gps=tuple(
            copy.deepcopy(model_error_gp) for _ in range(data.num_data_sets)
        ),
        use_independent_model_error_gps=True,
        data=data,
        residual_calculator=residual_calculator,
        num_model_parameters=num_model_parameters,
        device=device,
    )
    likelihood_strategy.prediction_mode()
    parameters = torch.tensor([1.0])
    return likelihood_strategy, parameters


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_gps_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training_independent_gps,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction_independent_gps,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_gps_single_data_set_single_dimension(
    likelihood_factory_func: NoiseAndModelErrorGPsLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_single_dimension()
    num_data_sets = 1
    data_points_per_set = 2
    inputs = (torch.tensor([[1.0], [1.0]]),)
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0], [1.0]]),)
    dim_outputs = 1
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set,),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    initial_gp_parameters = torch.tensor([0.0, 0.0])
    model_error_gp = FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error)
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        model_error_gp,
        initial_gp_parameters,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_gps_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training_independent_gps,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction_independent_gps,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_gps_multiple_data_sets_single_dimension(
    likelihood_factory_func: NoiseAndModelErrorGPsLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_single_dimension()
    num_data_sets = 2
    data_points_per_set = 2
    inputs = (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [1.0]]))
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0], [1.0]]), torch.tensor([[1.0], [1.0]]))
    dim_outputs = 1
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set, data_points_per_set),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    initial_gp_parameters = torch.tensor([0.0, 0.0])
    model_error_gp = FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error)
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        model_error_gp,
        initial_gp_parameters,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_gps_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training_independent_gps,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction_independent_gps,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_gps_single_data_set_multiple_dimension(
    likelihood_factory_func: NoiseAndModelErrorGPsLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_multiple_dimension()
    num_data_sets = 1
    data_points_per_set = 2
    inputs = (torch.tensor([[0.5, 0.5], [0.5, 0.5]]),)
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (torch.tensor([[1.0, 1.0], [1.0, 1.0]]),)
    dim_outputs = 2
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set,),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    initial_gp_parameters = torch.tensor([0.0, 0.0, 0.0, 0.0])
    model_error_gp = IndependentMultiOutputGP(
        [
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
        ],
        device=device,
    )
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        model_error_gp,
        initial_gp_parameters,
        data,
        num_model_parameters,
    )

    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("likelihood_factory_func"),
    [
        _create_noise_and_model_error_gps_likelihood_strategy_for_sampling,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_training_independent_gps,
        _create_optimized_noise_and_model_error_gps_likelihood_strategy_for_prediction_independent_gps,
    ],
)
def test_standard_calibration_likelihood_for_noise_and_model_error_gps_multiple_data_sets_multiple_dimension(
    likelihood_factory_func: NoiseAndModelErrorGPsLikelihoodFactoryFunc,
):
    model = _create_fake_standard_ansatz_multiple_dimension()
    num_data_sets = 2
    data_points_per_set = 2
    inputs = (
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
        torch.tensor([[0.5, 0.5], [0.5, 0.5]]),
    )
    num_model_parameters = 1
    parameters = torch.tensor([1.0])
    outputs = (
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
        torch.tensor([[1.0, 1.0], [1.0, 1.0]]),
    )
    dim_outputs = 2
    num_total_data_points = num_data_sets * data_points_per_set
    num_flattened_outputs = dim_outputs * num_total_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    std_model_error = 1 / math.sqrt(4)
    variance_model_error = std_model_error**2
    total_variance = variance_noise + variance_model_error
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = PreprocessedCalibrationData(
        num_data_sets=num_data_sets,
        inputs=inputs,
        outputs=outputs,
        std_noise=std_noise,
        num_data_points_per_set=(data_points_per_set, data_points_per_set),
        num_total_data_points=num_total_data_points,
        dim_outputs=dim_outputs,
    )
    residual_calculator = StandardResidualCalculator(model=model, device=device)
    initial_gp_parameters = torch.tensor([0.0, 0.0, 0.0, 0.0])
    model_error_gp = IndependentMultiOutputGP(
        [
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
            FakeZeroMeanScaledRBFKernelGP(variance_error=variance_model_error),
        ],
        device=device,
    )
    likelihood_strategy, parameters = likelihood_factory_func(
        residual_calculator,
        model_error_gp,
        initial_gp_parameters,
        data,
        num_model_parameters,
    )
    sut = StandardPPINNLikelihood(
        likelihood_strategy=likelihood_strategy,
        device=device,
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


# Likelihood for Bayesian PPINNs
# 1) The inputs and parameters are selected such that the sum of them minus the true output is 1.
# 2) The standard deviations of the noise and the error of the predicted output (predicted by the Baxesian PPINN)
# are selected such that the variance is 0.5.
# From 1 and 2 follows that the exponent of the exponential function in the normal distribution
# is -1 times the number of flattened outputs.


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
        return torch.tensor(0.0)

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
        return torch.tensor(0.0)

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
    std_bppinn = 1 / math.sqrt(4)
    variance_bppinn = std_bppinn**2
    model = _create_fake_bayesian_ansatz_single_dimension(
        variance_output=variance_bppinn
    )
    num_data_points = 1
    inputs = torch.tensor([[1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0]])
    dim_outputs = 1
    num_flattened_outputs = dim_outputs * num_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_variance = variance_noise + variance_bppinn
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = ConcatenatedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
        std_noise=std_noise,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_bayesian_calibration_likelihood_for_noise_multiple_data_single_dimension():
    std_bppinn = 1 / math.sqrt(4)
    variance_bppinn = std_bppinn**2
    model = _create_fake_bayesian_ansatz_single_dimension(
        variance_output=variance_bppinn
    )
    num_data_points = 2
    inputs = torch.tensor([[1.0], [1.0]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0], [1.0]])
    dim_outputs = 1
    num_flattened_outputs = dim_outputs * num_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_variance = variance_noise + variance_bppinn
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = ConcatenatedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
        std_noise=std_noise,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_bayesian_calibration_likelihood_for_noise_single_data_multiple_dimension():
    std_bppinn = 1 / math.sqrt(4)
    variance_bppinn = std_bppinn**2
    model = _create_fake_bayesian_ansatz_multiple_dimension(
        variance_output=variance_bppinn
    )
    num_data_points = 1
    inputs = torch.tensor([[0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0]])
    dim_outputs = 2
    num_flattened_outputs = dim_outputs * num_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_variance = variance_noise + variance_bppinn
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = ConcatenatedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
        std_noise=std_noise,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)


def test_bayesian_calibration_likelihood_for_noise_multiple_data_multiple_dimension():
    std_bppinn = 1 / math.sqrt(4)
    variance_bppinn = std_bppinn**2
    model = _create_fake_bayesian_ansatz_multiple_dimension(
        variance_output=variance_bppinn
    )
    num_data_points = 2
    inputs = torch.tensor([[0.5, 0.5], [0.5, 0.5]])
    parameters = torch.tensor([1.0])
    outputs = torch.tensor([[1.0, 1.0], [1.0, 1.0]])
    dim_outputs = 2
    num_flattened_outputs = dim_outputs * num_data_points
    std_noise = 1 / math.sqrt(4)
    variance_noise = std_noise**2
    total_variance = variance_noise + variance_bppinn
    covariance_matrix = torch.diag(torch.full((num_flattened_outputs,), total_variance))
    data = ConcatenatedCalibrationData(
        inputs=inputs,
        outputs=outputs,
        num_data_points=num_data_points,
        dim_outputs=dim_outputs,
        std_noise=std_noise,
    )
    sut = BayesianPPINNLikelihood(
        model=model, model_parameter_samples=torch.tensor([]), data=data, device=device
    )

    actual = sut.prob(parameters)

    expected = (
        1
        / torch.sqrt(
            (
                torch.tensor(2 * math.pi) ** num_flattened_outputs
                * torch.det(covariance_matrix)
            )
        )
        * torch.exp(torch.tensor(-num_flattened_outputs))
    ).type(torch.float64)
    torch.testing.assert_close(actual, expected)
