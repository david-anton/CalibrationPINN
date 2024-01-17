import pytest
import torch

from parametricpinn.data.parameterssampling import (
    sample_quasirandom_sobol,
    sample_uniform_grid,
)
from parametricpinn.settings import set_default_dtype
from parametricpinn.types import Tensor

min_parameter_1 = 0.0
max_parameter_1 = 1.0
min_parameter_2 = 0.0
max_parameter_2 = 10.0
min_parameter_3 = 0.0
max_parameter_3 = 100.0

set_default_dtype(torch.float64)
device = torch.device("cpu")


### sample uniform grid


@pytest.mark.parametrize(
    ("num_steps", "expected"),
    [
        ([1], torch.tensor([[min_parameter_1]])),
        ([2], torch.tensor([[min_parameter_1], [max_parameter_1]])),
        (
            [3],
            torch.tensor(
                [
                    [min_parameter_1],
                    [(min_parameter_1 + max_parameter_1) / 2],
                    [
                        max_parameter_1,
                    ],
                ]
            ),
        ),
    ],
)
def test_sample_sample_uniform_grid_1_parameter(
    num_steps: list[int], expected: Tensor
) -> None:
    sut = sample_uniform_grid

    actual = sut(
        min_parameters=[min_parameter_1],
        max_parameters=[max_parameter_1],
        num_steps=num_steps,
        device=device,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("num_steps", "expected"),
    [
        (
            [1, 1],
            torch.tensor([[min_parameter_1, min_parameter_2]]),
        ),
        (
            [1, 2],
            torch.tensor(
                [[min_parameter_1, min_parameter_2], [min_parameter_1, max_parameter_2]]
            ),
        ),
        (
            [2, 2],
            torch.tensor(
                [
                    [min_parameter_1, min_parameter_2],
                    [min_parameter_1, max_parameter_2],
                    [max_parameter_1, min_parameter_2],
                    [max_parameter_1, max_parameter_2],
                ]
            ),
        ),
        (
            [3, 3],
            torch.tensor(
                [
                    [min_parameter_1, min_parameter_2],
                    [min_parameter_1, (min_parameter_2 + max_parameter_2) / 2],
                    [min_parameter_1, max_parameter_2],
                    [(min_parameter_1 + max_parameter_1) / 2, min_parameter_2],
                    [
                        (min_parameter_1 + max_parameter_1) / 2,
                        (min_parameter_2 + max_parameter_2) / 2,
                    ],
                    [(min_parameter_1 + max_parameter_1) / 2, max_parameter_2],
                    [max_parameter_1, min_parameter_2],
                    [max_parameter_1, (min_parameter_2 + max_parameter_2) / 2],
                    [max_parameter_1, max_parameter_2],
                ]
            ),
        ),
    ],
)
def test_sample_sample_uniform_grid_2_parameters(
    num_steps: list[int], expected: Tensor
) -> None:
    sut = sample_uniform_grid

    actual = sut(
        min_parameters=[min_parameter_1, min_parameter_2],
        max_parameters=[max_parameter_1, max_parameter_2],
        num_steps=num_steps,
        device=device,
    )

    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("num_steps", "expected"),
    [
        (
            [1, 1, 1],
            torch.tensor([[min_parameter_1, min_parameter_2, min_parameter_3]]),
        ),
        (
            [2, 2, 2],
            torch.tensor(
                [
                    [min_parameter_1, min_parameter_2, min_parameter_3],
                    [min_parameter_1, min_parameter_2, max_parameter_3],
                    [min_parameter_1, max_parameter_2, min_parameter_3],
                    [min_parameter_1, max_parameter_2, max_parameter_3],
                    [max_parameter_1, min_parameter_2, min_parameter_3],
                    [max_parameter_1, min_parameter_2, max_parameter_3],
                    [max_parameter_1, max_parameter_2, min_parameter_3],
                    [max_parameter_1, max_parameter_2, max_parameter_3],
                ]
            ),
        ),
    ],
)
def test_sample_sample_uniform_grid_3_parameters(
    num_steps: list[int], expected: Tensor
) -> None:
    sut = sample_uniform_grid

    actual = sut(
        min_parameters=[min_parameter_1, min_parameter_2, min_parameter_3],
        max_parameters=[max_parameter_1, max_parameter_2, max_parameter_3],
        num_steps=num_steps,
        device=device,
    )

    torch.testing.assert_close(actual, expected)


### sample quasirandom sobol


def _create_quasirandom_sobol_sampled_parameters(
    min_parameters: list[float], max_parameters: list[float], num_samples: int
) -> Tensor:
    assert len(min_parameters) == len(
        max_parameters
    ), "It is expected that the length of the minimum and maximum parameters is the same."
    num_dimensions = len(min_parameters)
    sobol_engine = torch.quasirandom.SobolEngine(num_dimensions)
    normalized_parameters = (
        sobol_engine.draw(num_samples).to(device).requires_grad_(True)
    )
    parameters = torch.tensor(min_parameters) + normalized_parameters * (
        torch.tensor(max_parameters) - torch.tensor(min_parameters)
    )
    if num_dimensions == 1:
        parameters = parameters.reshape((-1, 1))
    return parameters


@pytest.mark.parametrize(
    ("num_samples"),
    [1, 2, 3],
)
def test_sample_sample_quasirandom_sobol_1_parameter(num_samples: int) -> None:
    min_parameters = [min_parameter_1]
    max_parameters = [max_parameter_1]
    sut = sample_quasirandom_sobol

    actual = sut(
        min_parameters=min_parameters,
        max_parameters=max_parameters,
        num_samples=num_samples,
        device=device,
    )

    expected = _create_quasirandom_sobol_sampled_parameters(
        min_parameters, max_parameters, num_samples
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("num_samples"),
    [1, 2, 4],
)
def test_sample_sample_quasirandom_sobol_2_parameter(num_samples: int) -> None:
    min_parameters = [min_parameter_1, min_parameter_2]
    max_parameters = [max_parameter_1, max_parameter_2]
    sut = sample_quasirandom_sobol

    actual = sut(
        min_parameters=min_parameters,
        max_parameters=max_parameters,
        num_samples=num_samples,
        device=device,
    )

    expected = _create_quasirandom_sobol_sampled_parameters(
        min_parameters, max_parameters, num_samples
    )
    torch.testing.assert_close(actual, expected)


@pytest.mark.parametrize(
    ("num_samples"),
    [1, 3, 9],
)
def test_sample_sample_quasirandom_sobol_3_parameter(num_samples: int) -> None:
    min_parameters = [min_parameter_1, min_parameter_2, min_parameter_3]
    max_parameters = [max_parameter_1, max_parameter_2, max_parameter_3]
    sut = sample_quasirandom_sobol

    actual = sut(
        min_parameters=min_parameters,
        max_parameters=max_parameters,
        num_samples=num_samples,
        device=device,
    )

    expected = _create_quasirandom_sobol_sampled_parameters(
        min_parameters, max_parameters, num_samples
    )
    torch.testing.assert_close(actual, expected)
