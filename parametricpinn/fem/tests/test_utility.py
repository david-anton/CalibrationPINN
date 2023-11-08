import dolfinx
import numpy as np
import pytest
from dolfinx.fem import Function, functionspace
from dolfinx.mesh import create_rectangle
from mpi4py import MPI

from parametricpinn.fem.base import DFunction
from parametricpinn.fem.utility import evaluate_function
from parametricpinn.tests.asserts import assert_numpy_arrays_equal
from parametricpinn.types import NPArray

element_family = "Lagrange"
element_shape = None  # None = (1,)
edge_length = 4.0
half_edge_length = edge_length / 2
quarter_edge_length = edge_length / 4
number_elements = 2
cell_type = dolfinx.mesh.CellType.triangle
communicator = MPI.COMM_WORLD


def solution(x: NPArray) -> float:
    return x[0] ** 2 + x[1] ** 2


def expected_solution(points: NPArray) -> NPArray:
    return points[:, 0] ** 2 + points[:, 1] ** 2


def expected_solution_between_nodes_linear(points: NPArray) -> NPArray:
    return 1 / 2 * ((2 * points[:, 0]) ** 2 + (2 * points[:, 1]) ** 2)



def create_fem_function(element_degree: int) -> DFunction:
    element = (element_family, element_degree, element_shape)
    mesh = create_rectangle(
        comm=communicator,
        points=[
            [-half_edge_length, -half_edge_length],
            [half_edge_length, half_edge_length],
        ],
        n=[number_elements, number_elements],
        cell_type=cell_type,
    )
    function_space = functionspace(mesh, element)
    function = Function(function_space)
    function.interpolate(solution)
    return function



@pytest.fixture
def fem_function_linear() -> DFunction:
    return create_fem_function(element_degree=1)


@pytest.mark.parametrize(
    ("points"),
    [
        np.array([[-half_edge_length, -half_edge_length, 0.0]]),
        np.array([[-half_edge_length, half_edge_length, 0.0]]),
        np.array([[half_edge_length, half_edge_length, 0.0]]),
        np.array([[half_edge_length, -half_edge_length, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
    ],
)
def test_evaluate_function_on_grid_nodes_linear(
    fem_function_linear: DFunction, points: NPArray
) -> None:
    sut = evaluate_function

    actual = sut(function=fem_function_linear, points=points)
    expected = expected_solution(points)

    assert_numpy_arrays_equal(actual, expected)


@pytest.mark.parametrize(
    ("points"),
    [
        np.array([[-quarter_edge_length, -quarter_edge_length, 0.0]]),
        np.array([[-quarter_edge_length, quarter_edge_length, 0.0]]),
        np.array([[quarter_edge_length, quarter_edge_length, 0.0]]),
        np.array([[quarter_edge_length, -quarter_edge_length, 0.0]]),
    ],
)
def test_evaluate_function_between_grid_nodes_linear(
    fem_function_linear: DFunction, points: NPArray
) -> None:
    sut = evaluate_function

    actual = sut(function=fem_function_linear, points=points)
    expected = expected_solution_between_nodes_linear(points)

    assert_numpy_arrays_equal(actual, expected)


@pytest.fixture
def fem_function_quadratic() -> DFunction:
    return create_fem_function(element_degree=2)


@pytest.mark.parametrize(
    ("points"),
    [
        np.array([[-half_edge_length, -half_edge_length, 0.0]]),
        np.array([[-half_edge_length, half_edge_length, 0.0]]),
        np.array([[half_edge_length, half_edge_length, 0.0]]),
        np.array([[half_edge_length, -half_edge_length, 0.0]]),
        np.array([[0.0, 0.0, 0.0]]),
        np.array([[-quarter_edge_length, -quarter_edge_length, 0.0]]),
        np.array([[-quarter_edge_length, quarter_edge_length, 0.0]]),
        np.array([[quarter_edge_length, quarter_edge_length, 0.0]]),
        np.array([[quarter_edge_length, -quarter_edge_length, 0.0]]),
    ],
)
def test_evaluate_function_quadratic(
    fem_function_quadratic: DFunction, points: NPArray
) -> None:
    sut = evaluate_function

    actual = sut(function=fem_function_quadratic, points=points)
    expected = expected_solution(points)

    assert_numpy_arrays_equal(actual, expected)