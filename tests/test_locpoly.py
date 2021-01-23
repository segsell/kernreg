"""Test cases for the locpoly module."""
from typing import Any, Callable, Tuple, Union

import numpy as np
import pandas as pd
import pytest

from kernreg.locpoly import (
    combine_bincounts_kernelweights,
    get_curve_estimator,
    get_kernelweights,
    is_sorted,
    locpoly,
)
from kernreg.utils import sort_by_x


@pytest.fixture
def array_sorted() -> np.ndarray:
    """Input array sorted ascendingly."""
    return np.linspace(-2, 2, 1000)


@pytest.fixture
def array_unsorted() -> np.ndarray:
    """Unsorted input array."""
    return np.array([-1, 2, 3, 2, 1])


@pytest.fixture
def array_sort_descend() -> np.ndarray:
    """Input array sorted descendingly."""
    return np.linspace(2, -2, 1000)


@pytest.fixture
def output_positive_counts() -> Tuple[np.ndarray, np.ndarray]:
    """Expected toy output for weighted x and y."""
    x = np.array([9.63, 12.63, 13.23, 13.23, 13.23, 13.23, 13.23, 13.2, 12.6, 9.6])
    y = np.array([16.05, 21.05, 22.05, 22.05, 22.05, 22.05, 22.05, 22, 21, 16])

    return x, y


@pytest.fixture
def output_zero_count() -> Tuple[np.ndarray, np.ndarray]:
    """Expected toy output for weighted x and y, where xcounts contains a zero."""
    x = np.array(
        [
            [9.63000000e00, 4.76666667e-01, 7.00000000e-02],
            [1.26300000e01, 1.43333333e-01, 1.07037037e-01],
            [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
            [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
            [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
            [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
            [1.32000000e01, 0.00000000e00, 1.33333333e-01],
            [1.26000000e01, -1.33333333e-01, 1.03703704e-01],
            [9.60000000e00, -4.66666667e-01, 6.66666667e-02],
            [3.60000000e00, -4.66666667e-01, 6.66666667e-02],
        ]
    )
    y = np.array(
        [
            [1.60500000e01, 7.94444444e-01],
            [2.10500000e01, 2.38888889e-01],
            [2.20500000e01, 1.66666667e-02],
            [2.20500000e01, 1.66666667e-02],
            [2.20500000e01, 1.66666667e-02],
            [2.20500000e01, 1.66666667e-02],
            [2.20000000e01, 0.00000000e00],
            [2.10000000e01, -2.22222222e-01],
            [1.60000000e01, -7.77777778e-01],
            [6.00000000e00, -7.77777778e-01],
        ]
    )

    return x, y


@pytest.fixture
def output_integration_default() -> np.ndarray:
    """Output data for default arguments of gridsize, binned, truncate."""
    return np.genfromtxt("tests/resources/toy_expect_default.csv")


@pytest.fixture
def output_integration_false_true() -> np.ndarray:
    """Output array for binned=False and truncate=True."""
    arr = np.array(
        [
            -8.21970601,
            -8.78389651,
            -9.30816104,
            -9.79135319,
            -10.23230278,
            -10.62981337,
            -10.9826599,
            -11.28958658,
            -11.54930484,
            -11.76049163,
            -11.92178784,
        ]
    )
    return arr


@pytest.fixture
def output_integration_true_true() -> np.ndarray:
    """Output array for binned=True and truncate=True."""
    arr = np.array(
        [
            -3.0005839,
            -2.98618264,
            -2.97177475,
            -2.95736025,
            -2.94293914,
            -2.92851141,
            -2.91407707,
            -2.89963612,
            -2.88518855,
            -2.87073435,
            -2.85627352,
        ]
    )
    return arr


@pytest.fixture
def output_integration_false_false() -> np.ndarray:
    """Output array for binned=False and truncate=False."""
    arr = np.array(
        [
            -3.38271026,
            -4.5850677,
            -5.74380798,
            -6.85848024,
            -7.92856605,
            -8.9534749,
            -9.93253988,
            -10.86501327,
            -11.75006237,
            -12.58676545,
            -13.37410782,
        ]
    )
    return arr


@pytest.mark.parametrize("arr", ["array_unsorted", "array_sort_descend"])
def test_input_arr_not_sorted_ascend(arr: Callable, request: Any) -> None:
    """It raises an Exception if unsorted x-array is put into locpoly."""
    x = request.getfixturevalue(arr)
    y = np.linspace(3, -20, 1000)

    msg = "Input arrays x and y must be sorted by x before estimation!"

    with pytest.raises(Exception) as error:
        assert locpoly(
            x,
            y,
            derivative=0,
            degree=1,
            gridsize=11,
            bandwidth=2,
            a=0,
            b=2,
            truncate=False,
        )
    assert str(error.value) == msg


@pytest.mark.parametrize(
    "arr, expected",
    [("array_sorted", True), ("array_unsorted", False)],
)
def test_arr_not_sorted_ascend(arr: Callable, expected: bool, request: Any) -> None:
    """It exits with a zero status if array is sorted ascendingly."""
    assert is_sorted(request.getfixturevalue(arr)) is expected


@pytest.mark.parametrize(
    "degree, count, expected",
    [
        (0, 6, "output_positive_counts"),
        (1, 0, "output_zero_count"),
    ],
)
def test_combine_weights_degree_zero(
    degree: int, count: int, expected: Callable, request: Any
) -> None:
    """Combines bincounts and weights where degree of polynomial is zero."""
    # if degree = 1, no ravel needed --> weightedx multidimensional
    # check regression/integration test if array (shape) handling still works
    # if degree 0 and weightedx of the form [[1], [2], [3] ]

    bandwidth = 0.1
    a, b, gridsize = 0, 1, 10
    binwidth = (b - a) / (gridsize - 1)

    xcounts = np.asarray(9 * [6] + [count])
    ycounts = np.asarray(10 * [10])

    symmetric_weights = [0.005, 0.1, 0.5]
    weights = np.asarray(symmetric_weights + [1] + symmetric_weights[::-1])

    weightedx, weightedy = combine_bincounts_kernelweights(
        xcounts, ycounts, weights, degree, gridsize, bandwidth, binwidth
    )

    if degree == 0:
        weightedx, weightedy = weightedx.ravel(), weightedy.ravel()

    expected_x, expected_y = request.getfixturevalue(expected)

    np.testing.assert_allclose(weightedx, expected_x)
    np.testing.assert_allclose(weightedy, expected_y)


def test_kernelweights() -> None:
    """Computes symmetric kernelweights for a small grid."""
    bandwidth = 0.1
    a, b, gridsize = 0, 1, 10
    binwidth = (b - a) / (gridsize - 1)  # Set the bin width = delta
    kernelweights = get_kernelweights(bandwidth, binwidth)

    symmetric_weights = [0.00386592, 0.08465799, 0.53940751]
    expected_weights = np.asarray(symmetric_weights + [1] + symmetric_weights[::-1])
    np.testing.assert_allclose(kernelweights, expected_weights)


def test_curve_estimation() -> None:
    """Computes estimate for first column of beta, i.e. zero-th derivative."""
    weigthedx = np.asarray([[10]] + [[13]] * 8 + [[10]])
    weigthedy = np.reshape(np.asarray([16] + [22] * 8 + [16]), (-1, 1))

    expected = np.array(
        [
            1.6,
            1.6923077,
            1.6923077,
            1.6923077,
            1.6923077,
            1.6923077,
            1.6923077,
            1.6923077,
            1.6923077,
            1.6,
        ]
    )

    estimate = get_curve_estimator(
        weigthedx, weigthedy, degree=0, derivative=0, gridsize=10
    )
    np.testing.assert_almost_equal(estimate, expected)


@pytest.mark.parametrize(
    "gridsize, binned, truncate, expected",
    [
        (None, False, True, "output_integration_default"),
        (11, False, True, "output_integration_false_true"),
        (11, True, True, "output_integration_true_true"),
        (11, False, False, "output_integration_false_false"),
    ],
)
def test_integration(
    gridsize: int, binned: bool, truncate: bool, expected: Callable, request: Any
) -> None:
    """It determines degree of the polynomial based on order of the derivative."""
    x = np.linspace(-1, 2, 1001)
    y = np.linspace(3, -20, 1001)

    rslt = locpoly(
        x,
        y,
        derivative=0,
        gridsize=gridsize,
        bandwidth=2,
        a=0,
        b=2,
        binned=binned,
        truncate=truncate,
    )

    gridsize = 401 if gridsize is None else gridsize
    np.testing.assert_almost_equal(rslt["gridpoints"], np.linspace(0, 2, gridsize))
    np.testing.assert_almost_equal(rslt["curvest"], request.getfixturevalue(expected))


@pytest.mark.parametrize(
    "degree, gridsize, bw, expected",
    [
        (1, 101, 3.3, np.genfromtxt("tests/resources/mcycle_expect_user_bw.csv")),
        (1, None, None, np.genfromtxt("tests/resources/mcycle_expect_auto_bw.csv")),
        (3, None, None, np.genfromtxt("tests/resources/mcycle_expect_degree_3.csv")),
    ],
)
def test_integration_motorcycle_data(
    degree: int, gridsize: int, bw: Union[float, None], expected: np.ndarray
) -> None:
    """It runs locpoly on example data with and without a user-specified bandwidth."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")

    time, accel = motorcycle["time"], motorcycle["accel"]

    rslt = locpoly(
        x=time, y=accel, derivative=0, degree=degree, gridsize=gridsize, bandwidth=bw
    )

    np.testing.assert_almost_equal(
        rslt["gridpoints"], np.linspace(min(time), max(time), len(rslt["gridpoints"]))
    )
    np.testing.assert_almost_equal(rslt["curvest"], expected)


@pytest.mark.parametrize("xcol", [0, "time"])
def test_df_sort_by_x(xcol: Union[int, str]) -> None:
    """Sort DataFrame based on column index or column name."""
    expected = pd.read_stata("tests/resources/motorcycle.dta")

    data = expected.sample(frac=1)  # Shuffle rows
    data_sorted = sort_by_x(data, xcol)  # Sort by xcol

    data_sorted.equals(expected)


def test_arr_sort_by_x() -> None:
    """Sort np.ndarray by first column."""
    arr = np.array([[1, 3, 2, 0], [7, 6, 6, 8]]).T
    expected = np.array([[0, 1, 2, 3], [8, 7, 6, 6]]).T

    data_sorted = sort_by_x(arr, xcol=0)

    np.testing.assert_equal(data_sorted, expected)
