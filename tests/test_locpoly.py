"""Test cases for the locpoly module."""
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


# @pytest.fixture  # (params=[array_unsorted, array_sort_descend])
# def raise_exception():

#     return

# array_sorted = np.linspace(-2, 2, 1000)
array_unsorted = np.array([-1, 2, 3, 2, 1])
array_sort_descend = np.linspace(2, -2, 1000)


@pytest.mark.parametrize("x", [array_unsorted, array_sort_descend])
def test_input_arr_not_sorted_ascend(x: np.ndarray) -> None:
    """It raises an Exception if unsorted x-array is put into locpoly."""
    y = np.linspace(3, -20, 1000)

    msg = "Input arrays x and y must be sorted by x before estimation!"

    with pytest.raises(Exception) as error:
        assert locpoly(
            x, y, derivative=0, degree=1, bandwidth=2, grid=11, a=0, b=2, truncate=False
        )
    assert str(error.value) == msg


@pytest.mark.parametrize(
    "arr,expected",
    [(np.linspace(-1, 2, 1000), True), (np.linspace(2, -1, 1000), False)],
)
def test_arr_not_sorted_ascend(arr: np.ndarray, expected: bool) -> None:
    """It exits with a zero status if array is sorted ascendingly."""
    assert is_sorted(arr) is expected


# def test_arr_not_sorted() -> None:
#     """It exits with a non-zero status if array is sorted ascendingly."""
#     sorted_arr = np.random.randint(-1, 2, 1001)
#     assert is_sorted(sorted_arr) is False


# @pytest.fixture(degree=[0, 1])
# def input():
#     bandwidth = 0.1
#     a, b, grid = 0, 1, 10
#     binwidth = (b - a) / (grid - 1)

#     xcounts = np.asarray(9 * [6] + [value])
#     ycounts = np.asarray(10 * [10])

x_positive_counts = np.array(
    [9.63, 12.63, 13.23, 13.23, 13.23, 13.23, 13.23, 13.2, 12.6, 9.6]
)
y_positive_counts = np.array(
    [16.05, 21.05, 22.05, 22.05, 22.05, 22.05, 22.05, 22, 21, 16]
)
x_zero_count = np.array(
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
y_zero_count = np.array(
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


@pytest.mark.parametrize(
    "degree,count,expected_x, expected_y",
    [(0, 6, x_positive_counts, y_positive_counts), (1, 0, x_zero_count, y_zero_count)],
)
def test_combine_weights_degree_zero(
    degree: int, count: int, expected_x: np.ndarray, expected_y: np.ndarray
) -> None:
    """Combines bincounts and weights where degree of polynomial is zero."""
    # if degree = 1, no ravel needed --> weightedx multidimensional
    # check regression/integration test if array (shape) handling still works
    # if degree 0 and weightedx of the form [[1], [2], [3] ]

    bandwidth = 0.1
    a, b, grid = 0, 1, 10
    binwidth = (b - a) / (grid - 1)

    xcounts = np.asarray(9 * [6] + [count])
    ycounts = np.asarray(10 * [10])

    symmetric_weights = [0.005, 0.1, 0.5]
    weights = np.asarray(symmetric_weights + [1] + symmetric_weights[::-1])

    weightedx, weightedy = combine_bincounts_kernelweights(
        xcounts, ycounts, weights, degree, grid, bandwidth, binwidth
    )

    if degree == 0:
        weightedx, weightedy = weightedx.ravel(), weightedy.ravel()

    np.testing.assert_allclose(weightedx, expected_x)
    np.testing.assert_allclose(weightedy, expected_y)


# def test_combine_weights_with_zeros() -> None:
#     """It combines bin counts and kernel weights, where some weights are zero."""
#     degree = 1  # if degree = 1, no ravel needed --> weightedx multidimensional
#     # check regression/integration test if array (shape) handling still works
#     # if degree 0 and weightedx of the form [[1], [2], [3] ]

#     bandwidth = 0.1
#     a, b, grid = 0, 1, 10
#     binwidth = (b - a) / (grid - 1)

#     xcounts = np.asarray(9 * [6] + [0])
#     ycounts = np.asarray(10 * [10])

#     symmetric_weights = [0.005, 0.1, 0.5]
#     weights = np.asarray(symmetric_weights + [1] + symmetric_weights[::-1])

#     weightedx, weightedy = combine_bincounts_kernelweights(
#         xcounts, ycounts, weights, degree, grid, bandwidth, binwidth
#     )

#     expected_x = np.array(
#         [
#             [9.63000000e00, 4.76666667e-01, 7.00000000e-02],
#             [1.26300000e01, 1.43333333e-01, 1.07037037e-01],
#             [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
#             [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
#             [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
#             [1.32300000e01, 1.00000000e-02, 1.36666667e-01],
#             [1.32000000e01, 0.00000000e00, 1.33333333e-01],
#             [1.26000000e01, -1.33333333e-01, 1.03703704e-01],
#             [9.60000000e00, -4.66666667e-01, 6.66666667e-02],
#             [3.60000000e00, -4.66666667e-01, 6.66666667e-02],
#         ]
#     )
#     expected_y = np.array(
#         [
#             [1.60500000e01, 7.94444444e-01],
#             [2.10500000e01, 2.38888889e-01],
#             [2.20500000e01, 1.66666667e-02],
#             [2.20500000e01, 1.66666667e-02],
#             [2.20500000e01, 1.66666667e-02],
#             [2.20500000e01, 1.66666667e-02],
#             [2.20000000e01, 0.00000000e00],
#             [2.10000000e01, -2.22222222e-01],
#             [1.60000000e01, -7.77777778e-01],
#             [6.00000000e00, -7.77777778e-01],
#         ]
#     )

#     np.testing.assert_allclose(weightedx, expected_x)
#     np.testing.assert_allclose(weightedy, expected_y)


def test_kernelweights() -> None:
    """Computes symmetric kernelweights for a small grid."""
    bandwidth = 0.1
    a, b, grid = 0, 1, 10
    binwidth = (b - a) / (grid - 1)  # Set the bin width = delta
    kernelweights = get_kernelweights(bandwidth, binwidth)

    symmetric_weights = [0.00386592, 0.08465799, 0.53940751]
    expected_weights = np.asarray(symmetric_weights + [1] + symmetric_weights[::-1])
    np.testing.assert_allclose(kernelweights, expected_weights)


def test_curve_estimation() -> None:
    """Computes estimator for first column of beta, i.e. zero-th derivative."""
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
        weigthedx, weigthedy, degree=0, derivative=0, grid=10
    )

    np.testing.assert_almost_equal(estimate, expected)


def test_integration() -> None:
    """It runs the main module to estimate beta."""
    x = np.linspace(-1, 2, 1001)
    y = np.linspace(3, -20, 1001)
    estimate = locpoly(x, y, derivative=0, degree=1, bandwidth=2, grid=11)

    expected = np.array(
        [
            2.78673,
            0.532652,
            -1.723597,
            -3.981458,
            -6.240384,
            -8.499833,
            -10.759265,
            -13.018143,
            -15.275922,
            -17.532054,
            -19.785975,
        ]
    )
    np.testing.assert_array_almost_equal(estimate, expected)


def test_locpoly_provide_start_endpoint() -> None:
    """Run the main module with user-specified start and endpoint of x."""
    x = np.linspace(-1, 2, 1001)
    y = np.linspace(3, -20, 1001)
    estimate = locpoly(x, y, derivative=0, degree=1, bandwidth=2, grid=11, a=0, b=2)

    expected = np.array(
        [
            -8.219706,
            -8.783897,
            -9.308161,
            -9.791353,
            -10.232303,
            -10.629813,
            -10.98266,
            -11.289587,
            -11.549305,
            -11.760492,
            -11.921788,
        ]
    )
    np.testing.assert_array_almost_equal(estimate, expected)


def test_integration_binned_true() -> None:
    """It runs the main module with binned set to True."""
    x = np.linspace(-10, 20, 500)
    y = np.linspace(30, -200, 500)
    estimate = locpoly(
        x, y, derivative=0, degree=1, bandwidth=2, grid=11, a=0, b=2, binned=True
    )

    expected = np.array(
        [
            -3.002416,
            -2.972707,
            -2.942971,
            -2.913206,
            -2.883414,
            -2.853594,
            -2.823746,
            -2.79387,
            -2.763967,
            -2.734035,
            -2.704076,
        ]
    )
    np.testing.assert_array_almost_equal(estimate, expected)


def test_integration_truncate_False() -> None:
    """It runs the main module with truncate set to False."""
    x = np.linspace(-1, 2, 1001)
    y = np.linspace(3, -20, 1001)
    estimate = locpoly(
        x, y, derivative=0, degree=1, bandwidth=2, grid=11, a=0, b=2, truncate=False
    )

    expected = np.array(
        [
            -3.38271,
            -4.585068,
            -5.743808,
            -6.85848,
            -7.928566,
            -8.953475,
            -9.93254,
            -10.865013,
            -11.750062,
            -12.586765,
            -13.374108,
        ]
    )
    np.testing.assert_array_almost_equal(estimate, expected)


def test_integration_motorcycle_data() -> None:
    """It runs locpoly with example data and estimates beta."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")

    time = np.asarray(motorcycle["time"])
    accel = np.asarray(motorcycle["accel"])

    estimate = locpoly(x=time, y=accel, derivative=0, degree=1, bandwidth=3.3, grid=101)
    # np.savetxt(
    #     "tests/resources/motorcycle_expected_estimate.csv", estimate, delimiter=","
    # )

    expected = np.genfromtxt("tests/resources/motorcycle_expected_estimate.csv")
    np.testing.assert_array_almost_equal(estimate, expected)
