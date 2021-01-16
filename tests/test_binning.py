"""Test cases for the linear_binning module."""
from typing import Tuple

import numpy as np
import pytest


from kernreg.linear_binning import include_weights_from_endpoints, linear_binning

xcounts_trunc_true = np.array(
    [
        38.06786787,
        75.68048048,
        75.68468468,
        75.68048048,
        75.68048048,
        75.68408408,
        75.68108108,
        75.68048048,
        75.68348348,
        188.47687688,
    ]
)
ycounts_trunc_true = np.array(
    [
        613.97600002,
        1146.67162057,
        1261.41141141,
        1376.01106211,
        1490.6743861,
        1605.41962984,
        1720.02656711,
        1834.67715163,
        1949.42238936,
        2758.90697905,
    ]
)
xcounts_trunc_false = np.array(
    [
        250.21711712,
        100.90900901,
        100.90900901,
        100.90900901,
        100.90900901,
        100.90900901,
        100.90900901,
        100.90990991,
        60.49009009,
        164.92882883,
    ]
)
ycounts_trunc_false = np.array(
    [
        3181.46857668,
        1579.88869049,
        1783.74494314,
        1987.60119579,
        2191.45744844,
        2395.31370109,
        2599.16995374,
        2803.05305305,
        1769.55160566,
        1858.50058166,
    ]
)


@pytest.fixture
def generate_test_input() -> Tuple[np.ndarray, np.ndarray]:
    """Reusable parameterization for x and y inputs."""
    x = np.linspace(2.2, 6.6, 1000)
    y = np.linspace(10, 30, 1000)

    return x, y


@pytest.mark.parametrize(
    "b, trunc, expected_xcounts, expected_ycounts",
    [
        (6, True, xcounts_trunc_true, ycounts_trunc_true),
        (7, False, xcounts_trunc_false, ycounts_trunc_false),
    ],
)
def test_binning_truncate_True_False(
    b: float,
    trunc: bool,
    expected_xcounts: np.ndarray,
    expected_ycounts: np.ndarray,
    generate_test_input: Tuple[np.ndarray, np.ndarray],
) -> None:
    """It bins inputs (x and y) with and without truncation of end points."""
    a, grid = 3, 10
    binwidth = (b - a) / (grid - 1)
    x, y = generate_test_input

    xcounts, ycounts = linear_binning(x, y, grid, a, binwidth, truncate=trunc)

    np.testing.assert_almost_equal(xcounts, expected_xcounts, decimal=8)
    np.testing.assert_almost_equal(ycounts, expected_ycounts, decimal=8)


# def test_binning_truncate_False(
#     generate_test_input: Tuple[np.ndarray, np.ndarray]
# ) -> None:
#     """Do not truncate endpoints."""
#     a, b, grid = 3, 7, 10
#     binwidth = (b - a) / (grid - 1)
#     x, y = generate_test_input

#     xcounts, ycounts = linear_binning(x, y, grid, a, binwidth, truncate=False)

#     expected_xcounts = np.array(
#         [
#             250.21711712,
#             100.90900901,
#             100.90900901,
#             100.90900901,
#             100.90900901,
#             100.90900901,
#             100.90900901,
#             100.90990991,
#             60.49009009,
#             164.92882883,
#         ]
#     )
#     expected_ycounts = np.array(
#         [
#             3181.46857668,
#             1579.88869049,
#             1783.74494314,
#             1987.60119579,
#             2191.45744844,
#             2395.31370109,
#             2599.16995374,
#             2803.05305305,
#             1769.55160566,
#             1858.50058166,
#         ]
#     )

#     np.testing.assert_almost_equal(xcounts, expected_xcounts, decimal=8)
#     np.testing.assert_almost_equal(ycounts, expected_ycounts, decimal=8)


def test_include_endpoints() -> None:
    """Calls inner function directly."""
    grid = 10
    y = np.linspace(10, 30, 100)

    xgrid = np.linspace(-2.6, 12, 100)
    xcounts = np.array([2.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 5.0, 13.0])
    ycounts = np.array(
        [
            32.52525253,
            73.23232323,
            78.28282828,
            83.33333333,
            88.38383838,
            93.43434343,
            98.48484848,
            103.53535354,
            108.58585859,
            179.09090909,
        ]
    )
    xcounts_expected = np.copy(xcounts)
    ycounts_expected = np.copy(ycounts)
    xcounts_expected[0], xcounts_expected[9] = 27, 27
    ycounts_expected[0], ycounts_expected[9] = 343.13131314, 580.70707071

    xcounts, ycounts = include_weights_from_endpoints(xcounts, ycounts, y, xgrid, grid)

    np.testing.assert_almost_equal(xcounts, xcounts_expected, decimal=8)
    np.testing.assert_almost_equal(ycounts, ycounts_expected, decimal=8)
