"""Test cases for the bandwidth selection."""
import numpy as np
import pandas as pd  # change motorcylce dta to csv?

from kernreg.residual_squares_criterion import (
    get_residual_squares_criterion,
    minimize_rsc,
)


def test_rsc() -> None:
    """Calls inner function directly."""
    x = np.linspace(0, 50, 1000)
    y = np.linspace(0, -20, 1000)

    # Look at one specific iteration of x0 and bw.
    x0, bw = 20, 3

    rsc = get_residual_squares_criterion(x, y, poly=3, input_arr=np.asarray([x0, bw]))
    expected = 4.1101045969245315e-29  # 1.443732936227206e-29

    # np.testing.assert_allclose(rsc, expected)
    np.testing.assert_equal(rsc, expected)
    # assert rsc == expected


def test_motorcycle_bw() -> None:
    """Computes optimal bandwidth for example data set."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")
    time = np.asarray(motorcycle["time"])
    accel = np.asarray(motorcycle["accel"])

    bw_rsc = minimize_rsc(x=time, y=accel, poly=1)

    expected_bw = 3.035999832153321  # 3.0360000000000005
    np.testing.assert_equal(bw_rsc, expected_bw)


def test_motorcycle_bw_with_x_range() -> None:
    """Compute bandwidth for user-specified x-range."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")
    time = np.asarray(motorcycle["time"])
    accel = np.asarray(motorcycle["accel"])

    bw_rsc = minimize_rsc(x=time, y=accel, poly=1, x_range=[3, 55])

    expected_bw = 2.8600000000000003  # 3.0360000000000005
    np.testing.assert_equal(bw_rsc, expected_bw)
