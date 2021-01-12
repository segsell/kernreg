"""This module provides the interface for the local polynomial
kernel regression.
"""
from __future__ import annotations

import math
from typing import Optional

from numba import njit
import numpy as np


from kernreg.funcs_to_jit import (
    combine_bincounts_kernelweights,
    get_kernelweights,
    is_sorted,
)
from kernreg.linear_binning import linear_binning

# Jit the functions
# Implemented as additional step since pytest-cov does not consider
# jitted functions when determining code coverage
# (however, looks like there will be a fix soon).
# The respective un-jitted functions are hence unit-tested in tests/test_locpoly.py
linear_binning_jitted = njit(linear_binning)
is_sorted_jitted = njit(is_sorted)
get_weights_jitted = njit(get_kernelweights)
combine_bincounts_weights_jitted = njit(combine_bincounts_kernelweights)


def locpoly(
    x: np.ndarray,
    y: np.ndarray,
    derivative: int,
    degree: int,
    bandwidth: float,
    grid: int = 401,
    a: Optional[float] = None,
    b: Optional[float] = None,
    binned: bool = False,
    truncate: bool = True,
) -> np.ndarray:
    """This function fits a regression function or their derivatives via
    local polynomials. A local polynomial fit requires a weighted
    least-squares regression at every point g = 1,..., M in the grid.
    The Gaussian density function is used as kernel weight.

    It is recommended that for a v-th derivative the order of the polynomial
    be p = v + 1.

    The local polynomial curve estimator beta_ and its derivatives are
    minimizers to the locally weighted least-squares problem. At each grid
    point, beta_ is computed as the solution to the linear matrix equation:

    X'W X * beta_ = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    X'W X and X'W y are approximated by weigthedx and weigthedy,
    which are the result of a direct convolution of bin counts (xcounts) and kernel
    weights, and bin averages (ycounts) and kernel weights, respectively.

    A binned approximation over an equally-spaced grid is used for fast
    computation. Fan and Marron (1994) recommend a defau of M = 400
    for the popular case of graphical analysis. They find that fewer than 400
    grid points results in distracting "granularity", while more grid points
    often give negligible improvements in resolution. Instead of a scalar
    bandwidth, local bandwidths of length gridsize may be chosen.

    The terms "kernel" and "kernel function" are used interchangeably
    throughout and denoted by K.

    This function builds on the R function "locpoly" from the "KernSmooth"
    package maintained by Brian Ripley and the original Fortran routines
    provided by M.P. Wand.

    Parameters
    ----------
    x: np.ndarry
        Array of x data. Missing values are not accepted. Must be sorted.
    y: np.ndarry
        1-D Array of y data. This must be same length as x. Missing values are
        not accepted. Must be presorted by x.
    derivative: int
        Order of the derivative to be estimated.
    degree: int:
        Degree of local polynomial used. Its value must be greater than or
        equal to the value of drv. Generally, users should choose a degree of
        size drv + 1.
    bandwidth: int, float, list or np.ndarry
        Kernel bandwidth smoothing parameter. It may be a scalar or a array of
        length gridsize.
    gridsize: int
        Number of equally-spaced grid points over which the function is to be
        estimated.
    a: float
        Start point of the grid mesh.
    b: float
        End point of the grid mesh.
    binned: bool
        If True, then x and y are taken to be bin counts rather than raw data
        and the binning step is skipped.
    truncate: bool
        If True, then endpoints are truncated.

    Returns
    -------
    gridpoints: np.ndarry
        Array of sorted x values, i.e. grid points, at which the estimate
        of E[Y|X] (or its derivative) is computed.
    curvest: np.ndarry
        Array of M local estimators.
    """
    # The input arrays x (predictor) and y (response variable)
    # must be sorted by x.
    if is_sorted_jitted(x) is False:
        raise Exception("Input arrays x and y must be sorted by x before estimation!")

    if a is None:
        a = min(x)

    if b is None:
        b = max(x)

    # tau is chosen so that the interval [-tau, tau] is the
    # "effective support" of the Gaussian kernel,
    # i.e. K is effectively zero outside of [-tau, tau].
    # According to Wand (1994) and Wand & Jones (1995), tau = 4 is a
    # reasonable choice for the Gaussian kernel.
    # tau = 4

    # Set the bin width
    binwidth = (b - a) / (grid - 1)

    # 1. Bin the data if not already binned
    if binned is False:
        xcounts, ycounts = linear_binning_jitted(x, y, grid, a, binwidth, truncate)
    else:
        xcounts, ycounts = x, y

    # 2. Obtain kernel weights
    # Note that only L < N kernel evaluations are required to obtain the
    # kernel weights regardless of the number of observations N.
    # get_weights_jitted = njit(get_kernelweights)
    weights = get_weights_jitted(bandwidth, binwidth)

    # 3. Combine bin counts and kernel weights
    # combine_bincounts_weights_jitted = njit(combine_bincounts_weights)
    weightedx, weigthedy = combine_bincounts_weights_jitted(
        xcounts, ycounts, weights, degree, grid, bandwidth, binwidth
    )

    # 4. Fit the curve and obtain estimator for the desired derivative
    curvest = get_curve_estimator(weightedx, weigthedy, degree, derivative, grid)

    # Generate grid points for visual representation
    # gridpoints = np.linspace(a, b, grid)

    return curvest


def get_curve_estimator(
    weigthedx: np.ndarray,
    weigthedy: np.ndarray,
    degree: int,
    derivative: int,
    grid: int,
) -> np.ndarray:
    """This functions solves the locally weighted least-squares regression
    problem and returns an estimator for the v-th derivative of beta,
    the local polynomial estimator, at all points in the grid.

    Before doing so, the function first turns each row in weightedx into
    an array of size (coly, coly) and each row in weigthedy into
    an array of size (coly,), called xmat and yvec, respectively.

    The local polynomial curve estimator beta and its derivatives are
    minimizers to a locally weighted least-squares problem. At each grid
    point, beta is computed as the solution to the linear matrix equation:

    X'W X * beta = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    X'W X and X'W y are approximated by weigthedx and weigthedy,
    which are the result of a direct convolution of bin counts (xcounts) and
    kernel weights, and bin averages (ycounts) and kernel weights, respectively.

    Note that for a v-th derivative the order of the polynomial
    should be p = v + 1.

    Parameters
    ----------
    weigthedx: np.ndarry
        Dimensions (M, colx). Binned approximation to X'W X.
    weigthedy: np.ndarry
        Dimensions (M, coly). Binned approximation to X'W y.
    coly: int
        Number of columns of output array weigthedy, i.e the binned
        approximation to X'W y.
    derivative: int
        Order of the derivative to be estimated.
    grid: int
        Number of equally-spaced grid points.

    Returns
    -------
    curvest: np.ndarray
        1-D array of length grid. Estimator for the specified
        derivative at each grid point.
    """
    coly = degree + 1
    xmat = np.zeros((coly, coly))
    yvec = np.zeros(coly)
    curvest = np.zeros(grid)

    for g in range(grid):
        for row in range(0, coly):
            for column in range(0, coly):
                colindex = row + column
                xmat[row, column] = weigthedx[g, colindex]

                yvec[row] = weigthedy[g, row]

        # Calculate beta as the solution to the linear matrix equation
        # X'W X * beta = X'W y.
        # Note that xmat and yvec are binned approximations to X'W X and
        # X'W y, respectively, evaluated at the given grid point.
        beta = np.linalg.solve(xmat, yvec)

        # Obtain curve estimator for the desired derivative of beta.
        curvest[g] = beta[derivative]

    curvest = math.gamma(derivative + 1) * curvest

    return curvest
