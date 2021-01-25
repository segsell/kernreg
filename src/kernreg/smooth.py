"""Estimate Functions Using Local Polynomials."""
import math
from typing import Optional, Union

from mypy_extensions import TypedDict  # Python>=3.8: from typing import TypedDict
from numba import njit
import numpy as np
import pandas as pd

from kernreg.bandwidth_selection import get_bandwidth_rsc
from kernreg.funcs_to_jit import (
    combine_bincounts_kernelweights,
    get_kernelweights,
    is_sorted,
)
from kernreg.linear_binning import linear_binning
from kernreg.utils import process_inputs


# Jit the functions
# Implemented as additional step since pytest-cov does not consider
# jitted functions when determining code coverage.
# The respective un-jitted functions are hence unit-tested in tests/test_locpoly.py
linear_binning_jitted = njit(linear_binning)
is_sorted_jitted = njit(is_sorted)
get_weights_jitted = njit(get_kernelweights)
combine_bincounts_weights_jitted = njit(combine_bincounts_kernelweights)


class Result(TypedDict):
    """Result dict for func ``locpoly``."""

    gridpoints: np.ndarray
    curvest: np.ndarray
    bandwidth: float


def locpoly(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    derivative: int = 0,
    degree: Optional[int] = None,
    gridsize: Optional[int] = None,
    bandwidth: Optional[float] = None,
    a: Optional[float] = None,
    b: Optional[float] = None,
    binned: bool = False,
    truncate: bool = True,
) -> Result:
    r"""Estimates a regression function or their derivatives using local polynomials.

    Non-parametrically fits a smooth curve between a predictor, ``x``,
    and a response variable, ``y``. A binned approximation to ``x`` and ``y``
    over an equally-spaced grid is used for fast computation.

    Note that for a `v`-th derivative the order of the polynomial
    must be :math:`p = v + 1`, :math:`v + 3`, :math:`v + 5`, or :math:`v + 7`.

    The local polynomial curve estimator ``beta`` and its derivatives are
    minimizers to the locally weighted least-squares problem.
    In particular, at each point :math:`g = 1,..., M` in the grid,
    ``beta`` is computed as the solution to the linear matrix equation:

    .. math::

        X'W X \ \beta = X'W y,

    where :math:`W` are kernel weights approximated by the Gaussian density function.
    :math:`X'W X` and :math:`X'W y` use binned approximations to ``X`` and ``y``,
    respectively; see :func:`~kernreg.linear_binning.linear_binning`.

    The terms "kernel" and "kernel function" are used interchangeably
    throughout.

    :cite:`Fan1994` recommend a default of ``gridsize`` = 400
    for the popular case of graphical analysis. They find that using fewer than 400
    grid points produces distracting "granularity", while more grid points
    often give negligible improvements in resolution.

    The user may pre-specifiy a bandwidth smoothing parameter.
    If not, the bandwidth is auto-selected. ``locpoly`` uses the bandwidth
    that minimzes the Residual Squares Criterion (RSC), see
    :func:`~kernreg.bandwidth_selection.get_bandwidth_rsc`.

    Arguments:
        x: Array of x data. Missing values are not accepted. Must be sorted.
        y: Array of y data. This must be same length as x. Missing values are
            not accepted. Must be presorted by x.
        derivative: Order of the derivative to be estimated.
            ``derivative`` = 0 means a function, i.e. `0`-th derivative, is fitted.
        degree: Degree of local polynomial used. Its value must be equal to
            ``derivative + 1`` or ``derivative + 3``.
        bandwidth: Kernel bandwidth smoothing parameter. If not specified,
            the bandwidth that minimizes the RSC is chosen.
        gridsize: Number of equally-spaced grid points over which the
            function is to be estimated.
        a: Start point of the grid.
        b: End point of the grid.
        binned: If True, then ``x`` and ``y`` are taken to be bin counts
            rather than raw data and the binning step is skipped.
        truncate: If True, trim endpoints.

    Returns:
        Result: Result Dictionary containing:
            - gridpoints (np.ndarray): Sorted grid points (``x-dimension``) at
                which the estimate of :math:`E[Y|X]` (or its derivative) is computed
            - curvest (np.ndarray): Curve estimate for the specified
                derivative of ``beta``
            - bandwidth (float): Bandwidth used. If not pre-specified by
                the user, this is the bandwidth that minimizes the RSC.

    Raises:
        Exception: Input data ``x`` and ``y`` must be sorted by ``x``
            before estimation.
        Exception: The degree of the polynomial must be equal to
            ``derivative + 1``, ``derivative + 3``, ``derivative + 5``, or
             ``derivative + 7``.

    """
    x, y, degree, gridsize, a, b = process_inputs(
        x, y, derivative, degree, gridsize, a, b
    )

    if is_sorted_jitted(x) is False:
        raise Exception("Input arrays x and y must be sorted by x before estimation!")

    if degree not in [derivative + 1, derivative + 3, derivative + 5, derivative + 7]:
        raise Exception(
            "The degree of the polynomial must be equal to derivative "
            "v + 1, v + 3, v + 5, or v + 7."
        )

    if bandwidth is None:
        bandwidth = get_bandwidth_rsc(x, y, derivative, degree, [a, b])

    # Set bin width
    binwidth = (b - a) / (gridsize - 1)

    # 1. Bin the data if not already binned
    if binned is False:
        xcounts, ycounts = linear_binning_jitted(x, y, gridsize, a, binwidth, truncate)
    else:
        xcounts, ycounts = x, y

    # 2. Obtain kernel weights
    # Note that only L < N kernel evaluations are required to obtain the
    # kernel weights regardless of the number of observations N.
    # get_weights_jitted = njit(get_kernelweights)
    weights = get_weights_jitted(bandwidth, binwidth)

    # 3. Combine bin counts and kernel weights
    # combine_bincounts_weights_jitted = njit(combine_bincounts_weights)
    x_weighted, y_weighted = combine_bincounts_weights_jitted(
        xcounts, ycounts, weights, degree, gridsize, bandwidth, binwidth
    )

    # 4. Fit the curve and obtain estimator for the desired derivative
    curvest = get_curve_estimator(x_weighted, y_weighted, degree, derivative, gridsize)

    # Generate grid points for visual representation
    gridpoints = np.linspace(a, b, gridsize)

    rslt = Result(gridpoints=gridpoints, curvest=curvest, bandwidth=bandwidth)

    return rslt


def get_curve_estimator(
    x_weighted: np.ndarray,
    y_weighted: np.ndarray,
    degree: int,
    derivative: int,
    gridsize: int,
) -> np.ndarray:
    r"""Solves the locally weighted least-squares regression problem.

    Returns an estimator for the `v`-th derivative of ``beta`` .

    The local polynomial curve estimator ``beta`` and its derivatives are
    minimizers to a locally weighted least-squares problem. At each grid
    point, ``beta`` is computed as the solution to the linear matrix equation:

    .. math::

        X'W X \ \beta = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    :math:`X'W X`and :math:`X'W y`are approximated by ``x_weighted`` and ``y_weighted``,
    which are the result of a direct convolution of
    bin counts (``xcounts``) and kernel weights, as well as
    bin averages (``ycounts``) and kernel weights, respectively.

    Arguments:
        x_weighted: Binned approximation to :math:`X'W X`.
        y_weighted: Binned approximation to :math:`X'W y`.
        degree: Degree of the polynomial.
        derivative: Order of the derivative to be estimated.
        gridsize: Number of equally-spaced grid points.

    Returns:
        curvest: Curve estimate for the specified derivative of ``beta``.

    """
    coly = degree + 1
    xmat = np.zeros((coly, coly))
    yvec = np.zeros(coly)
    curvest = np.zeros(gridsize)

    for g in range(gridsize):
        for row in range(0, coly):
            for column in range(0, coly):
                colindex = row + column
                xmat[row, column] = x_weighted[g, colindex]

                yvec[row] = y_weighted[g, row]

        # Calculate beta as the solution to the linear matrix equation
        # X'W X * beta = X'W y.
        # Note that xmat and yvec are binned approximations to X'W X and
        # X'W y, respectively, evaluated at the given grid point.
        beta = np.linalg.solve(xmat, yvec)

        # Obtain curve estimator for the desired derivative of beta.
        curvest[g] = beta[derivative]

    curvest = math.gamma(derivative + 1) * curvest

    return curvest
