import math
from typing import Tuple

import numpy as np


def include_weights_from_endpoints(
    xcounts: np.ndarray,
    ycounts: np.ndarray,
    y: np.ndarray,
    xgrid: np.ndarray,
    grid: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """ """
    for index, value in enumerate(xgrid):
        if value < 1:
            xcounts[0] += 1
            ycounts[0] += y[index]
        elif value >= grid:
            xcounts[grid - 1] += 1
            ycounts[grid - 1] += y[index]

    return xcounts, ycounts


def is_sorted(a: np.ndarray) -> bool:
    """This function checks if the input array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def combine_bincounts_kernelweights(
    xcounts: np.ndarray,
    ycounts: np.ndarray,
    weights: np.ndarray,
    degree: int,
    grid: int,
    bandwidth: float,
    binwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    """
    This function combines the bin counts (xcounts) and bin averages (ycounts) with
    kernel weights via a series of direct convolutions. As a result, binned
    approximations to X'W X and X'W y, denoted by weigthedx and weigthedy, are computed.

    Recall that the local polynomial curve estimator beta and its derivatives are
    minimizers to a locally weighted least-squares problem. At each grid
    point g = 1,..., M in the grid, beta is computed as the solution to the
    linear matrix equation:

    X'W X * beta = X'W y,

    where W are kernel weights approximated by the Gaussian density function.
    X'W X and X'W y are approximated by weigthedx and weigthedy,
    which are the result of a direct convolution of bin counts (xcounts) and kernel
    weights, and bin averages (ycounts) and kernel weights, respectively.

    The terms "kernel" and "kernel function" are used interchangeably
    throughout.

    For more information see the documentation of the main function locpoly
    under KernReg.locpoly.

    Parameters
    ----------
    xcounts: np.ndarry
        1-D array of binned x-values ("bin counts") of length grid.
    ycounts: np.ndarry
        1-D array of binned y-values ("bin averages") of length grid.
    grid: int
        Number of equally-spaced grid points.
    colx: int
        Number of columns of output array weigthedx, i.e. the binned
        approximation to X'W X.
    coly: int
        Number of columns of output array weigthedy, i.e the binned
        approximation to X'W y.
    length: int
        Length of 1-D array weights.
    weights: np.ndarry
        1-D array of length lenfkap containing
        approximated weights for the Gaussian kernel
        (W in the notation above).
    L: int
        Parameter defining the number of times the kernel function
        has to be evaluated.
        Note that L < N, where N is the total number of observations.
    mid: int
        Midpoint of weights.
    binwidth: float
        Bin width.

    Returns
    -------
    weigthedx: np.ndarry
        Dimensions (M, colx). Binned approximation to X'W X.
    weigthedy: np.ndarry
        Dimensions (M, coly). Binned approximation to X'W y.
    """
    tau = 4
    L = math.floor(tau * bandwidth / binwidth)
    length = 2 * L + 1
    mid = L + 1

    colx = 2 * degree + 1
    coly = degree + 1

    weigthedx = np.zeros((grid, colx))
    weigthedy = np.zeros((grid, coly))

    for g in range(grid):
        if xcounts[g] != 0:
            for i in range(max(0, g - L - 1), min(grid, g + L)):

                if 0 <= i <= grid - 1 and 0 <= g - i + mid - 1 <= length - 1:
                    fac_ = 1

                    weigthedx[i, 0] += xcounts[g] * weights[g - i + mid - 1]
                    weigthedy[i, 0] += ycounts[g] * weights[g - i + mid - 1]

                    for j in range(1, colx):
                        fac_ *= binwidth * (g - i)

                        weigthedx[i, j] += xcounts[g] * weights[g - i + mid - 1] * fac_

                        if j < coly:
                            weigthedy[i, j] += (
                                ycounts[g] * weights[g - i + mid - 1] * fac_
                            )
    return weigthedx, weigthedy


def get_kernelweights(bandwidth: float, delta: float) -> np.ndarray:  # removed tau
    """This function computes approximated weights for the Gaussian kernel."""
    tau = 4
    L = math.floor(tau * bandwidth / delta)  # what is L exactly? relation to N?
    length = 2 * L + 1
    weights = np.zeros(length)

    # Determine midpoint of weights
    mid = L + 1

    # Compute the kernel weights
    for j in range(L + 1):

        # Note that the mid point (weights[mid - 1]) receives a weight of 1.
        weights[mid - 1 + j] = math.exp(-((delta * j / bandwidth) ** 2) / 2)

        # Because of the kernel's symmetry, weights in equidistance
        # below and above the midpoint are identical.
        weights[mid - 1 - j] = weights[mid - 1 + j]

    return weights
