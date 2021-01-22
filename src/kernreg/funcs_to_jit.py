"""Functions to jit.

The following functions are imported by the main modules
locpoly and linear_binning. There, numba.njit will be applied.
"""
import math
from typing import Tuple

import numpy as np


def include_weights_from_endpoints(
    xcounts: np.ndarray,
    ycounts: np.ndarray,
    y: np.ndarray,
    xgrid: np.ndarray,
    gridsize: int,
) -> Tuple[np.ndarray, np.ndarray]:
    """Attach weight from values outside the grid to corresponding endpoints."""
    for index, value in enumerate(xgrid):
        if value < 1:
            xcounts[0] += 1
            ycounts[0] += y[index]
        elif value >= gridsize:
            xcounts[gridsize - 1] += 1
            ycounts[gridsize - 1] += y[index]

    return xcounts, ycounts


def is_sorted(a: np.ndarray) -> bool:
    """Checks if the input array is sorted ascendingly."""
    for i in range(a.size - 1):
        if a[i + 1] < a[i]:
            return False
    return True


def get_kernelweights(bandwidth: float, delta: float) -> np.ndarray:
    """Compute approximated weights for the Gaussian kernel.

    Arguments:
        bandwidth: Bandwidth.
        delta: Bin width.

    Returns:
        weights: Kernel weights.

    """
    # tau is chosen so that the interval [-tau, tau] is the
    # "effective support" of the Gaussian kernel,
    # i.e. K is effectively zero outside of [-tau, tau].
    # According to Wand (1994) and Wand & Jones (1995), tau = 4 is a
    # reasonable choice for the Gaussian kernel.
    tau = 4

    # Note that L < N, where N is the total number of observations.
    L = math.floor(tau * bandwidth / delta)
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


def combine_bincounts_kernelweights(
    xcounts: np.ndarray,
    ycounts: np.ndarray,
    weights: np.ndarray,
    degree: int,
    gridsize: int,
    bandwidth: float,
    binwidth: float,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Combine bin counts and bin averages with kernel weights.

    Use direct convolutions to compute binned
    approximations to :math:`X'W X` and :math:`X'W y`,
    denoted by ``x_weighted`` and ``y_weighted``.

    Arguments:
        xcounts: Binned x-values ("bin counts") of length ``gridsize``.
        ycounts: Binned y-values ("bin averages") of length ``gridsize``.
        weights: Approximated weights for the Gaussian kernel.
        degree: Degree of the polynomial.
        gridsize: Number of equally-spaced grid points.
        bandwidth: Bandwidth.
        binwidth: Bin width.

    Returns:
        x_weighted: Binned approximation to :math:`X'W X`.
        y_weighted: Binned approximation to :math:`X'W y`.

    """
    tau = 4
    L = math.floor(tau * bandwidth / binwidth)
    length = 2 * L + 1
    mid = L + 1

    colx = 2 * degree + 1
    coly = degree + 1

    x_weighted = np.zeros((gridsize, colx))
    y_weighted = np.zeros((gridsize, coly))

    for g in range(gridsize):
        if xcounts[g] != 0:
            for i in range(max(0, g - L - 1), min(gridsize, g + L)):

                if 0 <= i <= gridsize - 1 and 0 <= g - i + mid - 1 <= length - 1:
                    fac_ = 1.0

                    x_weighted[i, 0] += xcounts[g] * weights[g - i + mid - 1]
                    y_weighted[i, 0] += ycounts[g] * weights[g - i + mid - 1]

                    for j in range(1, colx):
                        fac_ *= binwidth * (g - i)

                        x_weighted[i, j] += xcounts[g] * weights[g - i + mid - 1] * fac_

                        if j < coly:
                            y_weighted[i, j] += (
                                ycounts[g] * weights[g - i + mid - 1] * fac_
                            )

    return x_weighted, y_weighted
