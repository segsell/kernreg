"""Implementation for the linear binning procedure."""
from typing import Tuple

from numba import njit
import numpy as np

from kernreg.funcs_to_jit import include_weights_from_endpoints


include_weights_from_endpoints_jitted = njit(include_weights_from_endpoints)


def linear_binning(
    x: np.ndarray,
    y: np.ndarray,
    gridsize: int,
    a: float,
    binwidth: float,
    truncate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    r"""Apply linear binning to x and y.

    Linear binning generates bin counts (x-dimension) and bin averages
    (y-dimension) over an equally spaced grid.
    In essence, bin counts are obtained by assigning the raw data to
    neighboring grid points. A bin count represents the
    amount of data in the neighborhood of its corresponding grid point.
    Counts on the y-axis display the respective bin averages.

    The linear binning strategy is based on the transformation
    ``xgrid[i] = ((x[i] - a) * delta) + 1``,
    which maps each ``x[i]`` onto its corresponding gridpoint.
    The integer part of ``xgrid[i]`` indicates the two nearest bin centers to ``x[i]``.
    Additionally, the "fractional part"
    or "bin weight" is computed, which contains the weight attached to the
    two nearest bin centers:

    ``binweights = xgrid - bincenters``.

    The corresponding weights are ``1 - binweights`` for the bin to the left and
    ``binweights`` for the bin to the right.

    If ``truncate`` is True, end observations are trimmed.
    Otherwise, weight from end observations is given to corresponding
    end grid points.


    Arguments:
        x: Array of the predictor variable. Shape (N,). Missing values are not accepted.
            Must be sorted ascendingly.
        y: Array of the response variable. Shape (N,). Missing values are not accepted.
            Must come pre-sorted by ``x``.
        gridsize: Number of equally-spaced grid points in the ``x``-dimension.
            Over this grid, ``x`` and ``y`` are binned.
        a: Start point of the grid.
        binwidth: Bin width.
        truncate: If True, trim endpoints.

    Returns:
        xcounts: Array of binned x-values ("bin counts"). Of length ``gridsize``.
        ycounts: Array of binned y-values ("bin averages"). Of length ``gridsize``.

    """
    N = len(x)

    xcounts = np.zeros(gridsize)
    ycounts = np.zeros(gridsize)
    xgrid = np.zeros(N)
    binweights = np.zeros(N)
    bincenters = [0] * N

    # Map x into set of corresponding grid points
    for i in range(N):
        xgrid[i] = ((x[i] - a) / binwidth) + 1

        # The integer part of xgrid indicates the two nearest bin centers to x[i]
        bincenters[i] = int(xgrid[i])
        binweights[i] = xgrid[i] - bincenters[i]

    for point in range(gridsize):
        for index, value in enumerate(bincenters):
            if value == point:
                xcounts[point - 1] += 1 - binweights[index]
                xcounts[point] += binweights[index]

                ycounts[point - 1] += (1 - binweights[index]) * y[index]
                ycounts[point] += binweights[index] * y[index]

    if truncate is False:
        xcounts, ycounts = include_weights_from_endpoints_jitted(
            xcounts, ycounts, y, xgrid, gridsize
        )

    return xcounts, ycounts
