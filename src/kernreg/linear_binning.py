"""This module contains an implementation for the linear binning procedure."""
from typing import Tuple

from numba import njit
import numpy as np

from kernreg.funcs_to_jit import include_weights_from_endpoints


include_weights_from_endpoints_jitted = njit(include_weights_from_endpoints)


def linear_binning(
    x: np.ndarray,
    y: np.ndarray,
    grid: int,
    a: float,
    binwidth: float,
    truncate: bool = True,
) -> Tuple[np.ndarray, np.ndarray]:
    """Apply linear binning to input variables x and y.

    Linear binning generates bin counts (x-dimension) and bin averages
    (y-dmension) over an equally spaced grid.
    In essence, bin counts are obtained by assigning the raw data to
    neighboring grid points. A bin count can be thought of as representing the
    amount of data in the neighborhood of its corresponding grid point.
    Counts on the y-axis display the respective bin averages.

    The linear binning strategy is based on the transformation
    xgrid = ((x - a) / delta) + 1, which maps each x_i onto its corresponding
    gridpoint. The integer part of xgrid_i indicates the two
    nearest bin centers to x_i. Additionally, we compute the
    "fractional part", i.e. binweights = xgrid - bincenters, which yields the weights
    attached to the two nearest bin centers; namely (1 - binweights) for the bin
    considered and binweights for the next bin.

    If truncate is True, end observations are truncated.
    Otherwise, weight from end observations is given to corresponding
    end grid points.

    Arguments:
        x: Array of the predictor variable. Shape (N,). Missing values are not accepted.
            Must be sorted ascendingly.
        y: Array of the response variable. Shape (N,). Missing values are not accepted.
            Must come pre-sorted by x.
        grid: Number of equally-spaced grid points in the x-dimension.
            Over this grid, the values of x and y are binned.
        a: Start point of the grid.
        binwidth: Width of each of the M bins.
        truncate: If True, truncate endpoints.

    Returns:
        xcounts: Array of binned x-values ("bin counts"). Of length M.
        ycounts: Array of binned y-values ("bin averages"). Of length M.
    """
    N = len(x)

    xcounts = np.zeros(grid)
    ycounts = np.zeros(grid)
    xgrid = np.zeros(N)
    binweights = np.zeros(N)
    bincenters = [0] * N

    # Map x into set of corresponding grid points
    for i in range(N):
        xgrid[i] = ((x[i] - a) / binwidth) + 1

        # The integer part of xgrid indicates the two nearest bin centers to x[i]
        bincenters[i] = int(xgrid[i])
        binweights[i] = xgrid[i] - bincenters[i]

    for point in range(grid):
        for index, value in enumerate(bincenters):
            if value == point:
                xcounts[point - 1] += 1 - binweights[index]
                xcounts[point] += binweights[index]

                ycounts[point - 1] += (1 - binweights[index]) * y[index]
                ycounts[point] += binweights[index] * y[index]

    if truncate is False:
        xcounts, ycounts = include_weights_from_endpoints_jitted(
            xcounts, ycounts, y, xgrid, grid
        )

    # By default, end observations are truncated.
    # elif truncate is True or (1 <= bincenters[0] and bincenters[N - 1] < grid):
    #     pass

    # Truncation is implicit if there are no points in bincenters
    # beyond the grid's boundary points.
    # Note that bincenters is sorted. So it is sufficient to check if
    # the conditions below hold for the bottom and top
    # observation, respectively
    # elif 1 <= bincenters[0] and bincenters[N - 1] < grid:
    #     pass

    return xcounts, ycounts
