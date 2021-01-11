"""This module contains an implementation for the linear binning procedure."""
from numba import njit
import numpy as np

from kernreg.funcs_to_jit import include_weights_from_endpoints


include_weights_from_endpoints_jitted = njit(include_weights_from_endpoints)


def linear_binning(x, y, grid, a, binwidth, truncate=True):
    """
    This function generates bin counts and bin averages over an equally spaced
    grid via the linear binning strategy.
    In essence, bin counts are obtained by assigning the raw data to
    neighboring grid points. A bin count can be thought of as representing the
    amount of data in the neighborhood of its corresponding grid point.
    Counts on the y-axis display the respective bin averages.

    The linear binning strategy is based on the transformation
    xgrid = ((x - a) / delta) + 1, which maps each x_i onto its corresponding
    gridpoint. The integer part of xgrid_i indicates the two
    nearest bin centers to x_i. This calculation already does the trick
    for simple binning. For linear binning, however, we additionally compute the
    "fractional part" or binweights = xgrid - bincenters, which gives the weights
    attached to the two nearest bin centers, namely (1 - binweights) for the bin
    considered and binweights for the next bin.

    If truncate is True, end observations are truncated.
    Otherwise, weight from end observations is given to corresponding
    end grid points.

    Parameters
    ----------
    x: np.ndarray
        Array of the predictor variable. Shape (N,).
        Missing values are not accepted. Must be sorted.
    y: np.ndarray
        Array of the response variable. Shape (N,).
        Missing values are not accepted. Must come presorted by x.
    grid: int
        Number of equally-spaced grid points
        over which x and y are to be evaluated.
    a: int
        Start point of the grid.
    binwidth: float
        Bin width.
    truncate: bool
        If True, then endpoints are truncated.

    Returns
    -------
    xcounts: np.ndarry
        Array of binned x-values ("bin counts") of length M.
    ycounts: np.ndarry
        Array of binned y-values ("bin averages") of length M.
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
