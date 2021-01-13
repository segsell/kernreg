"""Bandwidth selection via the Residual Squares Criterion (RSC)."""
from functools import partial
import math
from typing import List, Optional, Union

import numpy as np


def get_residual_squares_criterion(
    x: np.ndarray, y: np.ndarray, poly: int, input_arr: Union[List[float], np.ndarray]
) -> float:
    """Following the notation in Fan & Gijibell.

    Arguments:
        x: Dependent variable
        y: Independent variable
        poly: Degree of polynomial
        input_arr: Array or List containing one pair of
            a) x0: Value of x to evaluate bandwidth on
            b) bw: bandwidth

    Returns:
        Value of the residual squares criterion for given
            combination of x0 and bandwidth

    """
    x0, bw = input_arr
    delta = x - x0
    X = np.column_stack(
        [delta ** p for p in range(poly + 1)]
    )  # design matrix starting with np.ones column

    # Gaussian Kernel:  sigma = 1, mu = 0
    # https://en.wikipedia.org/wiki/Gaussian_function

    # normpdf = norm.pdf(delta / bw) # delta / bw = (X_i - x0) / h
    normpdf = (
        np.exp(-((delta / bw) ** 2) / 2) * 1 / np.sqrt(2 * np.pi)
    )  # because of Gaussian Kernel
    weights = np.diag(normpdf)  # kernel weights # weight matrix

    # Prepare matrices
    # X_and_y = (X.T @ weights) @ y
    S = (X.T @ weights) @ X
    S_inv = np.linalg.inv(S)  # z_inv = scipy.linalg.pinvh((z.T @ weights) @ z)
    # S_w2 = (X.T @ (weights ** 2)) @ X # former S_and_S
    # S_mat = (S_inv @ S_w2) @ S_inv

    cond_var_weights = (S_inv @ (X.T @ (weights ** 2)) @ X) @ S_inv
    V = cond_var_weights[
        0, 0
    ]  # first diagonal element of.., reflects the effective number of local data points

    b_hat = (
        S_inv @ (X.T @ weights) @ y
    )  # beta_hat: solution to weighted least-squares problem p.59 (3.5)
    y_hat = X @ b_hat

    nom = np.sum(((y - y_hat) ** 2) * weights)
    denom = np.trace(weights - (weights @ X @ S_inv @ (X.T @ weights)))
    weighted_res_sum_squares = nom / denom

    rsc = weighted_res_sum_squares * (1 + (poly + 1) * V)

    return rsc


def minimize_rsc(
    x: np.ndarray,
    y: np.ndarray,
    poly: int,
    x_range: Optional[Union[List[float], np.ndarray]] = None,
) -> float:
    """Minimize the residual squares criterion.

    This yields the most efficient (optimal?) bandwidth.

    Arguments:
        x: Dependent variable
        y: Independent variable
        poly: Degree of polynomial
        x_range: (User-specified) range of x-values to consider
            for bandwidth selection. Default is the entire x-range.

    Returns:
        Bandwidth that produces the minimal/minimizes residual squares
            criterion. Adjusted by constant for the Gaussian kernel.
            See Fan and Gjibels (1995, p. 133)

    """
    if x_range is not None:
        xmin = x_range[0]
        xmax = x_range[1]
    else:
        xmin = min(x)
        xmax = max(x)

    # Construct sequence of 11 x-values to evaluate bandwidth on
    xseq = np.linspace(xmin, xmax, 11)

    bw_min = (xmax - xmin) / 20
    bw_max = (xmax - xmin) / 2
    bw_cons = 1.1

    # Determine total number of bandwidths to iterate over
    total = math.ceil(np.log(bw_max / bw_min) / np.log(bw_cons))
    bw_arr = np.array([bw_min * (bw_cons ** (i + 1)) for i in range(total)])
    cartesian_pairs = np.transpose([np.tile(xseq, len(bw_arr)), np.repeat(bw_arr, 11)])

    rsc_p = partial(get_residual_squares_criterion, x, y, poly)

    sums_1d = np.zeros(total * 11)
    for index, value in enumerate(cartesian_pairs):
        rsc = rsc_p(value)
        sums_1d[index] += rsc

    rsc_sums = np.sum(sums_1d.reshape(total, 11), axis=1)

    # Adjusting constants for the Gaussian kernel, taking p = v + 1
    # (p = v + 3 would also be possible)
    # Allow for that
    # From Fan and Gjibels (1995a)
    const = np.array([1, 0.8403, 0.8285, 0.8085, 0.8146, 0.8098, 0.8159])

    bw = float(bw_arr[np.where(rsc_sums == np.min(rsc_sums))])
    bw_adj = bw * const[poly - 1]

    return bw_adj
