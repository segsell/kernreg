"""Bandwidth selection via the Residual Squares Criterion (RSC)."""
from functools import partial
import math
from typing import List, Optional, Union

import numpy as np


def get_bandwidth_rsc(
    x: np.ndarray,
    y: np.ndarray,
    derivative: int,
    degree: int,
    x_range: Optional[Union[List[float], np.ndarray]] = None,
) -> float:
    r"""Returns the bandwidth that minimizes the Residual Squares Criterion.

    The Residual Squares Criterion (RSC) is definied as (:cite:`Fan1996`, p. 118):

    .. math::

        \text{RSC}(x_0; h) = \hat{\sigma}^2(x_0) \{ 1 + (p + 1) V \},

    where `V` is the first diagonal element of the matrix
    :math:`(\textbf{X}^T \textbf{W} \textbf{X})^{-1}
    (\textbf{X}^T \textbf{W}^2 \textbf{X})
    (\textbf{X}^T \textbf{W} \textbf{X})^{-1}`.

    The expression :math:`(\textbf{X}^T \textbf{W} \textbf{X})^{-1}
    (\textbf{X}^T \textbf{W}^2 \textbf{X})
    (\textbf{X}^T \textbf{W} \textbf{X})^{-1} \sigma^2`
    is an approximation to the conditional variance
    :math:`\text{Var}(\hat{\beta} \,|\, \textbf{X})`.

    Weight matrix `W` contains the diagonal elements of :math:`K_n(X_i - x_0)`,
    where kernel function :math:`K_n` represents the Gaussian density.

    The unkown quantity :math:`\hat{\sigma}^2`
    can be estimated by the normalized weighted residual sum of squares,
    see :func:`~kernreg.bandwidth_selection.get_residual_squares_criterion`:

    .. math::
        \hat{\sigma}^2(x_0) = \frac{\sum\limits_{i=1}^n
        (Y_i - \hat{Y}_i)^2 K_{h^*} (X_i - x_0)}
        { \text{tr} \{ \textbf{W}^* - \textbf{W}^* \textbf{X}^*
        ( \textbf{X}^{*T} \textbf{W}^* \textbf{X}^*)^{-1}
        \textbf{X}^{*T} \textbf{W}^*\} }.

    The RSC is computed for various combinations of ``x_0`` and bandwidth ``bw``.
    The RSCs for a particular bandwidth are then summed together across
    the ``x_0``. Finally, the bandwidth with the smallest sum of RCSs
    is chosen.


    Arguments:
        x: Predictor variable.
        y: Response variable.
        derivative: Order of the derivative to be estimated.
        degree: Degree of the polynomial.
        x_range: Start and end point of the domain of ``x`` to consider.
            Default is the entire range of ``x``.

    Returns:
        Bandwidth that minimizes the residual squares criterion.
            Adjusted by constant for the Gaussian kernel.
            See :cite:`Fan1996`, p. 120, and :cite:`Fan1995`.

    """
    if x_range is not None:
        xmin = x_range[0]
        xmax = x_range[1]
    else:
        xmin = min(x)
        xmax = max(x)

    # Construct sequence of 11 x-values to evaluate bandwidth on
    n_evals = 11
    xseq = np.linspace(xmin, xmax, n_evals)

    bw_min = (xmax - xmin) / 20
    bw_max = (xmax - xmin) / 2
    bw_cons = 1.1

    # Determine total number of bandwidths to iterate over
    total = math.ceil(np.log(bw_max / bw_min) / np.log(bw_cons))
    bw_arr = np.array([bw_min * (bw_cons ** (i + 1)) for i in range(total)])
    cartesian_pairs = np.transpose(
        [np.tile(xseq, len(bw_arr)), np.repeat(bw_arr, n_evals)]
    )

    rsc_p = partial(get_residual_squares_criterion, x, y, degree)

    sums_1d = np.zeros(total * n_evals)
    for index, value in enumerate(cartesian_pairs):
        rsc = rsc_p(value)
        sums_1d[index] += rsc

    rsc_sums = np.sum(sums_1d.reshape(total, n_evals), axis=1)

    # Adjusting constants for the Gaussian kernel, see :cite:`Fan1995`.
    if degree == derivative + 1:
        const = np.array([1, 0.8403, 0.8285, 0.8085, 0.8146, 0.8098, 0.8159])
    elif degree == derivative + 3:
        const = [np.nan] * 2 + [0.9554, 0.8975, 0.8846, 0.8671, 0.8652]
    elif degree == derivative + 5:
        const = [np.nan] * 4 + [0.9495, 0.9165, 0.9055]
    else:  # degree = derivative + 7
        const = [np.nan] * 6 + [0.9470]

    bw = float(bw_arr[np.where(rsc_sums == np.min(rsc_sums))])
    bw_adj = bw * const[degree - 1]

    return bw_adj


def get_residual_squares_criterion(
    x: np.ndarray, y: np.ndarray, degree: int, input_arr: Union[List[float], np.ndarray]
) -> float:
    r"""Compute residual squares criterion for given ``x_0`` and ``bw``.

    The Residual Squares Criterion (RSC) is definied as (:cite:`Fan1996`, p. 118):

    .. math::

        \text{RSC}(x_0; h) = \hat{\sigma}^2(x_0) \{ 1 + (p + 1) V \},

    where `V` is the first diagonal element of the matrix
    :math:`(\textbf{X}^T \textbf{W} \textbf{X})^{-1}
    (\textbf{X}^T \textbf{W}^2 \textbf{X})
    (\textbf{X}^T \textbf{W} \textbf{X})^{-1}`.

    The expression :math:`(\textbf{X}^T \textbf{W} \textbf{X})^{-1}
    (\textbf{X}^T \textbf{W}^2 \textbf{X})
    (\textbf{X}^T \textbf{W} \textbf{X})^{-1} \sigma^2`
    is an approximation to the conditional variance
    :math:`\text{Var}(\hat{\beta} \,|\, \textbf{X})`.

    Weight matrix `W` contains the diagonal elements of :math:`K_n(X_i - x_0)`,
    where kernel function :math:`K_n` represents the Gaussian density.

    The unkown quantity :math:`\hat{\sigma}^2`
    can be estimated by the normalized weighted residual sum of squares:

    .. math::
        \hat{\sigma}^2(x_0) = \frac{\sum\limits_{i=1}^n
        (Y_i - \hat{Y}_i)^2 K_{h^*} (X_i - x_0)}
        { \text{tr} \{ \textbf{W}^* - \textbf{W}^* \textbf{X}^*
        ( \textbf{X}^{*T} \textbf{W}^* \textbf{X}^*)^{-1}
        \textbf{X}^{*T} \textbf{W}^*\} }.

    after fitting locally a `p`-th order polynomial for given
    ``x_0`` and pilot bandwidth :math:`h^*`. Analogously, the :math:`*`
    in the denominator indicate that this particular :math:`h^*` has been
    used to derive design matrix :math:`\textbf{X^*}` and
    weight matrix :math:`\textbf{W^*}`
    (:math:`\textbf{X}` and :math:`\textbf{W}` above).


    Arguments:
        x: Array of x data, predictor variable
        y: Array of y data, response variable
        degree: Degree of the polynomial.
        input_arr: Array or List with two entries
            0) ``x_0``: Evaluation point
            1) ``bw``: bandwidth

    Returns:
        Residual Squares Criterion for given
            combination of ``x_0`` and ``bw``.

    """
    # Unpack x0 and bandwidth
    x0, bw = input_arr

    delta = x - x0
    X = np.column_stack(
        [delta ** p for p in range(degree + 1)]
    )  # design matrix starting with np.ones column

    # Gaussian Kernel:  sigma = 1, mu = 0
    # https://en.wikipedia.org/wiki/Gaussian_function
    normpdf = np.exp(-((delta / bw) ** 2) / 2) * 1 / np.sqrt(2 * np.pi)
    weights = np.diag(normpdf)  # kernel weights

    # Wrap matrices
    S = X.T @ weights @ X
    S_inv = np.linalg.inv(S)

    cond_var_weights = S_inv @ (X.T @ (weights ** 2) @ X) @ S_inv
    V = cond_var_weights[
        0, 0
    ]  # first diagonal element reflects the effective number of local data points

    # Solution to weighted least-squares problem, see :cite:`Fan1996`, p.59 (3.5)
    beta_hat = S_inv @ X.T @ weights @ y
    y_hat = X @ beta_hat

    nominator = np.sum(((y - y_hat) ** 2) * weights)
    denominator = np.trace(weights - weights @ X @ S_inv @ X.T @ weights)
    weighted_res_sum_squares = nominator / denominator

    rsc = weighted_res_sum_squares * (1 + (degree + 1) * V)

    return rsc
