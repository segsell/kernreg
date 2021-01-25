"""Methods for preparing input data."""
from typing import Optional, Tuple, Union

import numpy as np
import pandas as pd

from kernreg.config import TEST_RESOURCES_DIR


def sort_by_x(
    data: Union[np.ndarray, pd.DataFrame], xcol: Union[int, str] = 0
) -> Union[np.ndarray, pd.DataFrame]:
    """Sort input data by x column."""
    if isinstance(data, pd.DataFrame):
        if isinstance(xcol, int):
            xcol = list(data)[xcol]
        data.sort_values(by=xcol, ascending=True, inplace=True)
    else:  # np.ndarray
        data = data[np.argsort(data[:, xcol])]

    return data


def process_inputs(
    x: Union[np.ndarray, pd.Series],
    y: Union[np.ndarray, pd.Series],
    derivative: int,
    degree: Optional[int],
    gridsize: Optional[int],
    a: Optional[float],
    b: Optional[float],
) -> Tuple[np.ndarray, np.ndarray, int, int, float, float]:
    """Process input arguments for func locpoly."""
    # Turn x (predictor) and y (response variable) into np.ndarrays
    if isinstance(x, pd.Series):
        x, y = np.asarray(x), np.asarray(y)

    if degree is None:
        degree = derivative + 1

    if gridsize is None:
        if len(x) > 400:
            gridsize = 401
        else:
            gridsize = len(x)

    if a is None:
        a = min(x)

    if b is None:
        b = max(x)

    return x, y, degree, gridsize, a, b


def get_example_data() -> pd.DataFrame:
    """Load motorcycle data."""
    # Already sorted by x
    return pd.read_stata(TEST_RESOURCES_DIR / "motorcycle.dta")
