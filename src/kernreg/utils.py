"""Methods for preparing input data."""
from typing import Union

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


def get_example_data() -> pd.DataFrame:
    """Load motorcycle data."""
    # Already sorted by x
    return pd.read_stata(TEST_RESOURCES_DIR / "motorcycle.dta")
