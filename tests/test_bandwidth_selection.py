"""Test cases for the bandwidth selection."""
from typing import List, Union

import numpy as np
import pandas as pd
import pytest

from kernreg.bandwidth_selection import get_bandwidth_rsc


@pytest.mark.parametrize(
    "x_range, expected", [(None, 3.035999832153321), ([3, 55], 2.8600000000000003)]
)
def test_motorcycle_bw(x_range: Union[None, List[int]], expected: float) -> None:
    """Computes optimal bandwidth for example data set."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")
    time = np.asarray(motorcycle["time"])
    accel = np.asarray(motorcycle["accel"])

    bw_rsc = get_bandwidth_rsc(x=time, y=accel, degree=1, x_range=x_range)

    np.testing.assert_equal(bw_rsc, expected)
