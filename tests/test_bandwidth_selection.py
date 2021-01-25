"""Test cases for the bandwidth selection."""
from typing import List, Union

import numpy as np
import pytest

from kernreg.bandwidth_selection import get_bandwidth_rsc
from kernreg.utils import get_example_data


@pytest.mark.parametrize(
    "x_range, degree, expected",
    [
        (None, 1, 3.035999832153321),
        (None, 3, 2.900594239639283),
        (None, 5, 3.8368495298779686),
        ([3, 55], 1, 2.8600000000000003),
        ([3, 55], 7, 4.361937494200002),
    ],
)
def test_motorcycle_bw(
    x_range: Union[None, List[int]], degree: int, expected: float
) -> None:
    """Computes optimal bandwidth for example data set."""
    motorcycle = get_example_data()
    time = np.asarray(motorcycle["time"])
    accel = np.asarray(motorcycle["accel"])

    bw_rsc = get_bandwidth_rsc(
        x=time, y=accel, derivative=0, degree=degree, x_range=x_range
    )

    np.testing.assert_equal(bw_rsc, expected)
