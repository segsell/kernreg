"""Test cases for the plot module."""
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

from kernreg.locpoly import Result
import kernreg.visualization as plot_module


@mock.patch(f"{__name__}.plot_module.plt")
def test_plot(mock_plt: Any) -> None:
    """Uses mock object to call plot() function."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")
    x, y = motorcycle["time"], motorcycle["accel"]

    gridpoints = np.linspace(min(x), max(x), 101)
    curvest = np.genfromtxt("tests/resources/motorcycle_expected_user_bw.csv")
    bandwidth = 3.3
    rslt = Result(gridpoints=gridpoints, curvest=curvest, bandwidth=bandwidth)

    plot_module.plot(x, y, rslt)
    plot_module.plot(np.asarray(x), np.asarray(y), rslt)

    assert mock_plt.figure.call_count == 2
