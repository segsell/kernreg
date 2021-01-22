"""Test cases for the plot module."""
from typing import Any
from unittest import mock

import numpy as np
import pandas as pd

import kernreg.plot as plot_module


@mock.patch("%s.plot_module.plt" % __name__)
def test_plot(mock_plt: Any) -> None:
    """Plot and compare data."""
    motorcycle = pd.read_stata("tests/resources/motorcycle.dta")
    x, y = motorcycle["time"], motorcycle["accel"]

    gridpoints = np.linspace(min(x), min(y), 101)
    curvest = np.genfromtxt("tests/resources/motorcycle_expected_user_bw.csv")
    bandwidth = 3.3

    plot_module.plot(x, y, gridpoints, curvest, bandwidth)
    plot_module.plot(np.asarray(x), np.asarray(y), gridpoints, curvest, bandwidth)

    assert mock_plt.figure.call_count == 2
