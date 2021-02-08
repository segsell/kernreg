"""Test cases for the plot module."""
from typing import Any, Callable
from unittest import mock

import numpy as np

from kernreg.config import TEST_RESOURCES
from kernreg.smooth import Result
from kernreg.utils import get_example_data
import kernreg.visualize as plot_module


@mock.patch(f"{__name__}.plot_module.plt")
def test_plot(mock_plt: Any, change_test_dir: Callable) -> None:
    """Uses mock object to call plot() function."""
    motorcycle = get_example_data()
    x, y = motorcycle["time"], motorcycle["accel"]

    gridpoints = np.linspace(min(x), max(x), 101)
    curvest = np.genfromtxt(TEST_RESOURCES / "mcycle_expect_user_bw.csv")
    bandwidth = 3.3
    rslt = Result(gridpoints=gridpoints, curvest=curvest, bandwidth=bandwidth)

    plot_module.plot(x, y, rslt)
    plot_module.plot(np.asarray(x), np.asarray(y), rslt)
    plot_module.plot(np.asarray(x), np.asarray(y), rslt, save_as="curvefit.png")

    assert mock_plt.figure.call_count == 3
