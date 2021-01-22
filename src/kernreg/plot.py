"""Visualize results."""
from typing import Union

import matplotlib.patches as mpatches
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from kernreg.locpoly import Result


def plot(
    x_raw: Union[np.ndarray, pd.Series],
    y_raw: Union[np.ndarray, pd.Series],
    rslt: Result,
) -> None:
    """Plot fitted curve against raw data."""
    ax = plt.figure(figsize=(17.5, 10)).add_subplot(111)

    if isinstance(x_raw, pd.Series) and isinstance(y_raw, pd.Series):
        xlabel, ylabel = x_raw.name, y_raw.name
    else:
        xlabel, ylabel = "Predictor", "Response"

    ax.set_xlabel(xlabel, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=20)
    ax.tick_params(axis="both", which="major", labelsize=18)

    ax.scatter(
        x_raw,
        y_raw,
        color="royalblue",
        s=33,
    )

    ax.plot(rslt["gridpoints"], rslt["curvest"], color="orange", linewidth=3)

    bw = rslt["bandwidth"]
    blue_patch = mpatches.Patch(color="royalblue", label="Raw data")
    orange_patch = mpatches.Patch(
        color="orange", label=f"Local polynomial smooth. Bandwidth = {bw:.3f}"
    )

    plt.legend(
        handles=[blue_patch, orange_patch],
        prop={"size": 16},
    )
    plt.show()
