"""Figure-saving helper.

Keeps all plots going to reports/figures/ with descriptive filenames so notebooks
don't each reinvent save logic.
"""

import matplotlib.pyplot as plt

from . import paths


def save_fig(fig, filename: str, *, dpi: int = 150):
    """Save a matplotlib figure to reports/figures/ with a descriptive name.

    Parameters
    ----------
    fig : matplotlib.figure.Figure
    filename : str
        Descriptive name, e.g. "scree_plot.png", "class_balance.png".
    dpi : int, default 150
    """
    paths.FIGURES.mkdir(parents=True, exist_ok=True)
    fig.savefig(paths.FIGURES / filename, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
