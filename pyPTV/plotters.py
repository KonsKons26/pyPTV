import matplotlib.pyplot as plt
import matplotlib.cm as cm
from matplotlib.transforms import blended_transform_factory
from matplotlib.ticker import MaxNLocator
from matplotlib.colors import Normalize
from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection

import seaborn as sns

import k3d

import numpy as np


class Plotter:

    XKCD_COLORS = [
        f"xkcd:{c}" for c in [
            "royal blue", "bright blue", "sky blue", "cyan", "baby blue",
            "bright red", "coral", "brick", "red orange", "salmon",
            "olive green", "neon green", "mint green", "aqua green",
            "golden yellow", "lemon", "sandy",
            "electric pink", "purpleish pink",
        ]
    ]

    def __init__(
        self,
        figsize: tuple[int, int] = (16, 12),
        style: str = "whitegrid",
        context: str = "talk",
        cmap: str = "plasma",
        tight_layout: bool = True,
        seed: int = 42,
    ):
        self.figsize = figsize
        self.style = style
        self.context = context
        self.cmap = cmap
        self.tight_layout = tight_layout
        self.rng = np.random.default_rng(seed)
        self.colors = self.XKCD_COLORS
        self.color_limit = len(self.colors)

        sns.set_theme(style=style, context=context)

    def plot_series(
        self,
        x: np.ndarray,
        y: np.ndarray | list[np.ndarray],
        title: str,
        xlabel: str,
        ylabel: str,
        labels: str | list[str],
        colors: list[str],
        conv_window: int | None = None,
        **scatter_kwargs
    ) -> plt.Axes:
        """
        Plot series data with optional convolution smoothing.

        Parameters
        ----------
        - x : np.ndarray
            X-axis data
        - y : np.ndarray or list of np.ndarray
            Y-axis data series
        - title : str
            Plot title
        - xlabel : str
            X-axis label
        - ylabel : str
            Y-axis label
        - labels : str or list of str
            Labels for each data series
        - conv_window : int, optional
            Window size for convolution smoothing
        - **scatter_kwargs
            Additional keyword arguments for scatter plotting, including
            's' for size and 'alpha' for alpha (transparency).
        """
        if not isinstance(y, list):
            y = [y]
        if not isinstance(labels, list):
            labels = [labels]

        _, ax = plt.subplots(figsize=self.figsize)

        for i, y_series in enumerate(y):
            sns.scatterplot(
                x=x,
                y=y_series,
                ax=ax,
                color=colors[i],
                label=labels[i] if not conv_window else None,
                **scatter_kwargs
            )
            if conv_window:
                kernel = np.ones(conv_window) / conv_window
                y_convolved = np.convolve(y_series, kernel, mode="valid")
                offset = (conv_window - 1) // 2
                x_convolved = x[offset : offset + len(y_convolved)]

                sns.lineplot(
                    x=x_convolved,
                    y=y_convolved,
                    ax=ax,
                    color=colors[i],
                    label=labels[i],
                )

        ax.set_title(title)
        ax.set_xlabel(xlabel)
        ax.set_ylabel(ylabel)
        ax.legend()

        if self.tight_layout:
            plt.tight_layout()

        return ax

    def plot_heatmap(
        self,
        matrix: np.ndarray,
        cmap: str = "plasma",
        cbar_keywords: dict | None = None,
        square: bool = True,
        xticklabels: bool | list[str] = False,
        yticklabels: bool | list[str] = False,
    ) -> plt.Axes:
        _, ax = plt.subplots(figsize=self.figsize)
        sns.heatmap(
            matrix,
            cmap=cmap,
            cbar_kws=cbar_keywords,
            square=square,
            ax=ax,
            xticklabels=xticklabels,
            yticklabels=yticklabels,
        )
