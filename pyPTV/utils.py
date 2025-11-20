import numpy as np

import pandas as pd

from sklearn.decomposition import PCA
from sklearn.manifold import TSNE

import umap

from matplotlib.collections import LineCollection
from mpl_toolkits.mplot3d.art3d import Line3DCollection



def dim_red(
    data: np.ndarray,
    method: str,
    **kwargs,
) -> pd.DataFrame:
    """
    """

    method = method.lower()

    if method == "pca":
        show_variance = kwargs.pop("show_variance", False)

        pca = PCA(**kwargs)
        reduced_matrix = pca.fit_transform(data)

        if show_variance:
            variance = pca.explained_variance_ratio_
            columns = [
                f"PC{i+1} {var:.2%}"
                for i, var in enumerate(variance)
            ]
        else:
            columns = [
                f"PC{i+1}"
                for i in range(reduced_matrix.shape[1])
            ]

    elif method == "tsne":
        if "n_components" in kwargs:
            if kwargs["n_components"] > 3:
                print(
                    "Warning:"
                    f" t-SNE with n_components={kwargs["n_components"]}"
                    " > 3 is computationally expensive and rarely used."
                )

        tsne = TSNE(**kwargs)
        reduced_matrix = tsne.fit_transform(data)
        columns = [
            f"TSNE{i+1}"
            for i in range(reduced_matrix.shape[1])
        ]

    elif method == "umap":
        umap_reducer = umap.UMAP(**kwargs)
        reduced_matrix = umap_reducer.fit_transform(data)
        columns = [
            f"UMAP{i+1}"
            for i in range(reduced_matrix.shape[1])
        ]

    else:
        raise ValueError(
            "Method must be one of 'pca', 'tsne', or 'umap'"
        )

    return pd.DataFrame(
        reduced_matrix,
        columns=columns,
    )


# Stolen from:
# https://matplotlib.org/stable/gallery/lines_bars_and_markers/multicolored_line.html
def colored_line_2D(x, y, c, ax, **lc_kwargs):
    """
    Plot a line with a color specified along the line by a third value.

    It does this by creating a collection of line segments. Each line segment is
    made up of two straight lines each connecting the current (x, y) point to the
    midpoints of the lines connecting the current point with its two neighbors.
    This creates a smooth line with no gaps between the line segments.

    Parameters
    ----------
    x, y : array-like
        The horizontal and vertical coordinates of the data points.
    c : array-like
        The color values, which should be the same size as x and y.
    ax : Axes
        Axis object on which to plot the colored line.
    **lc_kwargs
        Any additional arguments to pass to matplotlib.collections.LineCollection
        constructor. This should not include the array keyword argument because
        that is set to the color argument. If provided, it will be overridden.

    Returns
    -------
    matplotlib.collections.LineCollection
        The generated line collection representing the colored line.
    """

    # Default the capstyle to butt so that the line segments smoothly line up
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Compute the midpoints of the line segments. Include the first and last points
    # twice so we don't need any special syntax later to handle them.
    x = np.asarray(x)
    y = np.asarray(y)
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))

    # Determine the start, middle, and end coordinate pair of each line segment.
    # Use the reshape to add an extra dimension so each pair of points is in its
    # own list. Then concatenate them to create:
    # [
    #   [(x1_start, y1_start), (x1_mid, y1_mid), (x1_end, y1_end)],
    #   [(x2_start, y2_start), (x2_mid, y2_mid), (x2_end, y2_end)],
    #   ...
    # ]
    coord_start = np.column_stack((x_midpts[:-1], y_midpts[:-1]))[:, np.newaxis, :]
    coord_mid = np.column_stack((x, y))[:, np.newaxis, :]
    coord_end = np.column_stack((x_midpts[1:], y_midpts[1:]))[:, np.newaxis, :]
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = LineCollection(segments, **default_kwargs)
    lc.set_array(c)  # set the colors of each segment

    return ax.add_collection(lc)


def colored_line_3d(x, y, z, c, ax, **lc_kwargs):
    """
    Plot a 3D line with a color specified along the line by a fourth value.
    
    Parameters
    ----------
    x, y, z : array-like
        The coordinates of the data points.
    c : array-like
        The color values, same size as x, y, and z.
    ax : Axes3D
        The 3D axis object.
    **lc_kwargs
        Arguments passed to Line3DCollection.
    """
    # Default capstyle
    default_kwargs = {"capstyle": "butt"}
    default_kwargs.update(lc_kwargs)

    # Ensure arrays
    x = np.asarray(x)
    y = np.asarray(y)
    z = np.asarray(z)

    # Compute midpoints for x, y, and z
    x_midpts = np.hstack((x[0], 0.5 * (x[1:] + x[:-1]), x[-1]))
    y_midpts = np.hstack((y[0], 0.5 * (y[1:] + y[:-1]), y[-1]))
    z_midpts = np.hstack((z[0], 0.5 * (z[1:] + z[:-1]), z[-1]))

    # Determine start, middle, and end coordinates
    # Stack x, y, and z for start points
    coord_start = np.column_stack(
        (x_midpts[:-1], y_midpts[:-1], z_midpts[:-1])
    )[:, np.newaxis, :]
    
    # Stack x, y, and z for mid points
    coord_mid = np.column_stack(
        (x, y, z)
    )[:, np.newaxis, :]
    
    # Stack x, y, and z for end points
    coord_end = np.column_stack(
        (x_midpts[1:], y_midpts[1:], z_midpts[1:])
    )[:, np.newaxis, :]

    # Concatenate to form segments: (N, 3, 3)
    segments = np.concatenate((coord_start, coord_mid, coord_end), axis=1)

    lc = Line3DCollection(segments, **default_kwargs)
    lc.set_array(c)

    return ax.add_collection(lc)