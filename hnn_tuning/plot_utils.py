"""Shared plotting helper functions.
Author: Katharina Duecker katharina.duecker@gmail.com
Last checked: 18-Oct-2026
"""
import numpy as np


def get_cmap_neurons():
    """Line colors for the 4 different neuron types (L2 Int, L2 Pyr, L5 Int, L5 Pyr)
    """
    return np.array([[218, 166, 156], [164, 30, 79], [119, 161, 187], [92, 114, 183]]) / 255


# def assign_color(net, gid):
#     """Return the cell-type colour for a given gid in a network.

#     DEPRECATED: hnn-core now assigns per-cell-type colours when plotting spiking
#     data. Kept only until the remaining network_sim callers migrate; do not use
#     in new code.
#     """
#     import matplotlib

#     cm_neurons = get_cmap_neurons()
#     cmap = matplotlib.colors.ListedColormap(cm_neurons)

#     cell_type_colors = {'L2_inhibitory': cmap.colors[0], 'L2_pyramidal': cmap.colors[1],
#                         'L5_inhibitory': cmap.colors[2], 'L5_pyramidal': cmap.colors[3]}

#     for cell_type, gid_range in net.gid_ranges.items():
#         if cell_type in list(cell_type_colors.keys()):
#             if gid in gid_range:
#                 return cell_type_colors[cell_type]


def add_scalebar(ax, x_len, y_len, x_label=None, y_label=None,
                 anchor=(-0.03, -0.03), color='k', lw=1.25, fontsize=8,
                 label_pad=0.03, flip_x=False):
    """Draw an L-shaped scale bar and strip the axes frame.


    Parameters
    ----------
    ax : matplotlib.axes.Axes
        Axes to annotate. 
    x_len, y_len : float
        Scale-bar lengths in DATA units (e.g. ``x_len=200`` for 200 ms,
        ``y_len=50`` for 50 mV). These differ per panel / simulation / cell type.
    x_label, y_label : str, optional
        Text labels for the horizontal and vertical bars (e.g. ``'200 ms'``,
        ``'50 mV'``). ``None`` (default) draws the bar without a label.
    anchor : (float, float)
        Corner of the L in axes fraction, ``(0, 0)`` = lower-left. 
    flip_x : bool
        Mirror the L horizontally.
    color, lw, fontsize, label_pad : styling.

    Returns
    -------
    ax
    """
    ax.set_xticks([])
    ax.set_yticks([])
    for spine in ax.spines.values():
        spine.set_visible(False)

    x0, x1 = ax.get_xlim()
    y0, y1 = ax.get_ylim()
    # bar lengths as axes fractions
    fx = x_len / (x1 - x0)
    fy = y_len / (y1 - y0)
    ax_x, ax_y = anchor
    sx = -1 if flip_x else 1   # horizontal direction (flip mirrors it)

    # horizontal (x) and vertical (y) segments forming an L
    ax.plot([ax_x, ax_x + sx * fx], [ax_y, ax_y], transform=ax.transAxes,
            color=color, lw=lw, clip_on=False, solid_capstyle='butt')
    ax.plot([ax_x, ax_x], [ax_y, ax_y + fy], transform=ax.transAxes,
            color=color, lw=lw, clip_on=False, solid_capstyle='butt')

    if x_label is not None:
        ax.text(ax_x + sx * fx / 2, ax_y - label_pad, x_label, transform=ax.transAxes,
                ha='center', va='top', fontsize=fontsize)
    if y_label is not None:
        # y-label sits opposite the horizontal bar (left for the normal L, right when flipped)
        ax.text(ax_x - sx * label_pad, ax_y + fy / 2, y_label, transform=ax.transAxes,
                ha=('left' if flip_x else 'right'), va='center', rotation=90, fontsize=fontsize)

    return ax
