from collections.abc import Iterable

import matplotlib.pyplot as plt
import seaborn as sns


def easy_subplots(ncols=1, nrows=1, base_figsize=None, **kwargs):

    if base_figsize is None:
        base_figsize = (8, 5)

    fig, axes = plt.subplots(
        ncols=ncols,
        nrows=nrows,
        figsize=(base_figsize[0] * ncols, base_figsize[1] * nrows),
        **kwargs
    )

    # Lazy way of doing this
    try:
        axes = axes.reshape(-1)
    except:
        pass

    return fig, axes


def despine(fig=None, axes=None):
    if fig is not None:
        sns.despine(trim=True, offset=0.5, fig=fig)

    elif axes is not None:

        if not isinstance(axes, Iterable):  # to generalize to a single ax
            axes = [axes]

        for ax in axes:
            sns.despine(trim=True, offset=0.5, ax=ax)

    else:
        fig = plt.gcf()
        sns.despine(trim=True, offset=0.5, fig=fig)


def savefig(fig, filename, ft=None):
    if ft is None:
        ft = "jpg"
    fig.savefig("{}.{}".format(filename, ft), dpi=300, bbox_inches="tight")
