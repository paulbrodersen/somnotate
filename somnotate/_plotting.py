#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt

from ._utils import (
    robust_normalize,
    truncate_signals,
    pad_along_axis,
    remove_padding_along_axis,
)

def plot_signals(signals,
                 sampling_frequency = 1,
                 remove_outliers    = True,
                 rescale_signal     = True,
                 signal_labels      = None,
                 ax                 = None,
                 *args, **kwargs):
    """
    Convenience function to quickly plot several signals onto one
    axis. The signals are (optionally) rescaled and then offset from
    one another such that they do not overlap.

    Arguments:
    ----------
    signals -- (total samples, total signals) ndarray
        The signals to plot.

    sampling_frequency -- float
        The sampling frequency of the signals.

    remove_outliers -- boolean (default True)
        If True, truncate signals such to the 1st - 99th percentile.

    rescale_signal -- boolean (True)
        If True, rescale signals to the unit interval.

    signal_labels -- list of str or None (default None)
        The labels corresponding to each signal.

    ax -- matplotlib.axes._subplots.AxesSubplot or None (default None)
        The axis to plot onto. If None is provided, the current axis is used.

    *args, **kwargs -- passed through to plt.plot

    Returns:
    --------
    ax -- matplotlib.axes._subplots.AxesSubplot
        The axis plotted onto.

    """

    if not ax:
        ax = plt.gca()

    # ensure that the signals ndarray is 2-dimensional
    if signals.ndim == 1:
        signals = signals.reshape((len(signals), 1))

    if remove_outliers:
        signals = truncate_signals(signals, 0.05, 99.95, axis=0)

    if rescale_signal:
        signals = robust_normalize(signals, p=1., axis=0, method='min-max')

    # reverse order of signals and labels such that the first signal ends up at the top of the plot
    signals = signals[:, ::-1]
    if signal_labels:
        signal_labels = signal_labels[::-1]

    # offset signals from each other such that they can be plotted on one axis
    signals = signals - np.nanmin(signals, axis=0)[None, :]
    offsets = np.cumsum(np.nanmax(signals, axis=0))
    signals[:, 1:] = signals[:, 1:] + offsets[:-1][None, :]

    # shift median of first signal to zero
    signals -= np.median(signals[:, 0])

    # create a time vector
    T, N = signals.shape
    time = np.arange(0, float(T)/sampling_frequency, 1./sampling_frequency)

    for ii, signal in enumerate(signals.T):
        ax.plot(time, signal, *args, **kwargs)
        ax.axhline(np.median(signal), color='k', linewidth=1., linestyle=':')

    if signal_labels:
        ax.set_yticks(np.median(signals, axis=0))
        ax.set_yticklabels(signal_labels)

    return ax


def plot_states(states, intervals,
                unique_states=None,
                state_to_color=None,
                mode="lines",
                ax=None,
                *args, **kwargs):

    if not unique_states:
        unique_states = list(set(states))
    else:
        # compute intersection with available states to prevent KeyErrors down the line
        # unique_states = set(unique_states) & set(states)
        unique_states = [state for state in unique_states if state in set(states)] # keep original order

    if not state_to_color: # use default color cycle
        prop_cycle = plt.rcParams['axes.prop_cycle']
        colors = prop_cycle.by_key()['color']
        state_to_color = dict()
        for ii, state in enumerate(unique_states):
            state_to_color[state] = colors[ii]

    if not ax:
        ax = plt.gca()

    data = dict()
    for state, (start, stop) in zip(states, intervals):
        x = [np.nan, start, stop, np.nan]
        if state in data:
            data[state] = np.r_[data[state], x]
        else:
            data[state] = x

    if mode == 'background':

        min_y, max_y = ax.get_ylim()

        for unique_state in unique_states:
            x = data[unique_state]
            y = np.tile([np.nan, max_y, max_y, np.nan], len(x) / 4)
            ax.fill_between(x, y, y2=min_y,
                            facecolor=state_to_color[unique_state],
                            zorder=-1, label=unique_state,
                            *args, **kwargs)

    elif mode == 'lines':

        yticks       = []
        ytick_labels = []

        for ii, unique_state in enumerate(unique_states[::-1]):

            x = data[unique_state]
            y = np.full_like(x, ii)
            ax.plot(x, y, color=state_to_color[unique_state], *args, **kwargs)

            yticks.append(ii)
            ytick_labels.append(unique_state)

        ax.set_yticks(yticks)
        ax.set_yticklabels(ytick_labels)

    else:
        raise ValueError("Mode one of: 'background', 'lines'. Currently: {}".format)(mode)

    return ax


def subplots(nrows=1, ncols=1, *args, **kwargs):
    """
    Make plt.subplots return an array of axes even is there is only one axis.
    """
    import matplotlib.pyplot as plt
    fig, axes = plt.subplots(nrows, ncols, *args, **kwargs)
    return fig, np.reshape(axes, (nrows, ncols))
