#!/usr/bin/env python

"""
User defined variables and functions that are used across all scripts.
"""

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import (
    sosfilt,
    iirfilter,
    # cheby1,
)
from functools import partial
from somnotate._plotting import plot_signals, plot_states
from somnotate._utils import (
    pad_along_axis,
    remove_padding_along_axis,
)

# --------------------------------------------------------------------------------
# Raw signals used in the automated state annotation and their visual display
# --------------------------------------------------------------------------------

# define which columns in the spreadsheet/dataframe index the signals
# in the raw signal array that are to be used for state inference
state_annotation_signals = [
    'frontal_eeg_signal_label',
    'occipital_eeg_signal_label',
    'emg_signal_label',
]

# define the corresponding labels when plotting these signals
state_annotation_signal_labels = [
    'frontal\nEEG',
    'occipital\nEEG',
    'EMG'
]

# define the frequency bands to display when plotting
# (has no effect on signal processing and state inference)
state_annotation_signal_frequency_bands = [
    (0.5, 30.), # Frontal EEG
    (0.5, 30.), # Occipital EEG
    (10., 45.), # EMG
]

# define a function that plots the raw signals
def plot_raw_signals(raw_signals, frequency_bands, sampling_frequency, *args, **kwargs):
    """
    Thin wrapper around `plot_signals` that applies a Chebychev bandpass filter
    to the given signals before plotting.

    Arguments:
    ----------
    signals -- (total samples, total signals) ndarray
        The signals to plot.

    frequency_bands -- list of (float start, float stop)
        The frequency bands to use in the bandpass filter for each signal.

    sampling_frequency -- float
        The sampling frequency of the signals.

    *args, **kwargs -- passed through to plot_signals

    Returns:
    --------
    ax -- matplotlib.axes._subplots.AxesSubplot
        The axis plotted onto.

    See also:
    ---------
    plot_signals
    """

    # apply Chebychev bandpass filters to raw signals
    filtered = np.zeros_like(raw_signals)
    for ii, signal in enumerate(raw_signals.T):
        lowcut, highcut = frequency_bands[ii]
        filtered[:, ii] = chebychev_bandpass_filter(signal,
                                                    lowcut=lowcut,
                                                    highcut=highcut,
                                                    fs=sampling_frequency)

    ax = plot_signals(filtered,
                      sampling_frequency = sampling_frequency,
                      *args, **kwargs)

    return ax


def chebychev_bandpass_filter(data, lowcut, highcut, fs, rs=60, order=5, axis=-1):
    """
    Apply band pass filter with specified low and high cutoffs to data.

    Adapted from:
    -------------
    https://stackoverflow.com/a/12233959/2912349

    """

    # create filter
    chebychev = _chebychev_bandpass(lowcut, highcut, rs=rs, fs=fs, order=order)

    # pad data to minimize boundary effects
    pad_length = int(float(fs) / lowcut) # this is a very rough guess
    padded = pad_along_axis(data,
                            before=pad_length,
                            after=pad_length,
                            axis=axis,
                            mode='reflect')

    # apply filter to data
    filtered = sosfilt(chebychev, padded, axis=axis)

    # trim to remove padding
    trimmed = remove_padding_along_axis(filtered,
                                        before=pad_length,
                                        after=pad_length,
                                        axis=axis)
    return trimmed


def _chebychev_bandpass(lowcut, highcut, fs, rs, order=5):
    # https://stackoverflow.com/a/12233959/2912349
    nyq = 0.5 * fs
    low = lowcut / nyq
    high = highcut / nyq
    sos = iirfilter(order, [low, high], rs=rs, btype='band',
                    analog=False, ftype='cheby2', # fs=fs,
                    output='sos')
    # sos = cheby1(order, [low, high], rs=rs, analog=False, btype='band')
    return sos

plot_raw_signals = partial(plot_raw_signals,
                           frequency_bands=state_annotation_signal_frequency_bands,
                           signal_labels=state_annotation_signal_labels)

# --------------------------------------------------------------------------------
# States and their visual display
# --------------------------------------------------------------------------------

# Define a mapping of state labels (as they occur in a hypnogram) to integers.
# States with positive integers are used for training the LDA.
# When training the HMM, the sign of the state integer representation is ignored,
# such that artefacts have no influence on the computed state transition probabilities.
# TODO: explain the mapping of `sleep movement`.
state_to_int = dict([
    ('awake'              ,  1),
    ('awake (artefact)'   , -1),
    ('sleep movement'     ,  1),
    ('non-REM'            ,  2),
    ('non-REM (artefact)' , -2),
    ('REM'                ,  3),
    ('REM (artefact)'     , -3),
    ('undefined'          ,  0),
])

# Construct the inverse mapping to convert back from state predictions to human readabe labels.
int_to_state = {ii : state for state, ii in state_to_int.items() if state != 'sleep movement'}

# define the keymap used for the manual annotation
keymap = {
    'w' : 'awake',
    'W' : 'awake (artefact)',
    'n' : 'non-REM',
    'N' : 'non-REM (artefact)',
    'r' : 'REM',
    'R' : 'REM (artefact)',
    'x' : 'undefined',
    'X' : 'undefined (artefact)',
    'm' : 'sleep movement',
    'M' : 'sleep movement (artefact)',
}

# define the visual display of states
state_to_color = {
    'awake'               : 'crimson',
    'awake (artefact)'    : 'coral',
    'sleep movement'      : 'violet',
    'non-REM'             : 'blue',
    'non-REM (artefact)'  : 'cornflowerblue',
    'REM'                 : 'gold',
    'REM (artefact)'      : 'yellow',
    'sleep movement'      : 'purple',
    'undefined'           : 'gray',
    'undefined (artefact)': 'lightgray',
}

state_display_order = [
    'awake',
    'awake (artefact)',
    'sleep movement',
    'non-REM',
    'non-REM (artefact)',
    'REM',
    'REM (artefact)',
    'sleep movement',
    'sleep movement (artefact)',
    'undefined',
    'undefined (artefact)',
]

plot_states = partial(plot_states,
                      unique_states=state_display_order,
                      state_to_color=state_to_color,
                      mode='lines')

# --------------------------------------------------------------------------------
# Figure formatting
# --------------------------------------------------------------------------------

plt.rcParams['figure.figsize'] = (30, 15)
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['axes.labelsize']  = 'medium'

# --------------------------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------------------------

# desired time resolution of the automated annotation (in seconds)
time_resolution = 1
