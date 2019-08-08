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
    'actigraphy_label',
]

# define the corresponding labels when plotting these signals
state_annotation_signal_labels = [
    'actigraphy',
]


# define a function that plots the raw signals
plot_raw_signals = partial(plot_signals,
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
    ('awake'      , 1),
    ('sleep'     ,  2),
    ('undefined' ,  0),
])

# Construct the inverse mapping to convert back from state predictions to human readabe labels.
int_to_state = {ii : state for state, ii in state_to_int.items()}

# define the keymap used for the manual annotation
keymap = {
    'w' : 'awake',
    'W' : 'awake (artefact)',
    's' : 'sleep',
    'S' : 'sleep (artefact)',
    'x' : 'undefined',
    'X' : 'undefined (artefact)',
}

# define the visual display of states
state_to_color = {
    'awake'               : 'crimson',
    'awake (artefact)'    : 'coral',
    'sleep'               : 'blue',
    'sleep (artefact)'    : 'cornflowerblue',
    'undefined'           : 'gray',
    'undefined (artefact)': 'lightgray',
}

state_display_order = [
    'awake',
    'awake (artefact)',
    'sleep',
    'sleep (artefact)',
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

plt.rcParams['figure.figsize'] = (12, 6)
plt.rcParams['ytick.labelsize'] = 'medium'
plt.rcParams['xtick.labelsize'] = 'medium'
plt.rcParams['axes.labelsize']  = 'medium'

# --------------------------------------------------------------------------------
# Miscellaneous
# --------------------------------------------------------------------------------

# desired time resolution of the automated annotation (in seconds)
# TODO: might need changing to 10
time_resolution = 1

# default view length when manually annotating states
default_view_length = 60.

# default selection length when manually annotating states
default_selection_length = 4.
