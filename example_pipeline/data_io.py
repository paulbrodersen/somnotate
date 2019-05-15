#!/usr/bin/env python

"""
TODO:
- define __all__ or limit available functions in some other way.
"""

import numpy as np
import pandas

from argparse import ArgumentParser
from collections import Iterable
from pyedflib import EdfReader
from six import ensure_str

from somnotate._utils import convert_state_intervals_to_state_vector


def check_dataframe(df, columns, column_to_dtype=None):
    """
    This function tests if a pandas dataframe has certain columns.
    If the data type for a column is given, then the type of that column
    is also checked.

    Arguments:
    ----------
    df -- pandas.DataFrame object
        The imported spreadsheet.

    columns -- list of str
        The columns, whose existence should be checked.

    column_to_dtype -- dict str : type
        The expected type of a given column.
        Not all values in `columns` have to be present in `column_to_dtype`
        (and vice versa).

    """

    not_present = []
    for column in columns:
        if not (column in df.columns):
            not_present.append(column)

    if not_present: # empty list evaluates to False
        error_msg = "The provided spreadsheet misses the following columns:"
        for column in not_present:
            error_msg += "\n{}".format(column)
        raise Exception(error_msg)

    if column_to_dtype:
        wrong_types = []
        for column, expected_dtype in column_to_dtype.items():
            # pandas encodes all strings as objects,
            # which is not very helpful for type checking;
            # here we set the actual dtype to str, if the object is string-like.
            if pandas.api.types.is_string_dtype(df[column]):
                actual_dtype = str
            else:
                actual_dtype = df[column].dtype
            # print(expected_dtype, actual_dtype)
            if isinstance(expected_dtype, type):
                if not actual_dtype == expected_dtype:
                    wrong_types.append((column, actual_dtype, expected_dtype))
            elif isinstance(expected_dtype, Iterable):
                if not actual_dtype in expected_dtype:
                    wrong_types.append((column, actual_dtype, expected_dtype))
            else:
                type_error_msg = "Values in column_to_dtype have to be either instances of type or an iterable thereof. Currently:"
                type_error_msg += "\ntype(column_to_dtype[{}] = {})".format(column, type(column_to_type[column]))
                raise TypeError(type_error_msg)

        if wrong_types:
            error_msg = "The following columns have the wrong data type:"
            for entry in wrong_types:
                error_msg += "\n{}: dtype is {} but should be {};".format(*entry)
            raise Exception(error_msg)


def _load_edf_file(file_path, signal_labels=None):
    """Simple wrapper around pyedflib to load LFP/EEG/EMG traces from EDF
    files and return a numpy array. Currently, only loading of signals
    with the same sampling frequency and number of samples is supported.

    Arguments:
    ----------
    file_path -- str
        /path/to/file.edf

    signal_labels -- list of str or None (default None)
        Labels of the signals to load. If None, all signals are loaded.

    Returns:
    --------
    signals -- (total samples, total signals) ndarray
        The signals concatenated into a single numpy array.

    """
    with EdfReader(file_path) as reader:
        if signal_labels is None:
            signal_labels = reader.getSignalLabels()
        signals = _load_edf_channels(signal_labels, reader)
    return signals


def _load_edf_channels(signal_labels, edf_reader):

    indices = [idx for idx in range(edf_reader.signals_in_file) if ensure_str(edf_reader.signal_label(idx)).strip() in signal_labels]
    assert len(indices) == len(signal_labels), "Could not recover all given signals."

    total_samples = [edf_reader.samples_in_file(idx) for idx in indices]
    assert len(set(total_samples)) == 1, "All signals need to have the same length! Lengths of selected signals: {}".format(total_samples)
    total_samples, = set(total_samples)

    output_array = np.zeros((len(signal_labels), total_samples), dtype=np.int32)
    for jj, idx in enumerate(indices):
        edf_reader.read_digital_signal(idx, 0, total_samples, output_array[jj])

    return output_array.transpose()


def load_state_vector(file_path, mapping, time_resolution=1.):
    """
    Load hypnogram given in visbrain Stage-duration format, and convert to a state vector.

    Arguments:
    ----------
    file_path -- str
        /path/to/hypnogram/file.hyp

    mapping -- dict str : int or None (default state_to_int)
        Mapping of state representations in the hypnogram to integers.
        If None, states returned by `get_hypnogram` must already be integers.

    time_resolution -- float (default 1.)
        The (desired) duration of each entry in the state vector in seconds.

    Returns:
    --------
    state_vector -- (total samples, ) ndarray of int
        The state vector.

    References:
    -----------
    http://visbrain.org/sleep.html#save-hypnogram
    """

    states, intervals = load_hypnogram(file_path)

    from somnotate._utils import convert_state_intervals_to_state_vector
    state_vector = convert_state_intervals_to_state_vector(states, intervals, mapping=mapping)

    return state_vector


def _load_visbrain_hypnogram(file_path):
    """
    Load hypnogram given in visbrain Stage-duration format.

    Arguments:
    ----------
    file_path -- str
        /path/to/hypnogram/file.hyp

    Returns:
    --------
    states -- list of str
        List of annotated states.

    intervals -- list of (float start, float stop) tuples
        Corresponding time intervals.

    References:
    -----------
    http://visbrain.org/sleep.html#save-hypnogram
    """
    dtype = [('Stage', '|S30'), ('stop', float)]
    data = np.genfromtxt(file_path, skip_header=2, dtype=dtype)
    states = [state.astype(str) for state in data['Stage']]
    transitions = np.r_[0, data['stop']]
    intervals = list(zip(transitions[:-1], transitions[1:]))
    return states, intervals


def _export_visbrain_hypnogram(file_path, states, intervals, total_time=None, data_file=None):
    """
    Export hypnogram to visbrain Stage-duration format.

    Arguments:
    ----------
    file_path -- str
        /path/to/hypnogram/file.hyp

    states -- list of str
        List of annotated states.

    intervals -- list of (float start, float stop) tuples
        Corresponding time intervals.

    total_time -- int or None (default None)
        Total time covered by the annotation.
        If None, the maximum of `intervals` is used instead.

    data_file -- str or None (default None)
        /path/to/corresponding/data/file_name.edf or just file_name.edf
        Optional argument. If provided, a reference to the corrsponding data file is added to the hypnogram.

    References:
    -----------
    http://visbrain.org/sleep.html#save-hypnogram

    """

    export_string = ""

    # --------------------------------------------------------------------------------
    # header

    if total_time:
        export_string += "*Duration_sec\t{:.1f}\n".format(total_time)
    else:
        export_string += "*Duration_sec\t{:.1f}\n".format(np.max(intervals))

    if data_file:
        export_string += "*Datafile\t{}\n".format(data_file)
    else:
        export_string += "*Datafile\tUnspecified\n"

    # --------------------------------------------------------------------------------
    # body

    # determine the length of the longest state label
    unique_states = set(states)
    string_length = max([len(s) for s in unique_states])

    # sort by interval start
    intervals = np.array(intervals)
    order = np.argsort(intervals[:, 0])
    intervals = intervals[order]
    states = [states[ii] for ii in order]

    # assert that no two intervals overlap
    assert not np.any(intervals[:-1, 1] > intervals[1:, 0]), "The hypnogram format does not support overlapping intervals!"

    # assert that there are no un-annotated gaps between intervals
    assert np.all(intervals[:-1, 1] == intervals[1:, 0]), "The hypnogram format does not support having un-annotated time intervals!"

    # TODO: instead insert empty states for time intervals with no corresponding state

    export_string += "".join(["{:{}}\t{:.1f}\n".format(state, string_length, stop) \
                              for state, (start, stop) in zip(states, intervals)])

    with open(file_path, 'w') as f:
        f.write(export_string)


def export_review_intervals(file_path, intervals, scores=None, notes=None):
    """
    TODO: implement mode that appends data to existing file
    - load file if path exists
    - combine data frames
    - save out combined DF
    """

    data = dict()

    data['start'] = [start for start, stop in intervals]
    data['stop'] = [stop for start, stop in intervals]

    if not (scores is None):
        data['score'] = scores

    if not (notes is None):
        data['note'] = notes

    df = pandas.DataFrame.from_dict(data)
    df.to_csv(file_path)


def load_review_intervals(file_path):
    df = pandas.read_csv(file_path)
    intervals = np.c_[df['start'].values, df['stop'].values]
    scores = df['score'].values
    return intervals, scores

# --------------------------------------------------------------------------------
# aliases

load_dataframe = pandas.read_csv
load_raw_signals = _load_edf_file
load_preprocessed_signals = np.load
export_preprocessed_signals = np.save
load_hypnogram = _load_visbrain_hypnogram
export_hypnogram = _export_visbrain_hypnogram
