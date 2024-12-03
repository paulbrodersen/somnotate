#!/usr/bin/env python

"""Determine intervals with missing data in the raw time series data.
If *any* signal appears to have missing values, the corresponding
interval is marked as missing.

EEG/EMG/LFP recordings often contain intervals with missing
data.

These events typically have one of two origins:

1) Missing values may occur at the start or end of the experiment,
when the recording hardware has been disconnected but the recording
software is still running.

2) Missing values can also occur movement artefacts, when the recorded
values exceed preset minimum or maximum values. As researchers often
inspect only bandpass-filtered signals, they may be unaware of the
existence of these events, as the missing values manifest themselves
as large transients in the traces due to length of the impulse
response of the filter.

"""

DEBUG = True

import numpy as np
import matplotlib.pyplot as plt

from somnotate._plotting import plot_signals, plot_states

from somnotate._utils import (
    _get_intervals,
)

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_raw_signals,
    export_review_intervals,
)


def is_missing(arr, missing_value_identifier=None):
    return np.apply_along_axis(_is_missing, axis=0, arr=arr,
                               missing_value_identifier=missing_value_identifier)


def _is_missing(vec, missing_value_identifier):
    # Missing values are all represented as the _same_ value.
    # We hence identify intervals with missing values as sequences
    # that remain constant. Constant sequences have a local
    # gradient and curvature of zero.

    output = np.zeros_like(vec)

    if missing_value_identifier:
        gradient = np.diff(vec)
        output[:-1] = np.logical_and(gradient==0, vec[:-1]==missing_value_identifier)

    else:
        gradient = (vec[2:] - vec[:-2]) / 2.
        curvature = np.diff(vec, 2)
        is_constant = np.logical_and(gradient==0, curvature==0)
        output[1:-1] = is_constant

    return output


if __name__ == '__main__':

    from configuration import (
        # artefact_annotation_signals,
        state_annotation_signals as artefact_annotation_signals,
    )

    # --------------------------------------------------------------------------------
    # parse and check inputs

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument("-s", "--show", action="store_true", help="Plot the output figures of the script.")
    parser.add_argument('--only',
                        nargs = '+',
                        type  = int,
                        help  = 'Indices corresponding to the rows to use (default: all). Indexing starts at zero.'
    )
    parser.add_argument("--missing_value_identifier", type=int, help="Value assigned to missing values in the raw data. Often zero (0). If none is provided, the value is inferred from the data by looking for continuous segments with identical values.")
    parser.add_argument("--pad_missing_value_intervals", type=int, default=4, help="If nonzero, intervals of missing values are padded by as many seconds.")

    args = parser.parse_args()

    # load spreadsheet / data frame
    datasets = load_dataframe(args.spreadsheet_file_path)

    # check contents of spreadsheet
    check_dataframe(datasets,
                    columns = [
                        'file_path_raw_signals',
                        'file_path_missing_value_intervals',
                    ],
                    column_to_dtype = {
                        'file_path_raw_signals' : str,
                        'file_path_missing_value_intervals' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    if args.missing_value_identifier is None:
        missing_value_identifier = None
    else:
        missing_value_identifier = args.missing_value_identifier

    if args.pad_missing_value_intervals:
        pad = args.pad_missing_value_intervals
    else:
        pad = 0

    # We do not want to mark every data point that is close to the
    # missing value identifier as missing, as there may be data points
    # that have a value equal or close to the missing value identifier
    # (as this is often zero). Therefor, we only mark several
    # consecutive missing value identifiers as missing values. We use
    # three consecutive entries as the automated determination of the
    # missing value identifier requires at least as many entries with
    # the same value to mark the entries in the interval as missing.
    # Thus we ensure consistency between the results when the missing
    # value identifier is given and when it is not.
    missing_value_interval_minimum_length = 100

    # --------------------------------------------------------------------------------
    print("Finding missing values in...")

    signal_arrays = []
    missing_value_vectors = []
    for ii, (idx, dataset) in enumerate(datasets.iterrows()):
        print("{} ({}/{})".format(dataset['file_path_raw_signals'], ii+1, len(datasets)))

        # load
        signal_labels = [dataset[column_name] for column_name in artefact_annotation_signals]
        raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)

        # determine time points with missing values (presumably due to very large movement artifacts)
        missing_value_vector = np.any(is_missing(raw_signals, missing_value_identifier=missing_value_identifier), axis=1)

        # remove intervals that are too short
        missing_value_intervals = _get_intervals(missing_value_vector)
        too_short = np.squeeze(np.diff(missing_value_intervals, axis=1) < missing_value_interval_minimum_length)
        missing_value_intervals = missing_value_intervals[~too_short]

        # pad left and right
        missing_value_intervals[:, 0] -= pad * dataset['sampling_frequency_in_hz']
        missing_value_intervals[:, 1] += pad * dataset['sampling_frequency_in_hz']

        # padding may introduce overlaps between consecutive intervals;
        # we want to merge the intervals in these cases
        missing_value_vector = np.zeros((len(raw_signals)), dtype=bool)
        for start, stop in missing_value_intervals:
            missing_value_vector[start:stop] |= True
        missing_value_intervals = _get_intervals(missing_value_vector)

        # change time resolution to seconds
        missing_value_intervals = missing_value_intervals / dataset['sampling_frequency_in_hz']

        export_review_intervals(
            dataset['file_path_missing_value_intervals'],
            missing_value_intervals,
            scores=np.squeeze(np.diff(missing_value_intervals, axis=1)),
        )

        if DEBUG:
            fig, axes = plt.subplots(2, 1, sharex=True)
            plot_signals(raw_signals, sampling_frequency=dataset['sampling_frequency_in_hz'], ax=axes[0])
            plot_states(['missing' for _ in missing_value_intervals], missing_value_intervals, ax=axes[1])
            axes[-1].set_xlabel('Time [seconds]')
            fig.tight_layout()
            fig.suptitle(dataset['file_path_raw_signals'])
            plt.show()
