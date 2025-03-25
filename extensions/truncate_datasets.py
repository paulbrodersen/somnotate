#!/usr/bin/env python

"""
Truncate manual state annotations and preprocessed datasets to a given (start, stop) interval.
Truncated files will have the same name as the original apart from a suffix noting the start and stop time points.
"""

import numpy as np
import matplotlib.pyplot as plt
import pathlib

from somnotate._automated_state_annotation import StateAnnotator
from somnotate._utils import convert_state_vector_to_state_intervals
from somnotate._plotting import plot_signals

from example_pipeline.data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_preprocessed_signals,
    load_state_vector,
    export_hypnogram,
    export_preprocessed_signals,
)


if __name__ == '__main__':

    from example_pipeline.configuration import (
        state_to_int,
        int_to_state,
        time_resolution,
    )

    # --------------------------------------------------------------------------------
    # parse and check inputs

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument('start', help="Start of truncated data set in seconds.", type=float)
    parser.add_argument('stop', help="End of truncated data set in seconds.", type=float)
    parser.add_argument('--only',
                        nargs = '+',
                        type  = int,
                        help  = 'Indices corresponding to the rows to use (default: all). Indexing starts at zero.'
    )

    args = parser.parse_args()

    # load spreadsheet / data frame
    datasets = load_dataframe(args.spreadsheet_file_path)

    # check contents of spreadsheet
    check_dataframe(datasets,
                    columns = [
                        'file_path_preprocessed_signals',
                        'file_path_manual_state_annotation',
                    ],
                    column_to_dtype = {
                        'file_path_preprocessed_signals' : str,
                        'file_path_manual_state_annotation' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    start = int(args.start / time_resolution)
    stop = int(args.stop / time_resolution)

    # --------------------------------------------------------------------------------
    print("Loading data sets...")

    signal_arrays = []
    state_vectors = []
    for ii, dataset in datasets.iterrows():
        print("{} ({}/{})".format(dataset['file_path_preprocessed_signals'], ii+1, len(datasets)))

        signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
        state_vector = load_state_vector(dataset['file_path_manual_state_annotation'], mapping=state_to_int)

        assert len(signal_array) > start, "The start time point is outside the data range."
        assert len(signal_array) >= stop, "The stop time point is outside the data range."
        assert len(state_vector) > start, "The start time point is outside the data range."
        assert len(state_vector) >= stop, "The stop time point is outside the data range."

        signal_array = signal_array[start:stop]
        state_vector = state_vector[start:stop]

        old_path = dataset['file_path_preprocessed_signals']
        old_suffix = pathlib.Path(old_path).suffix
        new_suffix = f'_start_{start}_stop_{stop}' + old_suffix
        new_path = old_path.replace(old_suffix, new_suffix)
        export_preprocessed_signals(new_path, signal_array)

        states, intervals = convert_state_vector_to_state_intervals(state_vector, mapping=int_to_state)
        old_path = dataset['file_path_manual_state_annotation']
        old_suffix = pathlib.Path(old_path).suffix
        new_suffix = f'_start_{start}_stop_{stop}' + old_suffix
        new_path = old_path.replace(old_suffix, new_suffix)
        export_hypnogram(new_path, states, intervals)
