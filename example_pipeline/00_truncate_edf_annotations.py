#!/usr/bin/env python
"""
EDF annotations can have overhangs. We need to remove them prior to running the pipeline.
"""

from argparse import ArgumentParser
from pyedflib import EdfReader

from data_io import (
    _load_edf_hypnogram,
    _export_edf_hypnogram,
    load_dataframe,
    check_dataframe,
)

if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # parse and check inputs

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
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
                        'file_path_raw_signals',
                        'file_path_manual_state_annotation',
                    ],
                    column_to_dtype = {
                        'file_path_raw_signals' : str,
                        'file_path_manual_state_annotation' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    for ii, dataset in datasets.iterrows():
        print("{} ({}/{})".format(dataset['file_path_manual_state_annotation'], ii+1, len(datasets)))

        file_path_raw_signals = dataset['file_path_raw_signals']
        with EdfReader(file_path_raw_signals) as f:
            total_time_in_seconds = f.file_duration
            edf_header = f.getHeader()
            
        file_path_manual_state_annotation = dataset['file_path_manual_state_annotation']
        states, intervals = _load_edf_hypnogram(file_path_manual_state_annotation)

        start, stop = intervals[-1]
        while stop > total_time_in_seconds:
            if start >= total_time_in_seconds:
                intervals.pop()
                states.pop()
            elif (start < total_time_in_seconds) and (stop > total_time_in_seconds):
                intervals[-1] = (start, total_time_in_seconds)
            start, stop = intervals[-1]
          
        _export_edf_hypnogram(file_path_manual_state_annotation, states, intervals, edf_header)
