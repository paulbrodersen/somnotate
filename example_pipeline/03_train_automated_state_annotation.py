#!/usr/bin/env python

"""Train a model on data sets for which manual annotations exist.
This model can then be loaded at a later time point to automatically
annotate new data sets.
"""

import numpy as np

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_state_vector,
    load_preprocessed_signals,
)

from somnotate._automated_state_annotation import StateAnnotator

if __name__ == '__main__':

    from configuration import state_to_int

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument("trained_model_file_path", help="Save trained model at /path/to/trained_model.pickle")
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

    print("Loading data sets...")
    signal_arrays = []
    state_vectors = []
    for ii, dataset in datasets.iterrows():
        print("    {} ({}/{})".format(dataset['file_path_preprocessed_signals'], ii+1, len(datasets)))
        signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
        state_vector = load_state_vector(dataset['file_path_manual_state_annotation'], mapping=state_to_int)
        signal_arrays.append(signal_array)
        state_vectors.append(state_vector)

    print("Training model...")
    annotator = StateAnnotator()
    annotator.fit(signal_arrays, state_vectors)
    annotator.save(args.trained_model_file_path)
