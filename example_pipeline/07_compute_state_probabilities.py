#!/usr/bin/env python

"""
Using a pre-trained model, and given the preprocessed signal
arrays, compute the probability of each state for each sample in the
signal array.

"""

import numpy as np
import matplotlib.pyplot as plt

from somnotate._automated_state_annotation import StateAnnotator
from somnotate._utils import (
    convert_state_vector_to_state_intervals,
    _get_intervals,
)

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_preprocessed_signals,
    load_raw_signals,
    export_hypnogram,
    export_review_intervals,
)


if __name__ == '__main__':

    from configuration import (
        # state_to_int,
        int_to_state,
        state_to_color,
        plot_raw_signals,
        # plot_signals,
        # plot_states,
        state_annotation_signals,
        time_resolution,
    )

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument("trained_model_file_path", help="Use trained model saved at /path/to/trained_model.pickle")
    parser.add_argument("-s", "--show", action="store_true", help="Plot the output figures of the script.")
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
                        'file_path_state_probabilities',
                    ],
                    column_to_dtype = {
                        'file_path_preprocessed_signals' : str,
                        'file_path_state_probabilities' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    annotator = StateAnnotator()
    annotator.load(args.trained_model_file_path)

    for ii, (idx, dataset) in enumerate(datasets.iterrows()):

        print("{} ({}/{})".format(dataset['file_path_preprocessed_signals'], ii+1, len(datasets)))
        print("    Computing state probability...")
        signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
        state_probabilities = annotator.predict_all_probabilities(signal_array)
        state_probabilities = {int_to_state[k]: v for k, v in state_probabilities.items()}
        np.savez(dataset["file_path_state_probabilities"], **state_probabilities)

        if args.show:
            print("    Plotting...")
            fig, axes = plt.subplots(2, 1, sharex=True)

            # plot raw signals
            signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
            raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
            plot_raw_signals(
                raw_signals,
                sampling_frequency = dataset['sampling_frequency_in_hz'],
                ax                 = axes[0],
            )

            for state, probability in state_probabilities.items():
                time = np.arange(0, time_resolution * len(probability), time_resolution)
                axes[1].plot(time, probability, color=state_to_color[state])
            axes[1].set_ylabel("State probability")

            fig.tight_layout()
            fig.suptitle(dataset['file_path_preprocessed_signals'])

    plt.show()
