#!/usr/bin/env python

"""
Using a pre-trained model, and given the preprocessed signal arrays,
annotate the state for each sample in the signal array, and save out a
1) a hypnogram, and
2) the likelihood of the predicted state for each sample.
The latter can then be used to manually refine annotation.
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

def export_intervals_with_state_probability_below_threshold(file_path, state_probability, threshold=0.99, time_resolution=1):
    """
    Determine all intervals, in which the probability of the predicted
    state is below the given threshold, and save them out to a CSV file.

    Arguments:
    ----------
    file_path -- str
        /path/to/file.csv

    state_probability_vector -- (total_samples, ) vector of float
        The probability of the predicted state.

    threshold -- float
        The probability threshold.

    Returns:
    --------
    intervals -- list of (start, stop) tuples
        The interval, for which the state probability is below the threshold.

    """
    intervals = _get_intervals(state_probability < threshold)
    scores = [_get_score(state_probability[start:stop]) for start, stop in intervals]
    intervals = [(start * time_resolution, stop * time_resolution) for start, stop in intervals]
    notes = ['probability below threshold' for _ in intervals]
    export_review_intervals(file_path, intervals, scores, notes)


def _get_score(vec):
    return len(vec) - np.sum(vec)


if __name__ == '__main__':

    from configuration import (
        state_to_int,
        int_to_state,
        plot_raw_signals,
        plot_states,
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
                        'file_path_automated_state_annotation',
                        'file_path_review_intervals',
                    ],
                    column_to_dtype = {
                        'file_path_preprocessed_signals' : str,
                        'file_path_automated_state_annotation' : str,
                        'file_path_review_intervals' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    annotator = StateAnnotator()
    annotator.load(args.trained_model_file_path)

    for ii, dataset in datasets.iterrows():

        print("{} ({}/{})".format(dataset['file_path_preprocessed_signals'], ii+1, len(datasets)))
        print("    Annotating states...")
        signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
        predicted_state_vector = annotator.predict(signal_array)
        predicted_states, predicted_intervals = convert_state_vector_to_state_intervals(
            predicted_state_vector, mapping=int_to_state, time_resolution=time_resolution)
        export_hypnogram(dataset['file_path_automated_state_annotation'], predicted_states, predicted_intervals)

        # compute intervals for manual review
        state_probability = annotator.predict_proba(signal_array)
        export_intervals_with_state_probability_below_threshold(dataset['file_path_review_intervals'],
                                                                state_probability,
                                                                threshold=0.99)

        if args.show:
            print("    Plotting...")
            fig, axes = plt.subplots(3, 1, sharex=True)

            # plot raw signals
            signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
            raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
            plot_raw_signals(
                raw_signals,
                sampling_frequency = dataset['sampling_frequency_in_hz'],
                ax                 = axes[0],
            )

            axes[1].plot(state_probability)
            axes[1].set_ylabel("Predicted state probability")

            # plot predicted states
            plot_states(predicted_states, predicted_intervals, ax=axes[2])
            axes[2].set_ylabel("Automated annotation")

            fig.tight_layout()
            fig.suptitle(dataset['file_path_preprocessed_signals'])

    plt.show()
