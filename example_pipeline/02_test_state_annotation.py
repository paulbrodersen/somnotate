#!/usr/bin/env python

"""Train and test in a hold-one-out fashion on data sets for which a
manual created state annotation exists.
"""

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import confusion_matrix as get_confusion_matrix

from somnotate._automated_state_annotation import StateAnnotator
from somnotate._utils import convert_state_vector_to_state_intervals
from somnotate._plotting import plot_signals

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_preprocessed_signals,
    load_state_vector,
)


if __name__ == '__main__':

    from configuration import (
        state_to_int,
        int_to_state,
        time_resolution,
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
    parser.add_argument("--model", help="Use pre-trained model saved at /path/to/trained_model.pickle. If none is provided, the test is run in a hold-one-out fashion instead.")
    parser.add_argument("--savefile", help="If provided with a /path/to/save/file.npz, the script saves the results of the analysis to file." )

    args = parser.parse_args()

    # load spreadsheet / data frame
    datasets = load_dataframe(args.spreadsheet_file_path)

    # check contents of spreadsheet
    check_dataframe(datasets,
                    columns = [
                        'file_path_raw_signals',
                        'file_path_preprocessed_signals',
                        'file_path_manual_state_annotation',
                    ],
                    column_to_dtype = {
                        'file_path_raw_signals' : str,
                        'file_path_preprocessed_signals' : str,
                        'file_path_manual_state_annotation' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    # --------------------------------------------------------------------------------
    print("Loading data sets...")

    signal_arrays = []
    state_vectors = []
    for ii, dataset in datasets.iterrows():
        print("{} ({}/{})".format(dataset['file_path_preprocessed_signals'], ii+1, len(datasets)))

        signal_array = load_preprocessed_signals(dataset['file_path_preprocessed_signals'])
        state_vector = load_state_vector(dataset['file_path_manual_state_annotation'],
                                         mapping=state_to_int, time_resolution=time_resolution)

        signal_arrays.append(signal_array)
        state_vectors.append(state_vector)

    # --------------------------------------------------------------------------------
    print('Testing...')

    total_datasets = len(datasets)

    if (total_datasets < 2) and not args.model:
        raise Exception("Training and testing in a hold-one-out fashion requires two or more datasets!")

    accuracy = np.zeros((total_datasets))

    unique_states = np.unique(np.abs(state_vectors))
    confusion = np.zeros((total_datasets, np.max(unique_states)+1, np.max(unique_states)+1))

    for ii, dataset in datasets.iterrows():

        if args.model:
            annotator = StateAnnotator()
            annotator.load(args.model)
        else:
            training_signal_arrays = [arr for jj, arr in enumerate(signal_arrays) if jj != ii]
            training_state_vectors = [seq for jj, seq in enumerate(state_vectors) if jj != ii]
            annotator = StateAnnotator()
            annotator.fit(training_signal_arrays, training_state_vectors)

        # The loaded state sequence denotes artefact states as negative integers.
        # However, the state annotator does not distinguish between states and their corresponding artefact states.
        # Hence we need to remove the sign from the loaded state sequence.
        accuracy[ii] = annotator.score(signal_arrays[ii], np.abs(state_vectors[ii]))
        print("{} ({}/{}) accuracy : {:.1f}%".format(dataset['file_path_preprocessed_signals'], ii+1, len(datasets), 100 * accuracy[ii]))

        confusion[ii] = get_confusion_matrix(np.abs(state_vectors[ii]), annotator.predict(signal_arrays[ii]), labels=unique_states)

        if args.show:

            from data_io import load_raw_signals

            from configuration import (
                plot_raw_signals,
                state_annotation_signals,
                plot_states,
            )

            fig, axes = plt.subplots(4, 1, sharex=True)

            # plot raw signals
            check_dataframe(datasets,
                            columns = [
                                'file_path_raw_signals',
                                'sampling_frequency_in_hz',
                            ] + state_annotation_signals,
                            column_to_dtype = {
                                'file_path_raw_signals' : str,
                                'sampling_frequency_in_hz' : (int, float, np.int, np.float, np.int64, np.float64),
                            }
            )
            signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
            raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
            plot_raw_signals(
                raw_signals,
                sampling_frequency = dataset['sampling_frequency_in_hz'],
                ax                 = axes[0],
            )
            axes[0].set_ylabel("Raw signals")

            # LDA tranformed signals
            transformed_signals = annotator.transform(signal_arrays[ii])
            plot_signals(transformed_signals, ax=axes[1])
            axes[1].set_ylabel("Transformed signals")

            predicted_state_vector = annotator.predict(signal_arrays[ii])
            predicted_states, predicted_intervals = convert_state_vector_to_state_intervals(predicted_state_vector, mapping=int_to_state)
            plot_states(predicted_states, predicted_intervals, ax=axes[2])
            axes[2].set_ylabel("Automated\nannotation")

            states, intervals = convert_state_vector_to_state_intervals(state_vectors[ii], mapping=int_to_state)
            plot_states(states, intervals, ax=axes[3])
            axes[3].set_ylabel("Manual\nannotation")

            axes[-1].set_xlabel('Time [seconds]')

            fig.suptitle(dataset['file_path_raw_signals'])
            fig.align_ylabels()
            fig.tight_layout()

    print("Mean accuracy +/- MSE: {:.2f}% +/- {:.2f}%".format(100*np.mean(accuracy), 100*np.std(accuracy)/np.sqrt(len(accuracy))))

    if args.savefile:
        np.savez(args.savefile, accuracy=accuracy, confusion=confusion)

    plt.show()
