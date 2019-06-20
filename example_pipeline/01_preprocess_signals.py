#!/usr/bin/env python

"""
Example script demonstrating how to preprocess a set of datasets and saving out the results.

TODO:
- provide different entry point by command line argument parsing using argparse
- fix plotting (`time` encapsulated at the moment...)
"""

import numpy as np
import matplotlib.pyplot as plt

try:
    # Ideally, we use a multitaper approach to compute LFP/EEG/EMG spectrograms.
    # An implementation is available on github and can be installed using:
    # pip install git+https://github.com/hbldh/lspopt.git#egg=lspopt
    from lspopt import spectrogram_lspopt
    from functools import partial
    get_spectrogram = partial(spectrogram_lspopt, c_parameter=20.)

except ImportError:
    import warnings
    message = "Falling back to scipy.signal.spectrogram to compute the spectrogram,"
    message += "\nwhich computes the standard Baum-Welch spectrogram."
    message += "\nA multitaper approach may yield better results, "
    message += "\nfor which an implementation is available on github and can be installed with:"
    message += "\npip install git+https://github.com/hbldh/lspopt.git#egg=lspopt"
    warnings.warn(message)
    from scipy.signal import spectrogram as get_spectrogram

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_raw_signals,
    export_preprocessed_signals,
)

from somnotate._utils import (
    robust_normalize,
)

def preprocess(raw_signal, sampling_frequency_in_hz,
               time_resolution_in_sec   = 1,
               low_cut                  = 1.,
               high_cut                 = 90.,
               notch_low_cut            = 45.,
               notch_high_cut           = 55.,
):
    """Wrapper around get_spectrogram, that
    1) computes the spectrogram for the given LFP/EEG/EMG trace,
    2) normalizes it such that the power in a given frequency band is
    approximately normally distributed, and
    3) excludes frequencies that are contaminated by noise using the equivalent
    of a notch filter.

    Arguments:
    ----------
    raw_signal -- (total samples, ) ndarray
        The electrophysiological signal.

    sampling_frequency_in_hz -- float
        The sampling frequency of `raw_signals`.

    time_resolution_in_sec -- int (default 1)
        The time resolution of the output array.

    low_cut, high_cut -- float (default 1.)
        The minimum/maximum frequency for which to compute the power.

    notch_low_cut, notch_high_cut -- float (default 45.)
        The frequency band for which NOT to compute the power
        (to eliminate 50 Hz noise from the output signal).

    Returns:
    --------
    preprocessed_signal -- (total samples / (sampling frequency * time resolution), total frequencies)
        The normalized spectrogram of the given signal.

    """

    # compute spectrogram
    frequencies, time, spectrogram = get_spectrogram(raw_signal,
                                                     fs       = sampling_frequency_in_hz,
                                                     nperseg  = sampling_frequency_in_hz * time_resolution_in_sec,
                                                     noverlap = 0)

    # exclude ill-determined frequencies
    mask = (frequencies >= low_cut) & (frequencies < high_cut)
    frequencies = frequencies[mask]
    spectrogram = spectrogram[mask]

    # exclude noise-contaminated frequencies around 50 Hz;
    # this improves performance (generally, 0.1-0.5%, but 3% in at least one case)
    mask = (frequencies >= notch_low_cut) & (frequencies <= notch_high_cut)
    frequencies = frequencies[~mask]
    spectrogram = spectrogram[~mask]

    # the power in each frequency band tends to be log-normally distributed, and
    # taking the log hence transforms the distribution of power values to a normal distribution;
    # shift power values by +1 such that values close to zero remain close to zero
    # (and do not become large, negative values after log transformation)
    spectrogram = np.log(spectrogram + 1)

    # normalize the data by de-meaning and rescaling by the standard deviation
    spectrogram = robust_normalize(spectrogram, p=5., axis=1, method='standard score')

    return time, frequencies, spectrogram


if __name__ == '__main__':

    from configuration import (
        time_resolution,
        state_annotation_signals,
        plot_raw_signals,
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
    args = parser.parse_args()

    # load spreadsheet / data frame
    datasets = load_dataframe(args.spreadsheet_file_path)

    # check contents of spreadsheet
    check_dataframe(datasets,
                    columns = [
                        'file_path_raw_signals',
                        'sampling_frequency_in_hz',
                        'file_path_preprocessed_signals',
                    ] + state_annotation_signals,
                    column_to_dtype = {
                        'file_path_raw_signals' : str,
                        'sampling_frequency_in_hz' : (int, float, np.int, np.float, np.int64, np.float64),
                        'file_path_preprocessed_signals' : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    # --------------------------------------------------------------------------------
    # preprocess specified files

    for ii, dataset in datasets.iterrows():
        print("{} ({}/{})".format(dataset['file_path_raw_signals'], ii+1, len(datasets)))

        # determine edf signals to load
        signal_labels = [dataset[column_name] for column_name in state_annotation_signals]

        # load data
        raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)

        preprocessed_signals = []
        for signal in raw_signals.T:
            time, frequencies, preprocessed_signal = preprocess(signal, dataset['sampling_frequency_in_hz'],
                                                                time_resolution_in_sec   = time_resolution,
                                                                low_cut                  = 1.,
                                                                high_cut                 = 90.,
                                                                notch_low_cut            = 45.,
                                                                notch_high_cut           = 55.,
            )
            preprocessed_signals.append(preprocessed_signal)

        # show input and outputs (not concatenated) for first dataset for quality control
        if args.show:
            fig, axes = plt.subplots(1+len(preprocessed_signals), 1, sharex=True)
            plot_raw_signals(
                raw_signals,
                sampling_frequency = dataset['sampling_frequency_in_hz'],
                ax                 = axes[0],
            )
            for signal, ax in zip(preprocessed_signals, axes[1:]):
                ax.imshow(signal, aspect='auto', origin='lower', extent=[time[0], time[-1], frequencies[0], frequencies[-1]])
                ax.set_ylabel('Frequency')
            ax.set_xlabel('Time [seconds]')

        # concatenate spectrograms into one set of features and save out
        preprocessed_signals = np.concatenate([signal.T for signal in preprocessed_signals], axis=1)
        export_preprocessed_signals(dataset['file_path_preprocessed_signals'], preprocessed_signals)

    plt.show()
