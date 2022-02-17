#!/usr/bin/env python

import numpy as np
import matplotlib.pyplot as plt
from scipy.signal import welch

import matplotlib
matplotlib.rcParams['path.simplify'] = True
matplotlib.rcParams['path.simplify_threshold'] = 1.0
matplotlib.rcParams['agg.path.chunksize'] = 10000
# import matplotlib.style as mplstyle
# mplstyle.use('fast')

from somnotate._utils import convert_state_intervals_to_state_vector, _get_intervals
from somnotate._manual_state_annotation import TimeSeriesStateViewer

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_raw_signals,
    load_hypnogram,
)


if __name__ == '__main__':

    from matplotlib.gridspec import GridSpec

    from configuration import (
        state_annotation_signals,
        state_annotation_signal_labels,
        plot_raw_signals,
        plot_states,
        state_to_color,
        state_display_order,
        state_to_int,
        time_resolution,
        default_selection_length,
        default_view_length,
    )

    parser = ArgumentParser(description = "For each dataset, interactively show the differences betweeen annotation type a and annotation type b. Press '?' for interactive help.")
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument('--annotation_type_a',
                    default = 'automated',
                    choices = ['automated', 'refined', 'manual'],
                    help    = 'The first annotation (default: %(default)s).'
    )
    parser.add_argument('--annotation_type_b',
                    default = 'manual',
                    choices = ['automated', 'refined', 'manual'],
                    help    = 'The second annotation (default: %(default)s).'
    )
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
                        'file_path_{}_state_annotation'.format(args.annotation_type_a),
                        'file_path_{}_state_annotation'.format(args.annotation_type_b),
                    ] + state_annotation_signals,
                    column_to_dtype = {
                        'file_path_raw_signals' : str,
                        'sampling_frequency_in_hz' : (int, float, np.int, np.float, np.int64, np.float64),
                        'file_path_{}_state_annotation'.format(args.annotation_type_a) : str,
                        'file_path_{}_state_annotation'.format(args.annotation_type_b) : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    # turn off interactive mode if on
    plt.ioff()

    for ii, dataset in datasets.iterrows():

        print("{} ({}/{})".format(dataset['file_path_raw_signals'], ii+1, len(datasets)))

        # load data
        total_raw_signals = len(state_annotation_signals)
        signal_labels = [dataset[column_name] for column_name in state_annotation_signals]
        raw_signals = load_raw_signals(dataset['file_path_raw_signals'], signal_labels)
        predicted_states, predicted_intervals = load_hypnogram(dataset['file_path_{}_state_annotation'.format(args.annotation_type_a)])
        manual_states, manual_intervals = load_hypnogram(dataset['file_path_{}_state_annotation'.format(args.annotation_type_b)])

        # compute regions where annotations differ
        predicted_state_vector = convert_state_intervals_to_state_vector(predicted_states, predicted_intervals, state_to_int)
        manual_state_vector    = convert_state_intervals_to_state_vector(manual_states, manual_intervals, state_to_int)
        is_discrepancy = predicted_state_vector != manual_state_vector
        discrepancy_intervals = _get_intervals(is_discrepancy)
        regions_of_interest = sorted(discrepancy_intervals, key=np.diff, reverse=True)

        # plot power in each frequency band and define callback
        # that updates figure based on the selection
        frequency_bands = [
            (r'$\delta$' , 0.,   4., 'aqua'),
            (r'$\theta$' , 4.,   8., 'seagreen'),
            (r'$\alpha$' , 8.,  12., 'limegreen'),
            (r'$\beta$' , 12.,  30., 'darkorchid'),
            (r'$\gamma$', 30., 100., 'crimson'),
        ]
        psd_figure, axes = plt.subplots(1, total_raw_signals,
                                     sharex=True, sharey=True,
                                     figsize=(total_raw_signals * 4, 4))
        psd_collections = []
        for ii, (signal, ax, label) in enumerate(zip(raw_signals.T, axes, state_annotation_signal_labels)):
            frequencies, psd = welch(signal, dataset['sampling_frequency_in_hz'])
            for _, fmin, fmax, color in frequency_bands:
                mask = (frequencies >= fmin) & (frequencies <= fmax)
                psd_collections.append(ax.fill_between(frequencies[mask], psd[mask], color=color))
            ax.set_title(label)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_xlim(0, 30)
        axes[0].set_ylabel("Power")

        def update_psd_figure(selection_lower_bound, selection_upper_bound):
            fs = dataset['sampling_frequency_in_hz']
            start = int(fs * selection_lower_bound)
            stop  = int(fs * selection_upper_bound)
            # The function `fill_between` returns a collection of patches.
            # It would be a lot of work to cycle through each patch and recompute
            # re-compute the path coordinates;
            # Instead, we remove the obsolete artists and draw new ones.
            while psd_collections:
                collection = psd_collections.pop()
                collection.remove()
            psd_max = 0
            for signal, ax in zip(raw_signals.T, psd_figure.get_axes()):
                frequencies, psd = welch(signal[start:stop], fs)
                for _, fmin, fmax, color in frequency_bands:
                    mask = (frequencies >= fmin) & (frequencies <= fmax)
                    psd_collections.append(ax.fill_between(frequencies[mask], psd[mask], color=color))
                if psd_max < np.max(psd):
                    psd_max = np.max(psd)
                    ax.set_ylim(0, psd_max)
            psd_figure.canvas.draw_idle()

        # initialise state annotation figure
        fig = plt.figure()
        gs = GridSpec(5, 1)
        data_axis  = fig.add_subplot(gs[:3, 0])
        state_axis = fig.add_subplot(gs[3, 0], sharex=data_axis)
        manual_state_axis = fig.add_subplot(gs[4, 0], sharex=data_axis)
        fig.tight_layout(**{'rect': [0.05, 0, 1, 1], 'pad': 2., 'h_pad': 0.})

        # plot signals
        plot_raw_signals(
            raw_signals,
            sampling_frequency = dataset['sampling_frequency_in_hz'],
            ax                 = data_axis,
            linewidth          = 1.,
        )

        # plot manual annotations
        plot_states(manual_states, manual_intervals,
                    ax        = manual_state_axis,
                    linewidth = 5.,
        )
        manual_state_axis.set_ylabel("Manual annotation")

        # initialise annotator
        annotator = TimeSeriesStateViewer(data_axis, state_axis,
                                          interval_to_state        = zip(predicted_intervals, predicted_states),
                                          regions_of_interest      = regions_of_interest,
                                          state_to_color           = state_to_color,
                                          state_display_order      = state_display_order,
                                          selection_callback       = update_psd_figure,
                                          default_selection_length = default_selection_length,
                                          default_view_length      = default_view_length,
        )
        state_axis.set_ylabel("Automated annotation")
        state_axis.set_xlabel('Time [s]')
        plt.show()
