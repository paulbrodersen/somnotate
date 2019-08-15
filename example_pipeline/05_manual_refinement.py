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

from somnotate._manual_state_annotation import TimeSeriesStateAnnotator
from somnotate._plotting import subplots

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_raw_signals,
    load_hypnogram,
    load_review_intervals,
    export_hypnogram,
)


if __name__ == '__main__':

    from matplotlib.gridspec import GridSpec

    from configuration import (
        state_annotation_signals,
        state_annotation_signal_labels,
        plot_raw_signals,
        state_to_color,
        state_display_order,
        keymap,
        default_selection_length,
        default_view_length,
    )

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument('--only',
                        nargs = '+',
                        type  = int,
                        help  = 'Indices corresponding to the rows to use (default: all). Indexing starts at zero.'
    )
    parser.add_argument('--annotation_type',
                    default = 'automated',
                    choices = ['automated', 'refined', 'manual'],
                    help    = 'The annotation type to export (default: %(default)s).'
    )

    args = parser.parse_args()

    # load spreadsheet / data frame
    datasets = load_dataframe(args.spreadsheet_file_path)

    # check contents of spreadsheet
    check_dataframe(datasets,
                    columns = [
                        'file_path_raw_signals',
                        'sampling_frequency_in_hz',
                        'file_path_{}_state_annotation'.format(args.annotation_type),
                        'file_path_refined_state_annotation',
                        'file_path_review_intervals',
                    ] + state_annotation_signals,
                    column_to_dtype = {
                        'file_path_raw_signals' : str,
                        'sampling_frequency_in_hz' : (int, float, np.int, np.float, np.int64, np.float64),
                        'file_path_{}_state_annotation'.format(args.annotation_type) : str,
                        'file_path_refined_state_annotation' : str,
                        'file_path_review_intervals' : str,
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
        predicted_states, predicted_intervals = load_hypnogram(dataset['file_path_{}_state_annotation'.format(args.annotation_type)])
        review_intervals, review_scores = load_review_intervals(dataset['file_path_review_intervals'])

        # plot power in each frequency band and define callback
        # that updates figure based on the selection
        frequency_bands = [
            (r'$\delta$' , 0.,   4., 'aqua'),
            (r'$\theta$' , 4.,   8., 'seagreen'),
            (r'$\alpha$' , 8.,  12., 'limegreen'),
            (r'$\beta$' , 12.,  30., 'darkorchid'),
            (r'$\gamma$', 30., 100., 'crimson'),
        ]
        psd_figure, axes = subplots(1, total_raw_signals,
                                    sharex=True, sharey=True,
                                    figsize=(total_raw_signals * 4, 4))
        psd_collections = []
        for ii, (signal, ax, label) in enumerate(zip(raw_signals.T, axes.ravel(), state_annotation_signal_labels)):
            frequencies, psd = welch(signal, dataset['sampling_frequency_in_hz'])
            for _, fmin, fmax, color in frequency_bands:
                mask = (frequencies >= fmin) & (frequencies <= fmax)
                psd_collections.append(ax.fill_between(frequencies[mask], psd[mask], color=color))
            ax.set_title(label)
            ax.set_xlabel("Frequency [Hz]")
            ax.set_xlim(0, 30)
        axes.ravel()[0].set_ylabel("Power")

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

        # compute order for regions of interest
        order = np.argsort(review_scores)[::-1]
        regions_of_interest = review_intervals[order]

        # initialise state annotation figure
        fig = plt.figure()
        gs = GridSpec(4, 1)
        data_axis  = fig.add_subplot(gs[:3, 0])
        state_axis = fig.add_subplot(gs[3, 0], sharex=data_axis)
        fig.tight_layout(**{'rect': [0.05, 0, 1, 1], 'pad': 2., 'h_pad': 0.})

        state_axis.set_xlabel('Time [s]')

        # plot signals
        plot_raw_signals(
            raw_signals,
            sampling_frequency = dataset['sampling_frequency_in_hz'],
            ax                 = data_axis,
            linewidth          = 1.,
        )

        # initialise annotator
        annotator = TimeSeriesStateAnnotator(data_axis, state_axis, keymap,
                                             interval_to_state        = zip(predicted_intervals, predicted_states),
                                             regions_of_interest      = regions_of_interest,
                                             state_to_color           = state_to_color,
                                             state_display_order      = state_display_order,
                                             selection_callback       = update_psd_figure,
                                             default_selection_length = default_selection_length,
                                             default_view_length      = default_view_length,
        )
        plt.show()

        refined_intervals = list(annotator.interval_to_state.keys())
        refined_states = list(annotator.interval_to_state.values())
        export_hypnogram(dataset['file_path_refined_state_annotation'], refined_states,
                         refined_intervals)
