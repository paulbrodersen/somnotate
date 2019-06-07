#!/usr/bin/env python

"""
Convert hypnograms to matlab array.
"""

import numpy as np
from scipy.io import savemat

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    load_state_vector,
)


def convert_hypnogram_to_mat(file_path_hyp, file_path_mat, mapping, time_resolution=1.):
    state_vector = load_state_vector(file_path_hyp, mapping, time_resolution=1.)
    output = dict(state_vector=state_vector,
                  mapping=mapping,
                  time_resolution=time_resolution)
    savemat(file_path_mat, mdict=output)


if __name__ == '__main__':

    from configuration import state_to_int

    # --------------------------------------------------------------------------------
    # parse and check inputs

    parser = ArgumentParser(description="Convert hypnograms in visbrain stage-duration format to matlab state vectors and save out as matlab struct.")
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    parser.add_argument('--annotation_type',
                    default = 'automated',
                    choices = ['automated', 'refined', 'manual'],
                    help    = 'The annotation type to export (default: %(default)s).'
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
                        'file_path_{}_state_annotation'.format(args.annotation_type),
                        'file_path_{}_state_annotation_mat'.format(args.annotation_type)
                    ],
                    column_to_dtype = {
                        'file_path_{}_state_annotation'.format(args.annotation_type) : str,
                        'file_path_{}_state_annotation_mat'.format(args.annotation_type) : str,
                    }
    )

    if args.only:
        datasets = datasets.loc[np.in1d(range(len(datasets)), args.only)]

    for ii, dataset in datasets.iterrows():
        print("{} ({}/{})".format(dataset['file_path_{}_state_annotation'.format(args.annotation_type)], ii+1, len(datasets)))
        old_file_path = dataset['file_path_{}_state_annotation'.format(args.annotation_type)]
        new_file_path = dataset['file_path_{}_state_annotation_mat'.format(args.annotation_type)]
        convert_hypnogram_to_mat(old_file_path, new_file_path, mapping=state_to_int, time_resolution=1)
