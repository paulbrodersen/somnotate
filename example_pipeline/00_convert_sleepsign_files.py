#!/usr/bin/env python

"""
Utilities to convert sleepsign output files.
"""

import numpy as np
from somnotate._utils import convert_state_vector_to_state_intervals

from data_io import (
    ArgumentParser,
    load_dataframe,
    check_dataframe,
    export_hypnogram,
)


SLEEPSIGN_KEY = dict([
        ('w'  , 'awake'),
        ('wa' , 'awake (artefact)'),
        ('wb' , 'awake (artefact)'),
        ('m'  , 'sleep movement'),
        ('nr' , 'non-REM'),
        ('na' , 'non-REM (artefact)'),
        ('nb' , 'non-REM (artefact)'),
        ('r'  , 'REM'),
        ('ra' , 'REM (artefact)'),
        ('rb' , 'REM (artefact)'),
        ('no' , 'undefined'),
    ])


def convert_sleepsign_hypnogram(old_file_path, new_file_path,
                                sleepsign_key = SLEEPSIGN_KEY,
                                epoch_duration = 4):
    states, intervals = load_sleepsign_hypnogram(old_file_path,
                                                 epoch_duration=epoch_duration,
                                                 mapping=sleepsign_key)
    export_hypnogram(new_file_path, states, intervals)



def load_sleepsign_hypnogram(file_path,
                             epoch_duration = 1,
                             mapping        = None,
                             *args, **kwargs):

    """
    Load hypnogram given in sleepsign format.
    """

    # load data as array
    dtype = [('EpochNo', int),
             ('Stage', '|S2'),
             ('DateTime', '|S19')]

    lines = read_sleepsign_hypnogram(file_path)
    data = np.genfromtxt(lines, usecols=(0, 1, 2), dtype=dtype, skip_header=19, *args, **kwargs)

    # re-format text from a mixed case byte string to lower case string
    epochs = [state.astype(str).lower() for state in data['Stage']]

    # # for debugging purposes:
    # print(file_path)
    # print(epochs[:10])
    # print(epochs[-10:])
    # print(set(epochs))
    # print(len(epochs))
    # print(len(epochs)*epoch_duration)

    # rename epoch based on mapping
    if mapping:
        epochs = [mapping[state] for state in epochs]

    # convert list of epochs to lists of states and corresponding intervals
    states, intervals = convert_state_vector_to_state_intervals(epochs, epoch_duration)

    return states, intervals


def read_sleepsign_hypnogram(file_path):
    """
    Sleepsign appends state annotations to an existing file instead of overwriting them.
    As a result, sleepsign hypnograms often contain one or more annotations.
    This functions reads the file and returns the lines corresponding to the header
    and the last set of annotations.
    """

    with open(file_path, 'r') as f:
        lines = f.readlines()

    empty_lines = [ii for ii, line in enumerate(lines) if line == '\r\n']

    if len(empty_lines) <= 2: # only a linebreak at the of the header and at the end of the file
        return lines
    else:
        warnings.warn("{} seems to contain multiple sets of annotations. Reading only the last set.".format(file_path))
        new_lines = lines[:empty_lines[0]] + lines[empty_lines[-2]:]
        return new_lines


if __name__ == '__main__':

    # --------------------------------------------------------------------------------
    # parse and check inputs

    parser = ArgumentParser()
    parser.add_argument("spreadsheet_file_path", help="Use datasets specified in /path/to/spreadsheet.csv")
    args = parser.parse_args()

    # load spreadsheet / data frame
    datasets = load_dataframe(args.spreadsheet_file_path)

    # check contents of spreadsheet
    check_dataframe(datasets,
                    columns = [
                        'file_path_sleepsign_state_annotation',
                        'file_path_manual_state_annotation',
                    ],
                    column_to_dtype = {
                        'file_path_sleepsign_state_annotation' : str,
                        'file_path_manual_state_annotation' : str,
                    }
    )

    for ii, dataset in datasets.iterrows():
        print("{} ({}/{})".format(dataset['file_path_sleepsign_state_annotation'], ii+1, len(datasets)))
        old_file_path = dataset['file_path_sleepsign_state_annotation']
        new_file_path = dataset['file_path_manual_state_annotation']
        convert_sleepsign_hypnogram(old_file_path, new_file_path)
