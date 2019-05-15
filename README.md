# somnotate

Automatically annotate vigilance states using LFP, EEG, or EMG data.

## Installation instructions

1. Download this repository.

2. Open a console and change to the root folder of the repository.

``` shell
cd /path/to/somnotate
```

2. Optionally, create a new virtual environment.

3. Install all relevant dependencies.

Using pip:
``` shell
pip install -r ./sonotate/requirements.txt
pip install -r ./example_pipeline/requirements.txt
```

The example pipeline has one optional dependency, `lspopt`, which is
used to compute multitaper spectrograms. At the time of writing, the
module is not avalaible on the PIPy or anaconda servers, so please
follow the installation instructions on [the github homepage of the
lspopt project](https://github.com/hbldh/lspopt).

If you are using Anaconda, you also have to add the somnotate top
directory to your PYTHONPATH environment variable, if you want to use
the pipeline (it uses relative imports):

``` shell
conda develop /path/to/somnotate
```

## Quickstart guide

This repository comes in two parts, the core library, `somnotate`, and
an example pipeline. The core library implements the functionality to
automatically (or manually) annotate states using any type of time
series data "emitted" by these states, and visualize the
results. There is nothing specific to sleep staging in this part of
the code base.

The example pipeline is a collection of functions and scripts that
additionally manage data import/export, data preprocessing, and
testing. For most users, this is the part of the code base they will interact with.

Each script in the pipeline expects as mandatory input argument a path
to a spreadsheet in CSV format. This spreadsheet contains a number of
columns detailing the paths to the file pertaining to each dataset,
the properties of the dataset, such as e.g. the sampling frequency of
the EEG/EMG data, and the (desired) paths for the files created by the
pipeline. An example CSV file is provided with the test data.

For example:

``` shell
python /path/to/somnotate/example_pipeline/00_convert_sleepsign_files.py /path/to/spreadsheet.csv
```

Some scripts (specifically, scripts
`03_train_automated_state_annotation.py` and `04_annotate_states.py`)
have an additional mandatory argument, namely the path to the trained
model:

``` shell
python /path/to/somnotate/example_pipeline/03_train_automated_state_annotion.py /path/to/spreadsheet.csv /path/to/model.pickle
```

Some scripts produce output plots if the optional argument `--show` is
added to the list of arguments. For each script, a list of mandatory
and optional arguments can be accessed using the `--help` argument:

``` shell
python /path/to/somnotate/example_pipeline/script.py --help
```

