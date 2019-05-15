# somnotate

Automatically annotate vigilance states by applying linear
discriminant analysis (LDA) and hidden Markov models (HMM) to
timeseries data from EEGs, EMGs, or LFPs.

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

## The pipeline

### Contents

The currently available scripts are:

1. `00_convert_sleepsign_files.py`

    Extract the hypnogram from SleepSign FFT files (created in SleepSign
    via: Analysis -> FFT-Text Output -> Continuous FFT).

2. `01_preprocess.py`

    Convert the EEG and EMG signals into features to use in the state
    inference.  Currently, we simply (1) compute the spectrogram for
    each "raw" signal (i.e. the EEG, LFP, or EMG trace), (2)
    renormalize the data, such that within each frequency band, the
    power is approximately normally distributed, and (3) trim the
    spectrogram to exclude frequencies for which our estimate is very
    noisy, i.e. frequencies near the Nyquist limit and frequencies
    around 50 Hz. Finally, we concatenate the spectrograms into one
    set of features over time.

3. `02_test_automated_state_annotion.py`

   Test the performance of the automated state annotation in a
   hold-one-out fashion on a given set of training data sets
   (i.e. preprocessed data with corresponding manually created state
   annotations).

4. `03_train_automated_state_annotion.py`

   Train a model (LDA + HMM) using a set of training data sets and
   export it for later use.

5. `04_annotate_states.py`

   Use a previously trained model to automatically annotate the states in
   a given set of un-annotated data sets.

6. `05_manual_refinement.py`

   Manually check (and refine where necessary) the automatically generated state annotations.

Apart from these scripts, there are two additional files, `data_io.py` and `configuration.py`

- `data_io.py`

    provides a set of functions for data import and export.

- `configuration.py`

    defines a set of variables shared across all scripts. Most of
    these pertain to the states and their representation in the
    hypnograms, their represantation internally in the pipeline, and
    their visualisation in the plots created by the pipeline.


### Customization

Most pipeline customisations should only require changes in either `data_io.py` or `configuration.py`.

#### Changes in input or output data formats

Most function definitions in `data_io.py` are just aliases for other
functions. In many cases, changes in the format of the input or the
output files can hence be achieved by simply rebinding the aliases to
different functions.

For example, the aforementioned spreadsheet is loaded into memory by a
function called `load_dataframe`. However, `load_dataframe` is just an
alias to `read_csv` in the `pandas` module. To import spreadsheets in
Excel format instead, simply rebind `load_dataframe` to a function
that reads Excel files. Conveniently, such a function is also
available in `pandas` with `read_excel`. Therefor, it is sufficient to
replace the line


``` python
load_dataframe = pandas.read_csv
```

with

``` python
load_dataframe = pandas.read_excel
```

The function `load_dataframe` continues to be available, and can be
used by all scripts in the pipeline just as before.


#### Other changes

Most other changes can be made by changing the values of variables in
`configuration.py`. Please refer to the extensive comments in
`configuration.py` for further guidance.


