# somnotate

Automatically annotate vigilance states from timeseries data such
electroencephalograms (EEGs), electromyograph (EMG), or local field
potentials (LFPs).

The approach taken here consists of two parts: linear discriminant
analysis (LDA) and a hidden Markov model (HMM). Linear discriminant
analysis (LDA) performs automatic feature selection by projecting the
high-dimensional time series data to a lower dimensional feature space
that is optimal for state classification using hard, linear decision
boundaries. However, instead of applying these decision boundaries
immediately, the transformed time series data is annotated using a
hidden Markov model. This allows for "soft" decision boundaries that
apart from the feature values of each sample also take contextual
information into account. This contextual information includes the
surrounding state sequence and the previously learnt state transition
frequencies.

The combination of these two algorithms results in a fast, automated
state annotation that has a high accuracy compared to a human
consensus state sequence. Furthermore, this approach is remarkably
robust to mislabelled training data, artefacts, and other outliers.

## Installation instructions

1. Clone this repository. Git is available
   [here](https://git-scm.com/downloads).

    ```shell
    git clone https://github.com/paulbrodersen/somnotate.git
    ```

    Alternatively, you can just download the repository as a zip file,
    and unzip it. However, you will have to repeat this process each
    time a new version is released. If you use git, you can update the
    repository simply by changing to anywhere in the somnotate
    directory and running `git pull`.

    ```shell
    cd /path/to/somnotate
    git pull
    ```

2. Optionally, create a clean virtual environment.

   For example, to create a clean virtual environment using conda
   (available
   [here](https://www.anaconda.com/distribution/#download-section)),
   open a terminal (on Windows: Anaconda Prompt), and enter:

   ``` shell
   conda create --no-default-packages -n my_somnotate_virtual_environment_name python
   ```

    Then activate the environment:
   ``` shell
   conda activate my_somnotate_virtual_environment_name
   ```

   You will need to re-activate the environment each time you want to
   use somnotate.

3. Install all required dependencies.

    Using conda:
    ```shell
    cd /path/to/somnotate
    conda install --file ./somnotate/requirements.txt
    conda install -c conda-forge pyedflib
    conda install --file ./example_pipeline/requirements.txt
    ```

    Using pip:
    ``` shell
    cd /path/to/somnotate
    pip install -r ./somnotate/requirements.txt
    pip install -r ./example_pipeline/requirements.txt
    ```

    However, if you use `pip` and you don't have a C++ compiler, the
    installation will fail for `pyedflib`, which is used in the
    pipeline to load EDF files.  On Windows, you will need to install
    the "Build tools for Visual Studio"; on MacOS, you will need to
    install the "Command Line Tools for Xcode". Then rerun the last
    command.

4.  Install all optional dependencies.

    The example pipeline has one optional dependency, `lspopt`, which
    is used to compute multitaper spectrograms. At the time of
    writing, the module is not avalaible on the PIPy or anaconda
    servers, so please follow the installation instructions on [the
    github homepage of the lspopt project](https://github.com/hbldh/lspopt).
    Currently, the easiest way is to use pip and git:

    ```shell
    pip install git+https://github.com/hbldh/lspopt.git#egg=lspopt
    ```

5. Ensure that the somnotate folder is in your PYTHONPATH environment variable.

    In most python environments, it suffices to change the working
    directory to the somnotate root folder. If you are using Anaconda,
    you have to explicitly add the somnotate root directory to your
    PYTHONPATH environment variable:

    ``` shell
    conda develop /path/to/somnotate
    ```

## Quickstart guide

Assuming you have two sets of data sets, a set of previously
(manually) annotated data sets for training of the pipeline (data sets
A) and an un-annotated set of data sets that you would like to apply
the pipeline to (data sets B).

First, prepare two spreadsheets, spreadsheet A and spreadsheet B,
providing the paths to the files pertaining to each data set, and a
few other data sets properties.  Detailed instructions regarding the
spreadsheet format can be found below.

Then, in the shell of your choice, execute in order:

``` shell
# Preprocess the training data sets.
python /path/to/somnotate/example_pipeline/01_preprocess_signals.py /path/to/spreadsheet_A.csv

# Test the performance of the pipeline on the training data sets to ensure that everything is in working order.
python /path/to/somnotate/example_pipeline/02_test_state_annotation.py /path/to/spreadsheet_A.csv

# Train a model and save it for later use.
python /path/to/somnotate/example_pipeline/03_train_state_annotation.py /path/to/spreadsheet_A.csv /path/to/model.pickle

# Preprocess the un-annotated data sets.
python /path/to/somnotate/example_pipeline/01_preprocess_signals.py /path/to/spreadsheet_B.csv

# Apply the trained model to your un-annotated data.
python /path/to/somnotate/example_pipeline/04_run_state_annotation.py /path/to/spreadsheet_B.csv /path/to/model.pickle

# Manually check intervals in the predicted state annotations that have been flagged as ambiguous.
python /path/to/somnotate/example_pipeline/05_manual_refinement.py /path/to/spreadsheet_B.csv
```

## Documentation

This repository comes in two parts, the core library, `somnotate`, and
an example pipeline. The core library implements the functionality to
automatically (or manually) annotate states using any type of time
series data "emitted" by these states, and visualize the results. The
core library supports (and is designed for) interactive use. However,
there is nothing specific to sleep staging in this part of the code
base.

The example pipeline is a collection of functions and scripts that
additionally manage data import/export, data preprocessing, and
testing. The pipeline supports (and is designed for) batch processing
of multiple files. **For most users, the pipeline is the part of the
code base they will interact with.**

### The pipeline

#### Scope / Content

Currently available scripts are:

1. `00_convert_sleepsign_files.py`

    Extract the hypnogram from SleepSign FFT files (created in
    SleepSign via: Analysis -> FFT-Text Output -> Continuous FFT), and
    convert them to hypnogram in the [visbrain stage-duration
    format](http://visbrain.org/sleep.html#hypnogram). If you already
    have hypnograms in this format, this step is not necessary.

2. `01_preprocess_signals.py`

    Convert the raw signals into features that are useful for the
    state inference. Currently, we simply (1) compute the spectrogram
    for each raw signal (i.e. the EEG, LFP, or EMG trace), (2)
    renormalize the data, such that within each frequency band, the
    power is approximately normally distributed, and (3) trim the
    spectrogram to exclude frequencies for which our estimate is very
    noisy, i.e. frequencies near the Nyquist limit and frequencies
    around 50 Hz. Finally, we concatenate the spectrograms into one
    set of features.

3. `02_test_state_annotation.py`

   Test the performance of the automated state annotation in a
   hold-one-out fashion on a given set of preprocessed training data
   sets (i.e. preprocessed data with corresponding manually created
   state annotations).

4. `03_train_state_annotation.py`

   Train a model (LDA + HMM) using a set of preprocessed training data
   sets and export it for later use.

5. `04_run_state_annotation.py`

   Use a previously trained model to automatically annotate the states in
   a given set of preprocessed data sets.

6. `05_manual_refinement.py`

   This script launches a simple GUI that facilitates manual quality
   control and refinement of the automatically generated state
   annotations. Press the key "?" to read the documentation for all
   available commands.

7. `06_compare_state_annotations.py`

   This script launches a simple GUI that facilitates manual checking
   of all differences between two state annotations, e.g. the a manual
   state annotation and a corresponding automated state annotation.
   Press the key "?" to read the documentation for all
   available commands.

Apart from these scripts, there are two additional files, `data_io.py` and `configuration.py`

- `data_io.py`

    provides a set of functions for data import and export.

- `configuration.py`

    defines a set of variables shared across all scripts. Most of
    these pertain to the states and their representation in the
    hypnograms, their represantation internally in the pipeline, and
    their visualisation in the plots created by the pipeline.


#### Examples

Each script in the pipeline expects as mandatory command line argument
a path to a spreadsheet in CSV format. The exact format of the
spreadsheet is detailed below but basically it contains a number of
columns detailing the paths to the files pertaining to each data set,
the properties of the data set (e.g. the sampling frequency of the
EEG/EMG data), and the (desired) paths for the files created by the
pipeline. An example CSV file is provided with the test data.

For example:

``` shell
python /path/to/somnotate/example_pipeline/00_convert_sleepsign_files.py /path/to/spreadsheet.csv
```

Some scripts (specifically, `03_train_state_annotation.py` and
`04_run_state_annotation.py`) have an additional mandatory argument,
namely the path to the trained model:

``` shell
python /path/to/somnotate/example_pipeline/03_train_state_annotation.py /path/to/spreadsheet.csv /path/to/model.pickle
```

If the script is supposed to run for only a subset of datasets in the spreadsheet,
the `--only` flag can be used to supply the indices for the corresponding rows.
For example, to train a model using only the first, third and fourth dataset, use:

``` shell
/path/to/somnotate/example_pipeline/03_train_state_annotation.py /path/to/spreadsheet.csv /path/to/model.pickle --only 0 2 3
```

Some scripts produce output plots if the optional argument `--show` is
added to the list of arguments. For each script, a list of mandatory
and optional arguments can be accessed using the `--help` argument:

``` shell
python /path/to/somnotate/example_pipeline/script.py --help
```

#### The spreadsheet

For each data set, the spreadsheet details a number of properties, as
well as the paths to the corresponding input and output files. By
accessing these parameters via a spreadsheet, the user does not have
to manually provide these arguments repeatedly to each script in the
pipeline. Furthermore, it ensures that the arguments remain consistent
across tasks.

Not all arguments, i.e. columns in the spreadsheet, are required for
each script. However, it is often convenient to have a single
spreadsheet for a given set of data that are processed together that
details all parameters, i.e. contains all columns. These are:

-  `file_path_raw_signals`:
  the path to the EDF file containing the raw signals (i.e. the EEG/EMG or LFP traces)
- `file_path_preprocessed_signals`;
  the (desired) path to the file containing the corresponding preprocessed signal array
- `file_path_manual_state_annotation`:
  the path to the file containing the manually created state annotation (hypnogram) in visbrain stage-duration format (only required for training data sets)
- `file_path_automated_state_annotation`:
  the (desired) path to the file containing the automated state annotion (hypnogram)
- `file_path_refined_state_annotation`:
  the (desired) path to the file containing the automated state annotion (hypnogram) that has subsequently been manually refined
- `file_path_review_intervals`:
  the (desired) path for the file containing the time intervals highlighted by the automated annotation for manual review
- `sampling_frequency_in_hz`:
  the sampling frequency of the raw signal(s)

Additionally, the variable `state_annotation_signals` in
`configuration.py` defines another set of columns in the spreadsheet
(which can be arbitrarily named) that contain the indices to the
relevant raw signals in the file at `file_path_raw_signals`.

The order of columns in the spreadsheet is arbitrary.
However, you should avoid having empty rows, as these will be
interpreted as datasets for which all parameters are missing.

Each time you run a script, the spreadsheet is checked for the
existence of all relevant columns, and it is asserted that the entries
in the columns have the correct type. Should the spreadsheet not
contain the required columns with entries in the required format, an
error will be raised detailing the missing columns or misrepresented
column entries.


#### Customization

Many pipeline customisations will only require changes in either `data_io.py` or `configuration.py`.

##### Changes in input or output data formats

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


##### Other changes

Most other changes can be made by changing the values of variables in
`configuration.py`. Please refer to the extensive comments in
`configuration.py` for further guidance.

If you cannot change an important aspect of the pipeline by changing
either `data_io.py` or `configuration.py`, please raise an issue on
the github issue tracker. Pull requests are -- of course -- very welcome.
