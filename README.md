# Somnotate

*Automated polysomnography for experimental animal research: annotate vigilance states from arbitrary time series data.*

Somnotate combines linear discriminant analysis (LDA) with a hidden
Markov model (HMM). Linear discriminant analysis performs automatic
feature selection by projecting the high-dimensional time series data
to a lower dimensional feature space that is optimal for state
classification using hard, linear decision boundaries. However,
instead of applying these decision boundaries immediately, the
transformed time series data is annotated using a hidden Markov
model. This allows for "soft" decision boundaries that in addition of
the features of each sample also take contextual information into
account. This approach results in a fast, fully automated state
annotation that is typically more accurate than manual annotations by
human experts, while being remarkably robust to mislabelled training
data, artefacts, and other outliers.

A journal article thoroughly describing Somnotate and characterising
its performance has been accepted for publication by PLoS
Computationally Biology. A pre-print of the article is available at
[bioRxiv](https://doi.org/10.1101/2021.10.06.463356). The data
underpinning the presented results is archived [at
Zenodo](https://doi.org/10.5281/zenodo.10200481). While the article
focuses on Somnotate's performance in polysomnography based on mouse
EEG, EMG, and/or LFP data, Somnotate has been applied successfully to
human clinical data, as well as telemetry data from hibernating
Alaskan black bears. As the core components of Somnotate are
completely agnostic to the data modality, Somnotate can also be
applied to other indicators of vigilance state, such as heart rate,
blood pressure, or actigraphy. While we have performed encouraging
pilot tests in these directions, we lack access to annotated data
repositories that are comprehensive enough for a proper evaluation (do
be in touch if you have data and you would like to collaborate).


## Is this software the right choice for me?

Somnotate is designed to support animal experimental research, and
hence a core assumption is that the data is peculiar in some way: the
recording setup might be non-standard, the experimental manipulation
might be severe, the genotype and (sleep) phenotype of the animals
might be deviant, or the animal model might be entirely non-standard
(such as black bears). As a consequence, machine learning models that
have been pre-trained on other data are of limited use, and you have
to train your own model using data annotated by yourself or your
colleagues. Somnotate has been designed with two aims in mind: (1)
minimise the amount of manually annotated data required to surpass the
accuracy of human experts, and (2) make model training simple enough
that any motivated scientist can optimise and use the software to its
fullest potential without requiring prior programming experience or
machine learning knowledge.

These aims motivate the use of hidden Markov models as the core
classifier over and above deep neural networks, as the latter
necessitate 2-3 orders of magnitude more data to train, and have a
number of hyperparameters that have to be tailored to each data set
for optimal results. A welcome side effect of this core design
decision is that Somnotate -- unlike most other polysomnography
software -- computes the likelihood of each vigilance state for each
epoch (rather than just determining the most likely one). This allows
the identification of intermediate states that occur around vigilance
state transitions and failed transition attempts. Analysis of
intermediate states can yield unique insights into the dynamics of
vigilance state transitions. They may also provide a more sensitive
readout for experimental manipulations than, for example, the total
time spent in each vigilance state, as such traditional measures of
sleep quality might be more strongly controlled by the physiological
needs of the animal.

For annotating human clinical data, Somnotate likely is not the
optimal choice. The acquisition of human clinical data is relatively
standardised, and large repositories with annotated data are freely
available. These have been used to train sophisticated machine
learning models, such as U-Sleep and its successors, that are readily
available and which you should use instead (unless you are interested
in intermediate states).


## What do I need?

### Recordings

To train the classifier, you will need manually annotated
recordings. With a small data set, selecting the right recordings for
training can have a significant impact on performance. In general, the
performance of the classifier depends on (1) how well it is able to
estimate the mean and the variance of all provided features in the
training data set, and (2) how well the (automatically selected)
subset of features used for inference matches between the training and
the test data sets. Here are a few notes that spell out what that
means in practice:

1. The variance between recordings from different animals is typically
   greater than the variance within one long recording from one
   animal. Training on shorter recordings from multiple animals is
   hence preferable to training on longer recordings from fewer
   animals. In our experiments, training on manually annotated
   recordings from five or six different animals represented a
   sweet-spot, where adding further training data yielded strongly
   diminishing returns in performance improvements.

2. The length of the recordings is less important, as long as the
   recordings are continuous (splicing introduces artefacts) and cover
   the full spectrum of vigilance and arousal states (i.e. light NREM
   sleep, deep NREM sleep, REM sleep, awake & rested, awake with high
   sleep pressure, etc.). For laboratory animals on a 12-hours
   light-on / 12-hours light-off cycle, 12-hour recordings covering
   the cycle half dominated by sleep (the light-on phase in mice) is
   probably sufficient for training: in our tests, training on 24-hours
   recordings only marginally improved performance, but not
   statistically significantly so.

3. Do not exclusively use "clean" recordings for training, as
   underestimates of feature variance can negatively impact feature
   selection. Training on data sets with normal or even sub-standard
   signal-to-noise ratios may improve the robustness of the
   classifier, as features affected by artefacts or noise are typically
   weighted down, and thus affect inference less.

4. It is often sufficient to train on recordings from control
   experiments, and then apply the classifier to both, recordings from
   control experiments and recordings acquired during experimental
   manipulations. In this way, a single classifier can be applied to
   data from multiple experiments. However, if the different
   conditions alter the physiology of the animal strongly, the
   characteristics of the data may become too distinct. It may be
   necessary to either (1) train multiple classifiers, one for each
   condition, or (2) train one classifier on data from both
   conditions. The first approach tends to work a little bit better
   than the second approach but requires more annotated data sets. If
   you use the second approach, ensure that both conditions are
   represented equally in the training data.

The provided example pipeline expects recordings to be in the
[European data format (EDF)](https://www.edfplus.info/specs/edf.html).

### Manual Annotations

Annotations are expected to be in
[Visbrain's](https://github.com/EtienneCmb/visbrain) [stage-duration
format](http://visbrain.org/sleep.html#hypnogram). This is a very
simple text file with the first line specifying the length of the
corresponding recording in seconds, and the second line specifying the
file name of the recording (or `Unspecified`). The remaining lines
list each state and its end-point since the start of the recording in
seconds. In the example below, the duration of the first occurrence of
`Awake` is 1 minute, the duration of the following `NREM` period is 2
minutes, and the duration of the following `REM` period is 3
minutes. The label `Undefined` should be used if no state assignment
is appropriate, for example at the start of the recording when the
electrodes aren't connected, yet; do not use `Undefined` to denote
artefacts, as this will result in overestimates of the state
transition frequencies. List only one state per line. Use a single tab
to separate items within a line. The last entry in the file should
match the duration specified on the first line.

```
*Duration_sec	43200.0
*Datafile	Unspecified
Undefined	10.0
Awake	70.0
NREM	190.0
REM	370.0
Awake	372.0
NREM	672.0
...
Awake	42000.0
Undefined	43200.0
```

The quality of the annotations is not particularly important, as
Somnotate is highly robust to errors in the training data. In the
journal article linked above, we used data sets annotated by up to 10
experienced sleep researchers to show that Somnotate was able to match
the human consensus better than any individual expert in
testing. However, for the purpose of training, using a single manual
annotation per recording is fine. When we tested Somnotate's
robustness to errors in the training data, performance was unaffected,
even if a large fraction of the data used for training was
deliberately mislabelled (up to 50% of all epochs).

### Operating System and Computational Hardware Requirements

Somnotate was developed under Linux but also runs on Windows and
iOS. No dedicated hardware is required, as even a simple laptop should
complete all tasks within a reasonable time, provided it has
sufficient RAM to load a single recording into memory. As a rough
guide, after pre-processing, training and testing should not require
more than 1-2 seconds per 24 hours of recordings on any but the most
ancient hardware. The time it takes to preprocess a data set is highly
variable and depends on the file format, number of signals, and their
sampling frequency. Preprocessing recordings in EDF file format with
three signals at 256 Hz requires about 10 seconds per 24 hours.


## Installation instructions

1. Clone this repository. Git is available
   [here](https://git-scm.com/downloads).

    ```shell
    git clone https://github.com/paulbrodersen/somnotate.git
    ```
    Alternatively, you can download the repository as a zip file, and
    unzip it. However, you will have to repeat this process each time
    a new version is released. If you use git, you can update the
    repository simply by changing to anywhere in the somnotate
    directory and running `git pull`.

    ```shell
    cd /path/to/somnotate
    git pull
    ```

2. Optionally, create a clean virtual environment.

   For example, to create a clean virtual environment using conda
   (available [here](https://www.anaconda.com)),
   open a terminal (on Windows: Anaconda Prompt), and enter:

   ```shell
   conda create --no-default-packages -n my_somnotate_virtual_environment_name python
   ```

    Then activate the environment:

   ```shell
   conda activate my_somnotate_virtual_environment_name
   ```

   You will need to re-activate the environment each time you want to
   use somnotate.

3. Install all required dependencies.

    Using conda:

    ```shell
    cd /path/to/somnotate
    conda install --file requirements.txt
    conda install -c conda-forge pyedflib
    conda install --file example_pipeline/requirements.txt
    conda develop somnotate
    ```

    Using pip:

    ```shell
    cd /path/to/somnotate
    pip install -e .
    pip install -e .[pipeline]
    ```

    However, if you use `pip` and you don't have a C++ compiler, the
    installation may fail for `pyedflib`, which is used in the
    pipeline to load EDF files. On Windows, you will need to install
    the "Build tools for Visual Studio"; on MacOS, you will need to
    install the "Command Line Tools for Xcode". Then rerun the last
    command. If you are using `pip` inside a conda environment, you
    can use conda to install pyedflib from conda-forge as above.


## Quickstart Guide / Cheat Sheet

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

## Detailed Description of the Pipeline

This repository comes in two parts, the core library, `somnotate`, and
an example pipeline. The core library implements the functionality to
automatically (or manually) annotate states using any type of time
series data, and visualize the results. However, there is nothing
specific to sleep staging in this part of the code base.

The example pipeline is a collection of functions and scripts that
additionally manage data import/export, data preprocessing, and
testing. The pipeline supports (and is designed for) batch processing
of multiple files. **For most users, the pipeline is the part of the
code base they will interact with.**

### Content

Currently available scripts are:

1. `01_preprocess_signals.py`

    Convert the raw signals into features that are useful for the
    state inference. Currently, we simply (1) compute the spectrogram
    for each raw signal (i.e. the EEG, LFP, or EMG trace), (2)
    renormalize the data, such that within each frequency band, the
    power is approximately normally distributed, and (3) trim the
    spectrogram to exclude frequencies for which our estimate is very
    noisy, i.e. frequencies near the Nyquist limit and frequencies
    around 50 Hz. Finally, we concatenate the spectrograms of the
    difference signals into one set of features.

2. `02_test_state_annotation.py`

   Test the performance of the automated state annotation in a
   hold-one-out fashion on a given set of preprocessed training data
   sets (i.e. preprocessed data with corresponding manually created
   state annotations).

3. `03_train_state_annotation.py`

   Train a model using a set of preprocessed training data sets and
   export it for later use.

4. `04_run_state_annotation.py`

   Use a previously trained model to automatically annotate the states in
   a given set of preprocessed data sets.

5. `05_manual_refinement.py`

   This script launches a simple GUI that facilitates manual quality
   control and refinement of the automatically generated state
   annotations. Press the key "?" to read the documentation for all
   available commands.

6. `06_compare_state_annotations.py`

   This script launches a simple GUI that facilitates manual checking
   of all differences between two state annotations, e.g. the a manual
   state annotation and a corresponding automated state annotation.
   Press the key "?" to read the documentation for all
   available commands.

Apart from these scripts, there are two additional files in `example_pipeline`:

- `data_io.py`

    provides a set of functions for data import and export.

- `configuration.py`

    defines a set of variables shared across all scripts. Most of
    these pertain to the states and their representation in the
    hypnograms, their represantation internally in the pipeline, and
    their visualisation in the plots created by the pipeline.

The `extensions` folder has two additional scripts to facilitate data I/O:

- `convert_sleepsign_files.py`

    Extract the hypnogram from SleepSign FFT files (created in
    SleepSign via: Analysis -> FFT-Text Output -> Continuous FFT), and
    convert them to hypnogram in the stage-duration format. If you already
    have hypnograms in this format, this step is not necessary.

- `convert_hypnogram_to_matlab_struct.py`

    Convert hypnograms in stage-duration format to MATLAB structs.


### Examples

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

### The Spreadsheet

For each data set, the spreadsheet details a number of properties, as
well as the paths to the corresponding input and output files. By
providing these parameters via a spreadsheet, the user does not have
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


### Customization

Many pipeline customisations will only require changes in either `data_io.py` or `configuration.py`.

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

Most other changes can be made by changing the values of variables in
`configuration.py`. Please refer to the extensive comments in
`configuration.py` for further guidance.

If you cannot change an important aspect of the pipeline by changing
either `data_io.py` or `configuration.py`, please raise an issue on
the github issue tracker. Pull requests are -- of course -- very welcome.


## Citing Somnotate

If you use Somnotate in your scientific work, I would appreciate citations to the following paper:

Brodersen PJN, Alfonsa H, Krone LB, Duque CB, Fisk AS, Flaherty SJ, et al. Somnotate: A probabilistic sleep stage classifier for studying vigilance state transitions. bioRxiv. 2023;2021.10.06.463356.

Bibtex entry:

```bibtex
@article{Brodersen2023,
author = {Brodersen, Paul J N and Alfonsa, Hannah and Krone, Lukas B and Duque, Cristina Blanco and Fisk, Angus S and Flaherty, Sarah J and Guillaumin, Mathilde C C and Huang, Yi-Ge and Kahn, Martin C and McKillop, Laura E and Milinski, Linus and Taylor, Lewis and Thomas, Christopher W and Yamagata, Tomoko and Foster, Russell G and Vyazovskiy, Vladyslav V and Akerman, Colin J},
journal = {bioRxiv},
pages = {2021.10.06.463356},
title = {{Somnotate: A probabilistic sleep stage classifier for studying vigilance state transitions}},
url = {http://biorxiv.org/content/early/2023/06/20/2021.10.06.463356.abstract},
year = {2023}
}
```

## Recent changes

0.3.0 Simplified installation
0.2.0 Clean-up of pipeline: moved `convert_sleepsign_files.py` and `convert_hypnogram_to_matlab_struct.py` to extensions
0.1.0 Improved README: added sections "Is this software the right choice for me?" and "What do I need?"


## License

Somnotate has a dual license. It is licensed under the GPLv3 license
for academic use only. For commercial or any other use of Somnotate or
parts thereof, please be in contact: paul.brodersen@gmail.com.
