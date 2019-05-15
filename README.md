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
