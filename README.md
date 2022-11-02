[![CI-tests](https://github.com/GilesStrong/mode_muon_tomography/actions/workflows/tests.yml/badge.svg)](https://github.com/GilesStrong/mode_muon_tomography/actions)
[![CI-lints](https://github.com/GilesStrong/mode_muon_tomography/actions/workflows/linting.yml/badge.svg)](https://github.com/GilesStrong/mode_muon_tomography/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# TomOpt: Differential Muon Tomography Optimisation

## Installation


N.B. Whilst the repo is private, you will need to make sure that you have registered the public ssh key of your computer/instance with your [GitHub profile](https://github.com/settings/keys). Follow [these instructions](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/checking-for-existing-ssh-keys) to check for existing keys or [these](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/generating-a-new-ssh-key-and-adding-it-to-the-ssh-agent) to generate a new key. After that follow [this](https://docs.github.com/en/github/authenticating-to-github/connecting-to-github-with-ssh/adding-a-new-ssh-key-to-your-github-account) to associate the key.

Checkout package:

```
git clone git@github.com:GilesStrong/mode_muon_tomography.git
cd mode_muon_tomography
```

*N.B.* For GPU usage, it is recommended to manually setup conda and install PyTorch according to system, e.g.:
```
conda activate root
conda install nb_conda_kernels
conda create -n tomopt python=3.8 pip ipykernel
conda activate tomopt
pip install torch==1.8.1+cu111 -f https://download.pytorch.org/whl/torch_stable.html
pip install -r requirements.txt
```

Minimum python version is 3.8. Recommend creating a virtual environment, e.g. assuming your are using [Anaconda](https://www.anaconda.com/products/individual)/[Miniconda](https://docs.conda.io/en/latest/miniconda.html) (if installing conda for the first time, remember to restart the shell before attemting to use conda, and that by default conda writes the setup commands to `.bashrc`):

```
conda activate root
conda install nb_conda_kernels
conda env create -f environment.yml
conda activate tomopt
```

Otherwise set up a suitable environment using your python distribution of choice using the contents of `environment.yml`. Remember to activate the correct environment each time, via e.g. `conda activate tomopt`.

Install package and dependencies
```
pip install -r requirements.txt
pip install -e .
```

Install git-hooks:

```
pre-commit install
```

### Windows usage

Apparently when using Windows, the environment must also be activated within ipython using:

```
python -m ipykernel install --user --name tomopt --display-name "Python (tomopt)" 
```

## Testing

Testing is handled by `pytest` and is set up to run during pull requests. Tests can be manually ran locally via:

```
pytest tests/
```

to run all tests, or, e.g.:

```
pytest tests/test_muon.py
```

## External repos

- [tomo_deepinfer](https://github.com/GilesStrong/mode_muon_tomo_inference) (contact @GilesStrong for access) separately handles training and model definition of GNNs used for passive volume inference. Models are exported as JIT-traced scripts, and loaded here using the `DeepVolumeInferer` class. We still need to find a good way to host the trained models for easy download.
- [mode_muon_tomography_scattering](https://github.com/GilesStrong/mode_muon_tomography_scattering)  (contact @GilesStrong for access) separately handles conversion of PGeant model from root to HDF5, and Geant validation data from csv to HDF5.
- [tomopt_sphinx_theme](https://github.com/GilesStrong/tomopt_sphinx_theme) public. Controls the appearance of the docs.

## Authors

The TomOpt project, and its continued development and support, is the result of the combined work of many people, whose contributions are summarised in [the author list](https://github.com/GilesStrong/mode_muon_tomography/blob/main/AUTHORS.md)