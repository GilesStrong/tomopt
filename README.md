[![CI-tests](https://github.com/GilesStrong/mode_muon_tomography/actions/workflows/tests.yml/badge.svg)](https://github.com/GilesStrong/mode_muon_tomography/actions)
[![CI-lints](https://github.com/GilesStrong/mode_muon_tomography/actions/workflows/linting.yml/badge.svg)](https://github.com/GilesStrong/mode_muon_tomography/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)

# TomOpt: Differential Muon Tomography Optimisation

## Installation


Pyenv https://github.com/pyenv/pyenv
pyenv install 3.10
pyenv local 3.10

Poetry https://python-poetry.org/docs/#installing-with-the-official-installer
poetry self add poetry-plugin-export


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

### Running notebooks in a remote cluster

If you want to run notebooks on a remote cluster but access them on the browser of your local machine, you need to forward the notebook server from the cluster to your local machine.

On the cluster, run:
```
jupyter notebook --no-browser --port=8889
```

On your local computer, you need to set up a forwarding that picks the flux of data from the cluster via a local port, and makes it available on another port as if the server was in the local machine:
```
ssh -N -f -L localhost:8888:localhost:8889 username@cluster_hostname
```

The layperson version of this command is: *take the flux of info from the port `8889` of `cluster_hostname`, logging in as `username`, get it inside the local machine via the port `8889`, and make it available on the port `8888` as if the jupyter notebook server was running locally on the port `8888`*

You can now point your browser to [http://localhost:8888/tree](http://localhost:8888/tree) (you will be asked to copy the server authentication token, which is the number that is shown by jupyter when you run the notebook on the server)

If there is an intermediate machine (e.g. a gateway) between the cluster and your local machine, you need to set up a similar port forwarding on the gateway machine. The crucial point is that the input port of each machine must be the output port of the machine before it in the chain. For instance:
```
jupyter notebook --no-browser --port=8889 # on the cluster
ssh -N -f -L localhost:8888:localhost:8889 username@cluster_hostname # on the gateway. Makes the notebook running on the cluster port 8889 available on the local port 8888
ssh -N -f -L localhost:8890:localhost:8888 username@gateway_hostname # on your local machine. Picks up the server available on 8888 of the gateway and makes it available on the local port 8890 (or any other number, e.g. 8888)
```

## External repos

N.B. Not currently public

- [tomo_deepinfer](https://github.com/GilesStrong/mode_muon_tomo_inference) (contact @GilesStrong for access) separately handles training and model definition of GNNs used for passive volume inference. Models are exported as JIT-traced scripts, and loaded here using the `DeepVolumeInferer` class. We still need to find a good way to host the trained models for easy download.
- [mode_muon_tomography_scattering](https://github.com/GilesStrong/mode_muon_tomography_scattering)  (contact @GilesStrong for access) separately handles conversion of PGeant model from root to HDF5, and Geant validation data from csv to HDF5.
- [tomopt_sphinx_theme](https://github.com/GilesStrong/tomopt_sphinx_theme) public. Controls the appearance of the docs.

## Authors

The TomOpt project, and its continued development and support, is the result of the combined work of many people, whose contributions are summarised in [the author list](https://github.com/GilesStrong/mode_muon_tomography/blob/main/AUTHORS.md)