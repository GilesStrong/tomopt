[![CI-tests](https://github.com/GilesStrong/tomopt/actions/workflows/tests.yml/badge.svg)](https://github.com/GilesStrong/tomopt/actions)
[![CI-lints](https://github.com/GilesStrong/tomopt/actions/workflows/linting.yml/badge.svg)](https://github.com/GilesStrong/tomopt/actions)
[![Code style: black](https://img.shields.io/badge/code%20style-black-000000.svg)](https://github.com/psf/black)
[![pypi tomopt version](https://img.shields.io/pypi/v/tomopt.svg)](https://pypi.python.org/pypi/tomopt)
[![tomopt python compatibility](https://img.shields.io/pypi/pyversions/tomopt.svg)](https://pypi.python.org/pypi/tomopt) [![tomopt license](https://img.shields.io/pypi/l/tomopt.svg)](https://pypi.python.org/pypi/tomopt)
[![Documentation Status](https://readthedocs.org/projects/tomopt/badge/?version=stable)](https://tomopt.readthedocs.io/en/latest/?badge=stable)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.10673885.svg)](https://doi.org/10.5281/zenodo.10673885)

# TomOpt: Differential Muon Tomography Optimisation

This repo provides a library for the differential optimisation of scattering muon tomography systems. For an overview, please read our first publication [here](https://arxiv.org/abs/2309.14027).

As a disclaimer, this is a library designed to be extended by users for their specific tasks: e.g. passive volume definition, inference methods, and loss functions. Additionally, optimisation in TomOpt can be unstable, and requires careful tuning by users. This is to say that it is not a polished product for the general public, but rather fellow researchers in the field of optimisation and muon tomography.

If you are interested in using this library seriously, please contact us;  we would love to here if you have a specific use-case you wish to work on.


## Overview

The TomOpt library is designed to optimise the design of a muon tomography system. The detector system is defined by a set of parameters, which are used to define the geometry of the detectors. The optimisation is performed by minimising a loss function, which is defined by the user. The loss function is evaluated by simulating the muon scattering process through the detector system and passive volumes. The information recorded by the detectors is then passed through an inference system to arrive at a set of task-specific parameters. These are then compared to the ground truth, and the loss is calculated. The gradient of the loss with respect to the detector parameters is then used to update the detector parameters.

The TomOpt library is designed to be modular, and to allow for the easy addition of new inference systems, loss functions, and passive volume definitions. The library is also designed to be easily extensible to new optimisation algorithms, and to allow for the easy addition of new constraints on the detector parameters.

TomOpt consists of several submodules:

- benchmarks: and ongoing collection of concrete implementations and task-specific extensions that are used to test the library on real-world problems.
- inference: provides classes that infer muon-trajectories from detector data, and infer properties of passive volumes from muon-trajectories.
- muon: provides classes for handling muon batches, and generating muons from literature flux-distributions
- optimisation: provides classes for handling the optimisation of detector parameters, and an extensive callback system to modify the optimisation process.
- plotting: various plotting utilities for visualising the detector system, the optimisation process, and results
- volume: contains classes for defining passive volumes and detector systems
- core: core objects used by all parts of the code
- utils: various utilities used throughout the codebase

## Installation

### As a dependency

For dependency usage, `tomopt` can be installed via e.g. 

```bash
pip install tomopt
```

### For development

Check out the repo locally:

```bash
git clone git@github.com:GilesStrong/tomopt.git
cd tomopt
```

For development usage, we use [`poetry`](https://python-poetry.org/docs/#installing-with-the-official-installer) to handle dependency installation.
Poetry can be installed via, e.g.

```bash
curl -sSL https://install.python-poetry.org | python3 -
poetry self update
```

and ensuring that `poetry` is available in your `$PATH`

TomOpt requires `python >= 3.10`. This can be installed via e.g. [`pyenv`](https://github.com/pyenv/pyenv):

```bash
curl https://pyenv.run | bash
pyenv update
pyenv install 3.10
pyenv local 3.10
```

Install the dependencies:

```bash
poetry install
poetry self add poetry-plugin-export
poetry config warnings.export false
poetry run pre-commit install
```

Finally, make sure everything is working as expected by running the tests:
 
```bash
poetry run pytest tests
```

For those unfamiliar with `poetry`, basically just prepend commands with `poetry run` to use the stuff installed within the local environment, e.g. `poetry run jupyter notebook` to start a jupyter notebook server. This local environment is basically a python virtual environment. To correctly set up the interpreter in your IDE, use `poetry run which python` to see the path to the correct python executable.

## Examples

A few examples are included to introduce users and developers to the TomOpt library. These take the form of Jupyter notebooks. In `examples/getting_started` there are four ordered notebooks:

- `00_Hello_World.ipynb` aims to show the user the high-level classes in TomOpt and the general workflow.
- `01_Indepth_tutorial_single_cycle.ipynb` aims to show developers what is going on in a single update iteration.
- `02_Indepth_tutotial_optimisation_and_callbacks.ipynb` aims to show users and developers the workings of the callback system in TomOpt
- `03_fixed_budget_mode.ipynb` aims to show users and developers how to optimise such that the detector maintains a constant cost.

In `examples/benchmarks` there is a single notebook that covers the optimisation performed in our first publication, in which we optimised a detector to estimate the fill-height of a ladle furnace at a steel plant. As a disclaimer, this notebook may not fully reproduce our result, and is designed to be used in an interactive manner by experienced users.


### Running notebooks in a remote cluster

If you want to run notebooks on a remote cluster but access them on the browser of your local machine, you need to forward the notebook server from the cluster to your local machine.

On the cluster, run:
```
poetry run jupyter notebook --no-browser --port=8889
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

N.B. Most are not currently public

- [tomo_deepinfer](https://github.com/GilesStrong/mode_muon_tomo_inference) (contact @GilesStrong for access) separately handles training and model definition of GNNs used for passive volume inference. Models are exported as JIT-traced scripts, and loaded here using the `DeepVolumeInferer` class. We still need to find a good way to host the trained models for easy download.
- [mode_muon_tomography_scattering](https://github.com/GilesStrong/mode_muon_tomography_scattering)  (contact @GilesStrong for access) separately handles conversion of PGeant model from root to HDF5, and Geant validation data from csv to HDF5.
- [tomopt_sphinx_theme](https://github.com/GilesStrong/tomopt_sphinx_theme) public. Controls the appearance of the docs.

## Authors

The TomOpt project, and its continued development and support, is the result of the combined work of many people, whose contributions are summarised in [the author list](https://github.com/GilesStrong/tomopt/blob/main/AUTHORS.md)