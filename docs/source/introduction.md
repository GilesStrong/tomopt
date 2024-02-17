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