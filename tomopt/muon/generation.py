import math
import torch
from torch import Tensor
import numpy as np
from torch import tensor
from typing import Union, List, Tuple

__all__ = ["generate_batch", "MuonGenerator"]


def generate_batch(n: int, p: float = 5) -> Tensor:
    r"""
    Return tensor is (muons, coords),
    coords = (x~Uniform[0,1], y~Uniform[0,1], momentum (fixed), theta_x~cos2(a) a~Uniform[0,0.5pi], theta_y~Uniform[0,2pi])

    TODO:  specify initial x,y range
    """

    batch = torch.stack(
        [
            torch.rand(n),
            torch.rand(n),
            torch.zeros(n) + p,
            torch.clamp(torch.randn(n) / 10, -math.pi / 2, math.pi / 2),  # Fix this
            torch.clamp(torch.randn(n) / 10, -math.pi / 2, math.pi / 2),  # Fix this
        ],
        dim=1,
    )
    return batch


def flux(energy: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
    """
    Function returning modified Gaisser formula for cosmic muon flux given energy (float/np.array) and incidence angle (float/np.array)
    """
    cosTheta = np.cos(theta)
    P1 = 0.102573
    P2 = -0.068287
    P3 = 0.958633
    P4 = 0.0407253
    P5 = 0.817285
    cosine = np.sqrt((cosTheta ** 2 + P1 ** 2 + P2 * cosTheta ** P3 + P4 * cosTheta ** P5) / (1 + P1 ** 2 + P2 + P4))
    flux = (
        0.14
        * (energy * (1 + 3.64 / (energy * cosine ** 1.29))) ** (-2.7)
        * ((1 / (1 + (1.1 * energy * cosine) / 115)) + (0.054 / (1 + (1.1 * energy * cosine) / 850)))
    )
    return flux


class MuonGenerator:
    def __init__(self, max_x: float, max_y: float):
        """
        Initializer. Specify dimensions x,y of the impinging surface
        """
        self._dimensions = (max_x, max_y)

    @property
    def dimensions(self) -> Tuple[float, float]:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dim: Tuple[float, float]) -> None:
        self._dimensions = dim

    def generate_set(self, n_muons: int) -> Tensor:
        """
        Function to generate a set of muons [x, y, momentum, theta_x, theta_y] distributed according to the modified Gaisser formula given the size of the sample set.
        Note: optimized for log binning in (0.5,500)GeV
        """
        n_bins_energy = 200
        n_bins_theta = 200
        # Define intervals and centres
        energy_edges = np.geomspace(0.5, 500, n_bins_energy + 1)
        theta_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, n_bins_theta + 1)
        energy_centres = np.mean(np.vstack([energy_edges[0:-1], energy_edges[1:]]), axis=0)
        theta_centres = np.mean(np.vstack([theta_edges[0:-1], theta_edges[1:]]), axis=0)
        # Calculate 2d flux function
        xx, yy = np.meshgrid(energy_centres, theta_centres)
        edges_1d = np.append(0, np.cumsum(flux(xx, yy)))
        # Sample 2d function in 1d intervals
        muon_sample = np.random.uniform(0.0, edges_1d[-1], n_muons)
        indices_1d = edges_1d.searchsorted(muon_sample)
        # Get corresponding values
        theta_indices, energy_indices = [indices_1d // n_bins_energy, indices_1d % n_bins_energy]
        momentum = np.sqrt(np.square(energy_centres[energy_indices]).squeeze() - (0.106 * 0.106))  # Momentum [GeV/c]
        phi = theta_centres[theta_indices]
        # Generate randomly theta in [-pi/2,pi/2]
        theta = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, n_muons)
        # Get theta_x and theta_y from theta and phi
        theta_x = theta * np.cos(phi)
        theta_y = theta * np.sin(phi)
        # Generate x and y randomly
        coord_x = np.random.uniform(0, self.dimensions[1], n_muons)
        coord_y = np.random.uniform(0, self.dimensions[0], n_muons)
        return torch.Tensor(np.stack([coord_x, coord_y, momentum, theta_x, theta_y], axis=1))
