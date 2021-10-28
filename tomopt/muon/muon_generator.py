import numpy as np
import torch
from torch import tensor
from torch.functional import Tensor
from typing import Union, List

__all__ = ["MuonGenerator"]


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
    def __init__(self, x: float, y: float):
        """
        Initializer. Specify dimensions x,y of the impinging surface
        """
        self._dimensions = [x, y]

    @property
    def dimensions(self) -> List:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dim: List) -> None:
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
