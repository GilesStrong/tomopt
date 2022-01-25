import math
import torch
from torch import Tensor
import numpy as np
from torch import tensor
from typing import Union, List, Tuple
from particle import Particle

__all__ = ["MuonGenerator"]


class MuonGenerator:
    def __init__(self, max_x: float = 1.0, max_y: float = 1.0, sample_mom: bool = False) -> None:
        """
        Initializer. Specify dimensions x,y of the impinging surface flag (True/False) for sampled vs uniform muon momenta respectively
        """
        self._muon_mass = Particle.from_pdgid(13).mass * 1e-3
        self._sample_momentum = sample_mom
        self._dimensions = (max_x, max_y)
        self._n_bins_energy = 200
        self._n_bins_phi = 200
        # Define intervals and centres
        energy_edges = np.geomspace(0.5, 500, self._n_bins_energy + 1)
        phi_edges = np.linspace(-0.5 * np.pi, 0.5 * np.pi, self._n_bins_phi + 1)
        self._energy_centres = np.mean(np.vstack([energy_edges[0:-1], energy_edges[1:]]), axis=0)
        self._phi_centres = np.mean(np.vstack([phi_edges[0:-1], phi_edges[1:]]), axis=0)
        # Calculate 2d flux function
        xx, yy = np.meshgrid(self._energy_centres, self._phi_centres)
        self._edges_1d = np.append(0, np.cumsum(self.flux(xx, yy)))

    def __call__(self, n_muons: int) -> Tensor:
        return self.generate_set(n_muons)

    @property
    def dimensions(self) -> Tuple[float, float]:
        return self._dimensions

    @dimensions.setter
    def dimensions(self, dim: Tuple[float, float]) -> None:
        self._dimensions = dim

    def flux(self, energy: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Function returning modified Gaisser formula for cosmic muon flux given energy (float/np.array) and incidence angle (float/np.array)
        Uses model defined in arXiv:1509.06176
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

    def generate_set(self, n_muons: int) -> Tensor:
        """
        Function to generate a set of muons [x, y, momentum, theta_x, theta_y] distributed according to the modified Gaisser formula given the size of the sample set.
        Note: optimized for log binning in (0.5,500)GeV
        """
        # Sample 2d function in 1d intervals
        muon_sample = np.random.uniform(0.0, self._edges_1d[-1], n_muons)
        indices_1d = self._edges_1d.searchsorted(muon_sample)
        # Get corresponding values
        phi_indices, energy_indices = [indices_1d // self._n_bins_energy, indices_1d % self._n_bins_energy]
        if self._sample_momentum is False:
            momentum = torch.Tensor(np.ones(len(phi_indices))) * 5.0
        else:
            momentum = np.sqrt(np.square(self._energy_centres[energy_indices]).squeeze() - (self._muon_mass * self._muon_mass))  # Momentum [GeV/c]
        phi = self._phi_centres[phi_indices]
        # Generate randomly theta in [-pi/2,pi/2]
        theta = np.random.uniform(-0.5 * np.pi, 0.5 * np.pi, n_muons)
        # Get theta_x and theta_y from theta and phi
        theta_x = theta * np.cos(phi)
        theta_y = theta * np.sin(phi)
        # Generate x and y randomly
        coord_x = np.random.uniform(0, self.dimensions[1], n_muons)
        coord_y = np.random.uniform(0, self.dimensions[0], n_muons)
        return torch.Tensor(np.stack([coord_x, coord_y, momentum, theta_x, theta_y], axis=1))
