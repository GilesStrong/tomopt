import numpy as np
from typing import Union, Tuple, Optional
from particle import Particle

import torch
from torch import Tensor

__all__ = ["MuonGenerator"]


class MuonGenerator:
    _muon_mass2 = (Particle.from_pdgid(13).mass * 1e-3) ** 2
    _n_bins_energy = 200
    _n_bins_theta = 200

    def __init__(
        self, x_range: Tuple[float, float], y_range: Tuple[float, float], fixed_mom: Optional[float] = 5.0, energy_range: Tuple[float, float] = (0.5, 500)
    ) -> None:
        """
        Initializer. Specify dimensions x,y of the impinging surface flag (True/False) for sampled vs uniform muon momenta respectively
        Uses model defined in arXiv:1509.06176
        """

        self.x_range, self.y_range = x_range, y_range

        # Define intervals and centres
        self._fixed_mom = fixed_mom
        energy_edges = np.geomspace(energy_range[0], energy_range[1], self._n_bins_energy + 1)
        self._energy_centres = np.mean(np.vstack([energy_edges[0:-1], energy_edges[1:]]), axis=0)
        theta_edges = np.linspace(0, 0.5 * np.pi, self._n_bins_theta + 1)
        self._theta_centres = np.mean(np.vstack([theta_edges[0:-1], theta_edges[1:]]), axis=0)

        # Calculate 2d flux function
        xx, yy = np.meshgrid(self._energy_centres, self._theta_centres)
        self._edges_1d = np.cumsum(self.flux(xx, yy))  # theta x energy --> e0t0, e1t0, ...

    def __call__(self, n_muons: int) -> Tensor:
        return self.generate_set(n_muons)

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
        theta_indices, energy_indices = [indices_1d // self._n_bins_energy, indices_1d % self._n_bins_energy]
        if self._fixed_mom is None:
            momentum = np.sqrt(np.square(self._energy_centres[energy_indices]).squeeze() - self._muon_mass2)  # Momentum [GeV/c]
        else:
            momentum = np.ones(len(theta_indices)) * self._fixed_mom
        theta = self._theta_centres[theta_indices]

        # Get theta_x and theta_y from theta and phi
        phi = np.random.uniform(0, 2 * np.pi, n_muons)
        theta_x = theta * np.cos(phi)
        theta_y = theta * np.sin(phi)

        # Generate x and y randomly
        coord_x = np.random.uniform(self.x_range[0], self.x_range[1], n_muons)
        coord_y = np.random.uniform(self.y_range[0], self.y_range[1], n_muons)

        return torch.Tensor(np.stack([coord_x, coord_y, momentum, theta_x, theta_y], axis=1))
