from __future__ import annotations
from abc import abstractmethod
import numpy as np
from typing import Union, Tuple, Optional, TYPE_CHECKING
from particle import Particle

import torch
from torch import Tensor

if TYPE_CHECKING:
    from ..volume import Volume

__all__ = ["MuonGenerator2015", "MuonGenerator2016"]


class AbsMuonGenerator:
    _muon_mass2 = (Particle.from_pdgid(13).mass * 1e-3) ** 2  # GeV^2
    _n_bins_energy = 200
    _n_bins_theta = 200

    def __init__(
        self,
        x_range: Tuple[float, float],
        y_range: Tuple[float, float],
        fixed_mom: Optional[float] = 5.0,
        energy_range: Tuple[float, float] = (0.5, 500),
        theta_range: Tuple[float, float] = (0, 0.5 * np.pi),
    ) -> None:
        """
        Initializer. Specify dimensions x,y of the impinging surface flag (True/False) for sampled vs uniform muon momenta respectively
        """

        self.x_range, self.y_range = x_range, y_range

        # Define intervals and centres
        self._fixed_mom = fixed_mom
        energy_edges = np.geomspace(energy_range[0], energy_range[1], self._n_bins_energy + 1)
        self._energy_centres = np.mean(np.vstack([energy_edges[0:-1], energy_edges[1:]]), axis=0)
        theta_edges = np.linspace(theta_range[0], theta_range[1], self._n_bins_theta + 1)
        self._theta_centres = np.mean(np.vstack([theta_edges[0:-1], theta_edges[1:]]), axis=0)

        # Calculate 2d flux function
        xx, yy = np.meshgrid(self._energy_centres, self._theta_centres)
        self._edges_1d = np.cumsum(self.flux(xx, yy))  # theta x energy --> e0t0, e1t0, ...

    def __repr__(self) -> str:
        rep = f"Muon generator: x,y range: {self.x_range}, {self.y_range}."
        if self._fixed_mom is None:
            rep += f" Energy sampled from {self._energy_centres[0]}-{self._energy_centres[-1]} GeV."
        else:
            rep += f" Momentum is fixed at {self._fixed_mom} GeV"
        return rep

    def __call__(self, n_muons: int) -> Tensor:
        return self.generate_set(n_muons)

    @classmethod
    def from_volume(
        cls, volume: Volume, min_angle: float = np.pi / 12, fixed_mom: Optional[float] = 5.0, energy_range: Tuple[float, float] = (0.5, 500)
    ) -> AbsMuonGenerator:
        """
        Heuristically computes x,y generation range as (0-d,x+d), (0-d,y+d).
        Where d is such that a muon generated at (0-d,1) will only hit the last layer of the passive volume if it's initial angle is at least min_angle.
        This balances a trade-off between generation efficiency and generator realism.
        """

        x, y = volume.lw.detach().cpu().numpy().tolist()
        d = np.tan(min_angle) * (volume.h.detach().cpu().item() - volume.get_passive_z_range()[0].detach().cpu().item() + volume.passive_size)
        return cls(x_range=(0 - d, x + d), y_range=(0 - d, y + d), fixed_mom=fixed_mom, energy_range=energy_range)

    @abstractmethod
    def flux(self, energy: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        pass

    def generate_set(self, n_muons: int) -> Tensor:
        """
        Function to generate a set of muons [x, y, momentum, theta, phi] distributed according to the modified Gaisser formula given the size of the sample set.
        Note: optimized for log binning in (0.5,500)GeV
        """

        # Sample 2d function in 1d intervals
        muon_sample = np.random.uniform(0.0, self._edges_1d[-1], n_muons)
        indices_1d = self._edges_1d.searchsorted(muon_sample)

        # Get corresponding values
        theta_indices, energy_indices = [indices_1d // self._n_bins_energy, indices_1d % self._n_bins_energy]
        if self._fixed_mom is None:
            momentum = np.sqrt(np.square(self._energy_centres[energy_indices]) - self._muon_mass2)  # Momentum [GeV/c]
        else:
            momentum = np.ones(len(theta_indices)) * self._fixed_mom
        theta = self._theta_centres[theta_indices]

        phi = np.random.uniform(0, 2 * np.pi, n_muons)

        # Generate x and y randomly
        coord_x = np.random.uniform(self.x_range[0], self.x_range[1], n_muons)
        coord_y = np.random.uniform(self.y_range[0], self.y_range[1], n_muons)

        return torch.Tensor(np.stack([coord_x, coord_y, momentum, theta, phi], axis=1))


class MuonGenerator2015(AbsMuonGenerator):
    P1 = 0.102573
    P2 = -0.068287
    P3 = 0.958633
    P4 = 0.0407253
    P5 = 0.817285

    def flux(self, energy: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Function returning modified Gaisser formula for cosmic muon flux given energy (float/np.array) and incidence angle (float/np.array)
        Uses model defined in Guan et al. 2015 (arXiv:1509.06176)
        """

        cosTheta = np.cos(theta)
        cosine = np.sqrt(
            (cosTheta**2 + self.P1**2 + self.P2 * cosTheta**self.P3 + self.P4 * cosTheta**self.P5) / (1 + self.P1**2 + self.P2 + self.P4)
        )
        flux = (
            0.14
            * (energy * (1 + 3.64 / (energy * cosine**1.29))) ** (-2.7)
            * ((1 / (1 + (1.1 * energy * cosine) / 115)) + (0.054 / (1 + (1.1 * energy * cosine) / 850)))
        )
        return flux


class MuonGenerator2016(AbsMuonGenerator):
    I_0 = 88.0
    n = 3
    E_0 = 3.87
    E_c = 0.5
    epinv = 1 / 854.0
    Rod = 174.0
    N = (n - 1) * (E_0 + E_c) ** (n - 1)

    def flux(self, energy: Union[float, np.ndarray], theta: Union[float, np.ndarray]) -> Union[float, np.ndarray]:
        """
        Function returning modified Gaisser formula for cosmic muon flux given energy (float/np.array) and incidence angle (float/np.array)
        Uses model defined in Shukla and Sanskrith 2018 arXiv:1606.06907
        """
        #  initialize cosmic variables

        Cosine = (np.sqrt(self.Rod**2 * np.cos(theta) ** 2 + 2 * self.Rod + 1) - self.Rod * np.cos(theta)) ** (-(self.n - 1))
        flux = self.I_0 * self.N * (self.E_0 + energy) ** (-self.n) * (1 + energy * self.epinv) ** (-1) * Cosine
        return flux
