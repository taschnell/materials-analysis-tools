
import numpy as np
import numpy.typing as npt

from elli.dispersions.base_dispersion import Dispersion
from elli.utils import conversion_wavelength_energy

# pyElli lacks a harmonic oscillator, so i made one

class HarmonicOscillator(Dispersion):
    """Harmonic oscillator dispersion with energy parameters."""

    single_params_template = {}
    rep_params_template = {"A": 1, "E0": 1, "gamma": 1}

    def dielectric_function(self, lbda: npt.ArrayLike) -> npt.NDArray:
        energy = np.asarray(
            conversion_wavelength_energy(lbda),
            dtype=float
        )

        epsilon = np.zeros_like(energy, dtype=complex)

        for p in self.rep_params:
            A = p["A"]
            E0 = p["E0"]
            gamma = p["gamma"]

            epsilon += (
                A * gamma * E0
                / (E0**2 + gamma**2 / 4
                   - energy**2 - 1j * gamma * energy)
            )

        return epsilon
