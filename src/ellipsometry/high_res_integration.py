
import numpy as np

from elli.dispersions import CodyLorentz
from elli.dispersions.base_dispersion import DispersionSum
from elli.kkr import im2re_reciprocal
from elli.utils import conversion_wavelength_energy


def calculate_optical_constants_high_res(
    params, wavelengths_nm, model, grid_points=7500
):
    """High-resolution n,k for single- or multi-Cody models."""

    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)
    dispersion = model.build_film_dispersion(params)

    # Extract individual dispersion components
    if isinstance(dispersion, DispersionSum):
        components = dispersion.dispersions
    else:
        components = [dispersion]

    # High-resolution Kramers-Kronig integration grid
    grid = np.linspace(50, 10000, grid_points)
    energy_grid = conversion_wavelength_energy(grid)
    energy_out = conversion_wavelength_energy(wavelengths_nm)

    # Combined imaginary dielectric functions of Cody terms
    eps2_grid = np.zeros_like(grid)
    eps2_out = np.zeros_like(wavelengths_nm)

    # Other dispersions, e.g. harmonic oscillators
    eps_other = np.zeros_like(wavelengths_nm, dtype=complex)

    for component in components:
        if isinstance(component, CodyLorentz):
            eps2_grid += component.eps2(
                energy_grid, **component.single_params
            )
            eps2_out += component.eps2(
                energy_out, **component.single_params
            )
        else:
            # Keep analytical dielectric contribution
            eps_other += component.dielectric_function(
                wavelengths_nm
            )

    # One KK integration for all Cody contributions
    eps1_grid = im2re_reciprocal(eps2_grid, grid)
    eps1_out = np.interp(wavelengths_nm, grid, eps1_grid)

    # Sum dielectric contributions before taking square root
    epsilon = eps1_out + 1j * eps2_out + eps_other

    complex_n = np.sqrt(epsilon.astype(complex))
    return complex_n.real, complex_n.imag
