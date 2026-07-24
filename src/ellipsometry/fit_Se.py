from pathlib import Path
import sys

import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from load_data import load_elippso_data

import elli
from elli.structure import Structure, Layer
from elli.dispersions import CodyLorentz
from elli.materials import IsotropicMaterial


# =====================================================
# 1. Define fixed materials
# =====================================================
air = IsotropicMaterial(
    elli.ConstantRefractiveIndex(1.0)
)

glass = IsotropicMaterial(
    elli.ConstantRefractiveIndex(1.515)
)


# =====================================================
# 2. Helper functions
# =====================================================

def select_dataset(datasets, requested_name=None):
    if not datasets:
        raise RuntimeError("No datasets were loaded from the data directory.")

    if requested_name:
        if requested_name in datasets:
            return requested_name, datasets[requested_name]
        raise KeyError(f"{requested_name} dataset not found in the loaded data.")

    preferred_names = ["KH522_Ellipso" ,"TS3_Ellipso", "TS18_Ellipso"]
    for name in preferred_names:
        if name in datasets:
            return name, datasets[name]

    first_name = next(iter(datasets))
    return first_name, datasets[first_name]


def make_film(Eg, A, Et, gamma, Ep, E0, Eu):

    dispersion = CodyLorentz(
    Eg=Eg,
    A=A,
    Et=Et,
    gamma=gamma,
    Ep=Ep,
    E0=E0,
    Eu=Eu

)

    return IsotropicMaterial(dispersion)


# =====================================================
# 3. Forward model and residuals
# =====================================================

def model(params):

    thickness = params[0]
    Eg = params[1]
    A = params[2]
    Et = params[3]
    gamma = params[4]
    Ep = params[5]
    E0 = params[6]
    Eu = params[7]

    film = make_film(Eg, A, Et, gamma, Ep, E0, Eu)

    layer = Layer(
        film,
        thickness
    )

    sample = Structure(
        air,
        [layer],
        glass
    )


    psi_model = []
    delta_model = []


    wavelengths = wavelength_exp


    result = sample.evaluate(
            wavelengths,
            MEASURE_ANGLE
            )

    psi_model.extend(result.psi)
    delta_model.extend(result.delta)


    return (
        np.array(psi_model),
        np.array(delta_model)
    )


def residuals(params):

    psi_model, delta_model = model(params)

    delta_residual = (delta_model - delta_exp + 180) % 360 - 180

    psi_weight = 1.0
    delta_weight = 1.0

    return np.concatenate(
        [
            psi_weight * (psi_model - psi_exp),
            delta_weight * delta_residual
        ]
    )


# =====================================================
# 4. Load and process data
# =====================================================
angles = [65.02, 70.02, 75.02] 
MEASURE_ANGLE = 75.02  # degrees

requested_name = sys.argv[1] if len(sys.argv) > 1 else None
datasets = load_elippso_data()

print("Available files:")
print(list(datasets.keys()))

dataset_name, data = select_dataset(datasets, requested_name)
print(f"Using dataset: {dataset_name}")
print("Loaded columns:", list(data.keys()) if isinstance(data, dict) else data.shape)

# Filtering out rows that do not match the specified measurement angle
if not isinstance(data, dict):
    raise RuntimeError("Expected dataset rows as a dict of arrays")

mask = np.isclose(np.asarray(data['angle']), MEASURE_ANGLE)
print(f"Keeping {mask.sum()} rows where angle == {MEASURE_ANGLE} out of {mask.size}")

filtered_data = {key: np.asarray(value)[mask] for key, value in data.items()}

# for wl, ang, psi, delta in zip(filtered_data['wavelength'], filtered_data['angle'], filtered_data['psi'], filtered_data['delta']):
#     print(wl, ang, psi, delta)

wavelength_exp = filtered_data["wavelength"]
angle_exp = filtered_data["angle"]
psi_exp = filtered_data["psi"]
delta_exp = filtered_data["delta"]

print(f"Using {MEASURE_ANGLE} degrees measurement angle with {len(wavelength_exp)} data points.")


# =====================================================
# 5. Main execution
# =====================================================

initial_params = [150, 1.8, 50, 0.2, 2.4, 0.35, 5.9, 0.07]

psi_initial, delta_initial = model(initial_params)

fit = least_squares(
    residuals,
    x0=initial_params,
    bounds=(
        [50, 0.5, 1, 0.01, 0.1, 0.1, 1, 0.01],
        [200, 3, 500, 5, 15, 1.0, 20, 0.25]
    )
)


print("=" * 50)
print("FIT RESULTS")
print("=" * 50)
print(f"Success: {fit.success}")
print(f"Message: {fit.message}")
print(f"Number of iterations: {fit.nfev}")
print()
print("Best fit parameters:")
print(f"  Thickness (nm):  {fit.x[0]:.4f}")
print(f"  Eg (eV):         {fit.x[1]:.4f}")
print(f"  A (eV):          {fit.x[2]:.4f}")
print(f"  Et (eV):         {fit.x[3]:.4f}")
print(f"  Gamma:           {fit.x[4]:.4f}")
print(f"  Ep (eV):         {fit.x[5]:.4f}")
print(f"  E0 (eV):         {fit.x[6]:.4f}")
print(f"  Eu (eV):         {fit.x[7]:.4f}")
print()
print(f"Mean squared error: {np.mean(fit.fun**2):.6f}")
print("=" * 50)



# Get model predictions with fitted parameters
psi_model, delta_model = model(fit.x)

# Plot experimental vs model
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

# Plot Psi
ax1.plot(wavelength_exp, psi_exp, 'o', label='Experimental', alpha=0.6)
ax1.plot(wavelength_exp, psi_model, '-', label='Model', linewidth=2)
ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Psi (degrees)')
ax1.set_title('Psi: Experimental vs Model')
ax1.grid(True, alpha=0.3)
ax1.legend()

# Plot Delta
ax2.plot(wavelength_exp, delta_exp, 'o', label='Experimental', alpha=0.6)
ax2.plot(wavelength_exp, delta_model, '-', label='Model', linewidth=2)
ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Delta (degrees)')
ax2.set_title('Delta: Experimental vs Model')
ax2.grid(True, alpha=0.3)
ax2.legend()

plt.tight_layout()
plt.savefig('experimental_vs_model.png', dpi=150)
print("Saved comparison graph to experimental_vs_model.png")
plt.show()