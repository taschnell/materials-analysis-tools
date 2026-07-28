from pathlib import Path
import sys

import numpy as np

from datetime import datetime
from pathlib import Path


import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

from scipy.optimize import least_squares

from load_data import load_elippso_data

import elli
from elli.structure import Structure, Layer
from elli.dispersions import CodyLorentz
from elli.materials import IsotropicMaterial
from elli.materials import BruggemanEMA


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

    preferred_names = ["KH522_Ellipso"]
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

def make_roughness(host_material, guest_material, fraction):

    dispersion = BruggemanEMA(
        host_material,
        guest_material,
        fraction
    )

    return dispersion


# =====================================================
# 3. Forward model and residuals
# =====================================================
    
def model(params):

    film_thickness = params[0]
    roughness_thickness = params[1]
    Eg = params[2]
    A = params[3]
    Et = params[4]
    gamma = params[5]
    Ep = params[6]
    E0 = params[7]
    Eu = params[8]

    film = make_film(Eg, A, Et, gamma, Ep, E0, Eu)

    rough_film = make_roughness(film, air, 0.5)



    film_layer = Layer(
            film,
            film_thickness
        )

    roughness_layer = Layer(
        rough_film,
        roughness_thickness
    )

    sample = Structure(
        air,
        [
            roughness_layer,
            film_layer
        ],
        glass
    )

    psi_model = []
    delta_model = []

    for angle in fit_angles:
        angle_mask = np.isclose(angle_exp, angle)
        wavelengths = wavelength_exp[angle_mask]

        result = sample.evaluate(
            wavelengths,
            float(angle)
        )

        psi_model.extend(result.psi)
        delta_model.extend(result.delta)

    return (
        np.array(psi_model),
        np.array(delta_model)
    )


def residuals(params):

    psi_model, delta_model = model(params)

    if (
        np.any(~np.isfinite(psi_model)) or
        np.any(~np.isfinite(delta_model))
    ):
        return np.ones(len(psi_exp)*2)*1e6

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
requested_name = sys.argv[1] if len(sys.argv) > 1 else None
datasets = load_elippso_data()

print("Available files:")
print(list(datasets.keys()))

dataset_name, data = select_dataset(datasets, requested_name)
print(f"Using dataset: {dataset_name}")
print("Loaded columns:", list(data.keys()) if isinstance(data, dict) else data.shape)

if not isinstance(data, dict):
    raise RuntimeError("Expected dataset rows as a dict of arrays")

angles_arr = np.asarray(data['angle'], dtype=float)
fit_angles = np.sort(np.unique(angles_arr))
print(f"Using measurement angles: {fit_angles}")

wavelengths_all = []
angles_all = []
psi_all = []
delta_all = []

for angle in fit_angles:
    angle_mask = np.isclose(angles_arr, angle)
    subset = {key: np.asarray(value)[angle_mask] for key, value in data.items()}

    wavelengths_all.extend(subset["wavelength"])
    angles_all.extend(np.full(len(subset["wavelength"]), angle, dtype=float))
    psi_all.extend(subset["psi"])
    delta_all.extend(subset["delta"])

wavelength_exp = np.asarray(wavelengths_all, dtype=float)
angle_exp = np.asarray(angles_all, dtype=float)
psi_exp = np.asarray(psi_all, dtype=float)
delta_exp = np.asarray(delta_all, dtype=float)

print(f"Using {len(fit_angles)} measurement angles with {len(wavelength_exp)} total data points.")


# =====================================================
# 5. Main execution
# =====================================================

timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")

project_root = Path(__file__).resolve().parents[2]
output_dir = project_root / "plots"/ "ellipsometry" / dataset_name / timestamp

output_dir.mkdir(parents=True, exist_ok=True)

print(f"Saving output to: {output_dir}")


# initial_params = [106, 5, 1.8, 50, 0.2, 2.4, 0.35, 5.9, 0.07]
initial_params = [106,3.6,1.8,50,0.2,2.4,0.35,5.9,0.07]

psi_initial, delta_initial = model(initial_params)

fit = least_squares(
    residuals,
    x0=initial_params,
    bounds=(
        [50, 0, 1.8, 1, 0.01, 0.1, 0.1, 1, 0.01],
        [170, 10, 2.5, 200, 5, 10, 1.0, 20, 0.15]
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
print(f"  Roughness (nm):  {fit.x[1]:.4f}")
print(f"  Eg (eV):         {fit.x[2]:.4f}")
print(f"  A (eV):          {fit.x[3]:.4f}")
print(f"  Et (eV):         {fit.x[4]:.4f}")
print(f"  Gamma:           {fit.x[5]:.4f}")
print(f"  Ep (eV):         {fit.x[6]:.4f}")
print(f"  E0 (eV):         {fit.x[7]:.4f}")
print(f"  Eu (eV):         {fit.x[8]:.4f}")
print()
print(f"Mean squared error: {np.mean(fit.fun**2):.6f}")
print("=" * 50)


# Get model predictions with fitted parameters
psi_model, delta_model = model(fit.x)

psi_residual = psi_model - psi_exp

print(
    "Psi RMS:",
    np.sqrt(np.mean(psi_residual**2))
)


# Wrapped residual actually used by optimizer
delta_residual = (delta_model - delta_exp + 180) % 360 - 180

# Force model onto the same phase branch as experiment
delta_model_nearest = delta_exp + delta_residual


print(
    "Delta RMS:",
    np.sqrt(np.mean(delta_residual**2))
)

param_text = (
    f"Film Thickness = {fit.x[0]:.2f} nm\n"
    f"Roughness Thickness = {fit.x[1]:.2f} nm\n"
    f"Eg = {fit.x[2]:.4f} eV\n"
    f"A = {fit.x[3]:.4f} eV\n"
    f"Et = {fit.x[4]:.4f} eV\n"
    f"Gamma = {fit.x[5]:.4f}\n"
    f"Ep = {fit.x[6]:.4f} eV\n"
    f"E0 = {fit.x[7]:.4f} eV\n"
    f"Eu = {fit.x[8]:.4f} eV\n"
    f"\n"
    f"Psi RMS = {np.sqrt(np.mean(psi_residual**2)):.3f}°\n"
    f"Delta RMS = {np.sqrt(np.mean(delta_residual**2)):.3f}°"
)


# Plot experimental vs model
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

for angle in fit_angles:
    angle_mask = np.isclose(angle_exp, angle)

    # Psi
    ax1.plot(
        wavelength_exp[angle_mask],
        psi_exp[angle_mask],
        'o',
        label=f'Experimental {angle:.2f}°',
        alpha=0.6
    )

    ax1.plot(
        wavelength_exp[angle_mask],
        psi_model[angle_mask],
        '-',
        linewidth=2
    )

    # Delta
    ax2.plot(
        wavelength_exp[angle_mask],
        delta_exp[angle_mask],
        'o',
        label=f'Experimental {angle:.2f}°',
        alpha=0.6
    )

    ax2.plot(
        wavelength_exp[angle_mask],
        delta_model_nearest[angle_mask],
        '-',
        linewidth=2
    )

ax1.set_xlabel('Wavelength (nm)')
ax1.set_ylabel('Psi (degrees)')
ax1.set_title('Psi: Experimental vs Model')
ax1.grid(True, alpha=0.3)
ax1.legend()

ax2.set_xlabel('Wavelength (nm)')
ax2.set_ylabel('Delta (degrees)')
ax2.set_title('Delta: Experimental vs Model (Nearest Phase Branch)')
ax2.grid(True, alpha=0.3)
ax2.legend()

fig.text(
    0.765,          # x position
    0.09,           # y position
    param_text,
    fontsize=9,
    family="monospace",
    bbox=dict(
        boxstyle="round",
        facecolor="white",
        alpha=0.85
    )
)

plt.tight_layout()
plt.savefig(output_dir / 'experimental_vs_model.png', dpi=150)

plt.figure(figsize=(10,5))

psi_residual = psi_model - psi_exp
delta_residual = (delta_model - delta_exp + 180) % 360 - 180

fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 8), sharex=True)

for angle in fit_angles:
    mask = np.isclose(angle_exp, angle)

    ax1.plot(
        wavelength_exp[mask],
        psi_residual[mask],
        label=f'{angle:.2f}°'
    )

    ax2.plot(
        wavelength_exp[mask],
        delta_residual[mask],
        label=f'{angle:.2f}°'
    )

ax1.axhline(0, color='k', linestyle='--')
ax1.set_ylabel('Psi residual (deg)')
ax1.set_title('Psi Residual')
ax1.grid(True)

ax2.axhline(0, color='k', linestyle='--')
ax2.set_ylabel('Delta residual (deg)')
ax2.set_xlabel('Wavelength (nm)')
ax2.set_title('Wrapped Delta Residual')
ax2.grid(True)

ax1.legend()

plt.tight_layout()
plt.savefig(output_dir / 'residuals.png', dpi=150)

with open(output_dir / "fit_results.txt", "w") as f:
    f.write(f"Success: {fit.success}\n")
    f.write(f"Message: {fit.message}\n")
    f.write(f"Number of iterations: {fit.nfev}\n")
    f.write("Best fit parameters:\n")
    f.write(f"  Thickness (nm):  {fit.x[0]:.4f}\n")
    f.write(f"  Roughness (nm):  {fit.x[1]:.4f}\n")
    f.write(f"  Eg (eV):         {fit.x[2]:.4f}\n")
    f.write(f"  A (eV):          {fit.x[3]:.4f}\n")
    f.write(f"  Et (eV):         {fit.x[4]:.4f}\n")
    f.write(f"  Gamma:           {fit.x[5]:.4f}\n")
    f.write(f"  Ep (eV):         {fit.x[6]:.4f}\n")
    f.write(f"  E0 (eV):         {fit.x[7]:.4f}\n")
    f.write(f"  Eu (eV):         {fit.x[8]:.4f}\n")