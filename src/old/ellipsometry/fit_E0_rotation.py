from pathlib import Path

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
# 1. Load data
# =====================================================

datasets = load_elippso_data()

print("Available files:")
print(datasets.keys())


data = datasets["TS3_Ellipso"]


wavelength_exp = data["wavelength"]
angle_exp = data["angle"]
psi_exp = data["psi"]
delta_exp = data["delta"]


angles = np.unique(angle_exp)

print("Angles found:", angles)


# =====================================================
# 2. Define fixed materials
# =====================================================

air = IsotropicMaterial(
    elli.ConstantRefractiveIndex(1.0)
)

glass = IsotropicMaterial(
    elli.ConstantRefractiveIndex(1.52)
)



# =====================================================
# 3. Cody-Lorentz material
# =====================================================

def make_film(Eg, A, gamma,E0):

    dispersion = CodyLorentz(
    Eg=Eg,
    A=A,
    Et=0.2,
    gamma=gamma,
    Ep=0.35,
    E0=E0,
    Eu=0.07

)

    return IsotropicMaterial(dispersion)



# =====================================================
# 4. Forward model
# =====================================================

def model(params):

    thickness = params[0]
    Eg = params[1]
    A = params[2]
    gamma = params[3]
    E0 = params[4]

    film = make_film(Eg, A, gamma, E0)

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


    # evaluate each angle separately
    for angle in angles:

        mask = angle_exp == angle

        wavelengths = wavelength_exp[mask]


        result = sample.evaluate(
            wavelengths,
            angle
        )

        psi_model.extend(result.psi)
        delta_model.extend(result.delta)


    return (
        np.array(psi_model),
        np.array(delta_model)
    )



# =====================================================
# 5. Residual function
# =====================================================

def residuals(params):

    psi_model, delta_model = model(params)

    return np.concatenate(
        [
            psi_model - psi_exp,
            delta_model - delta_exp
        ]
    )


initial_params = [150, 2.0, 50, 2.4, 5.9]

psi_initial, delta_initial = model(initial_params)


# =====================================================
# 6. Fit thickness
# =====================================================

# fit = least_squares(
#     residuals,
#     x0=initial_params,
#     bounds=(
#         [120, 1.99, 1, 0.1, 1.0],
#         [180, 2.0, 200, 10.0, 10.0]
#     )
# )


for E0 in [1,2.5,5,7.5,10]:

    fit = least_squares(
        residuals,
        x0=[154.99, 2.0, 50, 2.4, E0],
        bounds=(
        [154.99, 1.99, 1, 0.1, E0-0.01],
        [155, 2.0, 200, 10.0, E0]
    )
    )

    print(
        E0,
        fit.x,
        fit.cost
    )



    print("--------------------")
    print(f"Thickness = {fit.x[0]:.2f} nm")
    print(f"Eg        = {fit.x[1]:.3f} eV")
    print(f"A         = {fit.x[2]:.2f}")
    print(f"Gamma     = {fit.x[3]:.2f}")
    print(f"E0        = {fit.x[4]:.2f}")
    print("--------------------")

    print("Cost:", fit.cost)
    print("Success:", fit.success)
    print("Message:", fit.message)

    # =====================================================
    # 7. Plot one angle
    # =====================================================

    plot_angle = angles[1]

    mask = angle_exp == plot_angle


    psi_fit, delta_fit = model(fit.x)


    # Because model is flattened by angle,
    # get index range for this angle
    start = 0
    for a in angles:
        n = np.sum(angle_exp == a)

        if a == plot_angle:
            break

        start += n


    stop = start + np.sum(mask)


    output_dir = Path(__file__).resolve().parent / "figures"
    output_dir.mkdir(exist_ok=True)

    fig, ax = plt.subplots()
    ax.plot(
        wavelength_exp[mask],
        psi_exp[mask],
        "o",
        label="Experiment"
    )

    ax.plot(
        wavelength_exp[mask],
        psi_initial[start:stop],
        label="Initial Guess"
    )

    ax.plot(
        wavelength_exp[mask],
        psi_fit[start:stop],
        label="Model"
    )

    ax.set_xlabel("Wavelength (nm)")
    ax.set_ylabel("Psi")
    ax.legend()
    ax.grid(True)

    output_file = output_dir / f"psi_fit_angle_{plot_angle}_E0_{E0}.png"
    fig.savefig(output_file, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved figure: {output_file}")