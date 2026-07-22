import numpy as np
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

def make_film():

    dispersion = CodyLorentz(
    Eg=1.68,
    A=50,
    Et=0.2,
    gamma=2.4,
    Ep=0.35,
    E0=5.9,
    Eu=0.07

)

    return IsotropicMaterial(dispersion)



# =====================================================
# 4. Forward model
# =====================================================

def model(params):

    thickness = params[0]

    film = make_film()

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


initial_params = [150]

psi_initial, delta_initial = model(initial_params)


# =====================================================
# 6. Fit thickness
# =====================================================

fit = least_squares(
    residuals,
    x0=[150],
    bounds=(
        [1],
        [2000]
    )
)


print("--------------------")
print("Best thickness:")
print(fit.x[0], "nm")
print("--------------------")



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


plt.figure()
plt.plot(
    wavelength_exp[mask],
    psi_exp[mask],
    "o",
    label="Experiment"
)

plt.plot(
    wavelength_exp[mask],
    psi_initial[start:stop],
    label="Initial Guess"
)


plt.plot(
    wavelength_exp[mask],
    psi_fit[start:stop],
    label="Model"
)

plt.xlabel("Wavelength (nm)")
plt.ylabel("Psi")
plt.legend()
plt.grid()
plt.show()