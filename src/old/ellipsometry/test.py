import numpy as np
import matplotlib.pyplot as plt

import elli
from elli.structure import Structure, Layer
from elli.dispersions import CodyLorentz
from elli.materials import IsotropicMaterial

help(CodyLorentz)

# =====================================================
# 1. Define wavelength range and measurement angle
# =====================================================

wavelength = np.linspace(400, 1000, 300)   # nm
theta_i = 70.5                             # degrees


# =====================================================
# 2. Define incident medium (air)
# =====================================================

air_dispersion = elli.ConstantRefractiveIndex(1.0)

air = IsotropicMaterial(
    air_dispersion
)


# =====================================================
# 3. Define substrate (glass)
# =====================================================

glass_dispersion = elli.ConstantRefractiveIndex(1.52)

glass = IsotropicMaterial(
    glass_dispersion
)


# =====================================================
# 4. Define Cody–Lorentz thin film
# =====================================================

film_dispersion = CodyLorentz(
    Eg=1.6,       # band gap (eV)
    A=47,        # amplitude
    Et=0.2,     # Urbach onset
    gamma=2.4,    # broadening
    Ep=0.8,       # Cody-Lorentz transition
    E0=5.9,    # Lorentz resonance
    Eu=0.05       # Urbach energy
)


film_material = IsotropicMaterial(
    film_dispersion
)


# =====================================================
# 5. Create thin film layer
# =====================================================

film_thickness = 150   # nm

film_layer = Layer(
    film_material,
    film_thickness
)


# =====================================================
# 6. Build sample structure
# =====================================================

sample = Structure(
    air,
    [film_layer],
    glass
)


# =====================================================
# 7. Calculate ellipsometry response
# =====================================================

result = sample.evaluate(
    wavelength,
    theta_i
)


# =====================================================
# 8. Extract Psi and Delta
# =====================================================

psi = result.psi
delta = result.delta


# =====================================================
# 9. Plot results
# =====================================================

plt.figure(figsize=(7,4))
plt.plot(wavelength, psi)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Psi (degrees)")
plt.title("Calculated Psi")
plt.grid()
plt.show()


plt.figure(figsize=(7,4))
plt.plot(wavelength, delta)
plt.xlabel("Wavelength (nm)")
plt.ylabel("Delta (degrees)")
plt.title("Calculated Delta")
plt.grid()
plt.show()

# =====================================================
# Plot optical constants n and k
# =====================================================

epsilon = film_material.get_tensor(wavelength)
# For isotropic material, take the xx component
epsilon = epsilon[:,0,0]

# Convert dielectric function to complex refractive index
nk = np.sqrt(epsilon)

n = np.real(nk)
k = np.imag(nk)


plt.figure(figsize=(7,4))
plt.plot(wavelength, n)
plt.xlabel("Wavelength (nm)")
plt.ylabel("n")
plt.title("Refractive index")
plt.grid()
plt.show()


plt.figure(figsize=(7,4))
plt.plot(wavelength, k)
plt.xlabel("Wavelength (nm)")
plt.ylabel("k")
plt.title("Extinction coefficient")
plt.grid()
plt.show()