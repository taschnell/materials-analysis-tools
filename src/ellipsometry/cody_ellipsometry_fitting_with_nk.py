# src/ellipsometry/cody_ellipsometry_fitting.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib

from harmonic_oscillator import HarmonicOscillator
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import elli
from elli.dispersions import CodyLorentz, LorentzEnergy
from elli.materials import BruggemanEMA, IsotropicMaterial
from elli.structure import Layer, Structure

from high_res_integration import calculate_optical_constants_high_res


class MeasurementDataset:
    def __init__(self, name: str, raw_data: dict):
        self.name = name
        self.raw_data = raw_data
        self._prepare_arrays()

    @classmethod
    def from_loaded_data(cls, datasets: dict, requested_name: Optional[str] = None):
        if not datasets:
            raise RuntimeError("No datasets were loaded from the data directory.")

        if requested_name:
            if requested_name in datasets:
                name = requested_name
                data = datasets[requested_name]
            else:
                raise KeyError(f"{requested_name} dataset not found in the loaded data.")
        else:
            preferred_names = ["KH522_Ellipso"]
            name = next((n for n in preferred_names if n in datasets), None) or next(iter(datasets))
            data = datasets[name]

        if not isinstance(data, dict):
            raise RuntimeError("Expected dataset rows as a dict of arrays")

        return cls(name, data)

    def _prepare_arrays(self) -> None:
        angles_arr = np.asarray(self.raw_data["angle"], dtype=float)
        self.fit_angles = np.sort(np.unique(angles_arr))

        wavelengths_all, angles_all, psi_all, delta_all = [], [], [], []

        for angle in self.fit_angles:
            angle_mask = np.isclose(angles_arr, angle)
            subset = {key: np.asarray(value)[angle_mask] for key, value in self.raw_data.items()}

            wavelengths_all.extend(subset["wavelength"])
            angles_all.extend(np.full(len(subset["wavelength"]), angle, dtype=float))
            psi_all.extend(subset["psi"])
            delta_all.extend(subset["delta"])

        self.wavelength_exp = np.asarray(wavelengths_all, dtype=float)
        self.angle_exp = np.asarray(angles_all, dtype=float)
        self.psi_exp = np.asarray(psi_all, dtype=float)
        self.delta_exp = np.asarray(delta_all, dtype=float)

    def angle_mask(self, angle: float) -> np.ndarray:
        return np.isclose(self.angle_exp, angle)


class BaseEllipsometryModel(ABC):
    def __init__(self, ambient_material=None, substrate_material=None, roughness_fraction: float = 0.5):
        self.ambient_material = ambient_material or IsotropicMaterial(elli.ConstantRefractiveIndex(1.0))
        self.substrate_material = substrate_material or IsotropicMaterial(elli.ConstantRefractiveIndex(1.515))
        self.roughness_fraction = roughness_fraction

    @property
    @abstractmethod
    def parameter_names(self) -> list[str]:
        ...

    @abstractmethod
    def build_structure(self, params: Sequence[float]):
        ...

    def evaluate(self, params: Sequence[float], dataset: MeasurementDataset):
        sample = self.build_structure(params)

        psi_model, delta_model = [], []
        for angle in dataset.fit_angles:
            wavelength_subset = dataset.wavelength_exp[dataset.angle_mask(angle)]
            result = sample.evaluate(wavelength_subset, float(angle))
            psi_model.extend(result.psi)
            delta_model.extend(result.delta)

        return np.asarray(psi_model), np.asarray(delta_model)


# class CodyLorentzRoughFilmModel(BaseEllipsometryModel):
#     @property
#     def parameter_names(self) -> list[str]:
#         return [
#             "film_thickness_nm",
#             "roughness_thickness_nm",
#             "Eg_eV",
#             "A_eV",
#             "Et_eV",
#             "gamma",
#             "Ep_eV",
#             "E0_eV",
#             "Eu_eV",
#             "void_fraction",
#         ]

#     def build_structure(self, params: Sequence[float]):
#         film_thickness, roughness_thickness, Eg, A, Et, gamma, Ep, E0, Eu, void_fraction = params

#         film_dispersion = CodyLorentz(
#             Eg=Eg,
#             A=A,
#             Et=Et,
#             gamma=gamma,
#             Ep=Ep,
#             E0=E0,
#             Eu=Eu,
#         )

#         self.roughness_fraction = void_fraction
#         film_material = IsotropicMaterial(film_dispersion)
#         rough_film_material = BruggemanEMA(film_material, self.ambient_material, self.roughness_fraction)

#         film_layer = Layer(film_material, film_thickness)
#         roughness_layer = Layer(rough_film_material, roughness_thickness)

#         return Structure(self.ambient_material, [roughness_layer, film_layer], self.substrate_material)


class CodyLorentzRoughFilmModel(BaseEllipsometryModel):
    def __init__(self, ambient_material=None, substrate_material=None):
        super().__init__(
            ambient_material=ambient_material,
            substrate_material=substrate_material,
            roughness_fraction=0.5,
        )

    @property
    def parameter_names(self) -> list[str]:
        return [
            "film_thickness_nm",
            "roughness_thickness_nm",
            "Eg_eV",
            "A_eV",
            "Et_offset_eV",
            "gamma",
            "Ep_eV",
            "E0_eV",
            "Eu_eV",
        ]

    def build_film_dispersion(self, params: Sequence[float]):
        _, _, Eg, A, Et_offset, gamma, Ep, E0, Eu = params

        return CodyLorentz(
            Eg=Eg,
            A=A,
            Et=Eg + Et_offset,
            gamma=gamma,
            Ep=Ep,
            E0=E0,
            Eu=Eu,
        )

    def build_structure(self, params: Sequence[float]):
        film_thickness, roughness_thickness = params[:2]

        film_material = IsotropicMaterial(
            self.build_film_dispersion(params)
        )

        rough_film_material = BruggemanEMA(
            film_material,
            self.ambient_material,
            0.5,
        )

        film_layer = Layer(film_material, film_thickness)
        roughness_layer = Layer(
            rough_film_material,
            roughness_thickness,
        )

        return Structure(
            self.ambient_material,
            [roughness_layer, film_layer],
            self.substrate_material,
        )

# class CodyLorentzRoughFilmModelOrdered(CodyLorentzRoughFilmModel):
#     @property
#     def parameter_names(self) -> list[str]:
#         names = super().parameter_names.copy()
#         names[4] = "Et_minus_Eg_eV"
#         return names

#     def build_structure(self, params: Sequence[float]):
#         ordered_params = list(params)
#         ordered_params[4] = ordered_params[2] + ordered_params[4]
#         return super().build_structure(ordered_params)


class CodyLorentzRoughFilmModelTwoCody(CodyLorentzRoughFilmModel):
    @property
    def parameter_names(self) -> list[str]:
        return [
            "film_thickness_nm",
            "roughness_thickness_nm",
            "Eg_eV",
            "A_1_eV",
            "Et_offset_1_eV",
            "gamma_1",
            "Ep_1_eV",
            "E0_1_eV",
            "Eu_1_eV",
            "A_2_eV",
            "Et_offset_2_eV",
            "gamma_2",
            "Ep_2_eV",
            "E0_2_eV",
            "Eu_2_eV",
            "A_ho (unitless?)",
            "E0_ho_eV",
            "gamma_ho",
        ]

    def build_film_dispersion(self, params: Sequence[float]):
        (
            _,
            _,
            Eg,
            A_1,
            Et_offset_1,
            gamma_1,
            Ep_1,
            E0_1,
            Eu_1,
            A_2,
            Et_offset_2,
            gamma_2,
            Ep_2,
            E0_2,
            Eu_2,
            A_ho,
            E0_ho,
            gamma_ho,
        ) = params

        film_dispersion_1 = CodyLorentz(
            Eg=Eg,
            A=A_1,
            Et=Eg + Et_offset_1,
            gamma=gamma_1,
            Ep=Ep_1,
            E0=E0_1,
            Eu=Eu_1,
        )
        film_dispersion_2 = CodyLorentz(
            Eg=Eg,
            A=A_2,
            Et=Eg + Et_offset_2,
            gamma=gamma_2,
            Ep=Ep_2,
            E0=E0_2,
            Eu=Eu_2,
        )
        harmonic_oscillator = HarmonicOscillator().add(
            A=A_ho,
            E0=E0_ho,
            gamma=gamma_ho,
        )
        return film_dispersion_1 + film_dispersion_2 + harmonic_oscillator

    def build_structure(self, params: Sequence[float]):
        film_thickness, roughness_thickness = params[:2]
        film_material = IsotropicMaterial(self.build_film_dispersion(params))

        rough_film_material = BruggemanEMA(film_material, self.ambient_material, self.roughness_fraction)

        film_layer = Layer(film_material, film_thickness)
        roughness_layer = Layer(rough_film_material, roughness_thickness)

        return Structure(self.ambient_material, [roughness_layer, film_layer], self.substrate_material)


def calculate_optical_constants(
    params: Sequence[float],
    wavelengths_nm: Sequence[float],
    model: Optional[CodyLorentzRoughFilmModel] = None,
) -> tuple[np.ndarray, np.ndarray]:
    """Calculate bulk-film n and k for either supported Cody-Lorentz fit.

    When no model is supplied, the parameter-vector length selects the
    single-Cody (10 parameters) or two-Cody (18 parameters) model.
    """
    params = np.asarray(params, dtype=float)
    wavelengths_nm = np.asarray(wavelengths_nm, dtype=float)

    if wavelengths_nm.ndim != 1 or not np.all(np.isfinite(wavelengths_nm)) or np.any(wavelengths_nm <= 0):
        raise ValueError("wavelengths_nm must be a 1-D array of positive finite values.")

    if model is None:
        if params.shape == (9,):
            model = CodyLorentzRoughFilmModel()
        elif params.shape == (18,):
            model = CodyLorentzRoughFilmModelTwoCody()
        else:
            raise ValueError("Expected 9 single-Cody or 18 two-Cody fit parameters.")
    elif params.shape != (len(model.parameter_names),):
        raise ValueError(f"Expected {len(model.parameter_names)} parameters for the supplied model.")

    dispersion = model.build_film_dispersion(params)
    complex_index = np.asarray(dispersion.get_refractive_index(wavelengths_nm))
    return complex_index.real, complex_index.imag


class CodyEllipsometryFitter:
    def __init__(self, dataset: MeasurementDataset, model: BaseEllipsometryModel, output_root: Optional[Path] = None):
        self.dataset = dataset
        self.model = model
        self.output_root = output_root or Path(__file__).resolve().parents[2] / "plots" / "ellipsometry"
        self.output_dir = None

    def residuals(self, params: Sequence[float]) -> np.ndarray:
        psi_model, delta_model = self.model.evaluate(params, self.dataset)

        if np.any(~np.isfinite(psi_model)) or np.any(~np.isfinite(delta_model)):
            self.dataset.delta_residual = np.zeros_like(self.dataset.delta_exp)
            return np.ones(len(self.dataset.psi_exp) * 2) * 1e6

        self.dataset.delta_residual = (delta_model - self.dataset.delta_exp + 180) % 360 - 180
        return np.concatenate([psi_model - self.dataset.psi_exp, self.dataset.delta_residual])

    def fit(self, initial_params, bounds):
        result = least_squares(
            self.residuals,
            x0=np.asarray(initial_params, dtype=float),
            bounds=bounds,
            method="trf",
            jac="2-point",
            x_scale="jac",
            max_nfev=200,
            verbose=2,
            ftol=1e-6,
            xtol=1e-6,
            gtol=1e-6,
        )

        self.psi_model, self.delta_model = self.model.evaluate(
            result.x, self.dataset
        )

        return result

    def run(self, initial_params: Sequence[float], bounds: Sequence[Sequence[float]]) -> None:
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_root / self.dataset.name / self.timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving output to: {self.output_dir}")

        fit = self.fit(initial_params, bounds)
        self._print_summary(fit)
        self._plot_psi_delta(fit)
        self._plot_residuals(fit)
        self._write_results(fit)
        if isinstance(self.model, CodyLorentzRoughFilmModel):
            self._save_optical_constants(fit)


    def _save_optical_constants(self, fit) -> None:
        """Export fitted n, k, and absorption coefficient vs photon energy."""

        # Calculate optical constants using wavelengths (required by pyElli)
        wavelengths_nm = np.arange(350, 1001, 1)
        n, k = calculate_optical_constants_high_res(
            fit.x, wavelengths_nm, self.model
        )

        # Absorption coefficient in cm^-1
        alpha = (4 * np.pi * k * 1e7) / wavelengths_nm

        # Convert wavelength to photon energy (eV)
        energy_eV = 1239.841984 / wavelengths_nm

        # Sort by increasing energy
        sort_idx = np.argsort(energy_eV)
        energy_eV = energy_eV[sort_idx]
        n = n[sort_idx]
        k = k[sort_idx]
        alpha = alpha[sort_idx]

        sample_name = self.dataset.name

        # Save optical constants CSV
        data = np.column_stack((energy_eV, n, k, alpha))
        np.savetxt(
            self.output_dir / f"optical_constants_{self.timestamp}_{sample_name}.csv",
            data,
            delimiter=",",
            header="energy_eV,n,k,alpha_cm^-1",
            comments="",
        )

        # --------------------------------------------------
        # Refractive index and extinction coefficient plot
        # --------------------------------------------------
        fig, ax1 = plt.subplots(figsize=(10, 6))
        ax2 = ax1.twinx()

        # Left Y-axis: Refractive index (n)
        line1, = ax1.plot(
            energy_eV, n,
            color="tab:blue",
            linewidth=2,
            label="Refractive index, n"
        )
        ax1.set_ylabel("Refractive index, n", color="tab:red")
        ax1.tick_params(axis="y", labelcolor="tab:red")

        # Right Y-axis: Extinction coefficient (k)
        line2, = ax2.plot(
            energy_eV, k,
            color="tab:red",
            linewidth=2,
            label="Extinction coefficient, k"
        )
        ax2.set_ylabel("Extinction coefficient, k", color="tab:blue")
        ax2.tick_params(axis="y", labelcolor="tab:blue")

        ax1.set_xlabel("Photon energy (eV)")
        ax1.set_title(f"{sample_name} - Fitted optical constants")
        ax1.set_xlim(energy_eV.min(), energy_eV.max())
        ax1.grid(True, alpha=0.3)
        ax1.legend(handles=[line1, line2], loc="lower right", framealpha=0.9)

        fig.tight_layout()
        fig.savefig(
            self.output_dir / f"optical_constants_{self.timestamp}_{sample_name}.png",
            dpi=150
        )
        plt.close(fig)

        # --------------------------------------------------
        # Absorption coefficient plot
        # --------------------------------------------------
        fig, ax = plt.subplots(figsize=(10, 6))

        ax.plot(
            energy_eV,
            alpha,
            color="tab:purple",
            linewidth=2,
            label="Absorption coefficient"
        )

        ax.set_xlabel("Photon energy (eV)")
        ax.set_ylabel(r"Absorption coefficient $\alpha$ (cm$^{-1}$)")
        ax.set_title(f"{sample_name} - Fitted absorption coefficient")
        ax.set_xlim(energy_eV.min(), energy_eV.max())
        ax.set_ylim(bottom=0)

        ax.grid(True, alpha=0.3)
        ax.legend(loc="best", framealpha=0.9)
        fig.tight_layout()
        fig.savefig(
            self.output_dir / f"absorption_coefficient_{self.timestamp}_{sample_name}.png",
            dpi=150
        )
        plt.close(fig)



    def _print_summary(self, fit) -> None:
        print("=" * 50)
        print("FIT RESULTS")
        print("=" * 50)
        print(f"Success: {fit.success}")
        print(f"Message: {fit.message}")
        print(f"Number of iterations: {fit.nfev}")
        print()
        print("Best fit parameters:")
        for idx, name in enumerate(self.model.parameter_names):
            print(f"  {name}:  {fit.x[idx]:.4f}")
        print()
        print(f"Mean squared error: {np.mean(fit.fun**2):.6f}")
        print("=" * 50)

    def _plot_psi_delta(self, fit) -> None:
        
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        
        delta_model_nearest = self.dataset.delta_exp + self.dataset.delta_residual


        for angle in self.dataset.fit_angles:
            mask = self.dataset.angle_mask(angle)

            ax1.plot(self.dataset.wavelength_exp[mask], self.dataset.psi_exp[mask], "o", alpha=0.6, label=f'Experimental {angle:.2f}°')
            ax1.plot(self.dataset.wavelength_exp[mask], self.psi_model[mask], "-", linewidth=2)

            ax2.plot(self.dataset.wavelength_exp[mask], self.dataset.delta_exp[mask], "o", alpha=0.6, label=f'Experimental {angle:.2f}°')
            ax2.plot(self.dataset.wavelength_exp[mask], delta_model_nearest[mask], "-", linewidth=2)

        param_text = f"Timestamp: {self.timestamp}\n"

        for idx, name in enumerate(self.model.parameter_names):
            param_text += (f"{name}: {fit.x[idx]:.3f}\n")
        
        param_text += f"\nMean squared error: {np.mean(fit.fun**2):.6f}"
        
        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Psi (degrees)")
        ax1.set_title("Psi: Experimental vs Model")
        ax1.grid(True, alpha=0.3)
        ax1.legend(loc="best", framealpha=0.9)

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Delta (degrees)")
        ax2.set_title("Delta: Experimental vs Model (Residuals adjusted to nearest equivalent)")
        ax2.grid(True, alpha=0.3)
        ax2.legend(loc="best", framealpha=0.9)

        # fig.text(
        #     0.765,          # x position
        #     0.09,           # y position
        #     param_text,
        #     fontsize=9,
        #     family="monospace",
        #     bbox=dict(
        #         boxstyle="round",
        #         facecolor="white",
        #         alpha=0.85
        #     )
        # )

        fig.tight_layout()
        fig.savefig(self.output_dir / f"experimental_vs_model_{self.timestamp}_{self.dataset.name}.png", dpi=150)
        plt.close(fig)

    def _plot_residuals(self, fit) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for angle in self.dataset.fit_angles:
            mask = self.dataset.angle_mask(angle)

            ax1.plot(self.dataset.wavelength_exp[mask], self.psi_model[mask] - self.dataset.psi_exp[mask], "-", alpha=0.6, label=f'Model Angle {angle:.2f}°')
            ax2.plot(self.dataset.wavelength_exp[mask], self.dataset.delta_residual[mask], "-", alpha=0.6, label=f'Model Angle {angle:.2f}°')
            ax1.legend(loc="best", framealpha=0.9)
            ax2.legend(loc="best", framealpha=0.9)

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Psi Residuals (degrees)")
        ax1.set_title("Psi Residuals: Model - Experimental")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Delta Residuals (degrees)")
        ax2.set_title("Delta Residuals: Model - Experimental (Adjusted to nearest equivalent)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / f"residuals_{self.timestamp}_{self.dataset.name}.png", dpi=150)
        plt.close(fig)

    def _write_results(self, fit) -> None:
        with open(self.output_dir / f"fit_results_{self.timestamp}_{self.dataset.name}.txt", "w", encoding="utf-8") as handle:
            handle.write(f"Success: {fit.success}\n")
            handle.write(f"Message: {fit.message}\n")
            handle.write(f"Number of iterations: {fit.nfev}\n")
            handle.write("Best fit parameters:\n")
            handle.write(f"Mean squared error: {np.mean(fit.fun**2):.6f}\n")
            handle.write(f"Timestamp: {self.timestamp}\n")
            handle.write(f"Using function: {self.model.__class__.__name__}\n")
            for idx, name in enumerate(self.model.parameter_names):
                handle.write(f"  {name}:\t{fit.x[idx]:.4f}\n")
