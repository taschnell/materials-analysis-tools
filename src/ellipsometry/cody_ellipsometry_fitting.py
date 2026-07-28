# src/ellipsometry/cody_ellipsometry_fitting.py
from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime
from pathlib import Path
from typing import Optional, Sequence

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from scipy.optimize import least_squares

import elli
from elli.dispersions import CodyLorentz
from elli.materials import BruggemanEMA, IsotropicMaterial
from elli.structure import Layer, Structure

from scipy.interpolate import BSpline


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


class CodyLorentzRoughFilmModel(BaseEllipsometryModel):
    @property
    def parameter_names(self) -> list[str]:
        return [
            "film_thickness_nm",
            "roughness_thickness_nm",
            "Eg_eV",
            "A_eV",
            "Et_eV",
            "gamma",
            "Ep_eV",
            "E0_eV",
            "Eu_eV",
        ]

    def build_structure(self, params: Sequence[float]):
        film_thickness, roughness_thickness, Eg, A, Et, gamma, Ep, E0, Eu = params

        film_dispersion = CodyLorentz(
            Eg=Eg,
            A=A,
            Et=Et,
            gamma=gamma,
            Ep=Ep,
            E0=E0,
            Eu=Eu,
        )
        film_material = IsotropicMaterial(film_dispersion)
        rough_film_material = BruggemanEMA(film_material, self.ambient_material, self.roughness_fraction)

        film_layer = Layer(film_material, film_thickness)
        roughness_layer = Layer(rough_film_material, roughness_thickness)

        return Structure(self.ambient_material, [roughness_layer, film_layer], self.substrate_material)


class CodyEllipsometryFitter:
    def __init__(self, dataset: MeasurementDataset, model: BaseEllipsometryModel, output_root: Optional[Path] = None):
        self.dataset = dataset
        self.model = model
        self.output_root = output_root or Path(__file__).resolve().parents[2] / "plots" / "ellipsometry"
        self.output_dir = None

    def residuals(self, params: Sequence[float]) -> np.ndarray:
        psi_model, delta_model = self.model.evaluate(params, self.dataset)

        if np.any(~np.isfinite(psi_model)) or np.any(~np.isfinite(delta_model)):
            return np.ones(len(self.dataset.psi_exp) * 2) * 1e6

        self.dataset.delta_residual = (delta_model - self.dataset.delta_exp + 180) % 360 - 180
        return np.concatenate([psi_model - self.dataset.psi_exp, self.dataset.delta_residual])

    def fit(self, initial_params: Sequence[float], bounds: Sequence[Sequence[float]]):
        result = least_squares(self.residuals, x0=np.asarray(initial_params, dtype=float), bounds=bounds)
        self.psi_model, self.delta_model = self.model.evaluate(result.x, self.dataset)
        return result

    def run(self, initial_params: Sequence[float], bounds: Sequence[Sequence[float]]) -> None:
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.output_dir = self.output_root / self.dataset.name / timestamp
        self.output_dir.mkdir(parents=True, exist_ok=True)
        print(f"Saving output to: {self.output_dir}")

        fit = self.fit(initial_params, bounds)
        self._print_summary(fit)
        self._plot_psi_delta(fit)
        self._plot_residuals(fit)
        self._write_results(fit)

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

            ax1.plot(self.dataset.wavelength_exp[mask], self.dataset.psi_exp[mask], "o", alpha=0.6)
            ax1.plot(self.dataset.wavelength_exp[mask], self.psi_model[mask], "-", linewidth=2)

            ax2.plot(self.dataset.wavelength_exp[mask], self.dataset.delta_exp[mask], "o", alpha=0.6)
            ax2.plot(self.dataset.wavelength_exp[mask], delta_model_nearest[mask], "-", linewidth=2)

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Psi (degrees)")
        ax1.set_title("Psi: Experimental vs Model")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Delta (degrees)")
        ax2.set_title("Delta: Experimental vs Model (Residuals adjusted to nearest equivalent)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "experimental_vs_model.png", dpi=150)
        plt.close(fig)

    def _plot_residuals(self, fit) -> None:
        fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))

        for angle in self.dataset.fit_angles:
            mask = self.dataset.angle_mask(angle)

            ax1.plot(self.dataset.wavelength_exp[mask], self.psi_model[mask] - self.dataset.psi_exp[mask], "-", alpha=0.6)
            ax2.plot(self.dataset.wavelength_exp[mask], self.dataset.delta_residual[mask], "-", alpha=0.6)

        ax1.set_xlabel("Wavelength (nm)")
        ax1.set_ylabel("Psi Residuals (degrees)")
        ax1.set_title("Psi Residuals: Model - Experimental")
        ax1.grid(True, alpha=0.3)

        ax2.set_xlabel("Wavelength (nm)")
        ax2.set_ylabel("Delta Residuals (degrees)")
        ax2.set_title("Delta Residuals: Model - Experimental (Adjusted to nearest equivalent)")
        ax2.grid(True, alpha=0.3)

        fig.tight_layout()
        fig.savefig(self.output_dir / "residuals.png", dpi=150)
        plt.close(fig)

    def _write_results(self, fit) -> None:
        with open(self.output_dir / "fit_results.txt", "w", encoding="utf-8") as handle:
            handle.write(f"Success: {fit.success}\n")
            handle.write(f"Message: {fit.message}\n")
            handle.write(f"Number of iterations: {fit.nfev}\n")
            handle.write("Best fit parameters:\n")
            for idx, name in enumerate(self.model.parameter_names):
                handle.write(f"  {name}:\t{fit.x[idx]:.4f}\n")

