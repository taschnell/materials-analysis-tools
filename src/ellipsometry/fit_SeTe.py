# src/ellipsometry/fit_SeTe.py
import sys

from load_data import load_elippso_data
from cody_ellipsometry_fitting_with_nk import (
    CodyEllipsometryFitter,
    CodyLorentzRoughFilmModelTwoCody,
    MeasurementDataset,
)


def main() -> None:
    requested_name = sys.argv[1] if len(sys.argv) > 1 else None

    datasets = load_elippso_data()
    dataset = MeasurementDataset.from_loaded_data(datasets, requested_name)

    print("Available files:")
    print(list(datasets.keys()))
    print(f"Using dataset: {dataset.name}")
    print("Loaded columns:", list(dataset.raw_data.keys()))
    print(f"Using measurement angles: {dataset.fit_angles}")
    print(f"Using {len(dataset.fit_angles)} measurement angles with {len(dataset.wavelength_exp)} total data points.")

    use_two_cody = True
    model = CodyLorentzRoughFilmModelTwoCody()
    fitter = CodyEllipsometryFitter(dataset, model)

    if use_two_cody:
        fitter.run(
            initial_params=[
                100.0,  # film_thickness_nm
                2.0,    # roughness_thickness_nm

                1.866,  # Eg_eV
                12.46,  # A_1_eV
                0.1,    # Et_1_eV; avoid the singular Et=0 boundary in elli
                1.15,   # gamma_1
                0.787,  # Ep_1_eV
                2.789,  # E0_1_eV
                0.1,    # Eu_1_eV; avoids overflow in elli's inactive tail branch

                13.1,   # A_2_eV
                0.391,  # Et_2_eV
                1.32,   # gamma_2
                2.02,   # Ep_2_eV
                4.08,   # E0_2_eV
                2.0,    # Eu_2_eV

                3.43,   # A_HO
                5.54,   # E0_HO_eV
                1.87,   # gamma_HO
            ],
            bounds=(
                                [
                    10.0,    # film
                    0.0,     # roughness

                    0.5,     # Eg
                    1.0,     # A1
                    0.001,   # Et1
                    0.1,     # gamma1
                    0.4,     # Ep1
                    1.8,     # E01
                    0.05,    # Eu1

                    1.0,     # A2
                    0.0,     # Et2
                    0.1,     # gamma2
                    0.5,     # Ep2
                    3.0,     # E02
                    0.0,     # Eu2
                    
                    0.1,     # A_HO
                    3.0,     # E0_HO
                    0.1,     # gamma_HO
                ],
                [
                    300.0,
                    10.0,

                    2.2,
                    100.0,
                    2.0,
                    3.0,
                    2.5,
                    4.0,
                    1.0,

                    100.0,
                    2.0,
                    5.0,
                    3.5,
                    4.5,
                    2.0,

                    50.0,
                    7.0,
                    3.0,
                ])
        )

if __name__ == "__main__":
    main()