# src/ellipsometry/fit_SeTe.py
import sys

from load_data import load_elippso_data
from cody_ellipsometry_fitting import (
    CodyEllipsometryFitter,
    CodyLorentzRoughFilmModel,
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
    model = CodyLorentzRoughFilmModelTwoCody() if use_two_cody else CodyLorentzRoughFilmModel()
    fitter = CodyEllipsometryFitter(dataset, model)

    if use_two_cody:
        fitter.run(
            initial_params=[
                100.0,  # film_1_thickness_nm
                2.0,    # roughness_thickness_nm

                1.5,    # Eg_1_eV
                50.0,   # A_1_eV
                0.2,    # Et_1_eV
                1.2,    # gamma_1
                0.8,    # Ep_1_eV
                2.6,    # E0_1_eV
                0.07,   # Eu_1_eV

                20.0,   # A_2_eV
                0.3,    # Et_2_eV
                1.2,    # gamma_2
                1.5,    # Ep_2_eV
                3.7,    # E0_2_eV
                0.1,    # Eu_2_eV
                3.4,
                1.9,
                5.5,
            ],
            bounds=(
                                [
                    5.0,     # film1
                    0.5,     # roughness

                    0.7,     # Eg1
                    5.0,     # A1
                    0.0,     # Et1
                    0.5,     # gamma1
                    0.4,     # Ep1
                    1.8,     # E01
                    0.0,     # Eu1

                    5.0,     # A2
                    0.0,     # Et2
                    0.5,     # gamma2
                    0.5,     # Ep2
                    3.0,     # E02
                    0.0,     # Eu2
                    
                    1,
                    1,
                    1,
                ],
                [
                    100.0,
                    10.0,

                    2.2,
                    150.0,
                    0.5,
                    2.0,
                    2.5,
                    3.5,
                    0.6,

                    50.0,
                    2.0,
                    2.0,
                    3.5,
                    4.5,
                    2.0,

                    4,
                    4,
                    7,
                ])
        )
    else:
        pass

if __name__ == "__main__":
    main()