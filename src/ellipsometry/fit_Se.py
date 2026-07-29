# src/ellipsometry/fit_Se.py
import sys

from load_data import load_elippso_data
from cody_ellipsometry_fitting import CodyLorentzRoughFilmModel, CodyEllipsometryFitter, MeasurementDataset


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

    model = CodyLorentzRoughFilmModel()
    fitter = CodyEllipsometryFitter(dataset, model)

    fitter.run(
        initial_params=[106.0, 3.6, 1.8, 50.0, 0.2, 2.4, 0.35, 5.9, 0.07,0.5],
        bounds=(
            [50.0, 0.0, 1.8, 1.0, 0.01, 0.1, 0.1, 1.0, 0.01,0.2],
            [170.0, 10.0, 2.5, 200.0, 5.0, 10.0, 1.0, 20.0, 0.15,0.8],
        ),
    )


if __name__ == "__main__":
    main()