# src/ellipsometry/b_spline.py
import sys

from load_data import load_elippso_data
from cody_ellipsometry_fitting import B_SplineEllipsometryModel, MeasurementDataset


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

    model = B_SplineEllipsometryModel(dataset)


    model.run()
    

if __name__ == "__main__":
    main()