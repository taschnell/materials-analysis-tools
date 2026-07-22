from pathlib import Path

from refellips.dataSE import DataSE
from refellips.reflect_modelSE import ReflectModelSE
from refellips.objectiveSE import ObjectiveSE
from refellips.dispersion import RI, Cauchy, load_material

from refnx.analysis import CurveFitter

# --------------------------------------------------
# Data directory
# --------------------------------------------------

DATA_DIR = Path(__file__).resolve().parents[2] / "data"

# --------------------------------------------------
# Materials
# --------------------------------------------------

air = RI([1.0, 0.0])
glass = Cauchy(1.50, 0.004)

# --------------------------------------------------
# Process each csv
# --------------------------------------------------

csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

for csv_file in csv_files:

    print("\n" + "=" * 60)
    print(csv_file.name)

    try:

        # Debug DataSE API
        import inspect

        print("\nDataSE signature:")
        print(inspect.signature(DataSE))

        # Load data
        data = DataSE(data=str(csv_file), delimiter=",")
        
        # Film model
        film_ri = Cauchy(2.5, 0.0)

        # Initial thickness guess
        film = film_ri(100)
        film.name = "Film"

        # Fit thickness
        film.thick.setp(vary=True, bounds=(1, 500))

        # Cauchy parameters belong to the dispersion,
        # not the SlabSE layer
        if hasattr(film_ri, "A"):
            film_ri.A.setp(vary=False)

        if hasattr(film_ri, "B"):
            film_ri.B.setp(vary=False)

        if hasattr(film_ri, "C"):
            film_ri.C.setp(vary=False)

        # Structure
        structure = air() | film | glass()

        # Model
        model = ReflectModelSE(structure)

        # Fit
        objective = ObjectiveSE(model, data)

        fitter = CurveFitter(objective)
        fitter.fit(method="least_squares")

        # Results
        print(f"Thickness: {film.thick.value:.2f} nm")

        if hasattr(film_ri, "A"):
            print(f"A = {film_ri.A.value:.6f}")

        if hasattr(film_ri, "B"):
            print(f"B = {film_ri.B.value:.6f}")

        if hasattr(film_ri, "C"):
            print(f"C = {film_ri.C.value:.6f}")

        fig, ax = objective.plot()
        fig.savefig(csv_file.with_suffix(".png"))
    
    except Exception as e:
        print(f"FAILED: {e}")

print("\nDone.")