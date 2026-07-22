from pathlib import Path

from refellips.dataSE import DataSE
from refellips.reflect_modelSE import ReflectModelSE
from refellips.objectiveSE import ObjectiveSE
from refellips.dispersion import RI, Cauchy, TaucLorentz

from refnx.analysis import CurveFitter

# --------------------------------------------------
# Paths
# --------------------------------------------------

ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = ROOT / "data"
FIT_DIR = ROOT / "ellipsometry_fits"

FIT_DIR.mkdir(exist_ok=True)

# --------------------------------------------------
# Materials
# --------------------------------------------------

air = RI([1.0, 0.0])

# Simple glass model
glass = Cauchy(1.50, 0.004)

# --------------------------------------------------
# Find data files
# --------------------------------------------------

csv_files = sorted(DATA_DIR.glob("*.csv"))

print(f"Found {len(csv_files)} CSV files")

# --------------------------------------------------
# Process each file
# --------------------------------------------------

for csv_file in csv_files:

    print("\n" + "=" * 60)
    print(csv_file.name)

    try:

        # -------------------------
        # Load data
        # -------------------------

        data = DataSE(
            data=str(csv_file),
            delimiter=","
        )

        # -------------------------
        # Tauc-Lorentz film
        # -------------------------

        film_ri = TaucLorentz(
            Am=50,
            C=2,
            En=3,
            Eg=1.8,
            Einf=1
        )

        film = film_ri(150)
        film.name = "Film"


        # -------------------------
        # Fit ONLY thickness
        # -------------------------

        film.thick.setp(
            vary=True,
            bounds=(50, 300)
        )

        # hold optical constants fixed
        # film_ri.Am.setp(vary=False)
        # film_ri.C.setp(vary=False)
        # film_ri.En.setp(vary=False)
        # film_ri.Eg.setp(vary=False)
        # film_ri.Einf.setp(vary=False)

                
        film_ri.Eg.setp(
            vary=True,
            bounds=(1.0, 3.0)
        )


        # -------------------------
        # Structure
        # -------------------------

        structure = (
            air()
            | film
            | glass()
        )

        # -------------------------
        # Model
        # -------------------------

        model = ReflectModelSE(structure)

        objective = ObjectiveSE(
            model,
            data
        )

        fitter = CurveFitter(objective)

        fitter.fit(
            method="least_squares"
        )

        print(objective)

        # -------------------------
        # Results
        # -------------------------

        print("\nFit Results")
        print("-" * 30)

        print(
            f"Thickness = {film.thick.value:.2f} nm"
        )

        
        print(
            f"Am   = {film_ri.Am[0].value:.4f}"
        )

        print(
            f"C    = {film_ri.C[0].value:.4f}"
        )

        print(
            f"En   = {film_ri.En[0].value:.4f}"
        )

        print(
            f"Eg   = {film_ri.Eg.value:.4f}"
        )

        print(
            f"Einf = {film_ri.Einf.value:.4f}"
        )

        # -------------------------
        # Save fit plot
        # -------------------------

        try:

            fig, ax = objective.plot()

            fig.savefig(
                FIT_DIR /
                f"{csv_file.stem}_fit.png",
                dpi=300,
                bbox_inches="tight"
            )

            print(
                f"Saved fit plot: "
                f"{csv_file.stem}_fit.png"
            )

        except Exception as plot_error:

            print(
                f"Plot error: "
                f"{plot_error}"
            )

    except Exception as e:

        print(f"FAILED: {e}")

print("\nDone.")