from pathlib import Path
import pandas as pd
import matplotlib.pyplot as plt

DATA_DIR = Path(__file__).resolve().parents[2] / "data" / "ellipsometry"

# New output directory
PLOT_DIR = Path(__file__).resolve().parents[2] / "plots" / "psi_delta_graphs"
PLOT_DIR.mkdir(exist_ok=True)

csv_files = sorted(DATA_DIR.glob("*.csv"))

for csv_file in csv_files:

    print(f"Plotting {csv_file.name}")

    df = pd.read_csv(csv_file)

    # ==========================
    # PSI
    # ==========================

    plt.figure(figsize=(8, 5))

    for angle in sorted(df["angle"].unique()):

        m = df["angle"] == angle

        plt.plot(
            df.loc[m, "wavelength"],
            df.loc[m, "psi"],
            label=f"{angle:.2f}°"
        )

    plt.title(f"{csv_file.stem} - Psi")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Psi (°)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{csv_file.stem}_psi.png",
        dpi=300
    )

    plt.close()

    # ==========================
    # DELTA
    # ==========================

    plt.figure(figsize=(8, 5))

    for angle in sorted(df["angle"].unique()):

        m = df["angle"] == angle

        plt.plot(
            df.loc[m, "wavelength"],
            df.loc[m, "delta"],
            label=f"{angle:.2f}°"
        )

    plt.title(f"{csv_file.stem} - Delta")
    plt.xlabel("Wavelength (nm)")
    plt.ylabel("Delta (°)")
    plt.legend()
    plt.tight_layout()

    plt.savefig(
        PLOT_DIR / f"{csv_file.stem}_delta.png",
        dpi=300
    )

    plt.close()

print(f"Saved plots to: {PLOT_DIR}")