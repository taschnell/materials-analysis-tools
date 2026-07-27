from pathlib import Path
import pandas as pd

PROJECT_ROOT = Path(__file__).resolve().parents[2]

DATA_DIR = PROJECT_ROOT / "data" / "ellipsometry" 

ANGLES = [65.02, 70.02, 75.02]


def convert_file(txt_path):
    print(f"Processing {txt_path.name}")

    rows = []

    with open(txt_path, "r", encoding="utf-8", errors="ignore") as f:
        for line in f:

            line = line.strip()

            # skip blank lines
            if not line:
                continue

            # skip html junk
            if "<br>" in line:
                continue

            # skip header lines
            if not line[0].isdigit():
                continue

            try:
                values = [float(x) for x in line.split()]
            except ValueError:
                continue

            # expected format:
            # wavelength psi65 delta65 psi70 delta70 psi75 delta75
            if len(values) < 7:
                continue

            wavelength = values[0]

            rows.append(
                [wavelength, ANGLES[0], values[1], values[2]]
            )
            rows.append(
                [wavelength, ANGLES[1], values[3], values[4]]
            )
            rows.append(
                [wavelength, ANGLES[2], values[5], values[6]]
            )

    if not rows:
        print(f"  No data found in {txt_path.name}")
        return

    out = pd.DataFrame(
        rows,
        columns=["wavelength", "angle", "psi", "delta"]
    )

    csv_path = txt_path.with_suffix(".csv")

    out.to_csv(csv_path, index=False)

    print(f"  Saved {csv_path.name}")


def main():

    txt_files = sorted(DATA_DIR.glob("*.txt"))

    print(f"Found {len(txt_files)} txt files")

    for txt_file in txt_files:
        convert_file(txt_file)

    print("Done")


if __name__ == "__main__":
    main()