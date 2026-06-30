import pandas as pd
import random
from pathlib import Path

def simulation(output_dir: Path):
    """
    Generate a CSV file with random values.

    Args:
        output_dir (Path): The directory where the CSV will be saved.
    """
    output_dir.mkdir(parents=True, exist_ok=True)
    df = pd.DataFrame({
        "id": range(1, 6),
        "value": [random.random() for _ in range(5)],
    })
    out_path = output_dir / "simulation_output.csv"
    df.to_csv(out_path, index=False)
    print(f"🧪 Simulation complete → {out_path}")

