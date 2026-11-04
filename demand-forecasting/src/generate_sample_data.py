"""Create SYNTHETIC daily sales + inventory CSVs for demoing the tool."""
from pathlib import Path

import numpy as np
import pandas as pd

DATA = Path(__file__).resolve().parents[1] / "data"


def main(seed: int = 7) -> None:
    rng = np.random.default_rng(seed)
    DATA.mkdir(exist_ok=True)
    dates = pd.date_range("2024-01-01", periods=730, freq="D")
    rows = []
    for sku, base, trend in [("SKU-A", 40, 0.02), ("SKU-B", 15, 0.0), ("SKU-C", 80, -0.01)]:
        weekly = 1 + 0.25 * np.sin(2 * np.pi * dates.dayofweek / 7)
        mean = (base + trend * np.arange(len(dates))) * weekly
        units = rng.poisson(np.clip(mean, 1, None))
        rows += [(d, sku, u) for d, u in zip(dates, units)]
    # A brand-new SKU with almost no history, to demo exception handling.
    for d in dates[-10:]:
        rows.append((d, "SKU-NEW", int(rng.integers(3, 9))))
    pd.DataFrame(rows, columns=["date", "sku", "units"]).to_csv(DATA / "sales.csv", index=False)
    pd.DataFrame(
        {"sku": ["SKU-A", "SKU-B", "SKU-C", "SKU-NEW"],
         "on_hand": [300, 400, 250, 20],
         "lead_time_days": [7, 14, 5, 7]}
    ).to_csv(DATA / "inventory.csv", index=False)
    print(f"Wrote sample data to {DATA}")


if __name__ == "__main__":
    main()
