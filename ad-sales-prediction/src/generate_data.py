"""Generate a SYNTHETIC advertising dataset so the pipeline runs out of the box.

Replace data/advertising.csv with your real data (same columns) to reproduce
your own results.
"""
from pathlib import Path

import numpy as np
import pandas as pd

OUT = Path(__file__).resolve().parents[1] / "data" / "advertising.csv"


def main(n: int = 200, seed: int = 42) -> None:
    rng = np.random.default_rng(seed)
    tv = rng.uniform(1, 300, n)
    radio = rng.uniform(0, 50, n)
    newspaper = rng.uniform(0, 100, n)
    # Newspaper has no true effect; noise level is arbitrary.
    sales = 3.0 + 0.045 * tv + 0.19 * radio + rng.normal(0, 1.6, n)
    df = pd.DataFrame({"TV": tv, "Radio": radio, "Newspaper": newspaper, "Sales": sales}).round(2)
    OUT.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT, index=False)
    print(f"Wrote {len(df)} rows to {OUT}")


if __name__ == "__main__":
    main()
