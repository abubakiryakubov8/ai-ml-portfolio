"""Train and evaluate a linear regression model that predicts Sales from ad spend."""
import argparse
import json
from pathlib import Path

import joblib
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split

ROOT = Path(__file__).resolve().parents[1]
FEATURES = ["TV", "Radio", "Newspaper"]
TARGET = "Sales"


def load_data(path: Path) -> pd.DataFrame:
    if not path.exists():
        raise FileNotFoundError(f"{path} not found. Run `python src/generate_data.py` or add your own CSV.")
    df = pd.read_csv(path)
    missing = [c for c in FEATURES + [TARGET] if c not in df.columns]
    if missing:
        raise ValueError(f"CSV is missing columns: {missing}")
    before = len(df)
    df = df.dropna(subset=FEATURES + [TARGET])
    if len(df) < before:
        print(f"Dropped {before - len(df)} rows with missing values")
    return df


def main() -> None:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--data", type=Path, default=ROOT / "data" / "advertising.csv")
    p.add_argument("--test-size", type=float, default=0.2)
    p.add_argument("--seed", type=int, default=42)
    args = p.parse_args()

    df = load_data(args.data)
    X_train, X_test, y_train, y_test = train_test_split(
        df[FEATURES], df[TARGET], test_size=args.test_size, random_state=args.seed
    )

    model = LinearRegression().fit(X_train, y_train)
    pred = model.predict(X_test)

    metrics = {
        "n_train": len(X_train),
        "n_test": len(X_test),
        "r2": round(float(r2_score(y_test, pred)), 4),
        "mse": round(float(mean_squared_error(y_test, pred)), 4),
        "mae": round(float(mean_absolute_error(y_test, pred)), 4),
        "intercept": round(float(model.intercept_), 4),
        "coefficients": {f: round(float(c), 4) for f, c in zip(FEATURES, model.coef_)},
    }

    reports = ROOT / "reports"
    reports.mkdir(exist_ok=True)
    (reports / "metrics.json").write_text(json.dumps(metrics, indent=2))
    joblib.dump(model, ROOT / "model.joblib")

    fig, ax = plt.subplots(figsize=(5, 5))
    ax.scatter(y_test, pred, alpha=0.7)
    lims = [min(y_test.min(), pred.min()), max(y_test.max(), pred.max())]
    ax.plot(lims, lims, "k--", linewidth=1)
    ax.set_xlabel("Actual sales")
    ax.set_ylabel("Predicted sales")
    ax.set_title(f"Actual vs predicted (R² = {metrics['r2']})")
    fig.tight_layout()
    fig.savefig(reports / "actual_vs_predicted.png", dpi=150)

    print(json.dumps(metrics, indent=2))


if __name__ == "__main__":
    main()
