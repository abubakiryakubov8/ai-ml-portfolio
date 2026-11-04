"""Demand forecasting and automated reorder planning.

Pipeline: validate sales history -> forecast daily demand per SKU (Prophet,
with a moving-average fallback) -> compute reorder point and order quantity.
"""
from __future__ import annotations

import argparse
import logging
import math
from pathlib import Path

import numpy as np
import pandas as pd

log = logging.getLogger("forecast")

MIN_HISTORY_DAYS = 30
FALLBACK_WINDOW = 14


# ---------- validation ----------
def validate_sales(df: pd.DataFrame) -> pd.DataFrame:
    """Return a clean copy of the sales table or raise ValueError."""
    required = {"date", "sku", "units"}
    if not required.issubset(df.columns):
        raise ValueError(f"sales data must have columns {sorted(required)}")
    out = df.copy()
    out["date"] = pd.to_datetime(out["date"], errors="coerce")
    out["units"] = pd.to_numeric(out["units"], errors="coerce")
    bad = out["date"].isna() | out["units"].isna()
    if bad.any():
        log.warning("Dropping %d rows with unparseable date/units", int(bad.sum()))
        out = out[~bad]
    if (out["units"] < 0).any():
        log.warning("Clipping negative units (returns?) to 0")
        out["units"] = out["units"].clip(lower=0)
    if out.empty:
        raise ValueError("no valid sales rows after cleaning")
    return out


def daily_series(sales: pd.DataFrame) -> pd.DataFrame:
    """Aggregate to one row per day and fill missing days with 0 sales."""
    s = sales.groupby("date", as_index=False)["units"].sum()
    full = pd.date_range(s["date"].min(), s["date"].max(), freq="D")
    s = s.set_index("date").reindex(full, fill_value=0).rename_axis("date").reset_index()
    return s


# ---------- forecasting ----------
def forecast_moving_average(history: pd.DataFrame, horizon: int) -> np.ndarray:
    avg = history["units"].tail(FALLBACK_WINDOW).mean()
    return np.full(horizon, avg, dtype=float)


def forecast_prophet(history: pd.DataFrame, horizon: int) -> np.ndarray:
    from prophet import Prophet  # imported lazily so the logic tests don't need it

    logging.getLogger("cmdstanpy").setLevel(logging.WARNING)
    logging.getLogger("prophet").setLevel(logging.WARNING)
    m = Prophet(weekly_seasonality=True, yearly_seasonality="auto", daily_seasonality=False)
    m.fit(history.rename(columns={"date": "ds", "units": "y"}))
    fc = m.predict(m.make_future_dataframe(periods=horizon)).tail(horizon)
    return np.clip(fc["yhat"].to_numpy(), 0, None)  # demand cannot be negative


def forecast_sku(history: pd.DataFrame, horizon: int) -> tuple[np.ndarray, str]:
    """Forecast with Prophet; fall back to a moving average if it fails."""
    try:
        return forecast_prophet(history, horizon), "prophet"
    except Exception as exc:  # noqa: BLE001 - any model failure should degrade gracefully
        log.warning("Prophet failed (%s); using moving average", exc)
        return forecast_moving_average(history, horizon), "moving_average"


# ---------- reorder logic ----------
Z_SCORES = {0.90: 1.2816, 0.95: 1.6449, 0.99: 2.3263}


def reorder_decision(
    forecast: np.ndarray,
    daily_std: float,
    on_hand: float,
    lead_time: int,
    review_days: int = 7,
    service_level: float = 0.95,
) -> dict:
    """Compute reorder point, safety stock, and suggested order quantity.

    lead-time demand = sum of forecast over the lead time
    safety stock     = z * daily_std * sqrt(lead_time)
    reorder point    = lead-time demand + safety stock
    order quantity   = enough to cover lead time + review period + safety stock
    """
    if lead_time < 1:
        raise ValueError("lead_time must be >= 1 day")
    if service_level not in Z_SCORES:
        raise ValueError(f"service_level must be one of {sorted(Z_SCORES)}")
    if len(forecast) < lead_time + review_days:
        raise ValueError("forecast horizon shorter than lead_time + review_days")

    lt_demand = float(np.sum(forecast[:lead_time]))
    safety = Z_SCORES[service_level] * daily_std * math.sqrt(lead_time)
    rop = lt_demand + safety
    target = float(np.sum(forecast[: lead_time + review_days])) + safety
    qty = max(0, math.ceil(target - on_hand))
    return {
        "lead_time_demand": round(lt_demand, 1),
        "safety_stock": round(safety, 1),
        "reorder_point": round(rop, 1),
        "reorder_now": bool(on_hand <= rop),
        "order_qty": qty if on_hand <= rop else 0,
    }


# ---------- orchestration ----------
def build_plan(sales: pd.DataFrame, inventory: pd.DataFrame, horizon: int, service_level: float) -> pd.DataFrame:
    sales = validate_sales(sales)
    rows = []
    for _, inv in inventory.iterrows():
        sku = inv["sku"]
        hist = daily_series(sales[sales["sku"] == sku]) if (sales["sku"] == sku).any() else pd.DataFrame()
        if len(hist) < MIN_HISTORY_DAYS:
            log.warning("%s skipped: only %d days of history (need %d)", sku, len(hist), MIN_HISTORY_DAYS)
            rows.append({"sku": sku, "status": "skipped_insufficient_history"})
            continue
        fc, method = forecast_sku(hist, horizon)
        decision = reorder_decision(
            fc,
            daily_std=float(hist["units"].tail(90).std(ddof=1)),
            on_hand=float(inv["on_hand"]),
            lead_time=int(inv["lead_time_days"]),
            service_level=service_level,
        )
        rows.append({"sku": sku, "status": "ok", "method": method, "on_hand": inv["on_hand"], **decision})
    return pd.DataFrame(rows)


def main() -> None:
    root = Path(__file__).resolve().parents[1]
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--sales", type=Path, default=root / "data" / "sales.csv")
    p.add_argument("--inventory", type=Path, default=root / "data" / "inventory.csv")
    p.add_argument("--horizon", type=int, default=30)
    p.add_argument("--service-level", type=float, default=0.95)
    p.add_argument("--out", type=Path, default=root / "reorder_plan.csv")
    args = p.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")
    plan = build_plan(pd.read_csv(args.sales), pd.read_csv(args.inventory), args.horizon, args.service_level)
    plan.to_csv(args.out, index=False)
    print(plan.to_string(index=False))
    print(f"\nSaved to {args.out}")


if __name__ == "__main__":
    main()
