import sys
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
import forecast as f  # noqa: E402


def test_reorder_triggers_when_stock_low():
    fc = np.full(30, 10.0)
    d = f.reorder_decision(fc, daily_std=2.0, on_hand=50, lead_time=7)
    assert d["lead_time_demand"] == 70.0
    assert d["reorder_now"] is True
    assert d["order_qty"] > 0


def test_no_reorder_when_stock_high():
    fc = np.full(30, 10.0)
    d = f.reorder_decision(fc, daily_std=2.0, on_hand=1000, lead_time=7)
    assert d["reorder_now"] is False
    assert d["order_qty"] == 0


def test_safety_stock_formula():
    fc = np.full(30, 10.0)
    d = f.reorder_decision(fc, daily_std=4.0, on_hand=0, lead_time=9, service_level=0.95)
    assert d["safety_stock"] == pytest.approx(1.6449 * 4.0 * 3, abs=0.1)


def test_invalid_inputs_raise():
    fc = np.full(30, 10.0)
    with pytest.raises(ValueError):
        f.reorder_decision(fc, 1.0, 10, lead_time=0)
    with pytest.raises(ValueError):
        f.reorder_decision(fc, 1.0, 10, lead_time=29)  # horizon too short
    with pytest.raises(ValueError):
        f.reorder_decision(fc, 1.0, 10, lead_time=7, service_level=0.5)


def test_validate_cleans_bad_rows_and_negatives():
    df = pd.DataFrame({"date": ["2024-01-01", "bad", "2024-01-03"], "sku": "A", "units": [5, 3, -2]})
    out = f.validate_sales(df)
    assert len(out) == 2
    assert out["units"].min() == 0


def test_validate_missing_columns():
    with pytest.raises(ValueError):
        f.validate_sales(pd.DataFrame({"date": [1]}))


def test_daily_series_fills_gaps():
    df = pd.DataFrame({"date": pd.to_datetime(["2024-01-01", "2024-01-04"]), "units": [5, 7]})
    out = f.daily_series(df)
    assert len(out) == 4
    assert out["units"].tolist() == [5, 0, 0, 7]


def test_short_history_is_skipped(monkeypatch):
    monkeypatch.setattr(f, "forecast_sku", lambda h, n: (np.full(n, 5.0), "stub"))
    sales = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=5), "sku": "X", "units": 3})
    inv = pd.DataFrame({"sku": ["X"], "on_hand": [10], "lead_time_days": [7]})
    plan = f.build_plan(sales, inv, horizon=30, service_level=0.95)
    assert plan.loc[0, "status"] == "skipped_insufficient_history"


def test_full_plan_with_stubbed_model(monkeypatch):
    monkeypatch.setattr(f, "forecast_sku", lambda h, n: (np.full(n, 5.0), "stub"))
    rng = np.random.default_rng(0)
    sales = pd.DataFrame({"date": pd.date_range("2024-01-01", periods=60), "sku": "X", "units": rng.poisson(5, 60)})
    inv = pd.DataFrame({"sku": ["X"], "on_hand": [10], "lead_time_days": [7]})
    plan = f.build_plan(sales, inv, horizon=30, service_level=0.95)
    assert plan.loc[0, "status"] == "ok"
    assert plan.loc[0, "reorder_now"]
