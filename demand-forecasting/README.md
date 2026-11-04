# Demand Forecasting and Auto-Reorder Tool

Forecasts daily demand per SKU with [Prophet](https://facebook.github.io/prophet/) and turns the forecast into a reorder plan (reorder point, safety stock, suggested order quantity).

## Run it

```bash
pip install -r requirements.txt
python src/generate_sample_data.py      # synthetic demo data (skip if you have real data)
python src/forecast.py                  # writes reorder_plan.csv
python -m pytest -q                     # run the tests
```

Options: `--horizon 30`, `--service-level 0.95` (0.90 / 0.95 / 0.99), `--sales`, `--inventory`, `--out`.

### Input formats

`sales.csv`: `date, sku, units` (one row per sale or per day per SKU)
`inventory.csv`: `sku, on_hand, lead_time_days`

## How it works

1. **Validate**: bad dates/units are dropped, negative units clipped to 0, missing columns raise a clear error
2. **Aggregate**: daily series per SKU, missing days filled with 0
3. **Forecast**: Prophet with weekly seasonality; predictions clipped to ≥ 0
4. **Reorder logic**
   - lead-time demand = sum of forecast over the lead time
   - safety stock = `z × daily_std × √lead_time`
   - reorder point = lead-time demand + safety stock
   - order quantity covers lead time + a 7-day review period + safety stock, minus stock on hand

## Exception handling

| Situation | Behaviour |
|---|---|
| Fewer than 30 days of history | SKU skipped with status `skipped_insufficient_history` |
| Prophet fails for a SKU | Falls back to a 14-day moving average and records `method = moving_average` |
| Malformed rows / missing columns | Rows dropped with a warning, or `ValueError` if the schema is wrong |
| Lead time or service level invalid | `ValueError` |

## Testing

`tests/test_forecast.py` covers the reorder math, validation, gap filling, and the skip/fallback paths. The model call is stubbed in the tests, so they run fast and without Prophet installed.

## Limitations

- Safety stock uses historical demand variability, not forecast error
- No promotions, holidays, or price effects
- Demo data is synthetic; evaluate forecast accuracy (e.g. MAPE on a held-out window) before trusting real output
