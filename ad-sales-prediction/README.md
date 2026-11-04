# Advertising Sales Prediction

An end-to-end machine learning pipeline: load data, validate, split, train a linear regression model, evaluate, and save a report and plot.

## Run it

```bash
pip install -r requirements.txt
python src/generate_data.py   # creates a synthetic demo dataset (skip if you use your own)
python src/train.py
```

Outputs go to `reports/` (`metrics.json`, `actual_vs_predicted.png`) and `model.joblib`.

## Using your own data

Put a CSV at `data/advertising.csv` with columns `TV, Radio, Newspaper, Sales`, or pass `--data path/to/file.csv`.

## Results

| Dataset | R² | MSE |
|---|---|---|
| Original coursework dataset | **FILL IN from your own run** | **FILL IN** |
| Synthetic demo data (this repo) | see `reports/metrics.json` | see `reports/metrics.json` |

> The synthetic data exists only so the code runs anywhere. Its numbers say nothing about real-world performance.

## Method

1. Validate columns and drop rows with missing values
2. 80/20 train/test split (fixed seed for reproducibility)
3. Fit `LinearRegression`
4. Evaluate on the held-out set: R², MSE, MAE, plus coefficients for interpretation
5. Save metrics, model, and an actual-vs-predicted plot

## Limitations

- Linear model: assumes additive effects and no interaction between channels
- Single random split; cross-validation would give a more stable estimate
