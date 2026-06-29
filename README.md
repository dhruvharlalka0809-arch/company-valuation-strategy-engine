# Company Valuation & Strategic Recommendation Engine

A Streamlit dashboard that values a company using DCF, size-adjusted comparable companies, scenarios, sensitivity analysis, and a strategic recommendation memo.

## What It Does

- Builds a five-year revenue, EBITDA, EBIT, and free cash flow forecast
- Calculates WACC from risk-free rate, beta, equity risk premium, cost of debt, tax rate, and capital structure
- Values the company through DCF and size-adjusted public comparables
- Blends DCF and comps into enterprise value, equity value, and value per share with a configurable weighting
- Compares implied value per share to current share price
- Produces Downside, Base, and Upside scenarios
- Shows terminal value as a percentage of DCF enterprise value
- Calculates target implied EV/Revenue, EV/EBITDA, valuation-implied ROIC, and FCF yield
- Applies downside-case context to the recommendation logic
- Generates a strategic memo with recommendation, valuation readout, risks, and management priorities
- Supports CSV upload for custom company financials

## Why This Project Matters

This project is designed as a bridge between finance, consulting, business analysis, and strategy. It proves valuation mechanics, scenario thinking, business judgment, and executive communication.

## Tech Stack

- Python
- Streamlit
- Pandas
- Standard-library tests with `unittest`

## Run Locally

```bash
python -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
streamlit run app.py
```

## Input Data Format

```csv
Year,Revenue,EBITDA,EBIT,Net_Income,Free_Cash_Flow,Net_Debt,Shares_Outstanding
2020,335.0,50.3,31.8,18.4,20.1,142.0,40.5
2021,374.0,60.2,39.6,23.3,25.8,128.0,41.2
2022,420.0,71.4,48.3,29.1,31.8,115.0,42.0
2023,462.0,81.3,56.8,35.2,38.6,108.0,42.0
2024,515.0,95.3,68.0,43.7,48.9,96.0,42.0
```

## Validate

```bash
python scripts/validate.py
```

## Portfolio Talking Points

- Built a valuation dashboard combining DCF, WACC, public comps, sensitivity, and scenarios
- Added size-adjusted comparable company valuation to avoid over-crediting large public-market multiples
- Added terminal value disclosure, target implied multiples, valuation-implied ROIC, FCF yield, and downside-aware recommendation logic
- Converted valuation output into a strategic recommendation memo for business and consulting audiences
- Designed the project to support PE, corporate finance, business analyst, and consulting applications

## Author

Dhruv Harlalka

MBA Finance, Middlesex University Dubai
