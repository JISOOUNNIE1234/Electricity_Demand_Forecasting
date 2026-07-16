# Forecasting German Electricity Demand

## Project Overview

This project develops and compares multiple time-series forecasting techniques for predicting German electricity demand using publicly available data from the Open Power System Data (OPSD) platform. The objective is to evaluate statistical, machine learning, and deep learning models and identify the most suitable approach for operational electricity demand forecasting.

## Objectives

* Forecast German electricity demand using historical data.
* Compare benchmark, statistical, machine learning, and deep learning forecasting models.
* Evaluate model performance using common forecasting metrics.
* Build a reproducible forecasting pipeline following software engineering best practices.

## Dataset

**Source:** Open Power System Data (OPSD)

* Country: Germany (DE)
* Frequency: Hourly
* Study Period: January 2015 – October 2020

The hourly electricity demand data were cleaned and aggregated into daily and weekly time series. Weekly data were used for statistical and machine learning models, while hourly data were retained for the LSTM model.

Temperature data for Berlin were retrieved from the Open-Meteo Archive API and used as an exogenous variable in the SARIMAX model.

## Forecasting Models

The following forecasting models were implemented:

### Benchmark Models

* Mean Forecast
* Naïve Forecast
* Seasonal Naïve Forecast
* Drift Forecast

### Statistical Models

* SARIMA
* SARIMAX

### Machine Learning Models

* Random Forest Regressor
* Gradient Boosting Regressor

### Deep Learning Model

* Long Short-Term Memory (LSTM)

## Evaluation Metrics

Models were evaluated using:

* Mean Absolute Error (MAE)
* Root Mean Squared Error (RMSE)
* Mean Absolute Scaled Error (MASE)
* Bias

The final two years of the dataset were used as the test period for all models.

## Repository Structure

```text
electricity-demand-forecasting/
│
├── data/
├── outputs/
│   ├── figures/
│   ├── forecasts/
│   ├── metrics/
│   └── models/
│
├── scripts/
│   ├── download_data.py
│   ├── make_features.py
│   ├── run_analysis.py
│   ├── run_benchmarks.py
│   ├── run_sarima.py
│   ├── run_sarimax.py
│   ├── run_feature_models.py
│   ├── run_lstm.py
│   ├── run_pipeline.py
│   └── build_model_comparison.py
│
├── src/
│   └── electricity_demand/
│
├── tests/
│
├── README.md
├── requirements.txt
└── requirements-lstm.txt
```

## Installation

Clone the repository:

```bash
git clone <repository-url>
cd electricity-demand-forecasting
```

Create and activate the virtual environment:

```bash
python -m venv .venv
```

Windows:

```bash
.venv\Scripts\activate
```

Install dependencies:

```bash
pip install -r requirements.txt
```

For the LSTM model, use the separate Python 3.12 environment:

```bash
pip install -r requirements-lstm.txt
```

## Running the Project

Download the data:

```bash
python scripts/download_data.py
```

Create the processed datasets:

```bash
python scripts/make_features.py
```

Run exploratory analysis:

```bash
python scripts/run_analysis.py
```

Run benchmark models:

```bash
python scripts/run_benchmarks.py
```

Run SARIMA:

```bash
python scripts/run_sarima.py
```

Run SARIMAX:

```bash
python scripts/run_sarimax.py
```

Run feature-based models:

```bash
python scripts/run_feature_models.py
```

Run the LSTM model:

```bash
python scripts/run_lstm.py
```

Run the complete forecasting pipeline:

```bash
python scripts/run_pipeline.py
```

Generate the final model comparison:

```bash
python scripts/build_model_comparison.py
```

## Outputs

The project generates:

* Forecast CSV files
* Model evaluation metrics
* Forecast comparison plots
* Residual diagnostics
* LSTM training history
* Final model comparison table

All outputs are saved under the `outputs/` directory.

## Testing

Run all unit tests using:

```bash
pytest
```

## Technologies Used

* Python
* Pandas
* NumPy
* Matplotlib
* Statsmodels
* Scikit-learn
* TensorFlow
* Pytest

## Key Findings

* Benchmark models provided a useful baseline for evaluating forecasting performance.
* SARIMA and SARIMAX effectively captured seasonal patterns in electricity demand.
* Temperature information improved forecasting when used as an exogenous variable.
* Feature-based machine learning models captured non-linear relationships in the data.
* The LSTM model successfully modelled hourly electricity demand but required greater computational resources.
* The recommended operational model was **SARIMAX**, offering a strong balance between forecasting accuracy, interpretability, and computational efficiency.

## Author

**G. Srujan**

MSc Data Science with Advanced Research

University of Hertfordshire
