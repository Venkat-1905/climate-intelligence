"""
Climate Intelligence System — Time-Series Forecasting
======================================================
Multi-model forecasting engine:
  - Facebook Prophet: trend-based forecasting
  - PyTorch LSTM: deep learning sequence model
  - Ensemble: weighted average of both models

All models output confidence intervals.

Outputs:
  - outputs/forecasts.csv
  - outputs/temperature_forecast.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from prophet import Prophet
from neuralprophet import NeuralProphet
from statsmodels.tsa.arima.model import ARIMA

warnings.filterwarnings("ignore")
import logging
logging.getLogger("prophet").setLevel(logging.ERROR)
logging.getLogger("cmdstanpy").setLevel(logging.ERROR)

# ─── LSTM imports (PyTorch) ───
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

FORECAST_HORIZON = 10  # years into the future
TOP_N_COUNTRIES = 50   # forecast for top emitters + key countries

# Key countries to always include in forecasts
KEY_COUNTRIES = {
    "USA", "CHN", "IND", "RUS", "JPN", "DEU", "GBR", "BRA", "IDN", "CAN",
    "KOR", "AUS", "SAU", "ZAF", "MEX", "FRA", "ITA", "TUR", "IRN", "POL",
    "THA", "EGY", "PAK", "ARG", "NGA", "VNM", "MYS", "PHL", "COL", "BGD",
}


# ═══════════════════════════════════════════════════
# MODEL 1: PROPHET
# ═══════════════════════════════════════════════════

def forecast_prophet(country_data, horizon=FORECAST_HORIZON):
    """Run Prophet forecast for a single country's CO₂ emissions."""
    df = country_data[["year", "co2"]].dropna().copy()
    if len(df) < 10:
        return None

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(df["year"], format="%Y"),
        "y": df["co2"].values,
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.1,
        seasonality_mode="multiplicative",
        interval_width=0.9,
    )
    model.fit(prophet_df)

    last_year = df["year"].max()
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    future = pd.DataFrame({
        "ds": pd.to_datetime(future_years, format="%Y"),
    })

    forecast = model.predict(future)
    forecast["year"] = future_years
    forecast["prophet_co2"] = forecast["yhat"].clip(lower=0)
    forecast["prophet_lower"] = forecast["yhat_lower"].clip(lower=0)
    forecast["prophet_upper"] = forecast["yhat_upper"].clip(lower=0)

    return forecast[["year", "prophet_co2", "prophet_lower", "prophet_upper"]]


# ═══════════════════════════════════════════════════
# MODEL 2: LSTM (PyTorch)
# ═══════════════════════════════════════════════════

class LSTMModel(nn.Module):
    """Simple LSTM for time-series forecasting."""
    def __init__(self, input_size=1, hidden_size=32, num_layers=2, dropout=0.1):
        super().__init__()
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers,
                            batch_first=True, dropout=dropout)
        self.fc = nn.Linear(hidden_size, 1)

    def forward(self, x):
        out, _ = self.lstm(x)
        out = self.fc(out[:, -1, :])
        return out


def create_sequences(data, seq_length):
    """Create input sequences for LSTM."""
    xs, ys = [], []
    for i in range(len(data) - seq_length):
        xs.append(data[i:i + seq_length])
        ys.append(data[i + seq_length])
    return np.array(xs), np.array(ys)


def forecast_lstm(country_data, horizon=FORECAST_HORIZON, seq_length=5):
    """
    Run LSTM forecast for a single country's CO₂ emissions.
    Returns predictions with bootstrap confidence intervals.
    """
    df = country_data[["year", "co2"]].dropna().copy()
    if len(df) < 15:  # Need enough data for LSTM sequences
        return None

    values = df["co2"].values.astype(np.float32)
    years = df["year"].values

    # Normalize
    mean_val = values.mean()
    std_val = values.std()
    if std_val == 0:
        return None
    normalized = (values - mean_val) / std_val

    # Create sequences
    X, y = create_sequences(normalized, seq_length)
    if len(X) < 5:
        return None

    X_tensor = torch.FloatTensor(X).unsqueeze(-1)
    y_tensor = torch.FloatTensor(y).unsqueeze(-1)

    # Train model
    model = LSTMModel(input_size=1, hidden_size=32, num_layers=2, dropout=0.1)
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)

    model.train()
    for epoch in range(100):
        optimizer.zero_grad()
        output = model(X_tensor)
        loss = criterion(output, y_tensor)
        loss.backward()
        optimizer.step()

    # Forecast
    model.eval()
    last_seq = normalized[-seq_length:]
    predictions = []

    with torch.no_grad():
        current_seq = last_seq.copy()
        for _ in range(horizon):
            x_input = torch.FloatTensor(current_seq).unsqueeze(0).unsqueeze(-1)
            pred = model(x_input).item()
            predictions.append(pred)
            current_seq = np.append(current_seq[1:], pred)

    # Denormalize
    predictions = np.array(predictions) * std_val + mean_val
    predictions = np.clip(predictions, 0, None)

    # Bootstrap confidence intervals (using training residuals)
    with torch.no_grad():
        train_preds = model(X_tensor).numpy().flatten() * std_val + mean_val
    residuals = y * std_val + mean_val - train_preds
    residual_std = np.std(residuals)

    last_year = int(years.max())
    future_years = list(range(last_year + 1, last_year + horizon + 1))

    result = pd.DataFrame({
        "year": future_years,
        "lstm_co2": predictions,
        "lstm_lower": np.clip(predictions - 1.645 * residual_std, 0, None),
        "lstm_upper": predictions + 1.645 * residual_std,
    })

    return result


# ═══════════════════════════════════════════════════
# MODEL 3: ARIMA (Statsmodels)
# ═══════════════════════════════════════════════════

def forecast_arima(country_data, horizon=FORECAST_HORIZON):
    """Run ARIMA (1,1,0) forecast for a single country."""
    df = country_data[["year", "co2"]].dropna().sort_values("year").copy()
    if len(df) < 10:
        return None

    try:
        y = df["co2"].values
        model = ARIMA(y, order=(1, 1, 0))
        model_fit = model.fit()

        forecast_obj = model_fit.get_forecast(steps=horizon)
        predictions = forecast_obj.predicted_mean
        conf_int = forecast_obj.conf_int(alpha=0.10) # 90% confidence

        last_year = df["year"].max()
        future_years = list(range(last_year + 1, last_year + horizon + 1))

        return pd.DataFrame({
            "year": future_years,
            "arima_co2": predictions.clip(min=0),
            "arima_lower": conf_int[:, 0].clip(min=0),
            "arima_upper": conf_int[:, 1].clip(min=0),
        })
    except Exception as e:
        return None

# ═══════════════════════════════════════════════════
# MODEL 4: NEURALPROPHET
# ═══════════════════════════════════════════════════

def forecast_neuralprophet(country_data, horizon=FORECAST_HORIZON):
    """Run NeuralProphet forecast for a single country."""
    import logging
    logging.getLogger("NP.config").setLevel(logging.ERROR)
    
    df = country_data[["year", "co2"]].dropna().copy()
    if len(df) < 15:
        return None
        
    np_df = pd.DataFrame({
        "ds": pd.to_datetime(df["year"], format="%Y"),
        "y": df["co2"].values,
    })

    try:
        model = NeuralProphet(epochs=50, batch_size=16)
        model.fit(np_df, freq="Y")
        
        future = model.make_future_dataframe(np_df, periods=horizon)
        forecast = model.predict(future)
        
        last_year = df["year"].max()
        future_years = list(range(last_year + 1, last_year + horizon + 1))
        
        # NeuralProphet might not output quantiles if not configured, we'll proxy confidence bonds
        # using a standard +/- 5% error bound for this MVP integration
        yhat = forecast['yhat1'].values[-horizon:]
        
        return pd.DataFrame({
            "year": future_years,
            "np_co2": np.clip(yhat, 0, None),
            "np_lower": np.clip(yhat * 0.95, 0, None),
            "np_upper": np.clip(yhat * 1.05, 0, None),
        })
    except Exception as e:
        return None


# ═══════════════════════════════════════════════════
# ENSEMBLE
# ═══════════════════════════════════════════════════

def ensemble_forecasts(prophet_result, lstm_result, arima_result, np_result, weights=None):
    """
    Combine all models into a final ensemble.
    Default weighting: Prophet(40%), LSTM(20%), ARIMA(20%), NeuralProphet(20%)
    """
    if weights is None:
        weights = {"prophet": 0.4, "lstm": 0.2, "arima": 0.2, "np": 0.2}

    results = []
    if prophet_result is not None:
        results.append(prophet_result.set_index("year"))
    if lstm_result is not None:
        results.append(lstm_result.set_index("year"))
    if arima_result is not None:
        results.append(arima_result.set_index("year"))
    if np_result is not None:
        results.append(np_result.set_index("year"))
        
    if not results:
        return None

    # Merge all available
    merged = pd.concat(results, axis=1).reset_index()
    
    # Calculate weighted average for available models
    active_weights = {}
    if "prophet_co2" in merged: active_weights["prophet"] = weights["prophet"]
    if "lstm_co2" in merged: active_weights["lstm"] = weights["lstm"]
    if "arima_co2" in merged: active_weights["arima"] = weights["arima"]
    if "np_co2" in merged: active_weights["np"] = weights["np"]
    
    total_w = sum(active_weights.values())
    if total_w == 0: return None
    
    merged["forecast_co2"] = 0.0
    merged["forecast_co2_lower"] = 0.0
    merged["forecast_co2_upper"] = 0.0
    
    model_flags = []
    
    if "prophet" in active_weights:
        w = active_weights["prophet"] / total_w
        merged["forecast_co2"] += merged["prophet_co2"].fillna(0) * w
        merged["forecast_co2_lower"] += merged["prophet_lower"].fillna(0) * w
        merged["forecast_co2_upper"] += merged["prophet_upper"].fillna(0) * w
        model_flags.append("Prophet")
        
    if "lstm" in active_weights:
        w = active_weights["lstm"] / total_w
        merged["forecast_co2"] += merged["lstm_co2"].fillna(0) * w
        merged["forecast_co2_lower"] += merged["lstm_lower"].fillna(0) * w
        merged["forecast_co2_upper"] += merged["lstm_upper"].fillna(0) * w
        model_flags.append("LSTM")
        
    if "arima" in active_weights:
        w = active_weights["arima"] / total_w
        merged["forecast_co2"] += merged["arima_co2"].fillna(0) * w
        merged["forecast_co2_lower"] += merged["arima_lower"].fillna(0) * w
        merged["forecast_co2_upper"] += merged["arima_upper"].fillna(0) * w
        model_flags.append("ARIMA")
        
    if "np" in active_weights:
        w = active_weights["np"] / total_w
        merged["forecast_co2"] += merged["np_co2"].fillna(0) * w
        merged["forecast_co2_lower"] += merged["np_lower"].fillna(0) * w
        merged["forecast_co2_upper"] += merged["np_upper"].fillna(0) * w
        model_flags.append("NeuralProphet")
    
    merged["forecast_co2"] = merged["forecast_co2"].clip(lower=0)
    merged["forecast_co2_lower"] = merged["forecast_co2_lower"].clip(lower=0)
    merged["model_used"] = "+".join(model_flags)

    return merged


# ═══════════════════════════════════════════════════
# TEMPERATURE FORECAST
# ═══════════════════════════════════════════════════

def forecast_temperature(fact_df, horizon=FORECAST_HORIZON):
    """Forecast global temperature anomaly."""
    print("🌡️  Forecasting global temperature anomaly...")

    temp_data = fact_df[["year", "temperature_anomaly"]].drop_duplicates("year").dropna()
    temp_data = temp_data.sort_values("year")

    if len(temp_data) < 5:
        print("   ⚠️  Not enough temperature data for forecasting")
        return pd.DataFrame()

    prophet_df = pd.DataFrame({
        "ds": pd.to_datetime(temp_data["year"], format="%Y"),
        "y": temp_data["temperature_anomaly"].values,
    })

    model = Prophet(
        yearly_seasonality=False,
        weekly_seasonality=False,
        daily_seasonality=False,
        changepoint_prior_scale=0.05,
        interval_width=0.9,
    )
    model.fit(prophet_df)

    last_year = temp_data["year"].max()
    future_years = list(range(last_year + 1, last_year + horizon + 1))
    future = pd.DataFrame({
        "ds": pd.to_datetime(future_years, format="%Y"),
    })

    forecast = model.predict(future)
    forecast["year"] = future_years
    forecast["forecast_temp_anomaly"] = forecast["yhat"]
    forecast["forecast_temp_lower"] = forecast["yhat_lower"]
    forecast["forecast_temp_upper"] = forecast["yhat_upper"]

    result = forecast[["year", "forecast_temp_anomaly", "forecast_temp_lower", "forecast_temp_upper"]]
    print(f"   ✅ Temperature forecast: {len(result)} years")
    return result


# ═══════════════════════════════════════════════════
# MAIN PIPELINE
# ═══════════════════════════════════════════════════

def run_forecasting():
    """Execute multi-model forecasting pipeline."""
    print("=" * 60)
    print("🔮 CLIMATE INTELLIGENCE — MULTI-MODEL FORECASTING ENGINE")
    print("   Models: Prophet + LSTM + ARIMA + NeuralProphet")
    print("=" * 60)

    # Load fact table
    fact_path = os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv")
    if not os.path.exists(fact_path):
        print("❌ fact_climate_metrics.csv not found. Run data_pipeline.py first.")
        sys.exit(1)

    fact = pd.read_csv(fact_path)

    # Select countries to forecast
    # Run forecasts for all countries in the fact table
    forecast_isos = set(fact["iso_code"].dropna().unique())

    print(f"\n📊 Forecasting CO₂ for {len(forecast_isos)} countries...")
    print(f"   Running Prophet + LSTM + ARIMA + NP ensemble for each country\n")

    all_forecasts = []
    
    # We will compute the model_stats from model_used afterwards
    done = 0

    for iso in sorted(forecast_isos):
        country_data = fact[fact["iso_code"] == iso].sort_values("year")
        country_name = country_data["country"].iloc[0] if len(country_data) > 0 else iso

        # Run all 4 models
        prophet_result = forecast_prophet(country_data)
        lstm_result = forecast_lstm(country_data)
        arima_result = forecast_arima(country_data)
        np_result = forecast_neuralprophet(country_data)

        # Ensemble
        result = ensemble_forecasts(prophet_result, lstm_result, arima_result, np_result)

        if result is not None:
            result["iso_code"] = iso
            result["country"] = country_name
            all_forecasts.append(result)

        done += 1
        if done % 10 == 0:
            print(f"   ⏳ {done}/{len(forecast_isos)} countries done...")

    # Combine CO₂ forecasts
    if all_forecasts:
        co2_forecasts = pd.concat(all_forecasts, ignore_index=True)
    else:
        co2_forecasts = pd.DataFrame()
        
    print(f"\n   ✅ Models completed successfully.")

    # Temperature forecast
    temp_forecast = forecast_temperature(fact)

    # Merge temperature into CO₂ forecasts
    if not temp_forecast.empty and not co2_forecasts.empty:
        co2_forecasts = co2_forecasts.merge(temp_forecast, on="year", how="left")

    # Historical latest for context
    historical_latest = fact.groupby("iso_code").agg(
        last_historical_year=("year", "max"),
        last_co2=("co2", "last"),
        last_co2_per_capita=("co2_per_capita", "last"),
    ).reset_index()

    if not co2_forecasts.empty:
        co2_forecasts = co2_forecasts.merge(historical_latest, on="iso_code", how="left")

    # Save
    output_path = os.path.join(OUTPUT_DIR, "forecasts.csv")
    if not co2_forecasts.empty:
        co2_forecasts.to_csv(output_path, index=False)
        print(f"\n💾 Saved forecasts: {output_path}")
        print(f"   {len(co2_forecasts)} forecast rows for {co2_forecasts['iso_code'].nunique()} countries")
    else:
        print("⚠️  No forecasts generated")

    # Save temperature forecast separately
    if not temp_forecast.empty:
        temp_path = os.path.join(OUTPUT_DIR, "temperature_forecast.csv")
        temp_forecast.to_csv(temp_path, index=False)

    print("=" * 60)
    return co2_forecasts


if __name__ == "__main__":
    run_forecasting()
