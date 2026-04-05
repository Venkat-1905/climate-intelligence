"""
Climate Intelligence System — FastAPI Backend
===============================================
REST API for dashboard, policy simulation, and data access.

Endpoints:
  GET  /api/countries           - All countries with latest metrics
  GET  /api/country/{iso}       - Detailed country data
  GET  /api/forecasts/{iso}     - Time series forecast
  GET  /api/risk-scores         - All risk scores with rankings
  GET  /api/shap/{iso}          - SHAP explanations
  GET  /api/clusters            - Country clusters
  POST /api/simulate            - Policy simulation
  GET  /api/recommendations/{iso} - AI recommendations
  GET  /api/global-overview     - Global KPIs and summaries
  GET  /api/historical/{iso}    - Full historical time series
  GET  /api/narratives/{iso}    - AI-generated narrative
  GET  /api/anomalies           - Anomaly detection results
  GET  /api/equity              - Emissions equity metrics
  GET  /api/paris-tracker       - Paris Agreement progress
"""

import os
import sys
import json
import pandas as pd
import numpy as np
from fastapi import FastAPI, Query, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from fastapi.responses import FileResponse
from pydantic import BaseModel
from typing import Optional

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DASHBOARD_DIR = os.path.join(BASE_DIR, "dashboard")

# Add src to path for imports
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from policy_simulator import simulate_policy

app = FastAPI(
    title="Climate Intelligence API",
    description="REST API for the Climate Intelligence System — multi-model forecasting, risk scoring, explainability, and policy simulation",
    version="2.0.0",
)

# CORS
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)


# ──────────────────────────────────────────────
# Data Loading (cached in memory)
# ──────────────────────────────────────────────
_cache = {}


def _load(name, filename):
    if name not in _cache:
        path = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(path):
            _cache[name] = pd.read_csv(path)
        else:
            _cache[name] = pd.DataFrame()
    return _cache[name]


def get_fact():
    return _load("fact", "fact_climate_metrics.csv")

def get_risk():
    return _load("risk", "risk_scores.csv")

def get_forecasts():
    return _load("forecasts", "forecasts.csv")

def get_clusters():
    return _load("clusters", "clusters.csv")

def get_shap():
    return _load("shap", "shap_values.csv")

def get_feature_importance():
    return _load("feat_imp", "feature_importance.csv")

def get_recommendations():
    return _load("recs", "recommendations.csv")

def get_dim_country():
    return _load("dim_country", "dim_country.csv")

def get_narratives():
    return _load("narratives", "narratives.csv")

def get_anomalies():
    return _load("anomalies", "anomalies.csv")


# ──────────────────────────────────────────────
# Pydantic Models
# ──────────────────────────────────────────────
class SimulationRequest(BaseModel):
    iso_code: str
    emission_reduction_pct: float = 0
    renewables_increase_pct: float = 0


# ──────────────────────────────────────────────
# API Routes
# ──────────────────────────────────────────────

@app.get("/api/global-overview")
def global_overview():
    """Global KPIs and summary statistics."""
    fact = get_fact()
    risk = get_risk()

    if fact.empty:
        raise HTTPException(status_code=404, detail="No data available")

    if not risk.empty:
        fact = fact.merge(risk[["iso_code", "year", "risk_score"]], on=["iso_code", "year"], how="left", suffixes=("", "_risk"))

    latest_year = int(fact["year"].max())
    latest = fact[fact["year"] == latest_year]

    total_co2 = float(latest["co2"].sum()) if "co2" in latest.columns else 0
    avg_temp_anomaly = float(latest["temperature_anomaly"].mean()) if "temperature_anomaly" in latest.columns else 0
    avg_renewables = float(latest["renewables_share"].mean()) if "renewables_share" in latest.columns else 0

    risk_col = "risk_score" if "risk_score" in latest.columns else "risk_score_risk"
    avg_risk = float(latest[risk_col].mean()) if risk_col in latest.columns else 0
    total_countries = int(latest["iso_code"].nunique())

    top_emitters = (
        latest.nlargest(10, "co2")[["iso_code", "country", "co2", "co2_per_capita", risk_col]]
        .rename(columns={risk_col: "risk_score"})
        .to_dict(orient="records")
    ) if "co2" in latest.columns else []

    yearly = fact.groupby("year").agg(
        total_co2=("co2", "sum"),
        avg_temp=("temperature_anomaly", "mean"),
    ).reset_index()

    if risk_col in fact.columns:
        yearly_risk = fact.groupby("year")[risk_col].mean().reset_index()
        yearly_risk.columns = ["year", "avg_risk"]
        yearly = yearly.merge(yearly_risk, on="year", how="left")

    trend = yearly.dropna(subset=["total_co2"]).replace({np.nan: None}).to_dict(orient="records")

    return {
        "latest_year": latest_year,
        "total_countries": total_countries,
        "kpis": {
            "total_co2_gt": round(total_co2 / 1000, 2),
            "avg_temperature_anomaly": round(avg_temp_anomaly, 3),
            "avg_renewables_share": round(avg_renewables, 1),
            "avg_risk_score": round(avg_risk, 1),
        },
        "top_emitters": top_emitters,
        "emissions_trend": trend,
    }


@app.get("/api/countries")
def list_countries():
    """List all countries with latest metrics."""
    risk = get_risk()
    dim = get_dim_country()

    if risk.empty:
        raise HTTPException(status_code=404, detail="No data available")

    latest_year = int(risk["year"].max())
    latest = risk[risk["year"] == latest_year].copy()

    if not dim.empty:
        latest = latest.merge(
            dim[["iso_code", "continent", "income_group"]],
            on="iso_code", how="left"
        )

    cols = ["iso_code", "country", "risk_score", "co2_per_capita",
            "coal_share", "renewables_share", "vulnerability_score",
            "continent", "income_group"]
    available = [c for c in cols if c in latest.columns]
    result = latest[available].sort_values("risk_score", ascending=False)

    return {
        "year": latest_year,
        "count": len(result),
        "countries": result.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/country/{iso_code}")
def get_country(iso_code: str):
    """Detailed data for a specific country."""
    fact = get_fact()
    dim = get_dim_country()

    iso_code = iso_code.upper()
    country_data = fact[fact["iso_code"] == iso_code].sort_values("year")

    risk = get_risk()
    if "risk_score" not in country_data.columns:
        if not risk.empty and not country_data.empty:
            country_data = country_data.merge(risk[["iso_code", "year", "risk_score"]], on=["iso_code", "year"], how="left")

    if country_data.empty:
        raise HTTPException(status_code=404, detail=f"Country {iso_code} not found")

    country_info = {}
    if not dim.empty:
        dim_row = dim[dim["iso_code"] == iso_code]
        if not dim_row.empty:
            country_info = dim_row.iloc[0].to_dict()

    latest = country_data.iloc[-1].to_dict()

    def clean(d):
        return {k: (None if isinstance(v, float) and np.isnan(v) else v) for k, v in d.items()}

    return {
        "info": clean(country_info),
        "latest": clean(latest),
        "historical": country_data.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/historical/{iso_code}")
def get_historical(iso_code: str):
    """Full historical time series for a country."""
    fact = get_fact()

    iso_code = iso_code.upper()
    data = fact[fact["iso_code"] == iso_code].sort_values("year")

    risk = get_risk()
    if "risk_score" not in data.columns:
        if not risk.empty and not data.empty:
            data = data.merge(risk[["iso_code", "year", "risk_score"]], on=["iso_code", "year"], how="left")

    if data.empty:
        raise HTTPException(status_code=404, detail=f"Country {iso_code} not found")

    cols = ["year", "co2", "co2_per_capita", "coal_share", "renewables_share",
            "risk_score", "temperature_anomaly", "energy_intensity",
            "sustainability_score", "population", "gdp"]
    available = [c for c in cols if c in data.columns]

    return {
        "iso_code": iso_code,
        "country": data["country"].iloc[0],
        "data": data[available].replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/forecasts/{iso_code}")
def get_forecast(iso_code: str):
    """Forecasted CO₂ emissions for a country."""
    forecasts = get_forecasts()
    fact = get_fact()

    iso_code = iso_code.upper()

    hist = fact[fact["iso_code"] == iso_code][["year", "co2"]].dropna()
    hist = hist.sort_values("year").tail(20)

    fc = forecasts[forecasts["iso_code"] == iso_code] if not forecasts.empty else pd.DataFrame()

    if hist.empty and fc.empty:
        raise HTTPException(status_code=404, detail=f"No data for {iso_code}")

    return {
        "iso_code": iso_code,
        "historical": hist.replace({np.nan: None}).to_dict(orient="records"),
        "forecast": fc.replace({np.nan: None}).to_dict(orient="records") if not fc.empty else [],
    }


@app.get("/api/risk-scores")
def get_risk_scores():
    """All risk scores ranked."""
    risk = get_risk()
    dim = get_dim_country()

    if risk.empty:
        raise HTTPException(status_code=404, detail="No risk scores available")

    latest_year = int(risk["year"].max())
    latest = risk[risk["year"] == latest_year].copy()

    if not dim.empty:
        latest = latest.merge(dim[["iso_code", "continent", "income_group"]], on="iso_code", how="left")

    latest = latest.sort_values("risk_score", ascending=False)

    return {
        "year": latest_year,
        "scores": latest.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/shap/{iso_code}")
def get_shap_values(iso_code: str):
    """SHAP feature importance for a country."""
    shap_df = get_shap()
    feat_imp = get_feature_importance()

    iso_code = iso_code.upper()

    if shap_df.empty:
        return {"iso_code": iso_code, "shap_values": {}, "feature_importance": []}

    country_shap = shap_df[shap_df["iso_code"] == iso_code]

    shap_values = {}
    if not country_shap.empty:
        row = country_shap.iloc[0]
        shap_cols = [c for c in row.index if c.startswith("shap_")]
        for col in shap_cols:
            feat_name = col.replace("shap_", "")
            val = row[col]
            shap_values[feat_name] = round(float(val), 4) if pd.notna(val) else 0

    feat_list = feat_imp.replace({np.nan: None}).to_dict(orient="records") if not feat_imp.empty else []

    return {
        "iso_code": iso_code,
        "shap_values": shap_values,
        "feature_importance": feat_list,
    }


@app.get("/api/clusters")
def get_cluster_data():
    """Country clusters."""
    clusters = get_clusters()

    if clusters.empty:
        raise HTTPException(status_code=404, detail="No cluster data")

    return {
        "clusters": clusters.replace({np.nan: None}).to_dict(orient="records"),
        "summary": clusters.groupby("cluster_name").size().to_dict(),
    }


import math

@app.post("/api/simulate")
def run_simulation(req: SimulationRequest):
    """Policy simulation."""
    risk_df = get_risk()
    forecast_df = get_forecasts()

    result = simulate_policy(
        iso_code=req.iso_code.upper(),
        emission_reduction_pct=req.emission_reduction_pct,
        renewables_increase_pct=req.renewables_increase_pct,
        risk_df=risk_df,
        forecast_df=forecast_df,
    )

    if "error" in result:
        raise HTTPException(status_code=404, detail=result["error"])

    # Handle JSON NaN serialization errors
    def clean_nans(obj):
        if isinstance(obj, dict):
            return {k: clean_nans(v) for k, v in obj.items()}
        elif isinstance(obj, list):
            return [clean_nans(v) for v in obj]
        elif isinstance(obj, float) and math.isnan(obj):
            return None
        return obj

    return clean_nans(result)


@app.get("/api/recommendations/{iso_code}")
def get_country_recommendations(iso_code: str):
    """Recommendations for a country."""
    recs = get_recommendations()
    iso_code = iso_code.upper()

    if recs.empty:
        raise HTTPException(status_code=404, detail="No recommendations available")

    country_rec = recs[recs["iso_code"] == iso_code]
    if country_rec.empty:
        raise HTTPException(status_code=404, detail=f"No recommendations for {iso_code}")

    row = country_rec.iloc[0]
    rec_list = str(row.get("recommendations", "")).split(" | ")

    return {
        "iso_code": iso_code,
        "country": row.get("country", iso_code),
        "risk_score": float(row.get("risk_score", 0)),
        "recommendations": rec_list,
    }


# ──────────────────────────────────────────────
# NEW ENDPOINTS (Phase 6+)
# ──────────────────────────────────────────────

@app.get("/api/narratives/{iso_code}")
def get_narrative(iso_code: str):
    """AI-generated narrative summary for a country."""
    narratives = get_narratives()
    iso_code = iso_code.upper()

    if narratives.empty:
        raise HTTPException(status_code=404, detail="No narratives available")

    country_narr = narratives[narratives["iso_code"] == iso_code]
    if country_narr.empty:
        raise HTTPException(status_code=404, detail=f"No narrative for {iso_code}")

    row = country_narr.iloc[0]
    return {
        "iso_code": iso_code,
        "country": row.get("country", iso_code),
        "narrative": row.get("narrative", ""),
        "risk_score": float(row.get("risk_score", 0)) if pd.notna(row.get("risk_score")) else None,
    }


@app.get("/api/anomalies")
def get_anomaly_data():
    """Anomaly detection results."""
    anomalies = get_anomalies()

    if anomalies.empty:
        return {"count": 0, "anomalies": []}

    return {
        "count": len(anomalies),
        "anomalies": anomalies.replace({np.nan: None}).to_dict(orient="records"),
    }


@app.get("/api/equity")
def get_equity_metrics():
    """Emissions equity metrics — CO₂ per GDP and per capita."""
    fact = get_fact()

    if fact.empty:
        raise HTTPException(status_code=404, detail="No data available")

    latest_year = int(fact["year"].max())
    latest = fact[fact["year"] == latest_year].copy()

    # CO₂ per GDP (tonnes per billion $)
    latest["co2_per_gdp"] = np.where(
        latest["gdp"] > 0,
        latest["co2"] / (latest["gdp"] / 1e9),
        np.nan
    )

    cols = ["iso_code", "country", "co2_per_capita", "co2_per_gdp", "co2",
            "population", "gdp"]
    available = [c for c in cols if c in latest.columns]

    return {
        "year": latest_year,
        "data": latest[available].dropna(subset=["co2_per_capita"]).replace({np.nan: None}).sort_values("co2_per_capita", ascending=False).to_dict(orient="records"),
    }


@app.get("/api/paris-tracker")
def paris_agreement_tracker():
    """Paris Agreement NDC target vs actual progress."""
    fact = get_fact()

    NDC_TARGETS = {
        "USA": {"target_year": 2030, "reduction_pct": 50, "baseline_year": 2005},
        "CHN": {"target_year": 2030, "reduction_pct": 65, "baseline_year": 2005},
        "IND": {"target_year": 2030, "reduction_pct": 45, "baseline_year": 2005},
        "DEU": {"target_year": 2030, "reduction_pct": 65, "baseline_year": 1990},
        "GBR": {"target_year": 2030, "reduction_pct": 68, "baseline_year": 1990},
        "JPN": {"target_year": 2030, "reduction_pct": 46, "baseline_year": 2013},
        "CAN": {"target_year": 2030, "reduction_pct": 40, "baseline_year": 2005},
        "AUS": {"target_year": 2030, "reduction_pct": 43, "baseline_year": 2005},
        "BRA": {"target_year": 2030, "reduction_pct": 50, "baseline_year": 2005},
        "KOR": {"target_year": 2030, "reduction_pct": 40, "baseline_year": 2018},
    }

    latest_year = int(fact["year"].max())
    results = []

    for iso, target in NDC_TARGETS.items():
        country_data = fact[fact["iso_code"] == iso].sort_values("year")
        if country_data.empty:
            continue

        baseline = country_data[country_data["year"] == target["baseline_year"]]
        current = country_data[country_data["year"] == latest_year]

        if baseline.empty or current.empty:
            continue

        baseline_co2 = float(baseline["co2"].iloc[0])
        current_co2 = float(current["co2"].iloc[0])
        target_co2 = baseline_co2 * (1 - target["reduction_pct"] / 100)
        actual_reduction = (1 - current_co2 / baseline_co2) * 100

        results.append({
            "iso_code": iso,
            "country": current["country"].iloc[0],
            "baseline_year": target["baseline_year"],
            "target_year": target["target_year"],
            "ndc_reduction_pct": target["reduction_pct"],
            "actual_reduction_pct": round(actual_reduction, 1),
            "baseline_co2": round(baseline_co2, 1),
            "current_co2": round(current_co2, 1),
            "target_co2": round(target_co2, 1),
            "gap_mt": round(current_co2 - target_co2, 1),
            "on_track": current_co2 <= target_co2,
        })

    return {
        "latest_year": latest_year,
        "countries": results,
    }


# ──────────────────────────────────────────────
# Serve Dashboard
# ──────────────────────────────────────────────
if os.path.exists(DASHBOARD_DIR):
    app.mount("/static", StaticFiles(directory=DASHBOARD_DIR), name="static")

    @app.get("/")
    def serve_dashboard():
        return FileResponse(os.path.join(DASHBOARD_DIR, "index.html"))


if __name__ == "__main__":
    import uvicorn
    print("🚀 Starting Climate Intelligence API v2.0 on http://localhost:8000")
    uvicorn.run(app, host="0.0.0.0", port=8000)
