"""
Climate Intelligence System — Policy Simulation Engine
========================================================
What-if analysis: simulates the impact of policy interventions
on a country's risk score and future emissions.

Parameters:
  - emission_reduction_pct: Reduce CO₂ by X%
  - renewables_increase_pct: Increase renewable share by X percentage points
"""

import os
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def load_latest_data():
    """Load the latest risk scores and forecasts."""
    risk_path = os.path.join(OUTPUT_DIR, "risk_scores.csv")
    forecast_path = os.path.join(OUTPUT_DIR, "forecasts.csv")

    risk_df = pd.read_csv(risk_path) if os.path.exists(risk_path) else pd.DataFrame()
    forecast_df = pd.read_csv(forecast_path) if os.path.exists(forecast_path) else pd.DataFrame()

    return risk_df, forecast_df


def simulate_policy(
    iso_code,
    emission_reduction_pct=0,
    renewables_increase_pct=0,
    risk_df=None,
    forecast_df=None,
):
    """
    Simulate policy intervention for a country.

    Args:
        iso_code: ISO3 country code
        emission_reduction_pct: % reduction in CO₂ (0-100)
        renewables_increase_pct: Percentage points increase in renewable share (0-100)

    Returns:
        dict with original and simulated metrics
    """
    if risk_df is None or forecast_df is None:
        risk_df, forecast_df = load_latest_data()

    # Get latest year data for this country
    country_risk = risk_df[risk_df["iso_code"] == iso_code].copy()
    if country_risk.empty:
        return {"error": f"Country {iso_code} not found"}

    latest_year = country_risk["year"].max()
    latest = country_risk[country_risk["year"] == latest_year].iloc[0].to_dict()

    # Get forecasts
    country_forecast = forecast_df[forecast_df["iso_code"] == iso_code].copy() if not forecast_df.empty else pd.DataFrame()

    # ── Original metrics ──
    original = {
        "iso_code": iso_code,
        "country": latest.get("country", iso_code),
        "risk_score": latest.get("risk_score", 50),
        "co2_per_capita": latest.get("co2_per_capita", 0),
        "renewables_share": latest.get("renewables_share", 0),
        "coal_share": latest.get("coal_share", 0),
        "sustainability_score": latest.get("sustainability_score", 0),
    }

    # ── Simulate changes ──
    sim_co2_per_capita = original["co2_per_capita"] * (1 - emission_reduction_pct / 100)
    sim_renewables = min(100, original["renewables_share"] + renewables_increase_pct)
    sim_coal_share = max(0, original["coal_share"] - renewables_increase_pct * 0.5)  # Renewables displace coal
    sim_sustainability = sim_renewables - sim_coal_share

    # Risk score adjustment
    # Each 1% emission reduction → ~0.3 risk reduction
    # Each 1pp renewable increase → ~0.5 risk reduction
    risk_delta = (
        - emission_reduction_pct * 0.3
        - renewables_increase_pct * 0.5
    )
    sim_risk = max(0, min(100, original["risk_score"] + risk_delta))

    simulated = {
        "risk_score": round(sim_risk, 1),
        "co2_per_capita": round(sim_co2_per_capita, 2),
        "renewables_share": round(sim_renewables, 1),
        "coal_share": round(sim_coal_share, 1),
        "sustainability_score": round(sim_sustainability, 1),
        "risk_delta": round(risk_delta, 1),
        "emission_reduction_pct": emission_reduction_pct,
        "renewables_increase_pct": renewables_increase_pct,
    }

    # ── Simulate forecast adjustments ──
    forecast_sim = []
    if not country_forecast.empty:
        for _, row in country_forecast.iterrows():
            yr = row["year"]
            # Apply reduction linearly over forecast period
            years_from_now = yr - latest_year
            gradual_reduction = emission_reduction_pct * min(years_from_now / 5, 1)
            sim_co2 = row["forecast_co2"] * (1 - gradual_reduction / 100)
            forecast_sim.append({
                "year": int(yr),
                "forecast_co2_original": round(row["forecast_co2"], 2),
                "forecast_co2_simulated": round(max(0, sim_co2), 2),
            })

    return {
        "original": original,
        "simulated": simulated,
        "forecast_comparison": forecast_sim,
    }


def run_simulation_demo():
    """Run a demo simulation."""
    print("=" * 60)
    print("🧪 POLICY SIMULATION DEMO")
    print("=" * 60)

    risk_df, forecast_df = load_latest_data()

    # Demo: India — reduce emissions 20%, increase renewables 15%
    demo_countries = [
        ("IND", 20, 15, "India: -20% emissions, +15% renewables"),
        ("CHN", 30, 20, "China: -30% emissions, +20% renewables"),
        ("USA", 25, 10, "USA: -25% emissions, +10% renewables"),
    ]

    for iso, em_red, ren_inc, label in demo_countries:
        print(f"\n{'─' * 50}")
        print(f"📋 {label}")
        result = simulate_policy(iso, em_red, ren_inc, risk_df, forecast_df)

        if "error" in result:
            print(f"   ⚠️  {result['error']}")
            continue

        orig = result["original"]
        sim = result["simulated"]

        print(f"   Risk Score:    {orig['risk_score']:>6.1f}  →  {sim['risk_score']:>6.1f}  (Δ{sim['risk_delta']:+.1f})")
        print(f"   CO₂/capita:   {orig['co2_per_capita']:>6.2f}  →  {sim['co2_per_capita']:>6.2f}")
        print(f"   Renewables:   {orig['renewables_share']:>6.1f}% →  {sim['renewables_share']:>6.1f}%")
        print(f"   Coal share:   {orig['coal_share']:>6.1f}% →  {sim['coal_share']:>6.1f}%")

    print(f"\n{'=' * 60}")


if __name__ == "__main__":
    run_simulation_demo()
