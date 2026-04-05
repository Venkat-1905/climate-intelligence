"""
Climate Intelligence System — Anomaly Detection Module
========================================================
Uses Isolation Forest to detect anomalous emission patterns:
  - Sudden unexplained CO₂ spikes
  - Unusual country-year combinations
  - Suspicious data quality signals

Outputs:
  - outputs/anomalies.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.preprocessing import StandardScaler

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Features to use for anomaly detection
ANOMALY_FEATURES = [
    "co2_per_capita",
    "co2_growth_rate",
    "coal_share",
    "renewables_share",
    "energy_intensity",
    "temperature_anomaly",
]


def detect_emission_anomalies(fact_df, contamination=0.05):
    """
    Detect anomalous emission patterns using Isolation Forest.

    Args:
        fact_df: The fact_climate_metrics DataFrame
        contamination: Expected proportion of anomalies (0.01 to 0.1)

    Returns:
        DataFrame with anomaly flags and scores
    """
    print("🔍 Running Isolation Forest anomaly detection...")

    features_available = [f for f in ANOMALY_FEATURES if f in fact_df.columns]
    if len(features_available) < 3:
        print("   ⚠️  Not enough features for anomaly detection")
        return pd.DataFrame()

    # Prepare data — drop rows with all-NaN features
    df = fact_df.copy()
    df_clean = df.dropna(subset=["co2_per_capita"]).copy()

    # Fill NaN in feature columns with median
    for f in features_available:
        if f in df_clean.columns:
            df_clean[f] = df_clean[f].replace([np.inf, -np.inf], np.nan)
            df_clean[f] = df_clean[f].fillna(df_clean[f].median())

    X = df_clean[features_available].values

    # Standardize
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # Isolation Forest
    iso_forest = IsolationForest(
        n_estimators=200,
        contamination=contamination,
        random_state=42,
        n_jobs=-1,
    )
    df_clean["anomaly_label"] = iso_forest.fit_predict(X_scaled)  # 1=normal, -1=anomaly
    df_clean["anomaly_score"] = iso_forest.decision_function(X_scaled)  # lower=more anomalous

    # Flag anomalies
    df_clean["is_anomaly"] = (df_clean["anomaly_label"] == -1).astype(int)

    anomaly_count = df_clean["is_anomaly"].sum()
    total = len(df_clean)
    print(f"   ✅ Detected {anomaly_count} anomalies out of {total} rows ({anomaly_count/total:.1%})")

    return df_clean


def detect_yoy_spikes(fact_df, threshold_pct=50):
    """
    Detect year-over-year CO₂ spikes/drops exceeding threshold.

    Args:
        fact_df: The fact_climate_metrics DataFrame
        threshold_pct: Minimum % change to flag as spike

    Returns:
        DataFrame of flagged spikes
    """
    print(f"\n📈 Detecting YoY CO₂ spikes (threshold: ±{threshold_pct}%)...")

    df = fact_df.copy()
    df = df.sort_values(["iso_code", "year"])
    df["co2_yoy_pct"] = df.groupby("iso_code")["co2"].pct_change() * 100

    spikes = df[df["co2_yoy_pct"].abs() > threshold_pct].copy()
    spikes["spike_direction"] = np.where(spikes["co2_yoy_pct"] > 0, "SPIKE_UP", "SPIKE_DOWN")

    print(f"   ✅ Found {len(spikes)} YoY spikes across {spikes['iso_code'].nunique()} countries")

    if len(spikes) > 0:
        print("\n   📊 Top 10 largest spikes:")
        top = spikes.nlargest(10, "co2_yoy_pct")
        for _, row in top.iterrows():
            print(f"      {row.get('country', row['iso_code']):>25s} ({row['year']})"
                  f"  {row['co2_yoy_pct']:+.1f}%  [{row['spike_direction']}]")

    return spikes


def detect_cross_sectional_outliers(fact_df):
    """
    Detect countries that are cross-sectional outliers in any given year
    using Z-score method.
    """
    print("\n🌍 Detecting cross-sectional outliers per year...")

    metrics = ["co2_per_capita", "renewables_share", "coal_share"]
    available = [m for m in metrics if m in fact_df.columns]

    all_outliers = []
    for year in fact_df["year"].unique():
        year_data = fact_df[fact_df["year"] == year].copy()
        if len(year_data) < 20:
            continue

        for metric in available:
            vals = year_data[metric].dropna()
            if len(vals) < 10:
                continue

            mean = vals.mean()
            std = vals.std()
            if std == 0:
                continue

            z_scores = (vals - mean) / std
            outlier_mask = z_scores.abs() > 3

            for idx in z_scores[outlier_mask].index:
                row = year_data.loc[idx]
                all_outliers.append({
                    "iso_code": row.get("iso_code", ""),
                    "country": row.get("country", ""),
                    "year": year,
                    "metric": metric,
                    "value": row[metric],
                    "z_score": z_scores[idx],
                    "mean": mean,
                    "std": std,
                })

    outlier_df = pd.DataFrame(all_outliers)
    print(f"   ✅ Found {len(outlier_df)} cross-sectional outliers across all years")
    return outlier_df


def run_anomaly_detection():
    """Execute full anomaly detection pipeline."""
    print("=" * 60)
    print("🔎 CLIMATE INTELLIGENCE — ANOMALY DETECTION ENGINE")
    print("=" * 60)

    # Load fact table
    fact_path = os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv")
    if not os.path.exists(fact_path):
        print("❌ fact_climate_metrics.csv not found. Run data_pipeline.py first.")
        sys.exit(1)

    fact = pd.read_csv(fact_path)

    # Run all anomaly detection methods
    anomaly_df = detect_emission_anomalies(fact)
    spikes_df = detect_yoy_spikes(fact, threshold_pct=50)
    outliers_df = detect_cross_sectional_outliers(fact)

    # Save results
    if not anomaly_df.empty:
        # Get anomalies only
        anomalies_only = anomaly_df[anomaly_df["is_anomaly"] == 1].copy()
        cols = ["iso_code", "country", "year", "co2_per_capita", "co2_growth_rate",
                "coal_share", "renewables_share", "anomaly_score", "is_anomaly"]
        available_cols = [c for c in cols if c in anomalies_only.columns]
        anomalies_only[available_cols].to_csv(
            os.path.join(OUTPUT_DIR, "anomalies.csv"), index=False
        )

    if not spikes_df.empty:
        spike_cols = ["iso_code", "country", "year", "co2", "co2_yoy_pct", "spike_direction"]
        available_spike_cols = [c for c in spike_cols if c in spikes_df.columns]
        spikes_df[available_spike_cols].to_csv(
            os.path.join(OUTPUT_DIR, "emission_spikes.csv"), index=False
        )

    if not outliers_df.empty:
        outliers_df.to_csv(
            os.path.join(OUTPUT_DIR, "cross_sectional_outliers.csv"), index=False
        )

    # Merge anomaly flags back to fact table
    if not anomaly_df.empty:
        merge_cols = ["iso_code", "year", "anomaly_score", "is_anomaly"]
        available_merge = [c for c in merge_cols if c in anomaly_df.columns]
        anomaly_subset = anomaly_df[available_merge].copy()
        fact = fact.merge(anomaly_subset, on=["iso_code", "year"], how="left")
        fact["is_anomaly"] = fact["is_anomaly"].fillna(0).astype(int)
        fact.to_csv(fact_path, index=False)

    # Summary
    total_anomalies = len(anomaly_df[anomaly_df["is_anomaly"] == 1]) if not anomaly_df.empty else 0
    total_spikes = len(spikes_df)
    total_outliers = len(outliers_df)

    print(f"\n{'=' * 60}")
    print(f"📊 ANOMALY DETECTION SUMMARY")
    print(f"{'=' * 60}")
    print(f"  Isolation Forest anomalies: {total_anomalies:>6}")
    print(f"  Year-over-year spikes:      {total_spikes:>6}")
    print(f"  Cross-sectional outliers:   {total_outliers:>6}")
    print(f"  Outputs saved to:           {OUTPUT_DIR}")
    print(f"{'=' * 60}")

    return anomaly_df, spikes_df, outliers_df


if __name__ == "__main__":
    run_anomaly_detection()
