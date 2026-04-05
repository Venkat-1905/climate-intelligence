"""
Climate Intelligence System — Test Suite
==========================================
Tests for data pipeline, API, ML models, and utilities.
Run: pytest tests/ -v
"""

import os
import sys
import pytest
import pandas as pd
import numpy as np

# Add project root to path
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, os.path.join(BASE_DIR, "src"))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


# ═══════════════════════════════════════════════
# PHASE 1: DATA PIPELINE TESTS
# ═══════════════════════════════════════════════

class TestDataPipeline:
    """Tests for the data engineering pipeline outputs."""

    def test_fact_table_exists(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))

    def test_dim_country_exists(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "dim_country.csv"))

    def test_dim_time_exists(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "dim_time.csv"))

    def test_fact_table_schema(self):
        fact = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))
        required_cols = ["iso_code", "country", "year", "co2", "co2_per_capita"]
        for col in required_cols:
            assert col in fact.columns, f"Missing column: {col}"

    def test_fact_table_rows(self):
        fact = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))
        assert len(fact) > 1000, f"Expected >1000 rows, got {len(fact)}"

    def test_fact_table_year_range(self):
        fact = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))
        assert fact["year"].min() >= 1990
        assert fact["year"].max() >= 2020

    def test_dim_country_count(self):
        dim = pd.read_csv(os.path.join(OUTPUT_DIR, "dim_country.csv"))
        assert len(dim) > 50, f"Expected >50 countries, got {len(dim)}"

    def test_no_duplicate_country_year(self):
        fact = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))
        dupes = fact.duplicated(subset=["iso_code", "year"]).sum()
        assert dupes == 0, f"Found {dupes} duplicate country-year rows"

    def test_feature_engineering(self):
        fact = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))
        engineered = ["coal_share", "gas_share", "oil_share",
                     "renewables_share", "sustainability_score"]
        for col in engineered:
            assert col in fact.columns, f"Missing engineered feature: {col}"


# ═══════════════════════════════════════════════
# PHASE 2: ML MODEL TESTS
# ═══════════════════════════════════════════════

class TestForecasting:
    """Tests for the forecasting outputs."""

    def test_forecasts_exist(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "forecasts.csv"))

    def test_forecast_schema(self):
        fc = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"))
        assert "iso_code" in fc.columns
        assert "year" in fc.columns
        assert "forecast_co2" in fc.columns

    def test_forecast_positive(self):
        fc = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"))
        assert (fc["forecast_co2"] >= 0).all(), "Forecast contains negative values"

    def test_forecast_countries(self):
        fc = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"))
        assert fc["iso_code"].nunique() >= 10, "Too few countries in forecast"

    def test_confidence_intervals(self):
        fc = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"))
        assert "forecast_co2_lower" in fc.columns, "Missing lower CI"
        assert "forecast_co2_upper" in fc.columns, "Missing upper CI"


class TestRiskModel:
    """Tests for risk scoring outputs."""

    def test_risk_scores_exist(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "risk_scores.csv"))

    def test_risk_score_range(self):
        risk = pd.read_csv(os.path.join(OUTPUT_DIR, "risk_scores.csv"))
        valid = risk["risk_score"].dropna()
        assert valid.min() >= 0, f"Risk score below 0: {valid.min()}"
        assert valid.max() <= 100, f"Risk score above 100: {valid.max()}"

    def test_clusters_exist(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "clusters.csv"))

    def test_cluster_labels(self):
        cl = pd.read_csv(os.path.join(OUTPUT_DIR, "clusters.csv"))
        assert "cluster_name" in cl.columns
        expected = {"Green Economies", "Transition Economies", "High Emitters"}
        actual = set(cl["cluster_name"].unique())
        assert actual == expected, f"Unexpected clusters: {actual}"

    def test_shap_values_exist(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "shap_values.csv"))

    def test_recommendations_exist(self):
        assert os.path.exists(os.path.join(OUTPUT_DIR, "recommendations.csv"))

    def test_recommendations_nonempty(self):
        recs = pd.read_csv(os.path.join(OUTPUT_DIR, "recommendations.csv"))
        assert len(recs) > 50, "Too few recommendations"
        assert recs["recommendations"].notna().all(), "Empty recommendations found"


# ═══════════════════════════════════════════════
# PHASE 3: API TESTS
# ═══════════════════════════════════════════════

class TestAPI:
    """Tests for API models and utilities (not requiring running server)."""

    def test_policy_simulator_import(self):
        from policy_simulator import simulate_policy
        assert callable(simulate_policy)

    def test_policy_simulation_basic(self):
        from policy_simulator import simulate_policy
        result = simulate_policy("IND", 20, 15)
        assert "error" not in result, f"Simulation error: {result.get('error')}"
        assert "original" in result
        assert "simulated" in result
        assert result["simulated"]["risk_score"] <= result["original"]["risk_score"]

    def test_policy_simulation_zero_change(self):
        from policy_simulator import simulate_policy
        result = simulate_policy("USA", 0, 0)
        if "error" not in result:
            assert result["simulated"]["risk_delta"] == 0

    def test_narrative_generation(self):
        from narrative_gen import generate_country_narrative
        row = {
            "country": "TestCountry", "iso_code": "TST", "year": 2023,
            "risk_score": 75, "co2_per_capita": 12.5, "coal_share": 45,
            "renewables_share": 10, "vulnerability_score": 0.6,
            "temperature_anomaly": 1.2, "co2_growth_rate": 3.5,
        }
        narrative = generate_country_narrative(row)
        assert len(narrative) > 50
        assert "TestCountry" in narrative
        assert "high" in narrative.lower()


# ═══════════════════════════════════════════════
# PHASE 4: DATA QUALITY TESTS
# ═══════════════════════════════════════════════

class TestDataQuality:
    """Tests for data quality checks."""

    def test_data_quality_module_import(self):
        from data_quality import DataQualityValidator
        v = DataQualityValidator()
        assert hasattr(v, "check_schema")
        assert hasattr(v, "check_null_rates")

    def test_validator_basic(self):
        from data_quality import DataQualityValidator
        v = DataQualityValidator()
        df = pd.DataFrame({"a": [1, 2, 3], "b": [4, 5, 6]})
        v.check_schema(df, "test", required_columns=["a", "b"])
        assert len(v.results) == 1
        assert v.results[0].passed

    def test_validator_null_detection(self):
        from data_quality import DataQualityValidator
        v = DataQualityValidator()
        df = pd.DataFrame({"a": [1, None, None, None, None]})
        v.check_null_rates(df, "test", critical_columns=["a"], max_null_pct=0.5)
        assert len(v.results) == 1
        assert not v.results[0].passed  # 80% null > 50% threshold


# ═══════════════════════════════════════════════
# PHASE 5: ANOMALY DETECTION TESTS
# ═══════════════════════════════════════════════

class TestAnomalyDetection:
    """Tests for anomaly detection module."""

    def test_anomaly_module_import(self):
        from anomaly_detection import detect_emission_anomalies
        assert callable(detect_emission_anomalies)

    def test_yoy_spike_detection(self):
        from anomaly_detection import detect_yoy_spikes
        df = pd.DataFrame({
            "iso_code": ["TST"] * 5,
            "year": [2018, 2019, 2020, 2021, 2022],
            "co2": [100, 105, 200, 110, 115],  # Spike in 2020
        })
        spikes = detect_yoy_spikes(df, threshold_pct=50)
        assert len(spikes) >= 1


if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
