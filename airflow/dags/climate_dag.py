"""
Climate Intelligence System — Airflow DAG
===========================================
Orchestrates the full data pipeline:
  1. Validate data files exist
  2. Run data pipeline (clean + transform + star schema)
  3. Run forecasting (Prophet)
  4. Run risk model (scoring + SHAP + clustering)
  5. Export final outputs

Schedule: Weekly (every Monday at 6 AM UTC)
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.operators.bash import BashOperator
import os
import sys

# Project paths
PROJECT_DIR = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
SRC_DIR = os.path.join(PROJECT_DIR, "src")
DATA_DIR = os.path.join(PROJECT_DIR, "data")
OUTPUT_DIR = os.path.join(PROJECT_DIR, "outputs")

# Default DAG arguments
default_args = {
    "owner": "climate-intelligence",
    "depends_on_past": False,
    "email_on_failure": False,
    "email_on_retry": False,
    "retries": 1,
    "retry_delay": timedelta(minutes=5),
    "start_date": datetime(2024, 1, 1),
}


def validate_data(**kwargs):
    """Validate that required data files exist."""
    required_files = [
        os.path.join(DATA_DIR, "owid-co2-data.csv"),
        os.path.join(DATA_DIR, "GLB.Ts+dSST.csv"),
        os.path.join(DATA_DIR, "WB_WDI_EG_ELC_ACCS_ZS.csv"),
        os.path.join(DATA_DIR, "WB_WDI_EG_USE_ELEC_KH_PC.csv"),
        os.path.join(DATA_DIR, "WB_WDI_EG_USE_PCAP_KG_OE.csv"),
    ]

    missing = []
    for f in required_files:
        if not os.path.exists(f):
            missing.append(f)

    if missing:
        raise FileNotFoundError(f"Missing data files: {missing}")

    print(f"✅ All {len(required_files)} data files validated")
    return True


def run_data_pipeline(**kwargs):
    """Execute the data engineering pipeline."""
    sys.path.insert(0, SRC_DIR)
    from data_pipeline import run_pipeline
    run_pipeline()


def run_forecasting(**kwargs):
    """Execute the forecasting pipeline."""
    sys.path.insert(0, SRC_DIR)
    from forecasting import run_forecasting as forecast
    forecast()


def run_risk_model(**kwargs):
    """Execute the risk scoring pipeline."""
    sys.path.insert(0, SRC_DIR)
    from risk_model import run_risk_model as risk
    risk()


def export_final(**kwargs):
    """Validate and summarize final outputs."""
    expected = [
        "fact_climate_metrics.csv",
        "dim_country.csv",
        "dim_time.csv",
        "forecasts.csv",
        "risk_scores.csv",
        "clusters.csv",
        "recommendations.csv",
    ]

    print("\n📦 FINAL OUTPUT SUMMARY:")
    for f in expected:
        path = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(path):
            size = os.path.getsize(path) / 1024
            print(f"   ✅ {f:>35s}  ({size:.0f} KB)")
        else:
            print(f"   ⚠️  {f:>35s}  MISSING")

    # Create a final consolidated export
    print("\n✅ Pipeline complete!")


# ──────────────────────────────────────────────
# DAG Definition
# ──────────────────────────────────────────────
with DAG(
    dag_id="climate_intelligence_pipeline",
    default_args=default_args,
    description="End-to-end Climate Intelligence data pipeline",
    schedule="0 6 * * 1",  # Weekly on Monday at 6 AM UTC
    catchup=False,
    tags=["climate", "ml", "pipeline"],
) as dag:

    t1_validate = PythonOperator(
        task_id="validate_data",
        python_callable=validate_data,
    )

    t2_pipeline = PythonOperator(
        task_id="run_data_pipeline",
        python_callable=run_data_pipeline,
    )

    t3_forecast = PythonOperator(
        task_id="run_forecasting",
        python_callable=run_forecasting,
    )

    t4_risk = PythonOperator(
        task_id="run_risk_model",
        python_callable=run_risk_model,
    )

    t5_export = PythonOperator(
        task_id="export_final_outputs",
        python_callable=export_final,
    )

    # DAG dependency chain
    t1_validate >> t2_pipeline >> [t3_forecast, t4_risk] >> t5_export
