# Climate Intelligence System

An end-to-end data pipeline, multi-model ML engine, and interactive dashboard for analyzing global climate risk, emissions trends, and testing policy interventions.

## Overview

1. **Data Pipeline**: Ingests, cleans, and merges datasets from Our World in Data (CO₂), NASA (temperature anomaly), World Bank (energy usage/electricity), and ND-GAIN (vulnerability) into a unified star schema. Includes a data quality validation layer with schema checks, null detection, and outlier flagging.
2. **Machine Learning Layer**:
   - **Prophet + LSTM Ensemble**: Multi-model 10-year time-series forecasting of CO₂ emissions with confidence intervals.
   - **XGBoost**: Climate risk scoring trained on ND-GAIN indices, blended with heuristic model.
   - **SHAP + LIME**: Dual explainability — SHAP for precise feature attributions, LIME for intuitive rule-based explanations.
   - **K-Means + DBSCAN**: Country clustering with silhouette score comparison to justify algorithm choice.
   - **Isolation Forest**: Anomaly detection to flag sudden unexplained emission spikes.
3. **Orchestration**: Apache Airflow DAG to run the full pipeline automatically (weekly).
4. **Backend**: FastAPI providing 15+ REST endpoints for data, forecasts, risk scores, narratives, equity metrics, and Paris Agreement tracking.
5. **Frontend**: Premium dark-themed HTML/JS/CSS dashboard with KPIs, maps, forecasts, SHAP charts, and policy simulation.
6. **Power BI**: Star schema CSVs with DAX measure guide for enterprise BI integration.
7. **Experiment Tracking**: MLflow for model versioning and comparison.
8. **CI/CD**: GitHub Actions for automated linting and testing.

## Architecture

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed architecture decisions (ADRs) explaining why each technology was chosen.

## Setup

```bash
pip install -r requirements.txt
```

## Running the System

### 1. Validate Data Quality
```bash
python src/data_quality.py
```

### 2. Execute the Data and ML Pipeline
```bash
python src/data_pipeline.py
python src/forecasting.py
python src/risk_model.py
python src/anomaly_detection.py
python src/narrative_gen.py
python src/powerbi_export.py
```

### 3. Start the API & Dashboard
```bash
uvicorn src.api:app --host 0.0.0.0 --port 8000
```
Navigate to `http://localhost:8000`.

### 4. Run Tests
```bash
pytest tests/ -v
```

## Project Structure

```
climate-intelligence/
├── data/
│   ├── raw/                    # Original datasets
│   ├── processed/              # Intermediate processing
│   └── versioned/              # Timestamped dataset snapshots
├── notebooks/
│   └── model_experiments/      # MLflow experiment notebooks
├── src/
│   ├── data_pipeline.py        # ETL → star schema
│   ├── data_quality.py         # Data validation layer
│   ├── great_expectations_layer.py # Advanced schema expectations
│   ├── duckdb_store.py         # DuckDB analytics layer
│   ├── forecasting.py          # Prophet + LSTM ensemble
│   ├── risk_model.py           # XGBoost + SHAP + LIME + K-Means + DBSCAN
│   ├── anomaly_detection.py    # Isolation Forest
│   ├── narrative_gen.py        # Auto-generated country narratives
│   ├── policy_simulator.py     # What-if engine
│   ├── powerbi_export.py       # Power BI data export
│   ├── pdf_export.py           # Auto-generated PDF reports
│   └── api.py                  # FastAPI backend (15+ endpoints)
├── dashboard/                  # Premium HTML/CSS/JS frontend
├── airflow/dags/               # Airflow DAG
├── powerbi/                    # Power BI setup guide + DAX
├── tests/                      # pytest test suite
├── .github/workflows/          # CI/CD (GitHub Actions)
├── outputs/                    # Generated star schema + model outputs
├── ARCHITECTURE.md             # Architecture Decision Records
├── requirements.txt            # Python dependencies
└── README.md
```


