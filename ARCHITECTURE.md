# Architecture Decision Records

This document explains the key technical decisions in the Climate Intelligence System and the rationale behind each choice.

## ADR-1: Why Prophet over ARIMA for Primary Forecasting

**Decision:** Use Facebook Prophet as the primary time-series forecasting model, with PyTorch LSTM as the secondary model in an ensemble.

**Context:** We needed a forecasting model for CO₂ emissions per country with 10-year horizon on yearly data (small datasets, ~30 data points per country).

**Considered alternatives:**
| Model | Pros | Cons |
|-------|------|------|
| **ARIMA** | Well-understood, statistically rigorous | Requires manual p,d,q tuning per country; struggles with trend changes; no built-in uncertainty intervals |
| **Prophet** | Auto-detects trends and changepoints; built-in confidence intervals; handles missing data well; minimal tuning needed for 190+ countries | Less flexible for multivariate inputs; sometimes under-fits short series |
| **LSTM** | Captures non-linear patterns; learns from sequences | Needs more data; prone to overfitting on small samples; harder to interpret |
| **NeuralProphet** | Combines Prophet's interface with neural network backend | Additional dependency; marginal improvement for yearly data; less mature library |

**Decision rationale:** Prophet was chosen as primary (60% weight) because it handles trend decomposition well with minimal tuning — critical when running 50+ countries automatically. LSTM was added as secondary (40% weight) because it captures non-linear patterns Prophet might miss. The ensemble consistently outperforms either model alone in internal testing.

**Consequences:** The weighted ensemble produces forecasts with confidence intervals from both models, giving a robust uncertainty estimate.

---

## ADR-2: Why DuckDB over PostgreSQL for Analytical Storage

**Decision:** Use DuckDB as the analytical database layer instead of PostgreSQL.

**Context:** We needed a database for analytical queries on the star schema output, primarily for Power BI connectivity and fast ad-hoc analysis.

**Considered alternatives:**
| Database | Pros | Cons |
|----------|------|------|
| **CSV files only** | Simple, no setup | Slow for complex queries; no indexing; poor for joins |
| **SQLite** | Zero-config, embedded | Not optimized for analytical (columnar) queries |
| **PostgreSQL** | Full-featured, enterprise-grade | Requires server setup; overkill for single-user analytical workloads |
| **DuckDB** | Zero-config like SQLite but columnar; blazing fast analytical queries; reads CSV/Parquet directly; embeddable | Newer, less ecosystem support; no concurrent writes |

**Decision rationale:** DuckDB fits perfectly because: (1) zero server setup — just `pip install duckdb`, (2) columnar storage optimized for our aggregation-heavy analytical queries, (3) can directly query CSV files, (4) embeds naturally in Python without infrastructure overhead. PostgreSQL would require Docker or a managed service — unnecessary complexity for a portfolio project.

---

## ADR-3: Why XGBoost + Heuristic Blend for Risk Scoring

**Decision:** Blend XGBoost ML predictions (60%) with a weighted heuristic formula (40%) for the final risk score.

**Context:** No explicit "climate risk" label exists, so we used ND-GAIN vulnerability scores as a proxy target to train the ML model.

**Why not pure ML:** The proxy target (vulnerability_score) doesn't perfectly capture "climate risk" — it's more about climate readiness than emission danger. A pure ML model would overfit to vulnerability_score's biases.

**Why not pure heuristic:** Hand-crafted weights miss complex feature interactions that XGBoost captures.

**Blending advantage:** The 60/40 blend captures ML's non-linear pattern recognition while the heuristic ensures domain-knowledge features (coal dependency, renewables share) always influence the score proportionally. This makes the model more interpretable and robust.

---

## ADR-4: Why SHAP + LIME Dual Explainability

**Decision:** Use both SHAP (TreeExplainer) and LIME for risk model explainability.

**Rationale:**
- **SHAP** (via TreeExplainer): Theoretically grounded in Shapley values; provides exact feature attributions for tree-based models; great for global feature importance.
- **LIME**: Model-agnostic; explains individual predictions by fitting local linear models; provides rule-based explanations that are easier for non-technical stakeholders to understand.

Using both provides complementary views: SHAP gives mathematically precise attributions while LIME gives intuitive "if-then" rules. This dual approach demonstrates explainability maturity.

---

## ADR-5: Why K-Means + DBSCAN Comparison for Clustering

**Decision:** Run both K-Means and DBSCAN, compare silhouette scores, and justify the choice quantitatively.

**K-Means (chosen):** Produces consistent, balanced clusters (Green / Transition / High Emitters). Silhouette score provides a quantitative quality measure.

**DBSCAN (compared):** Better at finding irregular shapes and handling outliers (noise points), but produced less interpretable clusters on this data because emission profiles tend to form roughly spherical groupings in feature space.

**Key insight:** By running both and saving `cluster_comparison.csv`, we demonstrate that the clustering choice is data-driven rather than arbitrary — exactly the kind of rigor interviewers look for.

---

## ADR-6: Why Custom Web Dashboard over Streamlit

**Decision:** Build a custom HTML/JS/CSS dashboard backed by FastAPI instead of relying on Streamlit.

**Rationale:** While Streamlit is excellent for quick prototyping, a custom frontend provides:
- Complete control over visually stunning premium dark-mode aesthetics.
- Finer-grained interactivity for complex features like policy simulation and dynamically formatted Plotly charts.
- Separation of backend API logic via FastAPI, proving end-to-end architecture skills (creating a robust interface instead of a monolithic data app).
- No framework-related limitations when updating state for maps, complex grids, or dynamic UI components.

This ensures the project delivers a true production-grade, highly engaging UX.

---

## ADR-7: Pipeline Orchestration Strategy

**Decision:** Use Apache Airflow with a simple 5-task DAG running weekly.

**Task chain:**
```
validate_data → run_data_pipeline → [run_forecasting ∥ run_risk_model] → export_final
```

**Key design choice:** Forecasting and risk model run in parallel after the data pipeline completes — they're independent computations that both read from the same fact table.

---

## Technology Stack Summary

| Layer | Technology | Why |
|-------|-----------|-----|
| Data Pipeline | pandas + numpy | Industry standard, sufficient for ~14MB datasets |
| Storage | CSV + DuckDB | Zero-config, columnar analytics, Power BI compatible |
| Forecasting | Prophet + LSTM (PyTorch) | Trend decomposition + non-linear patterns |
| Risk Scoring | XGBoost + heuristic blend | ML precision + domain knowledge robustness |
| Explainability | SHAP + LIME | Dual-lens: mathematical + intuitive explanations |
| Clustering | K-Means + DBSCAN | Systematic comparison with silhouette scoring |
| Anomaly Detection | Isolation Forest | Unsupervised, handles multivariate anomalies |
| Orchestration | Apache Airflow | Production-grade scheduling with monitoring |
| Backend API | FastAPI | Async, auto-documented, high-performance |
| Dashboard | HTML/CSS/Vanilla JS | Premium dark theme, Plotly-based charts, zero framework overhead |
| Experiment Tracking | MLflow | Model versioning and metric comparison |

