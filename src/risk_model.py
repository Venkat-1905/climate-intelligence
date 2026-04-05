"""
Climate Intelligence System — Risk Scoring & Explainability
============================================================
- Climate Risk Score (0–100) using XGBoost + heuristic model
- SHAP explainability for risk drivers
- LIME explainability as second lens
- K-Means + DBSCAN clustering with silhouette comparison
- MLflow experiment tracking

Outputs:
  - outputs/risk_scores.csv
  - outputs/shap_values.csv
  - outputs/lime_explanations.csv
  - outputs/clusters.csv
  - outputs/cluster_comparison.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler
from sklearn.cluster import KMeans, DBSCAN
from sklearn.metrics import silhouette_score
from xgboost import XGBRegressor

warnings.filterwarnings("ignore")

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

# Features used for risk scoring
RISK_FEATURES = [
    "co2_per_capita",
    "co2_growth_rate",
    "coal_share",
    "renewables_share",
    "energy_intensity",
    "temperature_anomaly",
    "vulnerability_score",
    "sustainability_score",
]

# Features for clustering
CLUSTER_FEATURES = [
    "co2_per_capita",
    "renewables_share",
    "energy_intensity",
    "co2_growth_rate",
    "coal_share",
]

CLUSTER_NAMES = {
    0: "High Emitters",
    1: "Green Economies",
    2: "Transition Economies",
}


def compute_heuristic_risk(df):
    """
    Compute a heuristic risk score (0–100) based on weighted features.
    Higher = more risk.
    """
    scaler = MinMaxScaler(feature_range=(0, 100))

    risk_up = ["co2_per_capita", "coal_share", "energy_intensity", "vulnerability_score"]
    risk_down = ["renewables_share", "sustainability_score", "readiness_score"]

    score = pd.Series(0.0, index=df.index)
    weights_total = 0

    feature_weights = {
        "co2_per_capita": 20,
        "coal_share": 15,
        "energy_intensity": 10,
        "vulnerability_score": 20,
        "renewables_share": 15,
        "sustainability_score": 10,
        "readiness_score": 10,
    }

    for feat, weight in feature_weights.items():
        if feat not in df.columns:
            continue
        vals = df[feat].copy()
        if vals.notna().sum() == 0:
            continue

        vmin, vmax = vals.min(), vals.max()
        if vmax == vmin:
            normalized = pd.Series(50.0, index=df.index)
        else:
            normalized = (vals - vmin) / (vmax - vmin) * 100

        if feat in risk_down:
            normalized = 100 - normalized

        score += normalized.fillna(50) * weight
        weights_total += weight

    if weights_total > 0:
        score = score / weights_total

    return score.clip(0, 100)


def train_risk_model(df):
    """
    Train XGBoost model on ND-GAIN vulnerability as proxy target.
    Tracks experiment with MLflow if available.
    """
    print("🧠 Training risk scoring model...")

    train_mask = df["vulnerability_score"].notna()
    features_available = [f for f in RISK_FEATURES if f in df.columns and f != "vulnerability_score"]

    train_data = df[train_mask].copy()
    train_data = train_data.dropna(subset=features_available)

    if len(train_data) < 100:
        print("   ⚠️  Insufficient training data, using heuristic model only")
        return None, features_available

    X = train_data[features_available].replace([np.inf, -np.inf], np.nan).fillna(0)
    y = MinMaxScaler(feature_range=(0, 100)).fit_transform(
        train_data[["vulnerability_score"]]
    ).ravel()

    model = XGBRegressor(
        n_estimators=200,
        max_depth=5,
        learning_rate=0.1,
        random_state=42,
    )
    model.fit(X, y)

    train_score = model.score(X, y)
    print(f"   ✅ XGBoost model trained — R² = {train_score:.3f} on {len(train_data)} samples")

    # ─── MLflow Tracking ───
    try:
        import mlflow
        mlflow.set_tracking_uri(os.path.join(BASE_DIR, "mlruns"))
        mlflow.set_experiment("climate_risk_scoring")

        with mlflow.start_run(run_name="xgboost_risk_model"):
            mlflow.log_params({
                "n_estimators": 200,
                "max_depth": 5,
                "learning_rate": 0.1,
                "features": ",".join(features_available),
                "train_samples": len(train_data),
            })
            mlflow.log_metric("r2_score", train_score)
            mlflow.sklearn.log_model(model, "xgboost_risk_model")
            print("   📊 MLflow: experiment logged")
    except Exception as e:
        print(f"   ℹ️  MLflow tracking skipped: {e}")

    return model, features_available


def compute_shap_values(model, df, features):
    """Compute SHAP values for risk model explainability."""
    print("🔍 Computing SHAP values...")

    try:
        import shap

        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0)
        explainer = shap.TreeExplainer(model)
        shap_values = explainer.shap_values(X)

        shap_df = pd.DataFrame(shap_values, columns=[f"shap_{f}" for f in features], index=df.index)

        feature_importance = pd.DataFrame({
            "feature": features,
            "importance": np.abs(shap_values).mean(axis=0),
        }).sort_values("importance", ascending=False)

        print(f"   ✅ SHAP values computed for {len(shap_df)} rows")
        print("\n   📊 Feature Importance (SHAP):")
        for _, row in feature_importance.iterrows():
            bar = "█" * int(row["importance"] / feature_importance["importance"].max() * 20)
            print(f"      {row['feature']:>25s}  {bar}  {row['importance']:.3f}")

        # Save SHAP summary plot
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        plt.figure()
        shap.summary_plot(shap_values, X, show=False)
        plt.savefig(os.path.join(OUTPUT_DIR, "shap_summary.png"), bbox_inches="tight")
        plt.close()

        return shap_df, feature_importance

    except ImportError:
        print("   ⚠️  SHAP not installed, skipping explainability")
        return pd.DataFrame(), pd.DataFrame()
    except Exception as e:
        print(f"   ⚠️  SHAP computation failed: {e}")
        return pd.DataFrame(), pd.DataFrame()


def compute_lime_explanations(model, df, features, n_samples=10):
    """
    Compute LIME explanations for top risk countries.
    Provides a second lens on feature importance alongside SHAP.
    """
    print("\n🍋 Computing LIME explanations...")

    try:
        import lime
        import lime.lime_tabular

        X = df[features].replace([np.inf, -np.inf], np.nan).fillna(0).values

        explainer = lime.lime_tabular.LimeTabularExplainer(
            X,
            feature_names=features,
            mode="regression",
            random_state=42,
        )

        # Explain top-N risk countries
        lime_results = []
        sample_indices = df.nlargest(n_samples, "vulnerability_score").index.tolist()

        for idx_pos, idx in enumerate(sample_indices):
            row_pos = df.index.get_loc(idx)
            if isinstance(row_pos, slice):
                row_pos = row_pos.start

            exp = explainer.explain_instance(
                X[row_pos],
                model.predict,
                num_features=len(features),
            )

            iso = df.loc[idx, "iso_code"] if "iso_code" in df.columns else f"row_{idx}"
            country = df.loc[idx, "country"] if "country" in df.columns else iso

            for feat_name, weight in exp.as_list():
                lime_results.append({
                    "iso_code": iso,
                    "country": country,
                    "feature_rule": feat_name,
                    "lime_weight": weight,
                })

        lime_df = pd.DataFrame(lime_results)
        print(f"   ✅ LIME explanations generated for {n_samples} countries")
        return lime_df

    except ImportError:
        print("   ⚠️  LIME not installed, skipping")
        return pd.DataFrame()
    except Exception as e:
        print(f"   ⚠️  LIME computation failed: {e}")
        return pd.DataFrame()


def cluster_countries_kmeans(latest, features_available, X_scaled):
    """K-Means clustering."""
    kmeans = KMeans(n_clusters=3, random_state=42, n_init=10)
    labels = kmeans.fit_predict(X_scaled)
    sil_score = silhouette_score(X_scaled, labels)
    return labels, sil_score, "kmeans"


def cluster_countries_dbscan(latest, features_available, X_scaled):
    """DBSCAN clustering — handles irregular cluster shapes."""
    # Try multiple eps values to find best silhouette
    best_labels = None
    best_sil = -1
    best_eps = 0.5

    for eps in [0.3, 0.5, 0.7, 1.0, 1.5]:
        dbscan = DBSCAN(eps=eps, min_samples=5)
        labels = dbscan.fit_predict(X_scaled)
        n_clusters = len(set(labels)) - (1 if -1 in labels else 0)
        if n_clusters >= 2:
            # Exclude noise for silhouette
            mask = labels != -1
            if mask.sum() > 10:
                sil = silhouette_score(X_scaled[mask], labels[mask])
                if sil > best_sil:
                    best_sil = sil
                    best_labels = labels
                    best_eps = eps

    if best_labels is None:
        return None, -1, "dbscan"

    return best_labels, best_sil, f"dbscan_eps{best_eps}"


def cluster_countries(df):
    """Cluster countries using K-Means AND DBSCAN, compare with silhouette scores."""
    print("\n🏷️  Clustering countries (K-Means vs DBSCAN)...")

    features_available = [f for f in CLUSTER_FEATURES if f in df.columns]

    df_clean = df.dropna(subset=["co2_per_capita"]).copy()
    df_clean = df_clean.sort_values(["iso_code", "year"])
    latest = df_clean.drop_duplicates(subset=["iso_code"], keep="last").copy()
    latest = latest.dropna(subset=["co2_per_capita"])

    for f in features_available:
        if f in latest.columns:
            latest[f] = latest[f].replace([np.inf, -np.inf], np.nan).fillna(0)

    if len(latest) < 10:
        print("   ⚠️  Not enough data for clustering")
        return pd.DataFrame()

    X = latest[features_available].values
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    # ─── K-Means ───
    kmeans_labels, kmeans_sil, _ = cluster_countries_kmeans(latest, features_available, X_scaled)

    # ─── DBSCAN ───
    dbscan_labels, dbscan_sil, dbscan_info = cluster_countries_dbscan(latest, features_available, X_scaled)

    # Compare
    print(f"\n   📊 Clustering Comparison:")
    print(f"      K-Means  silhouette: {kmeans_sil:.3f}")
    print(f"      DBSCAN   silhouette: {dbscan_sil:.3f}")

    # Use K-Means as primary (more stable), keep DBSCAN as alternate
    latest["cluster_id"] = kmeans_labels
    latest["kmeans_silhouette"] = kmeans_sil

    if dbscan_labels is not None:
        latest["dbscan_cluster"] = dbscan_labels
        latest["dbscan_silhouette"] = dbscan_sil

    # Auto-label K-Means clusters
    cluster_medians = latest.groupby("cluster_id")["co2_per_capita"].median().sort_values()
    label_map = {}
    cluster_ids_sorted = cluster_medians.index.tolist()

    if len(cluster_ids_sorted) >= 3:
        label_map[cluster_ids_sorted[0]] = "Green Economies"
        label_map[cluster_ids_sorted[1]] = "Transition Economies"
        label_map[cluster_ids_sorted[2]] = "High Emitters"
    else:
        for i, cid in enumerate(cluster_ids_sorted):
            label_map[cid] = f"Cluster {i}"

    latest["cluster_name"] = latest["cluster_id"].map(label_map)

    # Save comparison report
    comparison = pd.DataFrame({
        "algorithm": ["K-Means (k=3)", f"DBSCAN ({dbscan_info})"],
        "silhouette_score": [kmeans_sil, dbscan_sil],
        "n_clusters": [3, len(set(dbscan_labels)) - (1 if -1 in dbscan_labels else 0) if dbscan_labels is not None else 0],
        "chosen": [True, False],
        "reason": [
            "More stable, interpretable clusters" if kmeans_sil >= dbscan_sil else "Lower silhouette but stable",
            "Better at irregular shapes" if dbscan_sil > kmeans_sil else "Noise-sensitive with this data"
        ],
    })
    comparison.to_csv(os.path.join(OUTPUT_DIR, "cluster_comparison.csv"), index=False)

    result = latest[["iso_code", "country", "cluster_id", "cluster_name"] + features_available].copy()

    for cname in label_map.values():
        count = (result["cluster_name"] == cname).sum()
        median_co2 = result[result["cluster_name"] == cname]["co2_per_capita"].median()
        print(f"   {cname:>25s}: {count:>3} countries (median CO₂/cap: {median_co2:.1f})")

    print(f"   ✅ Clustering complete: {len(result)} countries in {len(label_map)} clusters")
    print(f"   ✅ Justification: K-Means chosen (silhouette={kmeans_sil:.3f} vs DBSCAN={dbscan_sil:.3f})")
    return result


def generate_recommendations(row):
    """Generate action recommendations based on a country's risk profile."""
    recs = []

    if pd.notna(row.get("coal_share")) and row["coal_share"] > 30:
        recs.append(f"Reduce coal dependency (currently {row['coal_share']:.0f}% of emissions)")

    if pd.notna(row.get("renewables_share")) and row["renewables_share"] < 20:
        recs.append(f"Increase renewable energy adoption (currently {row['renewables_share']:.0f}%)")

    if pd.notna(row.get("co2_growth_rate")) and row["co2_growth_rate"] > 2:
        recs.append(f"Implement emission reduction targets (growing at {row['co2_growth_rate']:.1f}%/yr)")

    if pd.notna(row.get("energy_intensity")) and row["energy_intensity"] > 0.5:
        recs.append("Improve energy efficiency across industrial sectors")

    if pd.notna(row.get("co2_per_capita")) and row["co2_per_capita"] > 10:
        recs.append(f"Per-capita emissions ({row['co2_per_capita']:.1f}t) above global target — consider carbon pricing")

    if pd.notna(row.get("vulnerability_score")) and row["vulnerability_score"] > 0.5:
        recs.append("Invest in climate adaptation infrastructure")

    if not recs:
        recs.append("Maintain current sustainability policies and monitor trends")

    return " | ".join(recs)


def run_risk_model():
    """Execute the risk scoring pipeline."""
    print("=" * 60)
    print("⚠️  CLIMATE INTELLIGENCE — RISK SCORING ENGINE")
    print("   Models: XGBoost + SHAP + LIME + K-Means + DBSCAN")
    print("=" * 60)

    fact_path = os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv")
    if not os.path.exists(fact_path):
        print("❌ fact_climate_metrics.csv not found. Run data_pipeline.py first.")
        sys.exit(1)

    fact = pd.read_csv(fact_path)

    # ── Risk Scoring ──
    fact["risk_score_heuristic"] = compute_heuristic_risk(fact)

    model, features_used = train_risk_model(fact)

    if model is not None:
        X_all = fact[features_used].replace([np.inf, -np.inf], np.nan).fillna(0)
        fact["risk_score_ml"] = model.predict(X_all).clip(0, 100)
        fact["risk_score"] = (
            0.6 * fact["risk_score_ml"] + 0.4 * fact["risk_score_heuristic"]
        ).round(1)
    else:
        fact["risk_score"] = fact["risk_score_heuristic"].round(1)

    # ── SHAP ──
    latest_year = fact["year"].max()
    latest = fact[fact["year"] == latest_year].copy()

    if model is not None:
        shap_df, feature_importance = compute_shap_values(model, latest, features_used)

        if not shap_df.empty:
            shap_output = latest[["iso_code", "country"]].copy()
            shap_output = pd.concat([shap_output.reset_index(drop=True), shap_df.reset_index(drop=True)], axis=1)
            shap_output.to_csv(os.path.join(OUTPUT_DIR, "shap_values.csv"), index=False)
            feature_importance.to_csv(os.path.join(OUTPUT_DIR, "feature_importance.csv"), index=False)
            print(f"   💾 SHAP values saved")

        # ── LIME ──
        lime_df = compute_lime_explanations(model, latest, features_used, n_samples=15)
        if not lime_df.empty:
            lime_df.to_csv(os.path.join(OUTPUT_DIR, "lime_explanations.csv"), index=False)
            print(f"   💾 LIME explanations saved")

    # ── Clustering (K-Means + DBSCAN) ──
    clusters = cluster_countries(fact)
    if not clusters.empty:
        clusters.to_csv(os.path.join(OUTPUT_DIR, "clusters.csv"), index=False)

    # ── Recommendations ──
    print("\n📋 Generating recommendations...")
    latest_with_scores = fact[fact["year"] == latest_year].copy()
    latest_with_scores["recommendations"] = latest_with_scores.apply(generate_recommendations, axis=1)

    recommendations = latest_with_scores[["iso_code", "country", "risk_score", "recommendations"]].copy()
    recommendations = recommendations.sort_values("risk_score", ascending=False)
    recommendations.to_csv(os.path.join(OUTPUT_DIR, "recommendations.csv"), index=False)

    # Print top 10
    print("\n   🔴 TOP 10 HIGH-RISK COUNTRIES:")
    for _, row in recommendations.head(10).iterrows():
        print(f"      {row['country']:>25s}  Risk: {row['risk_score']:5.1f}")

    # Save risk scores
    def categorize_risk(x):
        if pd.isna(x): return "Unknown"
        return "High" if x > 70 else "Medium" if x > 40 else "Low"
    fact["risk_category"] = fact["risk_score"].apply(categorize_risk)

    risk_output = fact[["iso_code", "country", "year", "risk_score", "risk_category",
                        "co2_per_capita", "coal_share", "renewables_share",
                        "vulnerability_score", "sustainability_score",
                        "co2_growth_rate", "energy_intensity",
                        "temperature_anomaly"]].copy()
    risk_output.to_csv(os.path.join(OUTPUT_DIR, "risk_scores.csv"), index=False)

    # Save enriched fact table
    fact.to_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"), index=False)

    print(f"\n💾 Saved: risk_scores.csv, clusters.csv, recommendations.csv, lime_explanations.csv")
    print("=" * 60)

    return fact


if __name__ == "__main__":
    run_risk_model()
