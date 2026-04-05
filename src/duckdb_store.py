"""
Climate Intelligence System — DuckDB Analytical Store
=======================================================
Stores the star schema in DuckDB for fast analytical queries.
DuckDB is chosen over PostgreSQL because it's zero-config,
columnar (optimised for aggregations), and embeds directly in Python.

Usage:
  python src/duckdb_store.py           # Load CSVs → DuckDB
  python src/duckdb_store.py --query   # Run sample queries
"""

import os
import sys
import duckdb
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
DB_PATH = os.path.join(OUTPUT_DIR, "climate_intelligence.duckdb")


def create_database():
    """Load all CSV outputs into DuckDB tables."""
    print("=" * 60)
    print("🦆 CLIMATE INTELLIGENCE — DuckDB Store")
    print("=" * 60)

    con = duckdb.connect(DB_PATH)

    tables = {
        "fact_climate_metrics": "fact_climate_metrics.csv",
        "dim_country": "dim_country.csv",
        "dim_time": "dim_time.csv",
        "dim_energy_type": "dim_energy_type.csv",
        "forecasts": "forecasts.csv",
        "risk_scores": "risk_scores.csv",
        "clusters": "clusters.csv",
        "recommendations": "recommendations.csv",
    }

    for table_name, filename in tables.items():
        filepath = os.path.join(OUTPUT_DIR, filename)
        if os.path.exists(filepath):
            con.execute(f"DROP TABLE IF EXISTS {table_name}")
            con.execute(f"CREATE TABLE {table_name} AS SELECT * FROM read_csv_auto('{filepath}')")
            count = con.execute(f"SELECT COUNT(*) FROM {table_name}").fetchone()[0]
            print(f"   ✅ {table_name:>30s}: {count:>8,} rows")
        else:
            print(f"   ⚠️  {table_name:>30s}: FILE NOT FOUND")

    # Optional: load anomalies and narratives if they exist
    for extra in ["anomalies", "narratives", "lime_explanations"]:
        filepath = os.path.join(OUTPUT_DIR, f"{extra}.csv")
        if os.path.exists(filepath):
            con.execute(f"DROP TABLE IF EXISTS {extra}")
            con.execute(f"CREATE TABLE {extra} AS SELECT * FROM read_csv_auto('{filepath}')")
            count = con.execute(f"SELECT COUNT(*) FROM {extra}").fetchone()[0]
            print(f"   ✅ {extra:>30s}: {count:>8,} rows")

    con.close()
    print(f"\n💾 Database saved: {DB_PATH}")
    print(f"   Size: {os.path.getsize(DB_PATH) / 1024:.0f} KB")
    print("=" * 60)


def run_sample_queries():
    """Run sample analytical queries to demonstrate DuckDB's power."""
    print("\n" + "=" * 60)
    print("🔍 SAMPLE ANALYTICAL QUERIES")
    print("=" * 60)

    con = duckdb.connect(DB_PATH, read_only=True)

    queries = [
        (
            "Top 5 emitters (latest year)",
            """
            SELECT country, year, co2, co2_per_capita
            FROM fact_climate_metrics
            WHERE year = (SELECT MAX(year) FROM fact_climate_metrics)
            ORDER BY co2 DESC
            LIMIT 5
            """
        ),
        (
            "Average risk by continent",
            """
            SELECT d.continent, 
                   ROUND(AVG(r.risk_score), 1) as avg_risk,
                   COUNT(DISTINCT r.iso_code) as countries
            FROM risk_scores r
            JOIN dim_country d ON r.iso_code = d.iso_code
            WHERE r.year = (SELECT MAX(year) FROM risk_scores)
            GROUP BY d.continent
            ORDER BY avg_risk DESC
            """
        ),
        (
            "Year-over-year global CO₂ growth",
            """
            WITH yearly AS (
                SELECT year, SUM(co2) as total_co2
                FROM fact_climate_metrics
                GROUP BY year
            )
            SELECT year, 
                   ROUND(total_co2, 0) as total_co2_mt,
                   ROUND((total_co2 - LAG(total_co2) OVER (ORDER BY year)) 
                         / LAG(total_co2) OVER (ORDER BY year) * 100, 2) as yoy_growth_pct
            FROM yearly
            ORDER BY year DESC
            LIMIT 10
            """
        ),
        (
            "Cluster summary",
            """
            SELECT cluster_name,
                   COUNT(*) as countries,
                   ROUND(AVG(co2_per_capita), 1) as avg_co2_pc,
                   ROUND(AVG(renewables_share), 1) as avg_renewables
            FROM clusters
            GROUP BY cluster_name
            ORDER BY avg_co2_pc DESC
            """
        ),
    ]

    for title, query in queries:
        print(f"\n📊 {title}:")
        print("-" * 50)
        try:
            result = con.execute(query).fetchdf()
            print(result.to_string(index=False))
        except Exception as e:
            print(f"   Error: {e}")

    con.close()
    print("\n" + "=" * 60)


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser()
    parser.add_argument("--query", action="store_true", help="Run sample queries")
    args = parser.parse_args()

    create_database()
    if args.query:
        run_sample_queries()
