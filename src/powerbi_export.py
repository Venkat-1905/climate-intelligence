import os
import pandas as pd

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")

def export_final_powerbi_data():
    print("=" * 60)
    print("🚀 CLIMATE INTELLIGENCE — POWER BI EXPORT")
    print("=" * 60)

    # Load data
    try:
        fact = pd.read_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"))
        risk = pd.read_csv(os.path.join(OUTPUT_DIR, "risk_scores.csv"))
        forecasts = pd.read_csv(os.path.join(OUTPUT_DIR, "forecasts.csv"))
    except FileNotFoundError as e:
        print(f"❌ Missing file: {e}. Please run the pipeline scripts first.")
        return

    print("🔗 Merging Fact Table + Risk Scores...")
    # Add risk_score to fact table
    risk_subset = risk[["iso_code", "year", "risk_score"]].copy()
    final_df = fact.merge(risk_subset, on=["iso_code", "year"], how="left")

    print("🔗 Merging Forecast Data...")
    # Add forecast_co2 (where year matches, which mostly applies to future years)
    # The requirement wants country, year, forecast_co2. We need to union historical data and forecasts.
    # Actually, fact_climate_metrics contains historical. forecasts.csv contains future.
    # To have everything in one file, let's append the future forecast rows.
    
    # Extract only future forecast rows (where actual CO2 is nan)
    future_forecasts = forecasts[forecasts["forecast_co2"].notna()].copy()
    
    # We just need common columns: iso_code, country, year, forecast_co2
    # Ensure they map correctly.
    
    # Add forecast column to final_df for historical (will be NaN or we can map last known value)
    final_df["forecast_co2"] = float("nan")
    
    # For forecasted years, we need to append them if they don't exist in fact table
    # Standardize column selection for requirements:
    # country, year, co2_per_capita, temperature_anomaly, renewables_share, risk_score, forecast_co2
    
    # Filter only future years not already in fact table
    max_year = final_df["year"].max()
    future_only = future_forecasts[future_forecasts["year"] > max_year].copy()
    
    # Append the future forecast rows
    append_df = pd.DataFrame({
        "country": future_only["country"],
        "iso_code": future_only["iso_code"],
        "year": future_only["year"],
        "co2_per_capita": float("nan"),
        "temperature_anomaly": future_only["forecast_temp_anomaly"] if "forecast_temp_anomaly" in future_only.columns else float("nan"),
        "renewables_share": float("nan"),
        "risk_score": float("nan"),
        "forecast_co2": future_only["forecast_co2"]
    })
    
    # Concat
    final_combined = pd.concat([final_df, append_df], ignore_index=True)

    # Select strictly the requested columns for Power BI
    final_output = final_combined[[
        "country", 
        "year", 
        "co2_per_capita", 
        "temperature_anomaly", 
        "renewables_share", 
        "risk_score", 
        "forecast_co2"
    ]].copy()
    
    # Ensure sorted cleanly
    final_output = final_output.sort_values(by=["country", "year"])

    # Output to CSV
    output_path = os.path.join(OUTPUT_DIR, "final_climate_data.csv")
    final_output.to_csv(output_path, index=False)
    
    print(f"✅ Created {output_path}")
    print(f"   Rows: {len(final_output)}")
    print(f"   Columns: {list(final_output.columns)}")
    print("=" * 60)

if __name__ == "__main__":
    export_final_powerbi_data()
