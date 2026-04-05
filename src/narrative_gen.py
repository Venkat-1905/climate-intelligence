"""
Climate Intelligence System — Narrative Generation Engine
============================================================
Auto-generates plain-English narrative summaries per country.
Turns metrics into actionable, recruiter-friendly insights.

Outputs:
  - outputs/narratives.csv
  - outputs/narrative_report.txt (full text report)
"""

import os
import sys
import pandas as pd
import numpy as np

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def risk_level(score):
    if pd.isna(score):
        return "unknown"
    if score > 70:
        return "high"
    if score > 40:
        return "moderate"
    return "low"


def trend_word(growth_rate):
    if pd.isna(growth_rate):
        return "stable"
    if growth_rate > 5:
        return "rapidly increasing"
    if growth_rate > 1:
        return "increasing"
    if growth_rate > -1:
        return "stable"
    if growth_rate > -5:
        return "decreasing"
    return "rapidly declining"


def generate_country_narrative(row):
    """
    Generate a narrative summary for a single country-year row.

    Returns:
        str: Plain-English narrative summary
    """
    country = row.get("country", "Unknown")
    iso = row.get("iso_code", "???")
    year = int(row.get("year", 0))
    risk_score = row.get("risk_score", None)
    co2_pc = row.get("co2_per_capita", None)
    coal = row.get("coal_share", None)
    renewables = row.get("renewables_share", None)
    vulnerability = row.get("vulnerability_score", None)
    temp_anomaly = row.get("temperature_anomaly", None)
    co2_growth = row.get("co2_growth_rate", None)
    energy_int = row.get("energy_intensity", None)
    sustainability = row.get("sustainability_score", None)

    parts = []

    # Opening sentence with risk level
    rlevel = risk_level(risk_score)
    if risk_score is not None:
        parts.append(
            f"{country}'s climate risk is {rlevel} "
            f"(score: {risk_score:.1f}/100 as of {year})."
        )
    else:
        parts.append(f"{country}: climate risk data as of {year}.")

    # CO2 per capita
    if co2_pc is not None:
        if co2_pc > 15:
            parts.append(
                f"Per-capita emissions are very high at {co2_pc:.1f} tonnes, "
                f"significantly above the global average."
            )
        elif co2_pc > 5:
            parts.append(
                f"Per-capita emissions stand at {co2_pc:.1f} tonnes."
            )
        else:
            parts.append(
                f"Per-capita emissions are relatively low at {co2_pc:.1f} tonnes."
            )

    # Coal dependency
    if coal is not None and coal > 30:
        parts.append(
            f"Coal accounts for {coal:.0f}% of CO₂ emissions, "
            f"indicating heavy fossil fuel dependency."
        )

    # Renewables
    if renewables is not None:
        if renewables > 40:
            parts.append(
                f"Renewable energy makes up {renewables:.0f}% of the energy mix — "
                f"a strong position for energy transition."
            )
        elif renewables < 15:
            parts.append(
                f"Renewable energy adoption is low at {renewables:.0f}%, "
                f"presenting a major opportunity for decarbonization."
            )

    # Emission trend
    trend = trend_word(co2_growth)
    if co2_growth is not None and abs(co2_growth) > 1:
        parts.append(f"Emissions are {trend} ({co2_growth:+.1f}% year-over-year).")

    # Vulnerability
    if vulnerability is not None and vulnerability > 0.5:
        parts.append(
            f"Climate vulnerability is elevated ({vulnerability:.3f}), "
            f"suggesting urgent need for adaptation investment."
        )

    # Temperature anomaly context
    if temp_anomaly is not None and temp_anomaly > 1.0:
        parts.append(
            f"Global temperature anomaly reached {temp_anomaly:.2f}°C, "
            f"reinforcing the urgency of emissions reduction."
        )

    # Policy recommendation summary
    recs = []
    if coal is not None and coal > 30:
        recs.append("phasing out coal")
    if renewables is not None and renewables < 20:
        recs.append("accelerating renewable adoption")
    if co2_pc is not None and co2_pc > 10:
        recs.append("implementing carbon pricing")
    if vulnerability is not None and vulnerability > 0.5:
        recs.append("investing in climate adaptation")

    if recs:
        parts.append(f"Key priorities: {', '.join(recs)}.")

    return " ".join(parts)


def generate_global_narrative(fact_df):
    """Generate a global summary narrative."""
    latest_year = fact_df["year"].max()
    latest = fact_df[fact_df["year"] == latest_year]

    total_co2 = latest["co2"].sum() / 1000  # Mt to Gt
    avg_risk = latest["risk_score"].mean() if "risk_score" in latest.columns else None
    avg_temp = latest["temperature_anomaly"].mean() if "temperature_anomaly" in latest.columns else None
    avg_renewables = latest["renewables_share"].mean() if "renewables_share" in latest.columns else None
    n_countries = latest["iso_code"].nunique()

    # Top emitters
    if "co2" in latest.columns:
        top3 = latest.nlargest(3, "co2")
        top3_names = ", ".join(top3["country"].tolist())
    else:
        top3_names = "N/A"

    parts = [
        f"GLOBAL CLIMATE INTELLIGENCE SUMMARY ({latest_year})",
        f"{'─' * 50}",
        f"Global CO₂ emissions totalled {total_co2:.1f} Gt across {n_countries} countries.",
        f"The top three emitters were {top3_names}.",
    ]

    if avg_temp is not None:
        parts.append(f"Average global temperature anomaly: {avg_temp:.2f}°C above pre-industrial baseline.")
    if avg_renewables is not None:
        parts.append(f"Average renewable energy share: {avg_renewables:.1f}%.")
    if avg_risk is not None:
        parts.append(f"Average climate risk score: {avg_risk:.1f}/100.")

    # High risk count
    if "risk_score" in latest.columns:
        high_risk = (latest["risk_score"] > 70).sum()
        parts.append(f"{high_risk} countries classified as high-risk (score > 70).")

    return "\n".join(parts)


def run_narrative_generation():
    """Execute narrative generation pipeline."""
    print("=" * 60)
    print("📝 CLIMATE INTELLIGENCE — NARRATIVE GENERATION")
    print("=" * 60)

    # Load data
    fact_path = os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv")
    if not os.path.exists(fact_path):
        print("❌ fact_climate_metrics.csv not found. Run pipeline first.")
        sys.exit(1)

    fact = pd.read_csv(fact_path)
    latest_year = fact["year"].max()
    latest = fact[fact["year"] == latest_year].copy()

    # Generate per-country narratives
    print(f"\n📋 Generating narratives for {len(latest)} countries...")
    latest["narrative"] = latest.apply(generate_country_narrative, axis=1)

    # Save narratives
    narrative_df = latest[["iso_code", "country", "risk_score", "narrative"]].copy()
    narrative_df = narrative_df.sort_values("risk_score", ascending=False)
    narrative_path = os.path.join(OUTPUT_DIR, "narratives.csv")
    narrative_df.to_csv(narrative_path, index=False)
    print(f"   ✅ Saved {len(narrative_df)} country narratives: {narrative_path}")

    # Generate global narrative
    global_narrative = generate_global_narrative(fact)

    # Generate full text report
    report_lines = [
        "═" * 60,
        global_narrative,
        "═" * 60,
        "",
        "TOP 10 HIGHEST RISK COUNTRIES:",
        "─" * 50,
    ]

    for _, row in narrative_df.head(10).iterrows():
        report_lines.append(f"\n▸ {row['narrative']}")

    report_lines.extend([
        "",
        "═" * 60,
        "LOWEST RISK COUNTRIES (Bottom 5):",
        "─" * 50,
    ])

    for _, row in narrative_df.tail(5).iterrows():
        report_lines.append(f"\n▸ {row['narrative']}")

    # Save report
    report_text = "\n".join(report_lines)
    report_path = os.path.join(OUTPUT_DIR, "narrative_report.txt")
    with open(report_path, "w") as f:
        f.write(report_text)
    print(f"   ✅ Saved narrative report: {report_path}")

    # Print preview
    print(f"\n{'─' * 60}")
    print("📖 NARRATIVE PREVIEW (Top 3 risk countries):")
    print(f"{'─' * 60}")
    for _, row in narrative_df.head(3).iterrows():
        print(f"\n{row['narrative']}")

    print(f"\n{'=' * 60}")
    return narrative_df


if __name__ == "__main__":
    run_narrative_generation()
