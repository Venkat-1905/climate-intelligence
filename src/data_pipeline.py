"""
Climate Intelligence System — Data Engineering Pipeline
========================================================
Loads, cleans, merges, and transforms raw climate datasets into a star schema.

Outputs:
  - outputs/fact_climate_metrics.csv
  - outputs/dim_country.csv
  - outputs/dim_time.csv
"""

import os
import sys
import warnings
import pandas as pd
import numpy as np

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# Paths
# ──────────────────────────────────────────────
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)

# Aggregate / region names to exclude from country-level analysis
AGGREGATES = {
    "World", "Asia", "Europe", "Africa", "North America", "South America",
    "Oceania", "Antarctica", "European Union (27)", "European Union (28)",
    "High-income countries", "Low-income countries",
    "Lower-middle-income countries", "Upper-middle-income countries",
    "Asia (excl. China and India)", "Europe (excl. EU-27)",
    "Europe (excl. EU-28)", "North America (excl. USA)",
    "International transport", "Kuwaiti Oil Fires",
    "Non-OECD (GCP)", "OECD (GCP)",
}

# Continent mapping by ISO code prefix (simplified)
CONTINENT_MAP = {
    "AFG": "Asia", "ALB": "Europe", "DZA": "Africa", "AND": "Europe",
    "AGO": "Africa", "ATG": "North America", "ARG": "South America",
    "ARM": "Asia", "AUS": "Oceania", "AUT": "Europe", "AZE": "Asia",
    "BHS": "North America", "BHR": "Asia", "BGD": "Asia", "BRB": "North America",
    "BLR": "Europe", "BEL": "Europe", "BLZ": "North America", "BEN": "Africa",
    "BTN": "Asia", "BOL": "South America", "BIH": "Europe", "BWA": "Africa",
    "BRA": "South America", "BRN": "Asia", "BGR": "Europe", "BFA": "Africa",
    "BDI": "Africa", "KHM": "Asia", "CMR": "Africa", "CAN": "North America",
    "CPV": "Africa", "CAF": "Africa", "TCD": "Africa", "CHL": "South America",
    "CHN": "Asia", "COL": "South America", "COM": "Africa", "COG": "Africa",
    "COD": "Africa", "CRI": "North America", "CIV": "Africa", "HRV": "Europe",
    "CUB": "North America", "CYP": "Europe", "CZE": "Europe", "DNK": "Europe",
    "DJI": "Africa", "DMA": "North America", "DOM": "North America",
    "ECU": "South America", "EGY": "Africa", "SLV": "North America",
    "GNQ": "Africa", "ERI": "Africa", "EST": "Europe", "SWZ": "Africa",
    "ETH": "Africa", "FJI": "Oceania", "FIN": "Europe", "FRA": "Europe",
    "GAB": "Africa", "GMB": "Africa", "GEO": "Asia", "DEU": "Europe",
    "GHA": "Africa", "GRC": "Europe", "GRD": "North America", "GTM": "North America",
    "GIN": "Africa", "GNB": "Africa", "GUY": "South America", "HTI": "North America",
    "HND": "North America", "HUN": "Europe", "ISL": "Europe", "IND": "Asia",
    "IDN": "Asia", "IRN": "Asia", "IRQ": "Asia", "IRL": "Europe",
    "ISR": "Asia", "ITA": "Europe", "JAM": "North America", "JPN": "Asia",
    "JOR": "Asia", "KAZ": "Asia", "KEN": "Africa", "KIR": "Oceania",
    "PRK": "Asia", "KOR": "Asia", "KWT": "Asia", "KGZ": "Asia",
    "LAO": "Asia", "LVA": "Europe", "LBN": "Asia", "LSO": "Africa",
    "LBR": "Africa", "LBY": "Africa", "LIE": "Europe", "LTU": "Europe",
    "LUX": "Europe", "MDG": "Africa", "MWI": "Africa", "MYS": "Asia",
    "MDV": "Asia", "MLI": "Africa", "MLT": "Europe", "MHL": "Oceania",
    "MRT": "Africa", "MUS": "Africa", "MEX": "North America", "FSM": "Oceania",
    "MDA": "Europe", "MCO": "Europe", "MNG": "Asia", "MNE": "Europe",
    "MAR": "Africa", "MOZ": "Africa", "MMR": "Asia", "NAM": "Africa",
    "NRU": "Oceania", "NPL": "Asia", "NLD": "Europe", "NZL": "Oceania",
    "NIC": "North America", "NER": "Africa", "NGA": "Africa", "MKD": "Europe",
    "NOR": "Europe", "OMN": "Asia", "PAK": "Asia", "PLW": "Oceania",
    "PSE": "Asia", "PAN": "North America", "PNG": "Oceania", "PRY": "South America",
    "PER": "South America", "PHL": "Asia", "POL": "Europe", "PRT": "Europe",
    "QAT": "Asia", "ROU": "Europe", "RUS": "Europe", "RWA": "Africa",
    "KNA": "North America", "LCA": "North America", "VCT": "North America",
    "WSM": "Oceania", "SMR": "Europe", "STP": "Africa", "SAU": "Asia",
    "SEN": "Africa", "SRB": "Europe", "SYC": "Africa", "SLE": "Africa",
    "SGP": "Asia", "SVK": "Europe", "SVN": "Europe", "SLB": "Oceania",
    "SOM": "Africa", "ZAF": "Africa", "SSD": "Africa", "ESP": "Europe",
    "LKA": "Asia", "SDN": "Africa", "SUR": "South America", "SWE": "Europe",
    "CHE": "Europe", "SYR": "Asia", "TWN": "Asia", "TJK": "Asia",
    "TZA": "Africa", "THA": "Asia", "TLS": "Asia", "TGO": "Africa",
    "TON": "Oceania", "TTO": "North America", "TUN": "Africa", "TUR": "Asia",
    "TKM": "Asia", "TUV": "Oceania", "UGA": "Africa", "UKR": "Europe",
    "ARE": "Asia", "GBR": "Europe", "USA": "North America", "URY": "South America",
    "UZB": "Asia", "VUT": "Oceania", "VEN": "South America", "VNM": "Asia",
    "YEM": "Asia", "ZMB": "Africa", "ZWE": "Africa",
}

INCOME_GROUP_MAP = {
    # High income (sample)
    "USA": "High", "GBR": "High", "DEU": "High", "FRA": "High", "JPN": "High",
    "CAN": "High", "AUS": "High", "KOR": "High", "ITA": "High", "ESP": "High",
    "NLD": "High", "CHE": "High", "SWE": "High", "NOR": "High", "DNK": "High",
    "FIN": "High", "AUT": "High", "BEL": "High", "IRL": "High", "ISR": "High",
    "SGP": "High", "NZL": "High", "PRT": "High", "GRC": "High", "CZE": "High",
    "SVK": "High", "SVN": "High", "EST": "High", "LTU": "High", "LVA": "High",
    "HRV": "High", "POL": "High", "HUN": "High", "CHL": "High", "URY": "High",
    "SAU": "High", "ARE": "High", "QAT": "High", "KWT": "High", "BHR": "High",
    "OMN": "High", "TWN": "High", "ISL": "High", "LUX": "High", "MLT": "High",
    "CYP": "High",
    # Upper middle income
    "CHN": "Upper-middle", "BRA": "Upper-middle", "MEX": "Upper-middle",
    "RUS": "Upper-middle", "TUR": "Upper-middle", "ZAF": "Upper-middle",
    "MYS": "Upper-middle", "THA": "Upper-middle", "COL": "Upper-middle",
    "ARG": "Upper-middle", "PER": "Upper-middle", "ROU": "Upper-middle",
    "BGR": "Upper-middle", "KAZ": "Upper-middle", "IRN": "Upper-middle",
    "IRQ": "Upper-middle", "DZA": "Upper-middle", "ECU": "Upper-middle",
    "DOM": "Upper-middle", "JOR": "Upper-middle", "LBN": "Upper-middle",
    "AZE": "Upper-middle", "GEO": "Upper-middle", "ARM": "Upper-middle",
    "IDN": "Upper-middle", "JAM": "Upper-middle",
    # Lower middle income
    "IND": "Lower-middle", "BGD": "Lower-middle", "PAK": "Lower-middle",
    "VNM": "Lower-middle", "PHL": "Lower-middle", "EGY": "Lower-middle",
    "NGA": "Lower-middle", "KEN": "Lower-middle", "GHA": "Lower-middle",
    "UKR": "Lower-middle", "MDA": "Lower-middle", "UZB": "Lower-middle",
    "KGZ": "Lower-middle", "TJK": "Lower-middle", "MMR": "Lower-middle",
    "KHM": "Lower-middle", "LAO": "Lower-middle", "NPL": "Lower-middle",
    "LKA": "Lower-middle", "MAR": "Lower-middle", "TUN": "Lower-middle",
    "CMR": "Lower-middle", "SEN": "Lower-middle", "CIV": "Lower-middle",
    "ZMB": "Lower-middle", "TZA": "Lower-middle", "BOL": "Lower-middle",
    "HND": "Lower-middle", "NIC": "Lower-middle", "SLV": "Lower-middle",
    # Low income
    "AFG": "Low", "ETH": "Low", "MOZ": "Low", "MDG": "Low", "MWI": "Low",
    "MLI": "Low", "BFA": "Low", "NER": "Low", "TCD": "Low", "SSD": "Low",
    "SDN": "Low", "SOM": "Low", "COD": "Low", "CAF": "Low", "BDI": "Low",
    "SLE": "Low", "LBR": "Low", "GNB": "Low", "ERI": "Low", "HTI": "Low",
    "RWA": "Low", "UGA": "Low", "YEM": "Low",
}


def load_owid():
    """Load and clean Our World in Data CO₂ dataset."""
    print("📦 Loading OWID CO₂ dataset...")
    df = pd.read_csv(os.path.join(DATA_DIR, "owid-co2-data.csv"))

    # Filter out aggregates
    df = df[~df["country"].isin(AGGREGATES)].copy()

    # Keep 1990+
    df = df[df["year"] >= 1990].copy()

    # Drop rows with no iso_code (non-country entities)
    df = df.dropna(subset=["iso_code"])

    # Select key columns
    cols = [
        "country", "year", "iso_code", "population", "gdp",
        "co2", "co2_per_capita", "co2_growth_prct",
        "coal_co2", "coal_co2_per_capita",
        "gas_co2", "oil_co2",
        "energy_per_capita", "energy_per_gdp",
        "primary_energy_consumption",
        "methane", "nitrous_oxide", "total_ghg",
        "share_global_co2",
        "temperature_change_from_ghg",
        "trade_co2",
    ]
    existing_cols = [c for c in cols if c in df.columns]
    df = df[existing_cols].copy()

    print(f"   ✅ OWID loaded: {len(df)} rows, {df['iso_code'].nunique()} countries")
    return df


def load_nasa_temp():
    """Load NASA GISTEMP global temperature anomaly data."""
    print("🌡️  Loading NASA GISTEMP data...")
    df = pd.read_csv(os.path.join(DATA_DIR, "GLB.Ts+dSST.csv"), skiprows=1)

    # Keep Year and annual mean (J-D column)
    df = df[["Year", "J-D"]].copy()
    df.columns = ["year", "temperature_anomaly"]

    # Convert to numeric
    df["temperature_anomaly"] = pd.to_numeric(df["temperature_anomaly"], errors="coerce")
    df["year"] = pd.to_numeric(df["year"], errors="coerce")

    # Drop rows where year couldn't be parsed
    df = df.dropna(subset=["year"])
    df["year"] = df["year"].astype(int)

    # Make sure we use the GHCNv4/ERSSTv5 dataset (the second table in the CSV)
    # by keeping the 'last' occurrence of each year.
    df = df.drop_duplicates(subset=["year"], keep="last")

    # Filter 1990+
    df = df[df["year"] >= 1990].copy()

    print(f"   ✅ NASA GISTEMP loaded: {len(df)} years ({df['year'].min()}–{df['year'].max()})")
    return df


def load_world_bank():
    """Load and merge 3 World Bank indicator files."""
    print("🏦 Loading World Bank indicators...")
    wb_files = {
        "electricity_access_pct": "WB_WDI_EG_ELC_ACCS_ZS.csv",
        "electric_power_kwh_pc": "WB_WDI_EG_USE_ELEC_KH_PC.csv",
        "energy_use_kg_oil_pc": "WB_WDI_EG_USE_PCAP_KG_OE.csv",
    }

    dfs = []
    for col_name, filename in wb_files.items():
        filepath = os.path.join(DATA_DIR, filename)
        raw = pd.read_csv(filepath)

        # Extract relevant columns
        subset = raw[["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]].copy()
        subset.columns = ["iso_code", "year", col_name]
        subset["year"] = pd.to_numeric(subset["year"], errors="coerce")
        subset[col_name] = pd.to_numeric(subset[col_name], errors="coerce")
        subset = subset.dropna(subset=["year"])
        subset["year"] = subset["year"].astype(int)

        # Filter 1990+
        subset = subset[subset["year"] >= 1990].copy()
        dfs.append(subset)

    # Merge all WB indicators on iso_code + year
    wb = dfs[0]
    for d in dfs[1:]:
        wb = wb.merge(d, on=["iso_code", "year"], how="outer")

    print(f"   ✅ World Bank loaded: {len(wb)} rows, {wb['iso_code'].nunique()} countries")
    return wb


def load_ndgain():
    """Load ND-GAIN country index data (gain score + vulnerability)."""
    print("🌍 Loading ND-GAIN index data...")
    gain_path = os.path.join(DATA_DIR, "ndgain_countryindex_2026", "resources")

    result_dfs = []

    # Load gain score
    gain_df = pd.read_csv(os.path.join(gain_path, "gain", "gain.csv"))
    gain_melted = gain_df.melt(id_vars=["ISO3", "Name"], var_name="year", value_name="ndgain_score")
    gain_melted["year"] = pd.to_numeric(gain_melted["year"], errors="coerce")
    gain_melted = gain_melted.dropna(subset=["year"])
    gain_melted["year"] = gain_melted["year"].astype(int)
    gain_melted = gain_melted.rename(columns={"ISO3": "iso_code"})
    gain_melted = gain_melted[["iso_code", "year", "ndgain_score"]]
    result_dfs.append(gain_melted)

    # Load vulnerability score
    vuln_df = pd.read_csv(os.path.join(gain_path, "vulnerability", "vulnerability.csv"))
    vuln_melted = vuln_df.melt(id_vars=["ISO3", "Name"], var_name="year", value_name="vulnerability_score")
    vuln_melted["year"] = pd.to_numeric(vuln_melted["year"], errors="coerce")
    vuln_melted = vuln_melted.dropna(subset=["year"])
    vuln_melted["year"] = vuln_melted["year"].astype(int)
    vuln_melted = vuln_melted.rename(columns={"ISO3": "iso_code"})
    vuln_melted = vuln_melted[["iso_code", "year", "vulnerability_score"]]
    result_dfs.append(vuln_melted)

    # Load readiness score
    ready_df = pd.read_csv(os.path.join(gain_path, "readiness", "readiness.csv"))
    ready_melted = ready_df.melt(id_vars=["ISO3", "Name"], var_name="year", value_name="readiness_score")
    ready_melted["year"] = pd.to_numeric(ready_melted["year"], errors="coerce")
    ready_melted = ready_melted.dropna(subset=["year"])
    ready_melted["year"] = ready_melted["year"].astype(int)
    ready_melted = ready_melted.rename(columns={"ISO3": "iso_code"})
    ready_melted = ready_melted[["iso_code", "year", "readiness_score"]]
    result_dfs.append(ready_melted)

    # Merge all ND-GAIN data
    ndgain = result_dfs[0]
    for d in result_dfs[1:]:
        ndgain = ndgain.merge(d, on=["iso_code", "year"], how="outer")

    # Filter 1990+
    ndgain = ndgain[ndgain["year"] >= 1990].copy()

    print(f"   ✅ ND-GAIN loaded: {len(ndgain)} rows, {ndgain['iso_code'].nunique()} countries")
    return ndgain


def build_fact_table(owid, nasa, wb, ndgain):
    """Merge all sources into the fact_climate_metrics table."""
    print("\n🔗 Building fact_climate_metrics...")

    # Merge OWID + NASA (global temperature overlay)
    fact = owid.merge(nasa, on="year", how="left")

    # Merge World Bank
    fact = fact.merge(wb, on=["iso_code", "year"], how="left")

    # Merge ND-GAIN
    fact = fact.merge(ndgain, on=["iso_code", "year"], how="left")

    # ──────────────────────────────────────────────
    # Feature Engineering
    # ──────────────────────────────────────────────
    print("⚙️  Feature engineering...")

    # CO₂ growth rate (year-over-year)
    fact = fact.sort_values(["iso_code", "year"])
    fact["co2_growth_rate"] = fact.groupby("iso_code")["co2"].pct_change() * 100

    # Energy intensity
    fact["energy_intensity"] = np.where(
        fact["gdp"] > 0,
        fact["energy_per_capita"] / (fact["gdp"] / fact["population"]),
        np.nan,
    )

    # Coal share (coal_co2 / co2 * 100)
    fact["coal_share"] = np.where(
        fact["co2"] > 0,
        (fact["coal_co2"] / fact["co2"]) * 100,
        np.nan,
    )

    # Gas share
    fact["gas_share"] = np.where(
        fact["co2"] > 0,
        (fact["gas_co2"] / fact["co2"]) * 100,
        np.nan,
    )

    # Oil share
    fact["oil_share"] = np.where(
        fact["co2"] > 0,
        (fact["oil_co2"] / fact["co2"]) * 100,
        np.nan,
    )

    # Renewables share (estimated: 100 - fossil shares)
    fact["renewables_share"] = 100 - fact[["coal_share", "gas_share", "oil_share"]].sum(axis=1, min_count=1)
    fact["renewables_share"] = fact["renewables_share"].clip(lower=0, upper=100)

    # Sustainability score
    fact["sustainability_score"] = fact["renewables_share"] - fact["coal_share"]

    # Rolling 5-year average for co2_per_capita
    fact["co2_per_capita_5yr_avg"] = (
        fact.groupby("iso_code")["co2_per_capita"]
        .transform(lambda x: x.rolling(5, min_periods=2).mean())
    )

    # Forward fill within country groups for sparse data
    fill_cols = [
        "electricity_access_pct", "electric_power_kwh_pc", "energy_use_kg_oil_pc",
        "ndgain_score", "vulnerability_score", "readiness_score",
    ]
    for col in fill_cols:
        if col in fact.columns:
            fact[col] = fact.groupby("iso_code")[col].transform(
                lambda x: x.ffill().bfill()
            )

    # Drop any duplicates that might have sneaked in
    fact = fact.drop_duplicates(subset=["iso_code", "year"], keep="first")

    # Add country_id
    country_ids = {iso: i + 1 for i, iso in enumerate(sorted(fact["iso_code"].unique()))}
    fact["country_id"] = fact["iso_code"].map(country_ids)

    print(f"   ✅ Fact table built: {len(fact)} rows, {fact['iso_code'].nunique()} countries, {fact.shape[1]} columns")
    return fact, country_ids


def build_dim_country(fact, country_ids):
    """Build dim_country dimension table."""
    print("🗺️  Building dim_country...")

    countries = fact.groupby("iso_code").agg(
        country_name=("country", "first"),
    ).reset_index()

    countries["country_id"] = countries["iso_code"].map(country_ids)
    countries["continent"] = countries["iso_code"].map(CONTINENT_MAP).fillna("Other")
    countries["income_group"] = countries["iso_code"].map(INCOME_GROUP_MAP).fillna("Unknown")

    dim_country = countries[["country_id", "country_name", "continent", "income_group", "iso_code"]].copy()
    print(f"   ✅ dim_country: {len(dim_country)} countries")
    return dim_country


def build_dim_time(fact):
    """Build dim_time dimension table."""
    print("📅 Building dim_time...")

    years = sorted(fact["year"].unique())
    dim_time = pd.DataFrame({"year": years})
    dim_time["date"] = pd.to_datetime(dim_time["year"].astype(str) + "-01-01").dt.date
    dim_time["decade"] = (dim_time["year"] // 10) * 10
    dim_time["period"] = dim_time["decade"].astype(str) + "–" + (dim_time["decade"] + 9).astype(str)

    print(f"   ✅ dim_time: {len(dim_time)} years ({dim_time['year'].min()}–{dim_time['year'].max()})")
    return dim_time


def build_dim_energy_type():
    """Build dim_energy_type dimension table."""
    print("⚡ Building dim_energy_type...")
    
    data = [
        {"energy_type": "coal", "category": "fossil"},
        {"energy_type": "gas", "category": "fossil"},
        {"energy_type": "oil", "category": "fossil"},
        {"energy_type": "renewables", "category": "clean"},
        {"energy_type": "nuclear", "category": "clean"}
    ]
    dim_energy = pd.DataFrame(data)
    
    print(f"   ✅ dim_energy_type: {len(dim_energy)} specific energy types")
    return dim_energy


def run_pipeline():
    """Execute the full data pipeline."""
    print("=" * 60)
    print("🚀 CLIMATE INTELLIGENCE — DATA PIPELINE")
    print("=" * 60)

    # Load sources
    owid = load_owid()
    nasa = load_nasa_temp()
    wb = load_world_bank()
    ndgain = load_ndgain()

    # Build star schema
    fact, country_ids = build_fact_table(owid, nasa, wb, ndgain)
    dim_country = build_dim_country(fact, country_ids)
    dim_time = build_dim_time(fact)
    dim_energy = build_dim_energy_type()

    # Save outputs
    print("\n💾 Saving outputs...")
    fact.to_csv(os.path.join(OUTPUT_DIR, "fact_climate_metrics.csv"), index=False)
    dim_country.to_csv(os.path.join(OUTPUT_DIR, "dim_country.csv"), index=False)
    dim_time.to_csv(os.path.join(OUTPUT_DIR, "dim_time.csv"), index=False)
    dim_energy.to_csv(os.path.join(OUTPUT_DIR, "dim_energy_type.csv"), index=False)

    # Summary stats
    print("\n" + "=" * 60)
    print("📊 PIPELINE SUMMARY")
    print("=" * 60)
    print(f"  Fact table:   {len(fact):>8,} rows × {fact.shape[1]} cols")
    print(f"  Countries:    {dim_country.shape[0]:>8,}")
    print(f"  Years:        {dim_time['year'].min()} – {dim_time['year'].max()}")
    print(f"  Output dir:   {OUTPUT_DIR}")
    print("=" * 60)

    return fact, dim_country, dim_time


if __name__ == "__main__":
    run_pipeline()
