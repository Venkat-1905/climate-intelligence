"""
Climate Intelligence System — Data Quality & Validation Layer
================================================================
Validates raw data before it enters the pipeline.
Checks schemas, null rates, outliers, and data freshness.

Inspired by Great Expectations but implemented as a lightweight,
dependency-free framework for portability.

Outputs:
  - Validation report (pass/fail per check)
  - Flagged rows written to outputs/data_quality_report.csv
"""

import os
import sys
import pandas as pd
import numpy as np
from datetime import datetime

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(OUTPUT_DIR, exist_ok=True)


class ValidationResult:
    """Single validation check result."""
    def __init__(self, check_name, dataset, passed, details="", severity="WARNING"):
        self.check_name = check_name
        self.dataset = dataset
        self.passed = passed
        self.details = details
        self.severity = severity  # INFO, WARNING, CRITICAL
        self.timestamp = datetime.now().isoformat()

    def __repr__(self):
        status = "✅ PASS" if self.passed else f"❌ FAIL [{self.severity}]"
        return f"  {status}  {self.dataset:>20s} | {self.check_name}: {self.details}"


class DataQualityValidator:
    """
    Lightweight data quality framework.
    Runs schema, null, outlier, and freshness checks on raw datasets.
    """

    def __init__(self):
        self.results = []
        self.flagged_rows = []

    def add_result(self, result):
        self.results.append(result)
        print(result)

    # ─── Schema Validation ───
    def check_schema(self, df, name, required_columns, optional_columns=None):
        """Verify expected columns exist."""
        missing = [c for c in required_columns if c not in df.columns]
        if missing:
            self.add_result(ValidationResult(
                "schema_check", name, False,
                f"Missing required columns: {missing}", "CRITICAL"
            ))
            return False
        else:
            extra = set(df.columns) - set(required_columns) - set(optional_columns or [])
            self.add_result(ValidationResult(
                "schema_check", name, True,
                f"All {len(required_columns)} required columns present, {len(extra)} extra"
            ))
            return True

    # ─── Null Rate Check ───
    def check_null_rates(self, df, name, critical_columns, max_null_pct=0.5):
        """Flag columns with null rates above threshold."""
        for col in critical_columns:
            if col not in df.columns:
                continue
            null_rate = df[col].isna().mean()
            passed = null_rate <= max_null_pct
            self.add_result(ValidationResult(
                f"null_rate[{col}]", name, passed,
                f"{null_rate:.1%} null (threshold: {max_null_pct:.0%})",
                "WARNING" if null_rate <= 0.8 else "CRITICAL"
            ))

    # ─── Duplicate Check ───
    def check_duplicates(self, df, name, key_columns):
        """Check for duplicate rows on key columns."""
        dupes = df.duplicated(subset=key_columns, keep=False).sum()
        passed = dupes == 0
        self.add_result(ValidationResult(
            "duplicate_check", name, passed,
            f"{dupes} duplicate rows on {key_columns}",
            "WARNING" if dupes < 100 else "CRITICAL"
        ))

    # ─── Range / Outlier Check ───
    def check_value_range(self, df, name, column, min_val=None, max_val=None):
        """Flag values outside expected range."""
        if column not in df.columns:
            return
        series = pd.to_numeric(df[column], errors="coerce").dropna()
        if len(series) == 0:
            return

        violations = 0
        if min_val is not None:
            violations += (series < min_val).sum()
        if max_val is not None:
            violations += (series > max_val).sum()

        passed = violations == 0
        self.add_result(ValidationResult(
            f"range_check[{column}]", name, passed,
            f"{violations} values outside [{min_val}, {max_val}]",
            "WARNING"
        ))

    # ─── Statistical Outlier Detection ───
    def check_outliers_iqr(self, df, name, column, multiplier=3.0):
        """Flag statistical outliers using IQR method."""
        if column not in df.columns:
            return
        series = df[column].dropna()
        if len(series) < 10:
            return

        q1 = series.quantile(0.25)
        q3 = series.quantile(0.75)
        iqr = q3 - q1
        lower = q1 - multiplier * iqr
        upper = q3 + multiplier * iqr

        outliers = ((series < lower) | (series > upper)).sum()
        pct = outliers / len(series)
        passed = pct < 0.05  # Less than 5% outliers is OK

        self.add_result(ValidationResult(
            f"outlier_iqr[{column}]", name, passed,
            f"{outliers} outliers ({pct:.1%}) outside [{lower:.2f}, {upper:.2f}]",
            "INFO" if passed else "WARNING"
        ))

    # ─── Data Freshness Check ───
    def check_freshness(self, df, name, year_column="year", min_expected_year=2020):
        """Verify data is sufficiently recent."""
        if year_column not in df.columns:
            return
        years = pd.to_numeric(df[year_column], errors="coerce").dropna()
        if len(years) == 0:
            return
        max_year = int(years.max())
        passed = max_year >= min_expected_year
        self.add_result(ValidationResult(
            "freshness_check", name, passed,
            f"Latest year: {max_year} (expected >= {min_expected_year})",
            "WARNING" if not passed else "INFO"
        ))

    # ─── Row Count Check ───
    def check_row_count(self, df, name, min_rows=100):
        """Ensure dataset has minimum expected rows."""
        count = len(df)
        passed = count >= min_rows
        self.add_result(ValidationResult(
            "row_count", name, passed,
            f"{count:,} rows (minimum: {min_rows:,})",
            "CRITICAL" if not passed else "INFO"
        ))

    # ─── Type Check ───
    def check_numeric_columns(self, df, name, columns):
        """Verify specified columns are numeric."""
        for col in columns:
            if col not in df.columns:
                continue
            is_numeric = pd.api.types.is_numeric_dtype(df[col])
            if not is_numeric:
                # Try coercion
                coerced = pd.to_numeric(df[col], errors="coerce")
                coerce_fails = coerced.isna().sum() - df[col].isna().sum()
                self.add_result(ValidationResult(
                    f"type_check[{col}]", name, coerce_fails == 0,
                    f"{coerce_fails} non-numeric values found",
                    "WARNING"
                ))
            else:
                self.add_result(ValidationResult(
                    f"type_check[{col}]", name, True,
                    f"Column is {df[col].dtype}", "INFO"
                ))

    # ─── Summary Report ───
    def generate_report(self):
        """Generate and save validation summary."""
        total = len(self.results)
        passed = sum(1 for r in self.results if r.passed)
        failed = total - passed
        critical = sum(1 for r in self.results if not r.passed and r.severity == "CRITICAL")

        print("\n" + "=" * 60)
        print("📋 DATA QUALITY REPORT")
        print("=" * 60)
        print(f"  Total checks:    {total}")
        print(f"  Passed:          {passed} ✅")
        print(f"  Failed:          {failed} ❌")
        print(f"  Critical fails:  {critical} 🔴")
        print(f"  Score:           {passed/total*100:.0f}%")
        print("=" * 60)

        # Save report
        report_df = pd.DataFrame([{
            "timestamp": r.timestamp,
            "dataset": r.dataset,
            "check": r.check_name,
            "passed": r.passed,
            "severity": r.severity,
            "details": r.details,
        } for r in self.results])

        report_path = os.path.join(OUTPUT_DIR, "data_quality_report.csv")
        report_df.to_csv(report_path, index=False)
        print(f"\n💾 Report saved: {report_path}")

        # Return pass/fail
        if critical > 0:
            print("🔴 CRITICAL failures detected — pipeline should not proceed!")
            return False
        return True


def validate_owid():
    """Validate OWID CO₂ dataset."""
    path = os.path.join(DATA_DIR, "owid-co2-data.csv")
    if not os.path.exists(path):
        print("❌ owid-co2-data.csv not found!")
        return None

    df = pd.read_csv(path)
    v = DataQualityValidator()

    v.check_row_count(df, "OWID", min_rows=1000)
    v.check_schema(df, "OWID",
        required_columns=["country", "year", "iso_code", "co2", "population"],
        optional_columns=["gdp", "co2_per_capita", "coal_co2", "gas_co2",
                         "oil_co2", "methane", "nitrous_oxide", "total_ghg"]
    )
    v.check_null_rates(df, "OWID",
        critical_columns=["country", "year", "co2"],
        max_null_pct=0.3
    )
    v.check_duplicates(df, "OWID", key_columns=["country", "year"])
    v.check_freshness(df, "OWID", min_expected_year=2020)
    v.check_value_range(df, "OWID", "year", min_val=1750, max_val=2030)
    v.check_value_range(df, "OWID", "co2", min_val=0)
    v.check_outliers_iqr(df, "OWID", "co2_per_capita")
    v.check_numeric_columns(df, "OWID", ["co2", "population", "gdp"])

    return v


def validate_nasa():
    """Validate NASA GISTEMP dataset."""
    path = os.path.join(DATA_DIR, "GLB.Ts+dSST.csv")
    if not os.path.exists(path):
        print("❌ GLB.Ts+dSST.csv not found!")
        return None

    df = pd.read_csv(path, skiprows=1)
    v = DataQualityValidator()

    v.check_row_count(df, "NASA", min_rows=50)
    v.check_schema(df, "NASA",
        required_columns=["Year", "J-D"]
    )
    v.check_freshness(df, "NASA", year_column="Year", min_expected_year=2020)
    v.check_value_range(df, "NASA", "Year", min_val=1880, max_val=2030)

    return v


def validate_world_bank():
    """Validate World Bank indicator files."""
    files = {
        "WB_Electricity": "WB_WDI_EG_ELC_ACCS_ZS.csv",
        "WB_ElecPower": "WB_WDI_EG_USE_ELEC_KH_PC.csv",
        "WB_EnergyUse": "WB_WDI_EG_USE_PCAP_KG_OE.csv",
    }

    validators = []
    for name, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            print(f"❌ {filename} not found!")
            continue

        df = pd.read_csv(path)
        v = DataQualityValidator()

        v.check_row_count(df, name, min_rows=500)
        v.check_schema(df, name,
            required_columns=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"]
        )
        v.check_null_rates(df, name,
            critical_columns=["REF_AREA", "TIME_PERIOD", "OBS_VALUE"],
            max_null_pct=0.3
        )
        v.check_numeric_columns(df, name, ["TIME_PERIOD", "OBS_VALUE"])
        validators.append(v)

    return validators


def validate_ndgain():
    """Validate ND-GAIN dataset."""
    gain_path = os.path.join(DATA_DIR, "ndgain_countryindex_2026", "resources")
    v = DataQualityValidator()

    for subdir, col_name in [("gain", "gain"), ("vulnerability", "vulnerability"),
                              ("readiness", "readiness")]:
        path = os.path.join(gain_path, subdir, f"{subdir}.csv")
        if not os.path.exists(path):
            print(f"❌ {subdir}.csv not found!")
            continue

        df = pd.read_csv(path)
        v.check_row_count(df, f"NDGAIN_{subdir}", min_rows=100)
        v.check_schema(df, f"NDGAIN_{subdir}",
            required_columns=["ISO3", "Name"]
        )

    return v


def run_all_validations():
    """Execute all data quality validations."""
    print("=" * 60)
    print("🔍 CLIMATE INTELLIGENCE — DATA QUALITY VALIDATION")
    print("=" * 60)

    all_results = []

    # Validate each source
    print("\n📦 Validating OWID CO₂ data...")
    v_owid = validate_owid()
    if v_owid:
        all_results.extend(v_owid.results)

    print("\n🌡️  Validating NASA GISTEMP data...")
    v_nasa = validate_nasa()
    if v_nasa:
        all_results.extend(v_nasa.results)

    print("\n🏦 Validating World Bank data...")
    wb_validators = validate_world_bank()
    for v in wb_validators:
        all_results.extend(v.results)

    print("\n🌍 Validating ND-GAIN data...")
    v_ndgain = validate_ndgain()
    if v_ndgain:
        all_results.extend(v_ndgain.results)

    # Generate combined report
    combined = DataQualityValidator()
    combined.results = all_results
    passed = combined.generate_report()

    return passed


def version_dataset():
    """Create a timestamped version snapshot of current processed outputs."""
    version_dir = os.path.join(BASE_DIR, "data", "versioned")
    os.makedirs(version_dir, exist_ok=True)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    snapshot_dir = os.path.join(version_dir, f"v_{timestamp}")
    os.makedirs(snapshot_dir, exist_ok=True)

    import shutil
    files_to_version = [
        "fact_climate_metrics.csv", "dim_country.csv", "dim_time.csv",
        "forecasts.csv", "risk_scores.csv", "clusters.csv",
    ]

    versioned = 0
    for f in files_to_version:
        src = os.path.join(OUTPUT_DIR, f)
        if os.path.exists(src):
            shutil.copy2(src, os.path.join(snapshot_dir, f))
            versioned += 1

    # Write version metadata
    meta = {
        "version": timestamp,
        "files_versioned": versioned,
        "created_at": datetime.now().isoformat(),
    }
    meta_path = os.path.join(snapshot_dir, "version_meta.json")
    import json
    with open(meta_path, "w") as f:
        json.dump(meta, f, indent=2)

    print(f"\n📦 Dataset versioned: {snapshot_dir} ({versioned} files)")
    return snapshot_dir


if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description="Data Quality Validation")
    parser.add_argument("--version", action="store_true", help="Create versioned snapshot")
    args = parser.parse_args()

    passed = run_all_validations()

    if args.version:
        version_dataset()

    sys.exit(0 if passed else 1)
