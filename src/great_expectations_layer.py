"""
Climate Intelligence System — Data Quality via Great Expectations
================================================================
Validates raw data (CSVs) using Great Expectations locally.
This script sets up an in-memory ephemeral context to 
validate schema, null limits, and outliers without needing
a heavy `.great_expectations/` directory setup.

Outputs:
  - Prints validation results locally per dataset
  - Returns pass/fail status
"""

import os
import sys
import pandas as pd
import great_expectations as gx

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR = os.path.join(BASE_DIR, "data")
OUTPUT_DIR = os.path.join(BASE_DIR, "outputs")


def validate_owid(context):
    """Validate OWID CO2 dataset using GE V1 API."""
    path = os.path.join(DATA_DIR, "owid-co2-data.csv")
    if not os.path.exists(path):
        print("❌ owid-co2-data.csv not found!")
        return False

    print("\n📦 Running Great Expectations on OWID...")
    
    # 1. Add Data Source & Asset
    ds = context.data_sources.add_pandas("owid_source")
    asset = ds.add_csv_asset("owid_co2", filepath_or_buffer=path)
    batch_def = asset.add_batch_definition_whole_dataframe("my_batch")
    batch = batch_def.get_batch()

    # 2. Build Expectation Suite
    suite = gx.ExpectationSuite(name="owid_suite")
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="country"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="year"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="co2"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="country"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToBeBetween(column="year", min_value=1750, max_value=2030))
    
    # 3. Validate
    result = batch.validate(suite)
    passed = result.success
    print(f"   {'✅ PASS' if passed else '❌ FAIL'}: OWID Validation")
    return passed


def validate_nasa(context):
    """Validate NASA GISTEMP dataset."""
    path = os.path.join(DATA_DIR, "GLB.Ts+dSST.csv")
    if not os.path.exists(path):
        print("❌ GLB.Ts+dSST.csv not found!")
        return False

    print("\n🌡️  Running Great Expectations on NASA GISTEMP...")
    
    ds = context.data_sources.add_pandas("nasa_source")
    # skiprows=1 is needed for NASA
    import pandas as pd
    df = pd.read_csv(path, skiprows=1)
    
    asset = ds.add_dataframe_asset(name="nasa_df")
    batch_def = asset.add_batch_definition_whole_dataframe("my_batch")
    batch = batch_def.get_batch(batch_parameters={"dataframe": df})
    
    suite = gx.ExpectationSuite(name="nasa_suite")
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="Year"))
    suite.add_expectation(gx.expectations.ExpectColumnToExist(column="J-D"))
    suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="Year"))
    
    result = batch.validate(suite)
    passed = result.success
    print(f"   {'✅ PASS' if passed else '❌ FAIL'}: NASA Validation")
    return passed


def validate_world_bank(context):
    """Validate World Bank indicator files."""
    files = {
        "WB_Electricity": "WB_WDI_EG_ELC_ACCS_ZS.csv",
        "WB_ElecPower": "WB_WDI_EG_USE_ELEC_KH_PC.csv",
    }
    
    all_passed = True
    print("\n🏦 Running Great Expectations on World Bank...")
    
    ds = context.data_sources.add_pandas("wb_source")
    
    for name, filename in files.items():
        path = os.path.join(DATA_DIR, filename)
        if not os.path.exists(path):
            continue
            
        asset = ds.add_csv_asset(name, filepath_or_buffer=path)
        batch_def = asset.add_batch_definition_whole_dataframe("my_batch")
        batch = batch_def.get_batch()
        
        suite = gx.ExpectationSuite(name=f"{name}_suite")
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column="REF_AREA"))
        suite.add_expectation(gx.expectations.ExpectColumnToExist(column="OBS_VALUE"))
        suite.add_expectation(gx.expectations.ExpectColumnValuesToNotBeNull(column="REF_AREA"))
        
        result = batch.validate(suite)
        passed = result.success
        print(f"   {'✅ PASS' if passed else '❌ FAIL'}: {name}")
        all_passed = all_passed and passed
        
    return all_passed


def run_all_validations():
    """Execute all GE data quality validations."""
    print("=" * 60)
    print("✨ CLIMATE INTELLIGENCE — GREAT EXPECTATIONS VALIDATION")
    print("=" * 60)

    # Note: Ephemeral context is lightweight and doesn't require init setup
    context = gx.get_context()
    
    p1 = validate_owid(context)
    p2 = validate_nasa(context)
    p3 = validate_world_bank(context)
    
    overall = p1 and p2 and p3
    print("\n" + "=" * 60)
    if overall:
        print("🌟 GE VALIDATION: ALL SUITES PASSED 🌟")
    else:
        print("⚠️  GE VALIDATION: SOME SUITES FAILED")
    print("=" * 60)
    
    return overall


if __name__ == "__main__":
    success = run_all_validations()
    sys.exit(0 if success else 1)
