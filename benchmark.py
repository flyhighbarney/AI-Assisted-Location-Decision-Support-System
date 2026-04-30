"""
benchmark.py
=============
Module 4 – Performance Benchmarking  |  Lead: Johnson

Compares runtime of:
  METHOD A – Old CSV approach  (loads all files from disk into RAM each run)
  METHOD B – New SQL approach  (queries only what is needed, uses pre-computation)

Instructions for Johnson:
  1. Make sure urban_ai_fixed.db exists (run migration_script.py first)
  2. Run:  python benchmark.py
  3. Screenshot the full terminal output
  4. Include the screenshot as the 'Screenshot Deliverable' in the submission

Note on results:
  On a local machine with small data (< 1 GB), the CSV method may appear
  comparable or slightly faster because everything fits in RAM.
  The database advantage becomes decisive at production scale — see the
  200-word reflection for the full explanation.
"""

import time
import sqlite3
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_SCRIPT_DIR, "Data")
DB_PATH     = os.path.join(_SCRIPT_DIR, "urban_ai_fixed.db")

CATEGORY    = "Gasoline Stations"
REPS        = 3   # number of timed runs per method


# ---------------------------------------------------------------
# Resolve test placekey from database
# ---------------------------------------------------------------
def _get_test_placekey() -> str:
    conn = sqlite3.connect(DB_PATH)
    row  = pd.read_sql_query(
        "SELECT d.placekey FROM cbg_poi_distance d "
        "JOIN poi_master p ON d.placekey = p.placekey "
        "WHERE p.top_category = ? LIMIT 1",
        conn, params=(CATEGORY,)
    )
    conn.close()
    return row.iloc[0, 0]


# ---------------------------------------------------------------
# METHOD A: Old CSV-based approach
# ---------------------------------------------------------------
def old_csv_method(placekey: str, category: str) -> float:
    """Mimic the original engine: load all CSVs, run the full Huff loop."""
    t0 = time.perf_counter()

    # Load every file into memory (the old way)
    dist_csv = os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv")
    dist_zip = os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv.zip")
    dist_src = dist_csv if os.path.exists(dist_csv) else dist_zip

    pois      = pd.read_csv(os.path.join(DATA_DIR, "worcester_pois.csv"))
    distances = pd.read_csv(dist_src)
    visits    = pd.read_csv(os.path.join(DATA_DIR, "worcester_cbg_poi_visits.csv"))
    cbgs      = pd.read_csv(os.path.join(DATA_DIR, "worcester_cbgs.csv"))
    params    = pd.read_csv(os.path.join(DATA_DIR, "calibrated_parameters_filtered.csv"))

    # Clean data
    pois      = pois[pois["wkt_area_sq_meters"] > 0]
    distances = distances[distances["distance_m"] > 0]
    visits    = visits[visits["visit_count"] > 0]

    # Find category parameters
    param_row = params[params["top_category"] == category].iloc[0]
    alpha     = float(param_row["alpha"])
    beta      = float(param_row["beta"])

    # Find area of the new site
    area = float(pois[pois["placekey"] == placekey]["wkt_area_sq_meters"].iloc[0])

    # Distances to the new site
    dist_new = distances[distances["placekey"] == placekey].copy()
    dist_new = dist_new.rename(columns={"GEOID10": "cbg"})

    # Competitor utility — the EXPENSIVE full loop
    cat_pois = pois[pois["top_category"] == category][["placekey", "wkt_area_sq_meters"]]
    cat_pks  = cat_pois["placekey"].unique()
    cat_dist = distances[distances["placekey"].isin(cat_pks)].merge(cat_pois, on="placekey")
    cat_dist = cat_dist.copy()
    cat_dist["utility"] = (
        cat_dist["wkt_area_sq_meters"] ** alpha / cat_dist["distance_m"] ** beta
    )
    comp_sum = (
        cat_dist.groupby("GEOID10")["utility"]
                .sum()
                .reset_index()
                .rename(columns={"GEOID10": "cbg", "utility": "competitor_utility_sum"})
    )

    # Merge and compute probability
    df = dist_new.merge(comp_sum, on="cbg", how="left")
    df["u_new"] = area ** alpha / df["distance_m"] ** beta
    df["competitor_utility_sum"] = df["competitor_utility_sum"].fillna(0)
    df["p_new"] = df["u_new"] / (df["u_new"] + df["competitor_utility_sum"])

    return time.perf_counter() - t0


# ---------------------------------------------------------------
# METHOD B: New SQL-based approach
# ---------------------------------------------------------------
def new_sql_method(placekey: str, category: str) -> float:
    """Use the refactored SQL engine."""
    from huff_engine_v2 import run_huff_model
    t0 = time.perf_counter()
    run_huff_model(new_placekey=placekey, category=category, db_path=DB_PATH)
    return time.perf_counter() - t0


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    TEST_PK = _get_test_placekey()

    print("=" * 64)
    print("  Performance Benchmark  |  Module 4  ")
    print("=" * 64)
    print(f"  Category  : {CATEGORY}")
    print(f"  Placekey  : {TEST_PK}")
    print(f"  Reps      : {REPS}\n")

    # Method A
    print("Running Method A (CSV-based) ...")
    a_times = []
    for i in range(REPS):
        t = old_csv_method(TEST_PK, CATEGORY)
        a_times.append(t)
        print(f"  Run {i+1}: {t:.4f}s")
    avg_a = sum(a_times) / REPS

    # Method B
    print("\nRunning Method B (SQL-based) ...")
    b_times = []
    for i in range(REPS):
        t = new_sql_method(TEST_PK, CATEGORY)
        b_times.append(t)
        print(f"  Run {i+1}: {t:.4f}s")
    avg_b = sum(b_times) / REPS

    speedup = avg_a / avg_b if avg_b > 0 else float("inf")

    print("\n" + "=" * 64)
    print("  RESULTS")
    print("=" * 64)
    print(f"  Method A (CSV) — average : {avg_a:.4f} s")
    print(f"  Method B (SQL) — average : {avg_b:.4f} s")
    if speedup >= 1:
        print(f"  SQL is {speedup:.1f}x faster than CSV")
    else:
        print(f"  CSV is {1/speedup:.1f}x faster locally (see reflection for why SQL wins at scale)")
    print("=" * 64)
    print()
    print("  --> Johnson: screenshot this terminal output for the deliverable.")
    print("=" * 64)
