"""
migration_v2.py
================
Module 5  |  One-Time Database Setup Script

WHO WROTE WHAT
--------------
Barnabas  – Steps 1, 2, 3  (Python: data loading, centroid projection,
                              utility computation, market potential)
Elizabeth – Step 4          (SQL: write tables to DB + apply indexes)

Run once from the repo root after Elizabeth completes Step 4:
    python migration_v2.py

Output:
    urban_ai_v2.db  (written to the repo root)
"""

import json
import math
import os
import sqlite3
import time

import numpy as np
import pandas as pd

# ================================================================
# PATHS
# ================================================================
_DIR     = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(_DIR, "Data")
DB_PATH  = os.path.join(_DIR, "urban_ai_v2.db")

_GEOJSON = os.path.join(DATA_DIR, "worcester_cbgs_map.geojson")
_DIST_SRC = (
    os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv")
    if os.path.exists(os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv"))
    else os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv.zip")
)

DISTANCE_FLOOR = 100.0   # metres — same clip used in huff_engine.py


# ================================================================
# BARNABAS — Step 1: UTM 19N projection
# (no pyproj needed — runs on Azure without extra packages)
# ================================================================
def latlon_to_utm19n(lat: float, lon: float) -> tuple:
    """
    Convert WGS-84 lat/lon degrees to UTM Zone 19N (EPSG:26919) metres.

    Why we need this:
    huff_engine.py (V1) calls gpd.GeoDataFrame(...).to_crs("EPSG:26919")
    on EVERY request. That is slow. We do this math ONCE here and store
    the result in CBG_Master so the engine never has to project again.
    """
    a    = 6_378_137.0
    f    = 1 / 298.257223563
    b    = a * (1 - f)
    e2   = 1 - (b / a) ** 2
    k0   = 0.9996
    E0   = 500_000.0
    lon0 = math.radians(-69)

    lr = math.radians(lat)
    lo = math.radians(lon)

    N   = a / math.sqrt(1 - e2 * math.sin(lr) ** 2)
    T   = math.tan(lr) ** 2
    C   = (e2 / (1 - e2)) * math.cos(lr) ** 2
    A   = math.cos(lr) * (lo - lon0)
    e4  = e2 ** 2
    e6  = e2 ** 3
    M   = a * (
          (1 - e2/4 - 3*e4/64 - 5*e6/256)  * lr
        - (3*e2/8 + 3*e4/32 + 45*e6/1024)  * math.sin(2 * lr)
        + (15*e4/256 + 45*e6/1024)          * math.sin(4 * lr)
        - (35*e6/3072)                       * math.sin(6 * lr)
    )
    ep2 = e2 / (1 - e2)
    x = k0 * N * (
        A + (1 - T + C) * A**3 / 6
          + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120
    ) + E0
    y = k0 * (
        M + N * math.tan(lr) * (
            A**2 / 2
            + (5 - T + 9*C + 4*C**2) * A**4 / 24
            + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720
        )
    )
    return x, y


# ================================================================
# BARNABAS — Step 2: Build CBG_Master DataFrame
#
# Combines worcester_cbgs.csv (demographics) with the real census
# centroids from worcester_cbgs_map.geojson (INTPTLAT10/INTPTLON10).
# Computes x_proj and y_proj (metres, EPSG:26919) for every CBG.
# ================================================================
def build_cbg_master(data: dict) -> pd.DataFrame:
    print("  Building CBG_Master ...")

    with open(_GEOJSON) as f:
        geo = json.load(f)

    geo_rows = []
    for feat in geo["features"]:
        p   = feat["properties"]
        lat = float(p["INTPTLAT10"])
        lon = float(p["INTPTLON10"])
        x, y = latlon_to_utm19n(lat, lon)
        geo_rows.append({
            "geoid":      str(p["GEOID10"]),
            "intptlat10": lat,
            "intptlon10": lon,
            "x_proj":     x,
            "y_proj":     y,
        })
    geo_df = pd.DataFrame(geo_rows)

    cbgs = data["cbgs"].copy()
    cbgs["geoid"] = cbgs["cbg"].astype(str)
    cbg_master = cbgs.merge(geo_df, on="geoid", how="left")

    # Fallback to Worcester city centre for any CBG missing coordinates
    wx, wy = latlon_to_utm19n(42.2626, -71.8023)
    cbg_master["x_proj"] = cbg_master["x_proj"].fillna(wx)
    cbg_master["y_proj"] = cbg_master["y_proj"].fillna(wy)

    print(f"    {len(cbg_master)} CBGs enriched — x_proj/y_proj in metres (EPSG:26919)")
    return cbg_master


# ================================================================
# BARNABAS — Step 3a: Compute Competitor_Summary DataFrame
#
# For every NAICS category and every CBG, calculates:
#     comp_utility_sum = sum( area^alpha / dist^beta )
# summed over all existing stores in that category.
#
# This eliminates Bottleneck 4 from huff_engine.py (V1) — the
# 600k-row loop that ran on every single user request.
# ================================================================
def compute_competitor_summary(data: dict) -> pd.DataFrame:
    print("  Computing Competitor_Summary ...")

    pois      = data["pois"][data["pois"]["wkt_area_sq_meters"] > 0].copy()
    distances = data["distances"][data["distances"]["distance_m"] > 0].copy()
    params    = data["params"].copy()
    params["top_category"] = params["top_category"].str.strip()

    rows = []
    for _, param in params.iterrows():
        naics = int(param["NAICS code"])
        alpha = float(param["alpha"])
        beta  = float(param["beta"])
        cat   = str(param["top_category"]).strip()

        cat_pois = pois[pois["naics_code"] == naics][["placekey", "wkt_area_sq_meters"]]
        if cat_pois.empty:
            continue

        cat_dist = distances[distances["placekey"].isin(cat_pois["placekey"].unique())].merge(
            cat_pois, on="placekey"
        )
        if cat_dist.empty:
            continue

        cat_dist = cat_dist.copy()
        dist_eff = np.maximum(cat_dist["distance_m"], DISTANCE_FLOOR)
        cat_dist["utility"] = cat_dist["wkt_area_sq_meters"] ** alpha / dist_eff ** beta

        sums = (
            cat_dist.groupby("GEOID10")["utility"]
                    .sum()
                    .reset_index()
                    .rename(columns={"GEOID10": "geoid", "utility": "comp_utility_sum"})
        )
        sums["geoid"]        = sums["geoid"].astype(str)
        sums["top_category"] = cat
        sums["naics_code"]   = naics
        sums["alpha"]        = alpha
        sums["beta"]         = beta
        rows.append(sums)

    result = pd.concat(rows, ignore_index=True)
    result["comp_utility_sum"] = result["comp_utility_sum"].fillna(0.0)
    print(f"    {len(result):,} (CBG x category) sums computed")
    return result


# ================================================================
# BARNABAS — Step 3b: Compute Market_Potential DataFrame
#
# Pre-aggregates total observed visits per CBG per NAICS.
# Eliminates Bottleneck 5 — the live visit sum in V1.
# ================================================================
def compute_market_potential(data: dict) -> pd.DataFrame:
    print("  Computing Market_Potential ...")

    pois   = data["pois"][data["pois"]["wkt_area_sq_meters"] > 0][["placekey", "naics_code"]].drop_duplicates()
    visits = data["visits"][data["visits"]["visit_count"] > 0].copy()
    merged = visits.merge(pois, on="placekey")
    result = (
        merged.groupby(["visitor_home_cbg", "naics_code"])["visit_count"]
              .sum()
              .reset_index()
              .rename(columns={"visitor_home_cbg": "geoid", "visit_count": "market_potential"})
    )
    result["geoid"]            = result["geoid"].astype(str)
    result["market_potential"] = result["market_potential"].fillna(0.0)
    print(f"    {len(result):,} (CBG x NAICS) market potential values computed")
    return result


# ================================================================
# ELIZABETH — Step 4a: Store tables in SQLite
#
# Barnabas has computed four DataFrames and passed them here:
#   cbg_master   — 149 rows, columns include geoid, x_proj, y_proj
#   comp_summary — 3,426 rows, columns include geoid, naics_code,
#                  top_category, comp_utility_sum
#   market_pot   — 8,311 rows, columns include geoid, naics_code,
#                  market_potential
#   data["params"] — 23 rows, the calibrated alpha/beta parameters
#   data["pois"]   — 4,069 rows, the POI records
#
# YOUR TASK: write each DataFrame into the SQLite database (conn)
# using .to_sql(). The table names must be exactly:
#   "CBG_Master"
#   "Competitor_Summary"
#   "Ref_Categories"
#   "Market_Potential"
#   "POI_Master"
#
# For Ref_Categories, rename "NAICS code" -> "naics_code" first
# and strip whitespace from top_category before storing.
#
# Use if_exists="replace" and index=False on every .to_sql() call.
# ================================================================
def store_tables(
    conn: sqlite3.Connection,
    cbg_master:   pd.DataFrame,
    comp_summary: pd.DataFrame,
    market_pot:   pd.DataFrame,
    data:         dict,
):
    print("  Storing tables (Elizabeth) ...")

    # TODO (Elizabeth): store cbg_master as table "CBG_Master"

    # TODO (Elizabeth): store comp_summary as table "Competitor_Summary"

    # TODO (Elizabeth): build ref DataFrame from data["params"],
    #                   rename "NAICS code" -> "naics_code",
    #                   strip top_category whitespace,
    #                   store as table "Ref_Categories"

    # TODO (Elizabeth): store market_pot as table "Market_Potential"

    # TODO (Elizabeth): filter data["pois"] to wkt_area_sq_meters > 0,
    #                   store as table "POI_Master"

    raise NotImplementedError("Elizabeth: complete store_tables() above.")


# ================================================================
# ELIZABETH — Step 4b: Apply SQL Indexes
#
# After all tables are stored, add indexes so the engine can look
# up a single NAICS or geoid in microseconds instead of scanning
# every row. Think of indexes like alphabetical tabs in a phonebook.
#
# YOUR TASK: call conn.execute() for each index below, then call
# conn.commit() at the end.
#
# Required indexes (assignment specifies geoid + top_category):
#
#   Table            Column(s)
#   ────────────────────────────────────────────
#   CBG_Master       geoid
#   Competitor_Summary  top_category
#   Competitor_Summary  geoid
#   Competitor_Summary  geoid, naics_code
#   Ref_Categories   top_category
#   Ref_Categories   naics_code
#   Market_Potential geoid, naics_code
#
# SQL syntax for each one:
#   conn.execute(
#       "CREATE INDEX IF NOT EXISTS <index_name> ON <TableName>(<column>)"
#   )
# ================================================================
def apply_indexes(conn: sqlite3.Connection):
    print("  Applying SQL indexes (Elizabeth) ...")

    # TODO (Elizabeth): add CREATE INDEX statements here
    # Example of one index to get you started:
    #   conn.execute(
    #       "CREATE INDEX IF NOT EXISTS idx_cbgm_geoid ON CBG_Master(geoid)"
    #   )

    # TODO (Elizabeth): call conn.commit() after all indexes are added

    raise NotImplementedError("Elizabeth: complete apply_indexes() above.")


# ================================================================
# MAIN
# ================================================================
if __name__ == "__main__":
    t0 = time.time()
    print("=" * 56)
    print("  migration_v2.py — Module 5 One-Time Setup")
    print("=" * 56)

    print("\n[1] Loading source data ...")
    comp_kwarg = "zip" if _DIST_SRC.endswith(".zip") else None
    data = {
        "cbgs":      pd.read_csv(os.path.join(DATA_DIR, "worcester_cbgs.csv")),
        "pois":      pd.read_csv(os.path.join(DATA_DIR, "worcester_pois.csv")),
        "visits":    pd.read_csv(os.path.join(DATA_DIR, "worcester_cbg_poi_visits.csv")),
        "distances": pd.read_csv(_DIST_SRC, compression=comp_kwarg),
        "params":    pd.read_csv(os.path.join(DATA_DIR, "calibrated_parameters_filtered.csv")),
    }
    for k, v in data.items():
        print(f"    {k}: {len(v):,} rows")

    # BARNABAS — compute all DataFrames
    print("\n[2] Barnabas — computing enriched data ...")
    cbg_master   = build_cbg_master(data)
    comp_summary = compute_competitor_summary(data)
    market_pot   = compute_market_potential(data)

    # ELIZABETH — write to DB and index
    print("\n[3] Elizabeth — writing tables and indexes ...")
    conn = sqlite3.connect(DB_PATH)
    conn.execute("PRAGMA journal_mode=WAL")
    store_tables(conn, cbg_master, comp_summary, market_pot, data)
    apply_indexes(conn)
    conn.close()

    print(f"\nDone in {time.time() - t0:.1f}s  ->  {DB_PATH}")
    print("=" * 56)
