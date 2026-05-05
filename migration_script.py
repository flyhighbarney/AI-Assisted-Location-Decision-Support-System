"""
migration_script.py
====================
Module 4 – Database Design & Migration  |  Lead: Elizabeth (fixed & completed)

What this script does:
  1. Loads all CSV source files from the Data/ folder
  2. Pre-computes projected UTM 19N coordinates (EPSG:26919) for every CBG
  3. Pre-computes the competitor utility sum for every (CBG x category) pair
  4. Writes a clean, indexed SQLite database: urban_ai_fixed.db

Tables created
--------------
  cbg_master            – demographics + lat/lon + x_proj/y_proj
  poi_master            – all point-of-interest records (area > 0)test comment
  cbg_poi_distance      – CBG <-> POI distances (> 0 m)
  cbg_poi_visits        – observed visit counts
  calibrated_parameters – Huff alpha/beta per category
  competitor_utility    – PRE-COMPUTED utility sums (CBG x category)

Run from the repo root:
    python migration_script.py
"""

import sqlite3
import math
import time
import os
import pandas as pd
import numpy as np

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR    = os.path.join(_SCRIPT_DIR, "Data")
DB_PATH     = os.path.join(_SCRIPT_DIR, "urban_ai_fixed.db")


# ---------------------------------------------------------------
# HELPER: manual UTM 19N projection (no pyproj dependency)
# ---------------------------------------------------------------
def latlon_to_utm19n(lat: float, lon: float) -> tuple:
    """Convert WGS-84 lat/lon to UTM Zone 19N (EPSG:26919) easting/northing (metres)."""
    a  = 6_378_137.0
    f  = 1 / 298.257223563
    b  = a * (1 - f)
    e2 = 1 - (b / a) ** 2
    k0 = 0.9996
    E0 = 500_000.0
    lon0 = math.radians(-69)          # central meridian UTM zone 19

    lat_r = math.radians(lat)
    lon_r = math.radians(lon)

    N = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T = math.tan(lat_r) ** 2
    C = (e2 / (1 - e2)) * math.cos(lat_r) ** 2
    A = math.cos(lat_r) * (lon_r - lon0)

    e4 = e2 ** 2
    e6 = e2 ** 3
    M  = a * (
          (1 - e2/4 - 3*e4/64 - 5*e6/256)  * lat_r
        - (3*e2/8 + 3*e4/32 + 45*e6/1024)  * math.sin(2 * lat_r)
        + (15*e4/256 + 45*e6/1024)          * math.sin(4 * lat_r)
        - (35*e6/3072)                       * math.sin(6 * lat_r)
    )

    ep2 = e2 / (1 - e2)
    x = k0 * N * (
        A + (1 - T + C) * A**3 / 6
          + (5 - 18*T + T**2 + 72*C - 58*ep2) * A**5 / 120
    ) + E0

    y = k0 * (
        M + N * math.tan(lat_r) * (
            A**2 / 2
            + (5 - T + 9*C + 4*C**2) * A**4 / 24
            + (61 - 58*T + T**2 + 600*C - 330*ep2) * A**6 / 720
        )
    )
    return x, y


# ---------------------------------------------------------------
# STEP 1 – Load source CSVs
# ---------------------------------------------------------------
def load_csvs() -> dict:
    print("Loading CSVs ...")

    dist_csv = os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv")
    dist_zip = os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv.zip")
    dist_src = dist_csv if os.path.exists(dist_csv) else dist_zip

    data = {
        "cbgs":      pd.read_csv(os.path.join(DATA_DIR, "worcester_cbgs.csv")),
        "pois":      pd.read_csv(os.path.join(DATA_DIR, "worcester_pois.csv")),
        "visits":    pd.read_csv(os.path.join(DATA_DIR, "worcester_cbg_poi_visits.csv")),
        "distances": pd.read_csv(dist_src),
        "params":    pd.read_csv(os.path.join(DATA_DIR, "calibrated_parameters_filtered.csv")),
    }
    for name, df in data.items():
        print(f"  {name}: {len(df):,} rows")
    return data


# ---------------------------------------------------------------
# STEP 2 – Enrich CBGs with lat/lon + projected coordinates
# ---------------------------------------------------------------
def enrich_cbgs(cbgs: pd.DataFrame, pois: pd.DataFrame) -> pd.DataFrame:
    """
    CBG CSV has no lat/lon column.
    Strategy: derive a centroid for each CBG from the mean lat/lon of its POIs.
    Fallback: Worcester city centre (42.2626, -71.8023) for CBGs with no POIs.
    """
    print("Computing CBG centroids and projected coordinates ...")
    centroids = (
        pois.groupby("poi_cbg")
            .agg(lat=("latitude", "mean"), lon=("longitude", "mean"))
            .reset_index()
            .rename(columns={"poi_cbg": "cbg"})
    )
    cbgs_geo = cbgs.merge(centroids, on="cbg", how="left")
    cbgs_geo["lat"] = cbgs_geo["lat"].fillna(42.2626)
    cbgs_geo["lon"] = cbgs_geo["lon"].fillna(-71.8023)

    coords = cbgs_geo.apply(
        lambda r: latlon_to_utm19n(r["lat"], r["lon"]), axis=1
    )
    cbgs_geo["x_proj"] = coords.apply(lambda c: c[0])
    cbgs_geo["y_proj"] = coords.apply(lambda c: c[1])
    print(f"  {len(cbgs_geo)} CBGs enriched with x_proj / y_proj")
    return cbgs_geo


# ---------------------------------------------------------------
# STEP 3 – Pre-compute competitor utility sums
# ---------------------------------------------------------------
def precompute_competitor_utility(
    pois: pd.DataFrame,
    distances: pd.DataFrame,
    params: pd.DataFrame,
) -> pd.DataFrame:
    """
    For every calibrated category and every CBG, compute:
        competitor_utility_sum = sum( area^alpha / dist^beta )
    This is the Huff denominator for existing stores — computed once, stored forever.
    """
    print("Pre-computing competitor utility sums ...")
    pois_clean = pois[pois["wkt_area_sq_meters"] > 0].copy()
    dist_clean = distances[distances["distance_m"] > 0].copy()

    rows = []
    for _, param in params.iterrows():
        cat   = param["top_category"]
        alpha = float(param["alpha"])
        beta  = float(param["beta"])

        cat_pois = pois_clean[pois_clean["top_category"] == cat][
            ["placekey", "wkt_area_sq_meters"]
        ]
        cat_pks  = cat_pois["placekey"].unique()
        cat_dist = dist_clean[dist_clean["placekey"].isin(cat_pks)].merge(
            cat_pois, on="placekey"
        )
        if cat_dist.empty:
            continue

        cat_dist = cat_dist.copy()
        cat_dist["utility"] = (
            cat_dist["wkt_area_sq_meters"] ** alpha
            / cat_dist["distance_m"] ** beta
        )
        cbg_sums = (
            cat_dist.groupby("GEOID10")["utility"]
                    .sum()
                    .reset_index()
                    .rename(columns={
                        "GEOID10":  "cbg",
                        "utility":  "competitor_utility_sum"
                    })
        )
        cbg_sums["top_category"] = cat
        cbg_sums["naics_code"]   = int(param["NAICS code"])
        cbg_sums["alpha"]        = alpha
        cbg_sums["beta"]         = beta
        rows.append(cbg_sums)

    result = pd.concat(rows, ignore_index=True)
    print(f"  {len(result):,} (CBG x category) utility sums computed")
    return result


# ---------------------------------------------------------------
# STEP 4 – Build the SQLite database
# ---------------------------------------------------------------
def build_database(
    data: dict,
    cbgs_geo: pd.DataFrame,
    comp_util: pd.DataFrame,
    db_path: str,
):
    print(f"Writing database -> {db_path} ...")
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA journal_mode=WAL")

    # cbg_master: demographics + spatial coordinates
    cbgs_geo.to_sql("cbg_master", conn, if_exists="replace", index=False)

    # poi_master: cleaned POI records (area > 0)
    pois_clean = data["pois"][data["pois"]["wkt_area_sq_meters"] > 0].copy()
    pois_clean.to_sql("poi_master", conn, if_exists="replace", index=False)

    # cbg_poi_distance: positive distances only
    dist_clean = data["distances"][data["distances"]["distance_m"] > 0].copy()
    dist_clean.to_sql("cbg_poi_distance", conn, if_exists="replace", index=False)

    # cbg_poi_visits: positive visits only
    visits_clean = data["visits"][data["visits"]["visit_count"] > 0].copy()
    visits_clean.to_sql("cbg_poi_visits", conn, if_exists="replace", index=False)

    # calibrated_parameters: Huff alpha/beta per category
    data["params"].to_sql("calibrated_parameters", conn, if_exists="replace", index=False)

    # competitor_utility: PRE-COMPUTED sums <- key deliverable
    comp_util.to_sql("competitor_utility", conn, if_exists="replace", index=False)

    # Performance indexes
    indexes = [
        ("idx_comp_cbg_cat",  "competitor_utility(cbg, top_category)"),
        ("idx_dist_geoid",    "cbg_poi_distance(GEOID10)"),
        ("idx_dist_pk",       "cbg_poi_distance(placekey)"),
        ("idx_visits_cbg",    "cbg_poi_visits(visitor_home_cbg)"),
        ("idx_poi_cat",       "poi_master(top_category)"),
        ("idx_cbg_pk",        "cbg_master(cbg)"),
    ]
    for name, spec in indexes:
        conn.execute(f"CREATE INDEX IF NOT EXISTS {name} ON {spec}")

    conn.commit()
    conn.close()
    print("Database written successfully.")


# ---------------------------------------------------------------
# STEP 5 – Verify
# ---------------------------------------------------------------
def verify_database(db_path: str):
    print("\n-- Verification -------------------------------------")
    conn = sqlite3.connect(db_path)
    tables = [
        "cbg_master", "poi_master", "cbg_poi_distance",
        "cbg_poi_visits", "calibrated_parameters", "competitor_utility",
    ]
    for t in tables:
        n = pd.read_sql(f"SELECT COUNT(*) AS n FROM {t}", conn).iloc[0, 0]
        print(f"  {t:<28} {n:>10,} rows")

    sample = pd.read_sql(
        "SELECT cbg, lat, lon, x_proj, y_proj FROM cbg_master LIMIT 3", conn
    )
    print("\n  CBG master sample (first 3):")
    print(sample.to_string(index=False))

    cu = pd.read_sql(
        "SELECT top_category, COUNT(*) AS cbgs, AVG(competitor_utility_sum) AS avg_util "
        "FROM competitor_utility GROUP BY top_category LIMIT 5",
        conn,
    )
    print("\n  Competitor utility sample (5 categories):")
    print(cu.to_string(index=False))
    conn.close()
    print("-----------------------------------------------------")


# ---------------------------------------------------------------
# MAIN
# ---------------------------------------------------------------
if __name__ == "__main__":
    t0 = time.time()
    data      = load_csvs()
    cbgs_geo  = enrich_cbgs(data["cbgs"], data["pois"])
    comp_util = precompute_competitor_utility(
        data["pois"], data["distances"], data["params"]
    )
    build_database(data, cbgs_geo, comp_util, DB_PATH)
    verify_database(DB_PATH)
    print(f"\nMigration complete in {time.time() - t0:.1f} s")
    print(f"Database saved to: {DB_PATH}")