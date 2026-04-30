"""
huff_engine_v2.py
==================
Module 4 – Refactored Huff Model Engine  |  Lead: Barnabas

Changes from predict_site.py (CSV-based engine):
  - ALL pd.read_csv() calls removed
  - Data fetched from SQLite via sqlite3 + pd.read_sql_query()
  - Every query uses ? parameterized placeholders (no SQL injection risk)
  - Competitor utility sum fetched from pre-computed table — no heavy loop
  - New-site utility calculated on the fly and added to the pre-stored sum
  - Outputs a clean DataFrame sorted by predicted visits

Usage:
    python huff_engine_v2.py
    -- or import run_huff_model() from another script --

Requires:
    urban_ai_fixed.db in the same folder (built by migration_script.py)
"""

import sqlite3
import time
import math
import os
import pandas as pd

# ---------------------------------------------------------------
# PATHS
# ---------------------------------------------------------------
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
DB_PATH     = os.path.join(_SCRIPT_DIR, "urban_ai_fixed.db")


# ---------------------------------------------------------------
# HELPER: UTM 19N projection (no external dependencies)
# ---------------------------------------------------------------
def latlon_to_utm19n(lat: float, lon: float) -> tuple:
    """Convert WGS-84 degrees to UTM Zone 19N metres (EPSG:26919)."""
    a    = 6_378_137.0
    f    = 1 / 298.257223563
    b    = a * (1 - f)
    e2   = 1 - (b / a) ** 2
    k0   = 0.9996
    E0   = 500_000.0
    lon0 = math.radians(-69)
    lat_r = math.radians(lat)
    lon_r = math.radians(lon)
    N    = a / math.sqrt(1 - e2 * math.sin(lat_r) ** 2)
    T    = math.tan(lat_r) ** 2
    C    = (e2 / (1 - e2)) * math.cos(lat_r) ** 2
    A    = math.cos(lat_r) * (lon_r - lon0)
    e4   = e2 ** 2
    e6   = e2 ** 3
    M    = a * (
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
# DATABASE HELPERS  — parameterized queries only
# ---------------------------------------------------------------

def get_connection(db_path: str = DB_PATH) -> sqlite3.Connection:
    """Open a read-optimised SQLite connection."""
    if not os.path.exists(db_path):
        raise FileNotFoundError(
            f"Database not found: {db_path}\n"
            "Run migration_script.py first to build urban_ai_fixed.db"
        )
    conn = sqlite3.connect(db_path)
    conn.execute("PRAGMA query_only = ON")
    return conn


def fetch_categories(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all calibrated Huff parameters."""
    query = """
        SELECT top_category,
               "NAICS code" AS naics_code,
               alpha,
               beta
        FROM calibrated_parameters
        ORDER BY top_category
    """
    return pd.read_sql_query(query, conn)


def fetch_cbg_master(conn: sqlite3.Connection) -> pd.DataFrame:
    """Return all CBGs with demographics and pre-projected coordinates."""
    query = """
        SELECT cbg,
               total_population,
               median_household_income,
               lat, lon,
               x_proj, y_proj,
               income_q, education_q, age_q
        FROM cbg_master
        WHERE x_proj IS NOT NULL
    """
    return pd.read_sql_query(query, conn)


def fetch_competitor_utility(
    conn: sqlite3.Connection,
    category: str,
) -> pd.DataFrame:
    """
    Fetch pre-computed competitor utility sums for one category.
    Uses a parameterized query — ? placeholder, never string formatting.
    """
    query = """
        SELECT cbg, competitor_utility_sum, alpha, beta
        FROM competitor_utility
        WHERE top_category = ?
    """
    return pd.read_sql_query(query, conn, params=(category,))


def fetch_total_visits_by_cbg(
    conn: sqlite3.Connection,
    category: str,
) -> pd.DataFrame:
    """
    Return observed visit totals per CBG for a category.
    Used to scale Huff probabilities into predicted visit counts.
    """
    query = """
        SELECT v.visitor_home_cbg AS cbg,
               SUM(v.visit_count)  AS total_category_visits
        FROM cbg_poi_visits v
        JOIN poi_master p ON v.placekey = p.placekey
        WHERE p.top_category = ?
        GROUP BY v.visitor_home_cbg
    """
    return pd.read_sql_query(query, conn, params=(category,))


def fetch_new_site_distances(
    conn: sqlite3.Connection,
    new_placekey: str,
) -> pd.DataFrame:
    """
    Fetch distances from all CBGs to a specific POI.
    Parameterized query — ? placeholder for the placekey.
    """
    query = """
        SELECT GEOID10 AS cbg,
               distance_m
        FROM cbg_poi_distance
        WHERE placekey = ?
          AND distance_m > 0
    """
    return pd.read_sql_query(query, conn, params=(new_placekey,))


def fetch_new_site_area(
    conn: sqlite3.Connection,
    new_placekey: str,
) -> float:
    """Return the floor area (sq metres) of a specific POI."""
    query = "SELECT wkt_area_sq_meters FROM poi_master WHERE placekey = ?"
    row = pd.read_sql_query(query, conn, params=(new_placekey,))
    if row.empty:
        raise ValueError(
            f"Placekey '{new_placekey}' not found in poi_master."
        )
    return float(row.iloc[0, 0])


# ---------------------------------------------------------------
# CORE ENGINE
# ---------------------------------------------------------------

def run_huff_model(
    new_placekey: str,
    category: str,
    db_path: str = DB_PATH,
) -> pd.DataFrame:
    """
    Run the Huff gravity model for a proposed new site.

    Parameters
    ----------
    new_placekey : str  Safegraph/Placekey identifier of the new POI
    category     : str  Business category (must match calibrated_parameters)
    db_path      : str  Path to the SQLite database

    Returns
    -------
    pd.DataFrame with columns:
        cbg, distance_m, u_new, competitor_utility_sum,
        p_new, total_category_visits, predicted_visits,
        total_population, median_household_income
    Sorted by predicted_visits descending.
    """
    conn = get_connection(db_path)

    # 1. Fetch model parameters for this category
    params_df = fetch_categories(conn)
    row = params_df[params_df["top_category"] == category]
    if row.empty:
        conn.close()
        available = params_df["top_category"].tolist()
        raise ValueError(
            f"Category '{category}' not in calibrated_parameters.\n"
            f"Available categories:\n  " + "\n  ".join(available)
        )
    alpha = float(row.iloc[0]["alpha"])
    beta  = float(row.iloc[0]["beta"])

    # 2. New site floor area
    area = fetch_new_site_area(conn, new_placekey)

    # 3. Distances from all CBGs to the new site
    dist_df = fetch_new_site_distances(conn, new_placekey)
    if dist_df.empty:
        conn.close()
        raise ValueError(
            f"No distance records found for placekey '{new_placekey}'.\n"
            "Ensure it exists in cbg_poi_distance."
        )

    # 4. Pre-computed competitor utility sum (one fast DB lookup)
    comp_df = fetch_competitor_utility(conn, category)

    # 5. Observed visits per CBG for demand scaling
    visits_df = fetch_total_visits_by_cbg(conn, category)

    # 6. CBG demographics
    cbg_df = fetch_cbg_master(conn)
    conn.close()

    # 7. Compute new-site utility: U = Area^alpha / Dist^beta
    df = dist_df.copy()
    df["u_new"] = area ** alpha / df["distance_m"] ** beta

    # 8. Merge pre-computed competitor sum (fast — no loop needed)
    df = df.merge(comp_df[["cbg", "competitor_utility_sum"]], on="cbg", how="left")
    df["competitor_utility_sum"] = df["competitor_utility_sum"].fillna(0)

    # 9. Huff probability: P_new = U_new / (U_new + sum_competitors)
    df["p_new"] = df["u_new"] / (df["u_new"] + df["competitor_utility_sum"])

    # 10. Scale to predicted visits
    df = df.merge(visits_df, on="cbg", how="left")
    df["total_category_visits"] = df["total_category_visits"].fillna(0).astype(int)
    df["predicted_visits"] = df["p_new"] * df["total_category_visits"]

    # 11. Attach demographics
    df = df.merge(
        cbg_df[["cbg", "total_population", "median_household_income"]],
        on="cbg",
        how="left",
    )

    return (
        df[[
            "cbg", "distance_m", "u_new", "competitor_utility_sum",
            "p_new", "total_category_visits", "predicted_visits",
            "total_population", "median_household_income",
        ]]
        .sort_values("predicted_visits", ascending=False)
        .reset_index(drop=True)
    )


# ---------------------------------------------------------------
# UTILITY HELPERS
# ---------------------------------------------------------------

def list_available_categories(db_path: str = DB_PATH) -> list:
    """Return all category names stored in the database."""
    conn = get_connection(db_path)
    cats = pd.read_sql_query(
        "SELECT top_category FROM calibrated_parameters ORDER BY top_category",
        conn,
    )
    conn.close()
    return cats["top_category"].tolist()


def get_sample_placekey(db_path: str = DB_PATH, category: str = None) -> str:
    """Return the first placekey in cbg_poi_distance for a given category."""
    conn = get_connection(db_path)
    if category:
        query = """
            SELECT d.placekey
            FROM cbg_poi_distance d
            JOIN poi_master p ON d.placekey = p.placekey
            WHERE p.top_category = ?
            LIMIT 1
        """
        row = pd.read_sql_query(query, conn, params=(category,))
    else:
        row = pd.read_sql_query(
            "SELECT placekey FROM cbg_poi_distance LIMIT 1", conn
        )
    conn.close()
    return row.iloc[0, 0] if not row.empty else None


# ---------------------------------------------------------------
# DEMO / SELF-TEST
# ---------------------------------------------------------------
if __name__ == "__main__":
    print("=" * 62)
    print("  Huff Engine v2  —  SQL-backed  |  Lead: Barnabas")
    print("=" * 62)

    CATEGORY = "Gasoline Stations"
    PLACEKEY = get_sample_placekey(DB_PATH, CATEGORY)

    if PLACEKEY is None:
        print(f"No placekey found for category: {CATEGORY}")
        raise SystemExit(1)

    print(f"  Category   : {CATEGORY}")
    print(f"  New site   : {PLACEKEY}")
    print(f"  Database   : {DB_PATH}\n")

    t0 = time.perf_counter()
    results = run_huff_model(
        new_placekey=PLACEKEY,
        category=CATEGORY,
        db_path=DB_PATH,
    )
    elapsed = time.perf_counter() - t0

    print(f"  CBGs evaluated     : {len(results)}")
    print(f"  Total pred. visits : {results['predicted_visits'].sum():.1f}")
    print(f"\n  Top 5 CBGs by predicted visits:")
    print(
        results[["cbg", "distance_m", "p_new", "predicted_visits"]]
        .head(5)
        .to_string(index=False)
    )
    print(f"\n  Engine runtime (SQL method): {elapsed:.4f} seconds")
    print("=" * 62)

    print("\nAll available categories in database:")
    for c in list_available_categories(DB_PATH):
        print(f"  - {c}")
