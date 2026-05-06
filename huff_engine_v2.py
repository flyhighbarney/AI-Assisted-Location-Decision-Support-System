"""
huff_engine_v2.py
==================
Module 5  |  Optimized Inference Engine

WHO WROTE WHAT
--------------
Barnabas  – All engine logic (run_huff_model, huff_v2, distance math,
             Huff formula, DB fetches, return structure)
Elizabeth – Security: replace every TODO below with a parameterized
             SQL query using ? placeholders instead of f-strings.
             This prevents SQL injection on all user-supplied inputs.

HOW THIS DIFFERS FROM huff_engine.py (V1)
------------------------------------------
Bottleneck 1 — params CSV scan      -> single indexed SQL SELECT
Bottleneck 2 — on-the-fly CRS proj  -> x_proj/y_proj already in DB
Bottleneck 3 — polygon .distance()  -> simple Euclidean point math
Bottleneck 4 — 600k-row loop        -> one SQL fetch of pre-computed sum
Bottleneck 5 — live visit sum       -> pre-aggregated Market_Potential

The function signature and return dict are IDENTICAL to huff_engine.py
so app.py requires zero changes.
"""

import math
import os
import sqlite3
import time

import pandas as pd

# ================================================================
# PATHS
# ================================================================
_DIR    = os.path.dirname(os.path.abspath(__file__))
DB_PATH = os.path.join(_DIR, "urban_ai_v2.db")

DISTANCE_FLOOR = 100.0   # metres — same as V1


# ================================================================
# BARNABAS — UTM 19N projection for the candidate site only
#
# V1 projected ALL 149 CBG polygons on every request (slow).
# V2 projects only the one new pin the user dropped.
# CBG centroids are already stored as x_proj/y_proj in the DB.
# ================================================================
def _latlon_to_utm19n(lat: float, lon: float) -> tuple:
    a=6378137.0; f=1/298.257223563; b=a*(1-f); e2=1-(b/a)**2
    k0=0.9996; E0=500000.0; lon0=math.radians(-69)
    lr=math.radians(lat); lo=math.radians(lon)
    N=a/math.sqrt(1-e2*math.sin(lr)**2); T=math.tan(lr)**2
    C=(e2/(1-e2))*math.cos(lr)**2; A=math.cos(lr)*(lo-lon0)
    e4=e2**2; e6=e2**3
    M=a*((1-e2/4-3*e4/64-5*e6/256)*lr
        -(3*e2/8+3*e4/32+45*e6/1024)*math.sin(2*lr)
        +(15*e4/256+45*e6/1024)*math.sin(4*lr)
        -(35*e6/3072)*math.sin(6*lr))
    ep2=e2/(1-e2)
    x=k0*N*(A+(1-T+C)*A**3/6+(5-18*T+T**2+72*C-58*ep2)*A**5/120)+E0
    y=k0*(M+N*math.tan(lr)*(A**2/2+(5-T+9*C+4*C**2)*A**4/24+(61-58*T+T**2+600*C-330*ep2)*A**6/720))
    return x, y


# ================================================================
# BARNABAS — Core V2 Huff computation
# ================================================================
def huff_v2(naics: int, candidate_lat: float, candidate_lon: float,
            floor_area: float, conn: sqlite3.Connection) -> tuple:

    # ELIZABETH — fetch alpha and beta (parameterized query)
    params_row = pd.read_sql_query(
        "SELECT alpha, beta FROM Ref_Categories WHERE naics_code = ?",
        conn, params=(naics,)
    )
    if params_row.empty:
        raise ValueError(f"No calibrated parameters found for NAICS {naics}.")
    alpha = float(params_row.iloc[0]["alpha"])
    beta  = float(params_row.iloc[0]["beta"])

    # BARNABAS — project the new candidate site to metres
    new_x, new_y = _latlon_to_utm19n(candidate_lat, candidate_lon)

    # BARNABAS — fetch all CBG centroids (already in metres from DB)
    cbg_df = pd.read_sql_query(
        "SELECT geoid, x_proj, y_proj FROM CBG_Master WHERE x_proj IS NOT NULL",
        conn
    )

    # BARNABAS — Euclidean distance (replaces polygon .distance())
    cbg_df["distance"] = (
        (cbg_df["x_proj"] - new_x) ** 2 + (cbg_df["y_proj"] - new_y) ** 2
    ) ** 0.5
    cbg_df["distance"] = cbg_df["distance"].clip(lower=DISTANCE_FLOOR)

    # ELIZABETH — fetch pre-computed competitor utility sums (parameterized query)
    comp_df = pd.read_sql_query(
        "SELECT geoid, comp_utility_sum FROM Competitor_Summary WHERE naics_code = ?",
        conn, params=(naics,)
    )

    # ELIZABETH — fetch pre-aggregated market potential (parameterized query)
    mkt_df = pd.read_sql_query(
        "SELECT geoid, market_potential FROM Market_Potential WHERE naics_code = ?",
        conn, params=(naics,)
    )

    # BARNABAS — merge all lookups into one working DataFrame
    df = cbg_df.merge(comp_df, on="geoid", how="left")
    df = df.merge(mkt_df, on="geoid", how="left")
    df["comp_utility_sum"] = df["comp_utility_sum"].fillna(0.0)
    df["market_potential"] = df["market_potential"].fillna(0.0)

    # Only CBGs with observed visits for this category
    df = df[df["market_potential"] > 0].copy()

    # BARNABAS — Huff formula
    # U_new = floor_area ^ alpha / distance ^ beta
    # P_ij  = U_new / (U_new + comp_utility_sum)
    # predicted_visits = P_ij * market_potential
    df["u_new"] = floor_area ** alpha / df["distance"] ** beta
    df["p_ij"]  = df["u_new"] / (df["u_new"] + df["comp_utility_sum"])
    df["predicted_visits"] = df["p_ij"] * df["market_potential"]

    total_predicted = float(df["predicted_visits"].sum())
    total_market    = float(df["market_potential"].sum())
    market_share    = total_predicted / total_market if total_market > 0 else 0.0

    # ELIZABETH — fetch competitor POIs for dashboard map (parameterized query)
    comp_pois = pd.read_sql_query(
        """SELECT location_name, placekey, latitude, longitude, wkt_area_sq_meters
           FROM POI_Master WHERE naics_code = ? LIMIT 20""",
        conn, params=(naics,)
    ).fillna("")

    # BARNABAS — format competitor list for the dashboard
    competitors = []
    for _, row in comp_pois.iterrows():
        competitors.append({
            "name":           str(row.get("location_name", "Unknown")),
            "placekey":       str(row.get("placekey", "")),
            "lat":            _safe_float(row.get("latitude")),
            "lon":            _safe_float(row.get("longitude")),
            "size":           _safe_float(row.get("wkt_area_sq_meters")),
            "distance_miles": None,
            "attraction":     None,
        })

    return total_predicted, market_share, competitors


# ================================================================
# BARNABAS — App-facing wrapper
#
# Signature is IDENTICAL to huff_engine.py (V1) so app.py needs
# zero changes when this file replaces the old engine.
# ================================================================
def run_huff_model(
    candidate_lat,
    candidate_lon,
    business_category,
    floor_area,
    db_connection=None,
):
    """
    Required app-facing function — signature identical to V1.

    Parameters
    ----------
    candidate_lat      : float  Latitude of proposed store (WGS-84)
    candidate_lon      : float  Longitude of proposed store (WGS-84)
    business_category  : int    NAICS code, e.g. 4441
    floor_area         : float  Floor area in square metres
    db_connection      : optional  Pass an open sqlite3 connection,
                                   or leave None to open urban_ai_v2.db

    Returns
    -------
    dict — same keys as V1: predicted_visits, market_share,
           competitors, runtime_ms, notes, inputs
    """
    start = time.perf_counter()

    try:
        naics = int(str(business_category).strip())
    except Exception as exc:
        raise ValueError(
            "business_category must be a NAICS code, e.g. 4441"
        ) from exc

    candidate_lat = float(candidate_lat)
    candidate_lon = float(candidate_lon)
    floor_area    = float(floor_area)

    _close_after = False
    if db_connection is None:
        if not os.path.exists(DB_PATH):
            raise FileNotFoundError(
                f"Database not found: {DB_PATH}\n"
                "Run migration_v2.py first to build urban_ai_v2.db"
            )
        db_connection = sqlite3.connect(DB_PATH)
        _close_after  = True

    try:
        total_predicted, market_share, competitors = huff_v2(
            naics         = naics,
            candidate_lat = candidate_lat,
            candidate_lon = candidate_lon,
            floor_area    = floor_area,
            conn          = db_connection,
        )
    finally:
        if _close_after:
            db_connection.close()

    runtime_ms = round((time.perf_counter() - start) * 1000, 2)

    return {
        "predicted_visits": round(total_predicted, 2),
        "market_share":     round(market_share, 6),
        "competitors":      competitors,
        "runtime_ms":       runtime_ms,
        "notes": (
            "V2 engine — all data from SQLite (urban_ai_v2.db). "
            "Competitor utility pre-computed; no CSV files loaded at runtime."
        ),
        "inputs": {
            "candidate_lat":     candidate_lat,
            "candidate_lon":     candidate_lon,
            "business_category": naics,
            "floor_area":        floor_area,
        },
    }


def _safe_float(value):
    try:
        return None if value == "" else float(value)
    except Exception:
        return None


# ================================================================
# SELF-TEST
# ================================================================
if __name__ == "__main__":
    print("=" * 56)
    print("  huff_engine_v2.py — self-test")
    print("  NOTE: requires Elizabeth to complete all 4 TODOs")
    print("  and urban_ai_v2.db to exist (run migration_v2.py)")
    print("=" * 56)

    result = run_huff_model(
        candidate_lat=42.24,
        candidate_lon=-71.78,
        business_category=4441,
        floor_area=1000,
    )

    print(f"  predicted_visits : {result['predicted_visits']}")
    print(f"  market_share     : {result['market_share']}")
    print(f"  runtime_ms       : {result['runtime_ms']} ms")
    print(f"  competitors      : {len(result['competitors'])} shown")
    print("=" * 56)
