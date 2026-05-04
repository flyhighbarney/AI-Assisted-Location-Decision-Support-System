"""
Huff Model Engine — ALSDS Baseline Version

Estimates predicted visits to a hypothetical new retail location using the
Huff Gravity Model.

Given a candidate store's location, NAICS category, and floor area, the model:
1. Finds existing competing POIs in the same NAICS category.
2. Computes attractiveness of existing competitors.
3. Computes attractiveness of the proposed candidate store.
4. Estimates the probability that consumers from each CBG visit the candidate.
5. Aggregates predicted visits across Worcester CBGs.

Spatial reference:
- Candidate input coordinates are expected in WGS84 (EPSG:4326).
- CBG geometries are projected to UTM Zone 19N (EPSG:26919) for distance calculations in meters.

Study area:
- Worcester, MA

Important:
- Teams may replace the internals of this file.
- However, they should keep the run_huff_model(...) function signature and return structure.
"""

import time
from pathlib import Path

import geopandas as gpd
import pandas as pd
from shapely.geometry import Point


# ---------------------------------------------------------------------
# Data loading
# ---------------------------------------------------------------------
# Files are loaded once at import time instead of being reloaded during
# every model call. This improves response time for the deployed app.

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "Data"

PARAMS_PATH = DATA_DIR / "calibrated_parameters_filtered.csv"
POIS_PATH = DATA_DIR / "worcester_pois.csv"
DISTANCE_PATH = DATA_DIR / "worcester_cbg_poi_distance.csv.zip"
VISITS_PATH = DATA_DIR / "worcester_cbg_poi_visits.csv"
GEOJSON_PATH = DATA_DIR / "worcester_cbgs_map.geojson"


params = pd.read_csv(PARAMS_PATH)
pois = pd.read_csv(POIS_PATH)
dist_matrix = pd.read_csv(DISTANCE_PATH, compression="zip")
visits = pd.read_csv(VISITS_PATH)

# Read GeoJSON and project to UTM 19N for distance calculations in meters.
try:
    gdf_cbgs = gpd.read_file(GEOJSON_PATH, engine="pyogrio")
except Exception:
    # Fallback for environments where pyogrio is unavailable.
    gdf_cbgs = gpd.read_file(GEOJSON_PATH)

gdf_cbgs = gdf_cbgs.to_crs("EPSG:26919")

gdf_cbgs.rename(
    columns={
        "GEOID10": "cbg_id",
        "INTPTLAT10": "centroid_Y",
        "INTPTLON10": "centroid_X",
    },
    inplace=True,
)

# Standardize column names across DataFrames.
dist_matrix.rename(columns={"GEOID10": "cbg_id"}, inplace=True)
visits.rename(columns={"visitor_home_cbg": "cbg_id"}, inplace=True)

# Standardize CBG IDs to strings to prevent merge type conflicts.
dist_matrix["cbg_id"] = dist_matrix["cbg_id"].astype(str)
visits["cbg_id"] = visits["cbg_id"].astype(str)
gdf_cbgs["cbg_id"] = gdf_cbgs["cbg_id"].astype(str)


# ---------------------------------------------------------------------
# Core Huff computation
# ---------------------------------------------------------------------

def huff(naics, X, Y, Aj, params, pois, dist_matrix, visits, geocbgs):
    """
    Estimate total predicted visits to a hypothetical new retail location.

    Parameters
    ----------
    naics : int
        NAICS code identifying the retail category.
    X : float
        Longitude of the candidate store in WGS84 (EPSG:4326).
    Y : float
        Latitude of the candidate store in WGS84 (EPSG:4326).
    Aj : float
        Floor area of the candidate store in square meters.
    params : pd.DataFrame
        Calibrated alpha/beta parameters per NAICS code.
    pois : pd.DataFrame
        POI table with floor area and NAICS codes.
    dist_matrix : pd.DataFrame
        Precomputed CBG-to-POI distance table in meters.
    visits : pd.DataFrame
        Observed CBG-to-POI visit counts.
    geocbgs : gpd.GeoDataFrame
        CBG polygons projected to UTM 19N (EPSG:26919).

    Returns
    -------
    tuple
        total_predicted_visits, market_share_proxy, competitors
    """

    # Step 1: Retrieve calibrated model parameters for the given NAICS.
    matching_params = params.loc[params["NAICS code"] == naics]

    if matching_params.empty:
        raise ValueError(
            f"No calibrated alpha/beta parameters found for NAICS code {naics}."
        )

    row = matching_params.iloc[0]
    alpha, beta = row["alpha"], row["beta"]

    # Step 2: Identify competing POIs in the same NAICS category.
    naics_pois = pois[pois["naics_code"] == naics][
        ["placekey", "wkt_area_sq_meters", "location_name", "latitude", "longitude"]
    ].copy()

    if naics_pois.empty:
        raise ValueError(f"No competing POIs found for NAICS code {naics}.")

    # Using a set improves isin(...) lookup speed.
    relevant_placekeys = set(naics_pois["placekey"])

    # Step 3: Build CBG × competitor-POI working table.
    # Filter dist_matrix first to reduce merge size.
    temp = dist_matrix[dist_matrix["placekey"].isin(relevant_placekeys)].merge(
        naics_pois[["placekey", "wkt_area_sq_meters"]],
        on="placekey",
    )

    # Step 4: Join observed visit counts.
    # Left join preserves all CBG-POI pairs. Missing observed visits are treated as 0.
    relevant_visits = visits[visits["placekey"].isin(relevant_placekeys)]
    temp = temp.merge(
        relevant_visits[["cbg_id", "placekey", "visit_count"]],
        on=["cbg_id", "placekey"],
        how="left",
    )
    temp["visit_count"] = temp["visit_count"].fillna(0)

    # Step 5: Compute attraction utility for each CBG-POI pair:
    # Uik = Ak^alpha / dik^beta
    # Distances are clipped at 100m to prevent division instability.
    temp["uik"] = (
        temp["wkt_area_sq_meters"] ** alpha
    ) / ((temp["distance_m"].clip(lower=100)) ** beta)

    # Step 6: Aggregate existing competitor utility and observed visits to CBG level.
    temp = (
        temp.groupby(["cbg_id"])[["uik", "visit_count"]]
        .sum()
        .reset_index()
        .rename(columns={"uik": "sum_uik", "visit_count": "sum_visits"})
    )

    # CBGs with zero observed category visits are excluded.
    temp = temp[temp["sum_visits"] != 0]

    # Step 7: Compute distance from each CBG geometry to candidate store.
    # Candidate location starts as WGS84 lon/lat and is projected to UTM 19N.
    new_poi = Point(X, Y)
    poi_gdf = gpd.GeoDataFrame([{"geometry": new_poi}], crs="EPSG:4326").to_crs(
        "EPSG:26919"
    )
    projected_poi = poi_gdf.geometry.iloc[0]

    cbg_geometry = geocbgs[["cbg_id", "geometry"]].copy()
    cbg_geometry["distance"] = cbg_geometry["geometry"].distance(projected_poi)

    temp = temp.merge(cbg_geometry[["cbg_id", "distance"]], on="cbg_id")

    # Step 8: Compute candidate store utility:
    # Uij = Aj^alpha / dij^beta
    Aj_alpha = Aj ** alpha
    temp["uij"] = Aj_alpha / ((temp["distance"].clip(lower=100)) ** beta)

    # Step 9: Compute predicted visits from each CBG:
    # Pij = Uij / (Uij + sum_Uik)
    # predicted visits = Pij * observed category visits from that CBG
    temp["predicted_visits"] = (
        temp["uij"] * temp["sum_visits"]
    ) / (temp["uij"] + temp["sum_uik"])

    total_predicted_visits = float(temp["predicted_visits"].sum())

    # Simple market-share proxy:
    # candidate predicted visits divided by total observed visits used in the calculation.
    total_market_visits = float(temp["sum_visits"].sum())
    market_share_proxy = (
        total_predicted_visits / total_market_visits
        if total_market_visits > 0
        else 0.0
    )

    # Competitor sample for map/table display.
    # This is intentionally lightweight for the baseline dashboard.
    # competitors aresample POIs from the same NAICS code as user input naics
    competitors = (
        naics_pois.head(20)
        .fillna("")
        .to_dict(orient="records")
    )

    cleaned_competitors = []
    for comp in competitors:
        cleaned_competitors.append(
            {
                "name": str(comp.get("location_name", "Unknown")),
                "placekey": str(comp.get("placekey", "")),
                "lat": _safe_float(comp.get("latitude")),
                "lon": _safe_float(comp.get("longitude")),
                "size": _safe_float(comp.get("wkt_area_sq_meters")),
                "distance_miles": None,
                "attraction": None,
            }
        )

    return total_predicted_visits, market_share_proxy, cleaned_competitors


# ---------------------------------------------------------------------
# App-facing wrapper
# ---------------------------------------------------------------------

def run_huff_model(
    candidate_lat,
    candidate_lon,
    business_category,
    floor_area,
    db_connection=None,
):
    """
    Required app-facing function.

    The Flask app calls this function directly.

    Parameters
    ----------
    candidate_lat : float
        Candidate store latitude.
    candidate_lon : float
        Candidate store longitude.
    business_category : str or int
        For this baseline, this should be a NAICS code such as 4441.
    floor_area : float
        Candidate store floor area in square meters.
    db_connection : optional
        Reserved for team implementations that use Azure SQL directly.

    Returns
    -------
    dict
        Structured result used by the dashboard and chatbot.
    """

    start_time = time.perf_counter()

    try:
        naics = int(str(business_category).strip())
    except Exception as exc:
        raise ValueError(
            "business_category must be a NAICS code for this baseline, for example: 4441."
        ) from exc

    candidate_lat = float(candidate_lat)
    candidate_lon = float(candidate_lon)
    floor_area = float(floor_area)

    total_predicted_visits, market_share, competitors = huff(
        naics=naics,
        X=candidate_lon,
        Y=candidate_lat,
        Aj=floor_area,
        params=params,
        pois=pois,
        dist_matrix=dist_matrix,
        visits=visits,
        geocbgs=gdf_cbgs,
    )

    runtime_ms = round((time.perf_counter() - start_time) * 1000, 2)

    return {
        "predicted_visits": round(total_predicted_visits, 2),
        "market_share": round(market_share, 6),
        "competitors": competitors,
        "runtime_ms": runtime_ms,
        "notes": (
            "Baseline Huff model completed successfully. "
            "This version uses local CSV/GeoJSON files loaded from the Data folder. "
            "Teams may optimize this implementation by migrating data to Azure SQL, "
            "adding indexes, precomputing intermediate tables, and improving runtime."
        ),
        "inputs": {
            "candidate_lat": candidate_lat,
            "candidate_lon": candidate_lon,
            "business_category": naics,
            "floor_area": floor_area,
        },
    }


def _safe_float(value):
    try:
        if value == "":
            return None
        return float(value)
    except Exception:
        return None


# Local quick test, the result should be 16.99:
# result = run_huff_model(
#     candidate_lat=42.24,
#     candidate_lon=-71.78,
#     business_category=4441,
#     floor_area=1000,
# )
# print(result)
