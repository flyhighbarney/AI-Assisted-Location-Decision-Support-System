"""
predict_site.py — Huff Gravity Model Site Prediction Engine
=============================================================
Worcester Urban Analytics | INT6940 Capstone Project
Team 1: Eden Wenberg, Elizabeth Cummins, Johnson Ebe, Barnabas Ebanezar Johnson

This script predicts the expected customer visits for a proposed new business
location in Worcester, MA using the Huff Gravity Model.

The Huff Model calculates:
    P(ij) = U(j) / Σ U(all stores in category)
    where U = Area^alpha / Distance^beta

Usage:
    python predict_site.py

The script will prompt for:
    1. Latitude & Longitude of the proposed site
    2. Business category (top_category or NAICS code)
    3. Store size in square meters
"""

import pandas as pd
import numpy as np
import os
import sys

# ============================================================
# CONFIGURATION — File paths
# ============================================================
# Auto-detect: script can live in the repo root OR inside the Data folder.
# It checks for a "Data" subfolder first; if not found, looks in the script's own directory.
_SCRIPT_DIR = os.path.dirname(os.path.abspath(__file__))
_DATA_SUBDIR = os.path.join(_SCRIPT_DIR, "Data")

if os.path.isdir(_DATA_SUBDIR):
    DATA_DIR = _DATA_SUBDIR
else:
    DATA_DIR = _SCRIPT_DIR

FILE_CBGS       = os.path.join(DATA_DIR, "worcester_cbgs.csv")
FILE_POIS       = os.path.join(DATA_DIR, "worcester_pois.csv")
FILE_VISITS     = os.path.join(DATA_DIR, "worcester_cbg_poi_visits.csv")
FILE_DISTANCES  = os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv")
FILE_DISTANCES_ZIP = os.path.join(DATA_DIR, "worcester_cbg_poi_distance.csv.zip")
FILE_PARAMS     = os.path.join(DATA_DIR, "calibrated_parameters_filtered.csv")

# Fallback parameters when category is not found in the calibrated file
DEFAULT_ALPHA = 1.0
DEFAULT_BETA  = 1.0


# ============================================================
# STEP A — Load & Prepare Data
# ============================================================
def load_data():
    """Load all required datasets and return them as a dictionary."""
    print(f"Loading datasets from: {DATA_DIR}")

    # --- Determine distance file: prefer raw CSV, fall back to .zip ---
    if os.path.exists(FILE_DISTANCES):
        distance_source = FILE_DISTANCES
    elif os.path.exists(FILE_DISTANCES_ZIP):
        distance_source = FILE_DISTANCES_ZIP
    else:
        distance_source = None

    # --- Check that all required files exist before loading ---
    required_files = {
        "worcester_cbgs.csv":                FILE_CBGS,
        "worcester_pois.csv":                FILE_POIS,
        "worcester_cbg_poi_visits.csv":      FILE_VISITS,
        "calibrated_parameters_filtered.csv": FILE_PARAMS,
    }

    missing = [name for name, path in required_files.items() if not os.path.exists(path)]
    if distance_source is None:
        missing.append("worcester_cbg_poi_distance.csv (or .zip)")

    if missing:
        print("\n  ❌ ERROR: The following required data files were not found:\n")
        for name in missing:
            print(f"     - {name}")
        print(f"\n  The script is looking in: {DATA_DIR}")
        print(f"  Please ensure the data files are in a 'Data' subfolder next to")
        print(f"  predict_site.py, or in the same folder as predict_site.py.\n")
        sys.exit(1)

    cbgs      = pd.read_csv(FILE_CBGS)
    pois      = pd.read_csv(FILE_POIS)
    visits    = pd.read_csv(FILE_VISITS)
    params    = pd.read_csv(FILE_PARAMS)

    # Load distances — pandas can read directly from a zip file
    print(f"  Loading distance matrix from: {os.path.basename(distance_source)}")
    distances = pd.read_csv(distance_source)

    # Clean: remove zero-area POIs and zero-distance records (consistent with calibration notebook)
    pois      = pois[pois['wkt_area_sq_meters'] > 0]
    distances = distances[distances['distance_m'] > 0]
    visits    = visits[visits['visit_count'] > 0]

    print(f"  CBGs: {len(cbgs)} | POIs: {len(pois)} | "
          f"Visit records: {len(visits)} | Distance records: {len(distances)}")
    return {
        "cbgs": cbgs,
        "pois": pois,
        "visits": visits,
        "distances": distances,
        "params": params,
    }


# ============================================================
# STEP A.1 — Build CBG Centroid Table
# ============================================================
def build_cbg_centroids(pois, cbgs):
    """
    Approximate CBG centroids by averaging the lat/lon of all POIs within each CBG.
    For CBGs with no POIs, interpolate from same-tract neighbors.
    """
    centroids = (
        pois.groupby("poi_cbg")
        .agg(latitude=("latitude", "mean"), longitude=("longitude", "mean"))
        .reset_index()
        .rename(columns={"poi_cbg": "cbg"})
    )

    # Identify any CBGs missing from the centroid table
    all_cbgs = set(cbgs["cbg"])
    covered  = set(centroids["cbg"])
    missing  = all_cbgs - covered

    if missing:
        # Interpolate from same census-tract neighbors (first 10 digits of FIPS)
        for m in missing:
            tract_prefix = str(m)[:10]  # e.g. "2502773020"
            neighbors = centroids[centroids["cbg"].astype(str).str.startswith(tract_prefix)]
            if len(neighbors) > 0:
                avg_lat = neighbors["latitude"].mean()
                avg_lon = neighbors["longitude"].mean()
            else:
                # Ultimate fallback: Worcester city center
                avg_lat, avg_lon = 42.2626, -71.8023
            new_row = pd.DataFrame({"cbg": [m], "latitude": [avg_lat], "longitude": [avg_lon]})
            centroids = pd.concat([centroids, new_row], ignore_index=True)

    return centroids


# ============================================================
# STEP B — User Input
# ============================================================
def get_user_input(params):
    """Prompt the user for site coordinates, category, and store size."""
    print("\n" + "=" * 60)
    print("  WORCESTER SITE PREDICTION ENGINE — Huff Gravity Model")
    print("=" * 60)

    # --- Coordinates (with Worcester bounding-box validation) ---
    # Worcester, MA approximate bounds:
    #   Latitude:  42.20 – 42.35
    #   Longitude: -71.88 – -71.72
    WORCESTER_LAT_RANGE = (42.15, 42.40)
    WORCESTER_LON_RANGE = (-71.92, -71.68)

    while True:
        lat = float(input("\nEnter Latitude  (e.g. 42.2625): "))
        lon = float(input("Enter Longitude (e.g. -71.8023): "))

        lat_ok = WORCESTER_LAT_RANGE[0] <= lat <= WORCESTER_LAT_RANGE[1]
        lon_ok = WORCESTER_LON_RANGE[0] <= lon <= WORCESTER_LON_RANGE[1]

        if lat_ok and lon_ok:
            break

        print(f"\n  ⚠ WARNING: ({lat}, {lon}) is outside the Worcester study area!")
        if not lat_ok:
            print(f"    Latitude should be between {WORCESTER_LAT_RANGE[0]} and {WORCESTER_LAT_RANGE[1]}")
        if not lon_ok:
            print(f"    Longitude should be between {WORCESTER_LON_RANGE[0]} and {WORCESTER_LON_RANGE[1]}")
            print(f"    (Did you forget the minus sign? Worcester longitudes are around -71.80)")

        retry = input("\n  Type 'y' to re-enter coordinates, or anything else to proceed anyway: ").strip().lower()
        if retry != "y":
            print("  Proceeding with out-of-range coordinates...")
            break

    # --- Category ---
    print("\nAvailable calibrated categories:")
    for i, row in params.iterrows():
        print(f"  [{i+1:>2}] {row['top_category']}  (NAICS {row['NAICS code']})")
    print(f"  [ 0] Other / Enter manually (will use fallback α=1.0, β=1.0)")

    choice = input("\nSelect a category number, or type a category name / NAICS code: ").strip()

    alpha, beta, category_name, naics_code = DEFAULT_ALPHA, DEFAULT_BETA, None, None
    using_fallback = True

    # Try numeric selection first
    if choice.isdigit():
        idx = int(choice)
        if 1 <= idx <= len(params):
            row = params.iloc[idx - 1]
            alpha        = row["alpha"]
            beta         = row["beta"]
            category_name = row["top_category"]
            naics_code    = row["NAICS code"]
            using_fallback = False
        # idx == 0 means fallback, keep defaults

    # Try matching by name or NAICS
    if using_fallback and not choice.isdigit():
        # Try NAICS match
        try:
            naics_try = int(choice)
            match = params[params["NAICS code"] == naics_try]
            if len(match) > 0:
                row = match.iloc[0]
                alpha, beta  = row["alpha"], row["beta"]
                category_name = row["top_category"]
                naics_code    = row["NAICS code"]
                using_fallback = False
        except ValueError:
            pass

        # Try name match (case-insensitive substring)
        if using_fallback:
            match = params[params["top_category"].str.lower().str.contains(choice.lower())]
            if len(match) > 0:
                row = match.iloc[0]
                alpha, beta  = row["alpha"], row["beta"]
                category_name = row["top_category"]
                naics_code    = row["NAICS code"]
                using_fallback = False

    if using_fallback:
        category_name = input("Enter the exact top_category name for filtering competitors: ").strip()

    # --- Store Size (must be positive) ---
    while True:
        size = float(input("\nEnter store size in square meters (e.g. 2500): "))
        if size > 0:
            break
        print("  ⚠ Store size must be greater than 0. Please re-enter.")

    print("\n--- Input Summary ---")
    print(f"  Location    : ({lat}, {lon})")
    print(f"  Category    : {category_name} (NAICS {naics_code})")
    print(f"  Store Size  : {size:,.0f} sq meters")
    print(f"  Alpha (α)   : {alpha}")
    print(f"  Beta  (β)   : {beta}")
    if using_fallback:
        print("  ⚠ WARNING: Using fallback parameters (α=1.0, β=1.0) — "
              "category not found in calibrated set.")
    print()

    return {
        "lat": lat,
        "lon": lon,
        "category": category_name,
        "naics_code": naics_code,
        "size": size,
        "alpha": alpha,
        "beta": beta,
        "using_fallback": using_fallback,
    }


# ============================================================
# STEP C — Distance Calculation (Projected Euclidean)
# ============================================================
def calc_euclidean_distance_m(lat1, lon1, lat2, lon2):
    """
    Calculate approximate Euclidean distance in meters between two WGS84 points.
    Uses a local projected approximation (meters per degree at Worcester's latitude).

    At latitude ~42.26°N:
        1° latitude  ≈ 111,035 m
        1° longitude ≈ 111,035 * cos(42.26°) ≈ 82,220 m
    """
    lat_ref = np.radians(42.26)  # Worcester reference latitude
    m_per_deg_lat = 111_035.0
    m_per_deg_lon = 111_035.0 * np.cos(lat_ref)

    dy = (lat2 - lat1) * m_per_deg_lat
    dx = (lon2 - lon1) * m_per_deg_lon

    return np.sqrt(dx**2 + dy**2)


def compute_new_site_distances(site_lat, site_lon, cbg_centroids):
    """
    Calculate distance in meters from the new site to every CBG centroid.
    Returns a DataFrame with columns: [cbg, distance_m].
    """
    distances = calc_euclidean_distance_m(
        site_lat, site_lon,
        cbg_centroids["latitude"].values,
        cbg_centroids["longitude"].values,
    )

    result = cbg_centroids[["cbg"]].copy()
    result["distance_m"] = distances

    # Replace zero distances with a small value to avoid division by zero
    result.loc[result["distance_m"] < 1.0, "distance_m"] = 1.0

    return result


# ============================================================
# STEP D — Huff Model Probability Calculation
# ============================================================
def run_huff_model(user_input, data, cbg_centroids):
    """
    For every CBG, compute the market share probability (Pij) for the new store.

    Steps:
        1. Compute the new store's utility for each CBG.
        2. Compute the sum of existing competitor utilities for each CBG.
        3. Pij = U_new / (U_new + Σ U_existing)
    """
    alpha = user_input["alpha"]
    beta  = user_input["beta"]
    category = user_input["category"]

    pois      = data["pois"]
    distances = data["distances"]

    # --- Identify competitors in the same category ---
    competitors = pois[pois["top_category"] == category][["placekey", "wkt_area_sq_meters"]].copy()
    competitor_pks = competitors["placekey"].unique()
    print(f"Competitors found in '{category}': {len(competitor_pks)}")

    if len(competitor_pks) == 0:
        print("  ⚠ No existing competitors found. New store will have 100% market share.")

    # --- New store distances to all CBGs ---
    new_distances = compute_new_site_distances(
        user_input["lat"], user_input["lon"], cbg_centroids
    )

    # --- New store utility per CBG ---
    new_distances["u_new"] = (
        (user_input["size"] ** alpha) / (new_distances["distance_m"] ** beta)
    )

    # --- Existing competitor utilities per CBG ---
    # Filter distance matrix to only competitors in this category
    comp_dist = distances[distances["placekey"].isin(competitor_pks)].copy()
    comp_dist = comp_dist.merge(competitors, on="placekey")

    # Compute utility for each competitor-CBG pair
    comp_dist["utility"] = (
        (comp_dist["wkt_area_sq_meters"] ** alpha) / (comp_dist["distance_m"] ** beta)
    )

    # Sum of all competitor utilities per CBG (the denominator component)
    sum_existing = (
        comp_dist.groupby("GEOID10")["utility"]
        .sum()
        .reset_index()
        .rename(columns={"GEOID10": "cbg", "utility": "u_existing_sum"})
    )

    # --- Merge and compute probability ---
    result = new_distances[["cbg", "distance_m", "u_new"]].merge(
        sum_existing, on="cbg", how="left"
    )

    # CBGs with no existing competitors get 0 for existing utility
    result["u_existing_sum"] = result["u_existing_sum"].fillna(0)

    # Huff probability: P = U_new / (U_new + Σ U_existing)
    result["p_new"] = result["u_new"] / (result["u_new"] + result["u_existing_sum"])

    return result


# ============================================================
# STEP E — Demand Estimation (Predicted Visits)
# ============================================================
def estimate_demand(huff_result, user_input, data):
    """
    Convert probabilities into predicted visits.

    For each CBG:
        predicted_visits = P_new × total_category_visits_from_that_CBG
    """
    category = user_input["category"]
    pois     = data["pois"]
    visits   = data["visits"]

    # Get placekeys for this category
    cat_pks = pois[pois["top_category"] == category]["placekey"].unique()

    # Total historical visits per CBG for this category
    cat_visits = visits[visits["placekey"].isin(cat_pks)]
    cbg_demand = (
        cat_visits.groupby("visitor_home_cbg")["visit_count"]
        .sum()
        .reset_index()
        .rename(columns={"visitor_home_cbg": "cbg", "visit_count": "total_category_visits"})
    )

    # Merge with Huff probabilities
    result = huff_result.merge(cbg_demand, on="cbg", how="left")
    result["total_category_visits"] = result["total_category_visits"].fillna(0)

    # Predicted visits = probability × historical demand
    result["predicted_visits"] = result["p_new"] * result["total_category_visits"]

    # Add demographic info for the report
    cbgs = data["cbgs"]
    result = result.merge(
        cbgs[["cbg", "total_population", "median_household_income", "income_q"]],
        on="cbg",
        how="left",
    )

    return result.sort_values("predicted_visits", ascending=False)


# ============================================================
# OUTPUT — Print Results
# ============================================================
def print_results(result, user_input):
    """Display a summary report of the site prediction."""
    total_predicted = result["predicted_visits"].sum()
    total_demand    = result["total_category_visits"].sum()
    avg_prob        = result.loc[result["total_category_visits"] > 0, "p_new"].mean()

    print("=" * 70)
    print("  SITE PREDICTION REPORT")
    print("=" * 70)
    print(f"  Location        : ({user_input['lat']}, {user_input['lon']})")
    print(f"  Category        : {user_input['category']}")
    print(f"  Store Size      : {user_input['size']:,.0f} sq meters")
    print(f"  Parameters      : α = {user_input['alpha']}, β = {user_input['beta']}")
    if user_input["using_fallback"]:
        print(f"  ⚠ Fallback parameters in use — results may be less reliable.")
    print("-" * 70)
    print(f"  Total Category Demand (historical visits) : {total_demand:,.0f}")
    print(f"  Total Predicted Visits for New Site        : {total_predicted:,.1f}")
    if total_demand > 0:
        capture_rate = (total_predicted / total_demand) * 100
        print(f"  Overall Market Capture Rate                : {capture_rate:.1f}%")
    print(f"  Avg. Probability (active CBGs)             : {avg_prob:.3f}")
    print("-" * 70)

    # Top 10 neighborhoods
    top10 = result[result["predicted_visits"] > 0].head(10)
    if len(top10) > 0:
        print("\n  Top 10 Neighborhoods by Predicted Visits:")
        print(f"  {'CBG':<16} {'Pop':>6} {'Income':>10} {'Dist(m)':>9} "
              f"{'P(new)':>8} {'Cat.Visits':>10} {'Predicted':>10}")
        print("  " + "-" * 75)
        for _, r in top10.iterrows():
            print(f"  {int(r['cbg']):<16} {int(r['total_population']):>6} "
                  f"{'${:,.0f}'.format(r['median_household_income']):>10} "
                  f"{r['distance_m']:>9,.0f} {r['p_new']:>8.3f} "
                  f"{int(r['total_category_visits']):>10} "
                  f"{r['predicted_visits']:>10.1f}")

    print("\n" + "=" * 70)

    return total_predicted


def save_results(result, user_input, output_path="prediction_output.csv"):
    """Save the full results table to CSV."""
    result.to_csv(output_path, index=False)
    print(f"  Full results saved to: {output_path}")


# ============================================================
# MAIN
# ============================================================
def main():
    # Step A: Load data
    data = load_data()

    # Step A.1: Build CBG centroids
    cbg_centroids = build_cbg_centroids(data["pois"], data["cbgs"])

    # Step B: Get user input
    user_input = get_user_input(data["params"])

    # Step D: Run Huff Model
    print("Running Huff Model simulation...")
    huff_result = run_huff_model(user_input, data, cbg_centroids)

    # Step E: Estimate demand
    print("Estimating demand...")
    result = estimate_demand(huff_result, user_input, data)

    # Output
    total = print_results(result, user_input)
    save_results(result, user_input)

    return total


if __name__ == "__main__":
    main()
