"""
SpoilGuard — Cold Chain Shipment Data Simulator
=================================================

Generates a realistic shipment-level dataset by combining:
  1. Real Indian city-pair routes with real approximate road distances
  2. Real historical weather (temperature + humidity) along the route,
     pulled from the free Open-Meteo Historical Weather API (no key needed)
  3. Published food-science shelf-life / safe-temperature parameters
     per product category

Output: data/raw/shipments.csv

Run:
    python data_simulator.py --n_shipments 10000 --start_date 2024-01-01 --end_date 2024-12-31

Note: Requires internet access to archive-api.open-meteo.com. If the API
is unreachable, the script falls back to seasonal climate normals for the
same cities so the pipeline still runs end-to-end.
"""

import argparse
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
import pandas as pd
import requests

RANDOM_SEED = 42
random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# ---------------------------------------------------------------------------
# 1. REAL ROUTES — Indian city pairs with real approximate road distance (km)
#    and lat/lon for weather lookups. Distances are public-knowledge road
#    distances (rounded), not fabricated.
# ---------------------------------------------------------------------------
ROUTES = [
    # (origin, dest, origin_lat, origin_lon, dest_lat, dest_lon, distance_km)
    ("Nashik", "Mumbai", 19.9975, 73.7898, 19.0760, 72.8777, 180),
    ("Ooty", "Chennai", 11.4064, 76.6932, 13.0827, 80.2707, 565),
    ("Nagpur", "Hyderabad", 21.1458, 79.0882, 17.3850, 78.4867, 500),
    ("Pune", "Bengaluru", 18.5204, 73.8567, 12.9716, 77.5946, 840),
    ("Ludhiana", "Delhi", 30.9010, 75.8573, 28.7041, 77.1025, 310),
    ("Kolkata", "Patna", 22.5726, 88.3639, 25.5941, 85.1376, 585),
    ("Ahmedabad", "Mumbai", 23.0225, 72.5714, 19.0760, 72.8777, 525),
    ("Coimbatore", "Kochi", 11.0168, 76.9558, 9.9312, 76.2673, 190),
    ("Indore", "Bhopal", 22.7196, 75.8577, 23.2599, 77.4126, 195),
    ("Guwahati", "Kolkata", 26.1445, 91.7362, 22.5726, 88.3639, 1000),
    ("Jaipur", "Delhi", 26.9124, 75.7873, 28.7041, 77.1025, 280),
    ("Vijayawada", "Chennai", 16.5062, 80.6480, 13.0827, 80.2707, 440),
]

# ---------------------------------------------------------------------------
# 2. PRODUCT CATEGORIES — shelf-life & safe-temperature parameters
#    Base values reflect commonly published food-safety guidance
#    (FSSAI / USDA cold-chain references) — used as simulation anchors.
# ---------------------------------------------------------------------------
PRODUCTS = {
    "Milk & Dairy":       {"base_shelf_life_hr": 48,  "safe_temp_min": 2, "safe_temp_max": 8,  "sensitivity": 0.9},
    "Leafy Greens":       {"base_shelf_life_hr": 96,  "safe_temp_min": 1, "safe_temp_max": 4,  "sensitivity": 0.8},
    "Tomatoes & Produce": {"base_shelf_life_hr": 168, "safe_temp_min": 10, "safe_temp_max": 15, "sensitivity": 0.5},
    "Seafood":            {"base_shelf_life_hr": 36,  "safe_temp_min": 0, "safe_temp_max": 4,  "sensitivity": 1.0},
    "Poultry & Meat":     {"base_shelf_life_hr": 72,  "safe_temp_min": 0, "safe_temp_max": 4,  "sensitivity": 0.85},
    "Frozen Goods":       {"base_shelf_life_hr": 720, "safe_temp_min": -20, "safe_temp_max": -12, "sensitivity": 0.3},
    "Pharma (Vaccines)":  {"base_shelf_life_hr": 240, "safe_temp_min": 2, "safe_temp_max": 8,  "sensitivity": 1.0},
    "Bakery":             {"base_shelf_life_hr": 48,  "safe_temp_min": 10, "safe_temp_max": 20, "sensitivity": 0.4},
}

VEHICLE_TYPES = {
    "Refrigerated Truck":   {"temp_control_quality": 0.9},
    "Insulated Non-Refrigerated": {"temp_control_quality": 0.5},
    "Standard Truck":       {"temp_control_quality": 0.2},
}

PACKAGING_QUALITY = {
    "Premium (Insulated + Gel Packs)": 0.9,
    "Standard": 0.6,
    "Basic": 0.3,
}

# Seasonal climate-normal fallback (approx avg temp °C / humidity % by month,
# generic North/South India averages) — used only if the live API call fails.
FALLBACK_CLIMATE = {
    1: (18, 55), 2: (21, 50), 3: (26, 45), 4: (32, 40), 5: (36, 35),
    6: (33, 60), 7: (30, 75), 8: (29, 78), 9: (28, 70), 10: (26, 60),
    11: (22, 55), 12: (18, 55),
}


def fetch_weather(lat, lon, date_str, cache={}):
    """Fetch historical daily mean temperature (°C) and humidity (%) for a
    location/date using the free Open-Meteo Historical Weather API.
    Falls back to seasonal climate normals if the request fails."""
    key = (round(lat, 2), round(lon, 2), date_str)
    if key in cache:
        return cache[key]

    try:
        url = (
            "https://archive-api.open-meteo.com/v1/archive"
            f"?latitude={lat}&longitude={lon}"
            f"&start_date={date_str}&end_date={date_str}"
            "&daily=temperature_2m_mean,relative_humidity_2m_mean"
            "&timezone=Asia%2FKolkata"
        )
        resp = requests.get(url, timeout=8)
        resp.raise_for_status()
        daily = resp.json()["daily"]
        temp = daily["temperature_2m_mean"][0]
        humidity = daily["relative_humidity_2m_mean"][0]
        if temp is None or humidity is None:
            raise ValueError("null weather values")
        cache[key] = (temp, humidity)
        return temp, humidity
    except Exception:
        month = int(date_str.split("-")[1])
        temp, humidity = FALLBACK_CLIMATE[month]
        # small deterministic jitter so cities don't look identical
        jitter = (lat + lon) % 3
        cache[key] = (temp + jitter - 1, humidity)
        return cache[key]


def simulate_shipment(shipment_id, route, product_name, date_str):
    origin, dest, o_lat, o_lon, d_lat, d_lon, distance_km = route
    product = PRODUCTS[product_name]

    vehicle_type = random.choices(
        list(VEHICLE_TYPES.keys()), weights=[0.35, 0.35, 0.30]
    )[0]
    packaging = random.choices(
        list(PACKAGING_QUALITY.keys()), weights=[0.25, 0.45, 0.30]
    )[0]

    # Average speed varies by vehicle + realistic road conditions (40-55 km/h)
    avg_speed = np.random.uniform(38, 55)
    base_transit_hr = distance_km / avg_speed
    # Loading/unloading + stops add overhead
    stop_overhead_hr = np.random.uniform(1, 8)
    transit_duration_hr = base_transit_hr + stop_overhead_hr

    # Weather at midpoint of route (approx average of origin/dest)
    mid_lat, mid_lon = (o_lat + d_lat) / 2, (o_lon + d_lon) / 2
    ambient_temp, humidity = fetch_weather(mid_lat, mid_lon, date_str)

    # Effective in-transit temperature depends on vehicle control quality
    control_quality = VEHICLE_TYPES[vehicle_type]["temp_control_quality"]
    pack_quality = PACKAGING_QUALITY[packaging]
    # Higher control/packaging quality pulls effective temp toward the safe band
    safe_mid = (product["safe_temp_min"] + product["safe_temp_max"]) / 2
    effective_temp = (
        ambient_temp * (1 - control_quality * pack_quality)
        + safe_mid * (control_quality * pack_quality)
    )
    effective_temp += np.random.normal(0, 1.2)  # sensor/measurement noise

    # Temperature excursion: how far outside safe band, weighted by duration
    if effective_temp > product["safe_temp_max"]:
        excursion_severity = effective_temp - product["safe_temp_max"]
    elif effective_temp < product["safe_temp_min"]:
        excursion_severity = product["safe_temp_min"] - effective_temp
    else:
        excursion_severity = 0.0
    temp_excursion_hours = excursion_severity * (transit_duration_hr / 10)

    humidity_deviation = max(0, humidity - 65) / 10  # high humidity accelerates spoilage

    # Remaining shelf life calculation (grounded formula + realistic noise)
    sensitivity = product["sensitivity"]
    shelf_life_consumed_fraction = (
        (transit_duration_hr / product["base_shelf_life_hr"])
        + (temp_excursion_hours / product["base_shelf_life_hr"]) * sensitivity * 3
        + (humidity_deviation / 100) * sensitivity
    )
    # Unobserved real-world factors (driver behavior, door-opening frequency,
    # batch-to-batch handling variation) are NOT in the feature set at all --
    # this keeps the label from being a pure deterministic function of the
    # recorded features, which is what happens in real cold-chain data too.
    hidden_handling_factor = np.clip(np.random.normal(1.0, 0.22), 0.55, 1.6)
    shelf_life_consumed_fraction *= hidden_handling_factor
    shelf_life_consumed_fraction = np.clip(shelf_life_consumed_fraction, 0, 1.5)

    remaining_shelf_life_hr = max(
        0, product["base_shelf_life_hr"] * (1 - shelf_life_consumed_fraction)
    )
    # Proportional measurement/reporting noise (bigger products -> bigger noise band)
    remaining_shelf_life_hr += np.random.normal(0, product["base_shelf_life_hr"] * 0.05)
    remaining_shelf_life_hr = max(0, round(remaining_shelf_life_hr, 1))

    # Risk label from remaining shelf life as a fraction of base shelf life
    pct_remaining = remaining_shelf_life_hr / product["base_shelf_life_hr"]
    if pct_remaining < 0.15:
        risk_label = "High"
    elif pct_remaining < 0.4:
        risk_label = "Medium"
    else:
        risk_label = "Low"

    return {
        "shipment_id": shipment_id,
        "date": date_str,
        "origin": origin,
        "destination": dest,
        "distance_km": distance_km,
        "product_category": product_name,
        "vehicle_type": vehicle_type,
        "packaging_quality": packaging,
        "avg_speed_kmph": round(avg_speed, 1),
        "transit_duration_hr": round(transit_duration_hr, 2),
        "ambient_temp_c": round(ambient_temp, 1),
        "humidity_pct": round(humidity, 1),
        "effective_transit_temp_c": round(effective_temp, 1),
        "safe_temp_min_c": product["safe_temp_min"],
        "safe_temp_max_c": product["safe_temp_max"],
        "temp_excursion_hours": round(temp_excursion_hours, 2),
        "base_shelf_life_hr": product["base_shelf_life_hr"],
        "remaining_shelf_life_hr": remaining_shelf_life_hr,
        "spoilage_risk": risk_label,
    }


def daterange_sample(start_date, end_date, n):
    start = datetime.strptime(start_date, "%Y-%m-%d")
    end = datetime.strptime(end_date, "%Y-%m-%d")
    total_days = (end - start).days
    return [
        (start + timedelta(days=random.randint(0, total_days))).strftime("%Y-%m-%d")
        for _ in range(n)
    ]


def main():
    parser = argparse.ArgumentParser(description="Generate SpoilGuard shipment dataset")
    parser.add_argument("--n_shipments", type=int, default=10000)
    parser.add_argument("--start_date", type=str, default="2024-01-01")
    parser.add_argument("--end_date", type=str, default="2024-12-31")
    parser.add_argument("--out", type=str, default="../data/raw/shipments.csv")
    args = parser.parse_args()

    dates = daterange_sample(args.start_date, args.end_date, args.n_shipments)
    records = []

    print(f"Generating {args.n_shipments} shipments... (this may take a while "
          f"due to weather API calls; cached per lat/lon/date)")

    for i, date_str in enumerate(dates):
        route = random.choice(ROUTES)
        product_name = random.choice(list(PRODUCTS.keys()))
        record = simulate_shipment(i + 1, route, product_name, date_str)
        records.append(record)

        if (i + 1) % 500 == 0:
            print(f"  {i + 1}/{args.n_shipments} shipments generated...")
            time.sleep(0.1)  # gentle pacing for the free API

    df = pd.DataFrame(records)
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)

    print(f"\nDone. Saved {len(df)} records to {out_path.resolve()}")
    print("\nRisk label distribution:")
    print(df["spoilage_risk"].value_counts())


if __name__ == "__main__":
    main()
