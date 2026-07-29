"""
SpoilGuard — SQL Database Setup
==================================
Loads the shipment dataset into a normalized SQLite database and
demonstrates analytical queries used to power the Power BI dashboard.

Run:
    python db_setup.py --data ../data/raw/shipments.csv --db ../data/spoilguard.db
"""

import argparse
import sqlite3
from pathlib import Path

import pandas as pd

SCHEMA_SQL = """
DROP TABLE IF EXISTS shipments;
DROP TABLE IF EXISTS products;
DROP TABLE IF EXISTS routes;

CREATE TABLE products (
    product_category TEXT PRIMARY KEY,
    base_shelf_life_hr REAL,
    safe_temp_min_c REAL,
    safe_temp_max_c REAL
);

CREATE TABLE routes (
    route_id INTEGER PRIMARY KEY AUTOINCREMENT,
    origin TEXT,
    destination TEXT,
    distance_km REAL,
    UNIQUE(origin, destination)
);

CREATE TABLE shipments (
    shipment_id INTEGER PRIMARY KEY,
    date TEXT,
    origin TEXT,
    destination TEXT,
    distance_km REAL,
    product_category TEXT,
    vehicle_type TEXT,
    packaging_quality TEXT,
    avg_speed_kmph REAL,
    transit_duration_hr REAL,
    ambient_temp_c REAL,
    humidity_pct REAL,
    effective_transit_temp_c REAL,
    safe_temp_min_c REAL,
    safe_temp_max_c REAL,
    temp_excursion_hours REAL,
    base_shelf_life_hr REAL,
    remaining_shelf_life_hr REAL,
    spoilage_risk TEXT,
    FOREIGN KEY (product_category) REFERENCES products(product_category)
);

CREATE INDEX idx_shipments_risk ON shipments(spoilage_risk);
CREATE INDEX idx_shipments_route ON shipments(origin, destination);
CREATE INDEX idx_shipments_product ON shipments(product_category);
"""

# Analytical queries — these directly power the Power BI dashboard visuals
ANALYTICAL_QUERIES = {
    "route_wise_risk": """
        SELECT origin, destination,
               COUNT(*) AS total_shipments,
               SUM(CASE WHEN spoilage_risk='High' THEN 1 ELSE 0 END) AS high_risk_count,
               ROUND(100.0 * SUM(CASE WHEN spoilage_risk='High' THEN 1 ELSE 0 END) / COUNT(*), 1) AS high_risk_pct
        FROM shipments
        GROUP BY origin, destination
        ORDER BY high_risk_pct DESC;
    """,
    "product_wise_loss_risk": """
        SELECT product_category,
               COUNT(*) AS total_shipments,
               ROUND(AVG(remaining_shelf_life_hr), 1) AS avg_remaining_shelf_life_hr,
               SUM(CASE WHEN spoilage_risk='High' THEN 1 ELSE 0 END) AS high_risk_count
        FROM shipments
        GROUP BY product_category
        ORDER BY high_risk_count DESC;
    """,
    "monthly_risk_trend": """
        SELECT strftime('%Y-%m', date) AS month,
               COUNT(*) AS total_shipments,
               SUM(CASE WHEN spoilage_risk='High' THEN 1 ELSE 0 END) AS high_risk_count
        FROM shipments
        GROUP BY month
        ORDER BY month;
    """,
    "vehicle_packaging_effectiveness": """
        SELECT vehicle_type, packaging_quality,
               COUNT(*) AS total_shipments,
               ROUND(100.0 * SUM(CASE WHEN spoilage_risk='High' THEN 1 ELSE 0 END) / COUNT(*), 1) AS high_risk_pct
        FROM shipments
        GROUP BY vehicle_type, packaging_quality
        ORDER BY high_risk_pct DESC;
    """,
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=str, default="../data/raw/shipments.csv")
    parser.add_argument("--db", type=str, default="../data/spoilguard.db")
    args = parser.parse_args()

    df = pd.read_csv(args.data)
    db_path = Path(args.db)
    db_path.parent.mkdir(parents=True, exist_ok=True)

    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    cursor.executescript(SCHEMA_SQL)

    # Populate products table (distinct product parameters)
    products_df = df[
        ["product_category", "base_shelf_life_hr", "safe_temp_min_c", "safe_temp_max_c"]
    ].drop_duplicates()
    products_df.to_sql("products", conn, if_exists="append", index=False)

    # Populate routes table
    routes_df = df[["origin", "destination", "distance_km"]].drop_duplicates()
    routes_df.to_sql("routes", conn, if_exists="append", index=False)

    # Populate shipments table
    shipment_cols = [
        "shipment_id", "date", "origin", "destination", "distance_km",
        "product_category", "vehicle_type", "packaging_quality", "avg_speed_kmph",
        "transit_duration_hr", "ambient_temp_c", "humidity_pct",
        "effective_transit_temp_c", "safe_temp_min_c", "safe_temp_max_c",
        "temp_excursion_hours", "base_shelf_life_hr", "remaining_shelf_life_hr",
        "spoilage_risk",
    ]
    df[shipment_cols].to_sql("shipments", conn, if_exists="append", index=False)
    conn.commit()

    print(f"Database created at {db_path.resolve()}")
    print(f"Loaded {len(df)} shipments, {len(products_df)} products, {len(routes_df)} routes.\n")

    # Run and preview each analytical query (also exported as CSV for Power BI)
    export_dir = db_path.parent / "processed"
    export_dir.mkdir(exist_ok=True)
    for name, query in ANALYTICAL_QUERIES.items():
        result = pd.read_sql_query(query, conn)
        print(f"=== {name} ===")
        print(result.head(8).to_string(index=False))
        print()
        out_csv = export_dir / f"{name}.csv"
        result.to_csv(out_csv, index=False)
        print(f"  -> exported to {out_csv}\n")

    conn.close()
    print("Done. Import the CSVs in data/processed/ into Power BI, "
          "or connect Power BI directly to spoilguard.db via ODBC.")


if __name__ == "__main__":
    main()
