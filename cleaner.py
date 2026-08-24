"""
data_utils.py

Shared data loading, cleaning, and feature definitions used by BOTH app.py
(the Streamlit app) and tune_model.py (the hyperparameter tuning script).

Keeping this in one place means a cleaning-logic change only has to be made
once, and app.py and tune_model.py can never quietly drift out of sync with
each other.
"""

import pandas as pd

CURRENT_YEAR = 2026

FEATURES = [
    "brand_grouped", "car_age", "milage", "mileage_per_year",
    "fuel_type_simple", "transmission_simple", "has_accident", "is_clean_title",
]
NUMERIC_FEATURES = ["car_age", "milage", "mileage_per_year", "has_accident", "is_clean_title"]
CATEGORICAL_FEATURES = ["brand_grouped", "fuel_type_simple", "transmission_simple"]


def simplify_transmission(text):
    text = str(text)
    if "A/T" in text or "Automatic" in text or "Auto" in text:
        return "Automatic"
    elif "M/T" in text or "Manual" in text:
        return "Manual"
    return "Other"


def load_and_clean_data(path="data/used_cars.csv"):
    df = pd.read_csv(path)
    clean = df.copy()

    # price: "$10,300" -> 10300.0
    clean["price"] = clean["price"].replace(r"[\$,]", "", regex=True).astype(float)

    # milage: "51,000 mi." -> 51000.0
    clean["milage"] = clean["milage"].replace(r"[a-zA-Z.,]", "", regex=True).str.strip().astype(float)

    # drop duplicates
    clean = clean.drop_duplicates().reset_index(drop=True)

    # fill missing categoricals
    clean["fuel_type"] = clean["fuel_type"].fillna(clean["fuel_type"].mode()[0])
    clean["accident"] = clean["accident"].fillna("None reported")
    clean["clean_title"] = clean["clean_title"].fillna("Unknown")

    # simplify transmission into 3 buckets
    clean["transmission_simple"] = clean["transmission"].apply(simplify_transmission)

    # simplify fuel type: fold rare values into "Other"
    valid_fuels = ["Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel"]
    clean["fuel_type_simple"] = clean["fuel_type"].apply(lambda x: x if x in valid_fuels else "Other")

    # outlier trim on price (1st-99th percentile)
    low, high = clean["price"].quantile([0.01, 0.99])
    clean = clean[(clean["price"] >= low) & (clean["price"] <= high)].reset_index(drop=True)

    # feature engineering
    clean["car_age"] = (CURRENT_YEAR - clean["model_year"]).clip(lower=0)
    clean["mileage_per_year"] = clean["milage"] / clean["car_age"].replace(0, 1)
    clean["has_accident"] = clean["accident"].apply(lambda x: 0 if "None" in str(x) else 1)
    clean["is_clean_title"] = clean["clean_title"].apply(lambda x: 1 if x == "Yes" else 0)

    # group rare brands into "Other"
    brand_counts = clean["brand"].value_counts()
    rare_brands = brand_counts[brand_counts < 30].index
    clean["brand_grouped"] = clean["brand"].apply(lambda x: "Other" if x in rare_brands else x)

    return clean