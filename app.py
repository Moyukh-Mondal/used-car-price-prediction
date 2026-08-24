import streamlit as st
import pandas as pd
import numpy as np
import joblib
import os
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.model_selection import train_test_split
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.ensemble import RandomForestRegressor
from sklearn.metrics import mean_absolute_error, r2_score

from cleaner import load_and_clean_data, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES, CURRENT_YEAR

# -----------------------------------------------------------------
# Page config
# -----------------------------------------------------------------
st.set_page_config(
    page_title="Used Car Price Predictor",
    page_icon="🚗",
    layout="wide",
)

# -----------------------------------------------------------------
# Data loading + cleaning (logic lives in data_utils.py, shared with tune_model.py;
# st.cache_data just avoids re-running it on every Streamlit rerun)
# -----------------------------------------------------------------
@st.cache_data
def get_cleaned_data():
    return load_and_clean_data("data/used_cars.csv")


# -----------------------------------------------------------------
# Model training (cached so it only trains once per deployment)
# -----------------------------------------------------------------
@st.cache_resource
def get_model(data: pd.DataFrame):
    """
    Prefer the tuned model saved by tune_model.py (models/best_model.pkl) —
    it's an XGBoost model with hyperparameters found via RandomizedSearchCV.
    If that file doesn't exist (e.g. you haven't run tune_model.py yet),
    fall back to training a quick Random Forest on the spot, so the app
    always works out of the box.
    """
    best_model_path = "models/best_model.pkl"

    if os.path.exists(best_model_path):
        saved = joblib.load(best_model_path)
        return saved["model"], saved["mae"], saved["r2"], "XGBoost (tuned)"

    # --- fallback: quick-trained Random Forest ---
    X = data[FEATURES]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    model = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", RandomForestRegressor(n_estimators=200, max_depth=15, random_state=42, n_jobs=-1)),
    ])
    model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)

    return model, mae, r2, "Random Forest (quick fallback)"


data = get_cleaned_data()
model, test_mae, test_r2, model_label = get_model(data)

if model_label.startswith("Random Forest"):
    st.info(
        "Using a quick fallback model. For better accuracy, run `python tune_model.py` "
        "locally to train a tuned XGBoost model, then commit `models/best_model.pkl` — "
        "the app will pick it up automatically.",
        icon="ℹ️",
    )

# -----------------------------------------------------------------
# Sidebar navigation
# -----------------------------------------------------------------
st.title("🚗 Used Car Price Predictor")
st.caption(f"A beginner-level ML project — {model_label} trained on cars.com listings")

tab1, tab2, tab3 = st.tabs(["🔮 Predict Price", "📊 Explore Data", "ℹ️ About"])

# -----------------------------------------------------------------
# TAB 1 — Predict
# -----------------------------------------------------------------
with tab1:
    st.subheader("Enter your car's details")

    col1, col2 = st.columns(2)

    with col1:
        brand_options = sorted(data["brand_grouped"].unique())
        brand = st.selectbox("Brand", brand_options)

        model_year = st.slider("Model Year", min_value=1990, max_value=CURRENT_YEAR, value=2018)

        mileage = st.number_input("Mileage (miles)", min_value=0, max_value=300000, value=45000, step=1000)

        fuel_type = st.selectbox("Fuel Type", ["Gasoline", "Hybrid", "Diesel", "Plug-In Hybrid", "E85 Flex Fuel", "Other"])

    with col2:
        transmission = st.selectbox("Transmission", ["Automatic", "Manual", "Other"])

        accident = st.radio("Accident History", ["None reported", "At least 1 accident or damage reported"])

        clean_title = st.radio("Clean Title", ["Yes", "No / Unknown"])

    if st.button("Predict Price", type="primary"):
        car_age = max(CURRENT_YEAR - model_year, 0)
        mileage_per_year = mileage / car_age if car_age > 0 else mileage

        input_df = pd.DataFrame([{
            "brand_grouped": brand,
            "car_age": car_age,
            "milage": mileage,
            "mileage_per_year": mileage_per_year,
            "fuel_type_simple": fuel_type,
            "transmission_simple": transmission,
            "has_accident": 0 if accident == "None reported" else 1,
            "is_clean_title": 1 if clean_title == "Yes" else 0,
        }])

        predicted_price = model.predict(input_df)[0]

        st.success(f"### Estimated Price: **${predicted_price:,.0f}**")
        st.caption(f"Model: {model_label} — Test MAE: ${test_mae:,.0f} | R²: {test_r2:.3f}. "
                    "Treat this as a ballpark estimate, not an appraisal.")

# -----------------------------------------------------------------
# TAB 2 — Explore Data
# -----------------------------------------------------------------
with tab2:
    st.subheader("Dataset overview")
    st.write(f"{len(data):,} listings after cleaning (duplicates removed, price outliers trimmed).")

    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Price distribution**")
        fig, ax = plt.subplots()
        sns.histplot(data["price"], bins=40, ax=ax, color="#4C72B0")
        ax.set_xlabel("Price ($)")
        st.pyplot(fig)

    with col2:
        st.markdown("**Price vs. Mileage**")
        fig, ax = plt.subplots()
        sns.scatterplot(data=data.sample(min(1000, len(data)), random_state=1),
                         x="milage", y="price", alpha=0.4, ax=ax, color="#DD8452")
        ax.set_xlabel("Mileage")
        ax.set_ylabel("Price ($)")
        st.pyplot(fig)

    st.markdown("**Average price by brand (top 10 most listed)**")
    top_brands = data["brand_grouped"].value_counts().head(10).index
    avg_price = data[data["brand_grouped"].isin(top_brands)].groupby("brand_grouped")["price"].mean().sort_values(ascending=False)
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.barplot(x=avg_price.values, y=avg_price.index, hue=avg_price.index, ax=ax, palette="viridis", legend=False)
    ax.set_xlabel("Average Price ($)")
    st.pyplot(fig)

    with st.expander("View raw sample data"):
        st.dataframe(data[["brand", "model_year", "milage", "fuel_type", "transmission", "price"]].sample(20, random_state=1))

# -----------------------------------------------------------------
# TAB 3 — About
# -----------------------------------------------------------------
with tab3:
    st.subheader("About this project")
    st.markdown(f"""
    This app predicts used car prices using **{model_label}**, trained on
    ~4,000 real used car listings scraped from cars.com.

    The app prefers a **tuned XGBoost model** (hyperparameters found via
    `RandomizedSearchCV` in `tune_model.py`) whenever `models/best_model.pkl`
    is present. If that file hasn't been generated yet, it falls back to
    training a quick Random Forest on the spot so the app still works out
    of the box — that's what's currently active if you saw a fallback
    notice above.

    **Pipeline:** data cleaning → duplicate removal → missing value handling →
    feature engineering (car age, mileage/year, accident & title flags) →
    one-hot encoding + scaling → {model_label.split(" (")[0]}.

    Engine specs (horsepower, cylinders) were intentionally left out of this app
    to keep the input form simple — see the full notebook for a version that includes them.

    - 📓 [Full analysis notebook](https://github.com/Moyukh-Mondal/used-car-price-prediction/blob/main/notebooks/used_car_price_prediction.ipynb)
    - 💻 [GitHub repo](https://github.com/Moyukh-Mondal/used-car-price-prediction)
    """)