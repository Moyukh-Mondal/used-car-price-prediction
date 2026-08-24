"""
tune_model.py

Run this ONCE (locally, or in Colab where you have internet access) to find the
best XGBoost hyperparameters and save the tuned model. The Streamlit app
(app.py) will automatically pick up models/best_model.pkl if it exists and use
it instead of training a quick fallback model on every startup.

Usage:
    pip install -r requirements.txt
    python tune_model.py
"""

import time
import joblib
import numpy as np

from sklearn.model_selection import train_test_split, RandomizedSearchCV, KFold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import mean_absolute_error, r2_score, mean_squared_error
from xgboost import XGBRegressor

from cleaner import load_and_clean_data, FEATURES, NUMERIC_FEATURES, CATEGORICAL_FEATURES

RANDOM_STATE = 42


def main():
    print("Loading and cleaning data...")
    data = load_and_clean_data()

    X = data[FEATURES]
    y = data["price"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=RANDOM_STATE
    )

    preprocessor = ColumnTransformer(transformers=[
        ("num", StandardScaler(), NUMERIC_FEATURES),
        ("cat", OneHotEncoder(handle_unknown="ignore"), CATEGORICAL_FEATURES),
    ])

    pipe = Pipeline(steps=[
        ("preprocessor", preprocessor),
        ("regressor", XGBRegressor(
            random_state=RANDOM_STATE,
            n_jobs=-1,
            verbosity=0,
            objective="reg:squarederror",
        )),
    ])

    # Reasonably wide search space for a tabular regression problem this size (~4k rows)
    param_dist = {
        "regressor__n_estimators": [200, 300, 400, 500, 700],
        "regressor__learning_rate": [0.01, 0.02, 0.03, 0.05, 0.08, 0.1],
        "regressor__max_depth": [3, 4, 5, 6, 7],
        "regressor__subsample": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__colsample_bytree": [0.6, 0.7, 0.8, 0.9, 1.0],
        "regressor__min_child_weight": [1, 3, 5, 7],
        "regressor__reg_alpha": [0, 0.01, 0.1, 1],
        "regressor__reg_lambda": [0.5, 1, 1.5, 2],
    }

    cv = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)

    print("Running RandomizedSearchCV (50 iterations x 5-fold CV = 250 fits)...")
    print("This can take a few minutes depending on your machine.")
    start = time.time()

    search = RandomizedSearchCV(
        pipe,
        param_distributions=param_dist,
        n_iter=50,
        cv=cv,
        scoring="neg_root_mean_squared_error",
        random_state=RANDOM_STATE,
        n_jobs=-1,
        verbose=1,
    )
    search.fit(X_train, y_train)

    elapsed = time.time() - start
    print(f"\nSearch finished in {elapsed:.0f}s")
    print("Best params:", search.best_params_)
    print(f"Best CV RMSE: {-search.best_score_:,.0f}")

    best_model = search.best_estimator_

    # Final evaluation on held-out test set
    y_pred = best_model.predict(X_test)
    mae = mean_absolute_error(y_test, y_pred)
    rmse = np.sqrt(mean_squared_error(y_test, y_pred))
    r2 = r2_score(y_test, y_pred)

    print(f"\nTest set performance:")
    print(f"  MAE:  {mae:,.0f}")
    print(f"  RMSE: {rmse:,.0f}")
    print(f"  R2:   {r2:.4f}")

    # Save the tuned pipeline + its test metrics together, so the app can display them
    import os
    os.makedirs("models", exist_ok=True)
    joblib.dump({
        "model": best_model,
        "mae": mae,
        "rmse": rmse,
        "r2": r2,
        "best_params": search.best_params_,
    }, "models/best_model.pkl")

    print("\nSaved tuned model to models/best_model.pkl")
    print("Commit this file to your repo — app.py will automatically use it.")


if __name__ == "__main__":
    main()