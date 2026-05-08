import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor



def prepare_features(df):
    df = df.copy()

    # Target
    y = df["price_usd"]

    # Feature engineering
    df["log_area"] = np.log1p(df["area_sqft"])
    df["log_lot"] = np.log1p(df["lot_area_sqft"])
    df["total_sf"] = df["area_sqft"] + df["basement_area_sqft"]
    df["total_bath"] = df["bathrooms_num"]
    df["remodel_age"] = df["year_sold"] - df["year_remodeled"]

    # Quality mapping
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    df["kitchen_quality_num"] = df["kitchen_quality"].map(quality_map).fillna(3)
    df["exterior_quality_num"] = df["exterior_quality"].map(quality_map).fillna(3)

    # Luxury score
    df["quality_score"] = (
        df["overall_quality"] * 0.5 +
        df["kitchen_quality_num"] * 0.2 +
        df["exterior_quality_num"] * 0.2 +
        df["fireplaces"] * 0.1
    )

    # Price per sqft
    df["price_per_sqft"] = df["price_usd"] / (df["area_sqft"] + 1)
    df["nbr_price_per_sqft"] = df.groupby("neighborhood")["price_per_sqft"].transform("median")

    # Neighborhood median price
    neighborhood_avg = df.groupby("neighborhood")["price_usd"].median()
    df["neighborhood_median_price"] = df["neighborhood"].map(neighborhood_avg)

    # Label encoding
    le_neighborhood = LabelEncoder()
    le_building = LabelEncoder()
    le_style = LabelEncoder()

    df["neighborhood_encoded"] = le_neighborhood.fit_transform(df["neighborhood"])
    df["building_encoded"] = le_building.fit_transform(df["building_type"])
    df["style_encoded"] = le_style.fit_transform(df["house_style"])

    # Final feature list (35 features)
    feature_cols = [
        "area_sqft", "log_area", "total_sf", "log_lot",
        "lot_area_sqft",
        "basement_area_sqft", "garage_area_sqft",
        "bedrooms_num", "bathrooms_num", "total_rooms",
        "total_bath", "fireplaces", "garage_cars",
        "overall_quality", "overall_condition",
        "kitchen_quality_num", "exterior_quality_num",
        "quality_score",
        "year_built", "property_age", "years_since_remodel",
        "remodel_age",
        "has_garage", "has_basement", "has_fireplace", "has_central_air",
        "neighborhood_encoded", "neighborhood_median_price",
        "nbr_price_per_sqft",
        "building_encoded", "style_encoded",
        "month_sold", "year_sold",
        "price_per_sqft"
    ]

    X = df[feature_cols]

    encoders = {
        "le_neighborhood": le_neighborhood,
        "le_building": le_building,
        "le_style": le_style,
        "neighborhood_avg": neighborhood_avg,
        "feature_cols": feature_cols,
    }

    return X, y, encoders


def remove_outliers(df):
    Q1 = df["price_usd"].quantile(0.05)
    Q3 = df["price_usd"].quantile(0.95)
    return df[(df["price_usd"] >= Q1) & (df["price_usd"] <= Q3)]


def train_all_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=300, max_depth=20,
            min_samples_leaf=3, random_state=42, n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=900,
            learning_rate=0.03,
            max_depth=10,
            subsample=0.9,
            colsample_bytree=0.9,
            reg_lambda=2,
            random_state=42,
            verbosity=0
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"⏳ Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ {name} done!")
    return trained


def evaluate_models(trained_models, X_test, y_test):
    results = {}

    for name, model in trained_models.items():
        y_pred = model.predict(X_test)

        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        within_20 = np.mean(
            np.abs(y_pred - y_test) / y_test < 0.2
        ) * 100

        results[name] = {
            "MAE (USD)": round(mae, 2),
            "RMSE (USD)": round(rmse, 2),
            "R² Score": round(r2, 4),
            "Accuracy ±20%": f"{within_20:.1f}%"
        }

        print(f"\n📊 {name}:")
        print(f"   MAE  : ${mae:,.0f}")
        print(f"   RMSE : ${rmse:,.0f}")
        print(f"   R²   : {r2:.4f}")
        print(f"   Accuracy ±20%: {within_20:.1f}%")

    return results


def save_best_model(trained_models, results, encoders, model_type="ames"):
    best_name = max(results, key=lambda x: results[x]["R² Score"])
    best_model = trained_models[best_name]

    print(f"\n🏆 Best model: {best_name}")

    os.makedirs("models", exist_ok=True)
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "encoders": encoders,
        "results": results
    }

    filepath = f"models/{model_type}_model.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f)

    print(f"💾 Saved → {filepath}")
    return best_name


def run_training_pipeline(df, model_type="ames"):
    print("\n" + "="*60)
    print(f"🚀 START TRAINING: {model_type.upper()}")
    print("="*60)

    df = remove_outliers(df)
    print(f"📦 After outlier removal: {len(df):,} records")

    X, y, encoders = prepare_features(df)
    print(f"✅ Features: {X.shape[1]} columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    trained_models = train_all_models(X_train, y_train)

    print("\n" + "="*60)
    print("📈 EVALUATION RESULTS:")
    results = evaluate_models(trained_models, X_test, y_test)

    best_name = save_best_model(trained_models, results, encoders, model_type)
    return results, best_name
