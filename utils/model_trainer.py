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
    """Prepare features for training"""
    df = df.copy()

    # Log transform target and area (very important for price prediction)
    df["log_price"] = np.log1p(df["price_million"])
    df["log_area"] = np.log1p(df["area_m2"])

    # Target encoding for location (median price)
    district_avg = df.groupby("district")["price_million"].median()
    city_avg = df.groupby("city")["price_million"].median()
    ward_avg = df.groupby("ward")["price_million"].median() if "ward" in df.columns else None

    df["district_median_price"] = df["district"].map(district_avg)
    df["city_median_price"] = df["city"].map(city_avg)
    if ward_avg is not None:
        df["ward_median_price"] = df["ward"].map(ward_avg)

    # Fill missing values
    df["bedrooms_num"] = df["bedrooms_num"].fillna(df["bedrooms_num"].median())
    df["bathrooms_num"] = df["bathrooms_num"].fillna(df["bathrooms_num"].median())
    df["floors"] = df["floors"].fillna(df["floors"].median()) if "floors" in df.columns else 2
    df["frontage_width"] = df["frontage_width"].fillna(df["frontage_width"].median()) if "frontage_width" in df.columns else 4
    df["road_width"] = df["road_width"].fillna(df["road_width"].median()) if "road_width" in df.columns else 6

    df["district"] = df["district"].fillna("Unknown")
    df["city"] = df["city"].fillna("Unknown")
    df["ward"] = df["ward"].fillna("Unknown") if "ward" in df.columns else "Unknown"
    df["property_type"] = df["property_type"].fillna("House") if "property_type" in df.columns else "House"
    df["direction"] = df["direction"].fillna("Unknown") if "direction" in df.columns else "Unknown"

    df["district_median_price"] = df["district_median_price"].fillna(df["price_million"].median())
    df["city_median_price"] = df["city_median_price"].fillna(df["price_million"].median())
    if "ward_median_price" in df.columns:
        df["ward_median_price"] = df["ward_median_price"].fillna(df["price_million"].median())

    # Label Encoding
    le_district = LabelEncoder()
    le_city = LabelEncoder()
    le_property = LabelEncoder()
    le_direction = LabelEncoder()
    le_ward = LabelEncoder()
    le_street = LabelEncoder()
    le_legal = LabelEncoder()
    le_condition = LabelEncoder()

    df["street_type"] = df["street_type"].fillna("unknown") if "street_type" in df.columns else "unknown"
    df["legal_status"] = df["legal_status"].fillna("unknown") if "legal_status" in df.columns else "unknown"
    df["condition"] = df["condition"].fillna("unknown") if "condition" in df.columns else "unknown"

    df["street_encoded"] = le_street.fit_transform(df["street_type"])
    df["legal_encoded"] = le_legal.fit_transform(df["legal_status"])
    df["condition_encoded"] = le_condition.fit_transform(df["condition"])

    df["district_encoded"] = le_district.fit_transform(df["district"])
    df["city_encoded"] = le_city.fit_transform(df["city"])
    df["property_encoded"] = le_property.fit_transform(df["property_type"])
    df["direction_encoded"] = le_direction.fit_transform(df["direction"])
    df["ward_encoded"] = le_ward.fit_transform(df["ward"])

    # Final feature list
    feature_cols = [
        "area_m2", "log_area",
        "bedrooms_num", "bathrooms_num",
        "floors", "frontage_width", "road_width",
        "district_encoded", "city_encoded",
        "ward_encoded", "property_encoded",
        "direction_encoded",
        "street_encoded",    
        "legal_encoded",    
        "condition_encoded", 
        "district_median_price", "city_median_price",
    ]
    if "ward_median_price" in df.columns:
        feature_cols.append("ward_median_price")

    X = df[feature_cols]
    y = df["log_price"]

    encoders = {
        "le_district": le_district,
        "le_city": le_city,
        "le_property": le_property,
        "le_direction": le_direction,
        "le_ward": le_ward,
        "district_avg": district_avg,
        "city_avg": city_avg,
        "ward_avg": ward_avg,
        "feature_cols": feature_cols,
        "le_street": le_street,
        "le_legal": le_legal,
        "le_condition": le_condition,
    }

    return X, y, encoders


def remove_outliers(df):
    """Remove extreme outliers using 5% - 95% quantile"""
    Q1 = df["price_million"].quantile(0.05)
    Q3 = df["price_million"].quantile(0.95)
    return df[(df["price_million"] >= Q1) & (df["price_million"] <= Q3)]


def train_all_models(X_train, y_train):
    """Train all models"""
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=200, 
            max_depth=15,
            min_samples_leaf=5, 
            random_state=42, 
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=500,      
            learning_rate=0.03,    
            max_depth=8,           
            subsample=0.8,
            colsample_bytree=0.8,
            min_child_weight=3,   
            gamma=0.1,             
            reg_alpha=0.1,        
            reg_lambda=1.0,       
            random_state=42,
            verbosity=0
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"⏳ Training {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ {name} completed!")
    
    return trained


def evaluate_models(trained_models, X_test, y_test):
    """Evaluate all models"""
    results = {}
    y_test_real = np.expm1(y_test)

    for name, model in trained_models.items():
        y_pred_log = model.predict(X_test)
        y_pred_real = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)
        within_20 = np.mean(np.abs(y_pred_real - y_test_real) / y_test_real < 0.2) * 100

        results[name] = {
            "MAE (million)": round(mae, 2),
            "RMSE (million)": round(rmse, 2),
            "R² Score": round(r2, 4),
            "Accuracy ±20%": f"{within_20:.1f}%"
        }

        print(f"\n📊 {name}:")
        print(f"   MAE  : {mae:,.0f} million")
        print(f"   RMSE : {rmse:,.0f} million")
        print(f"   R²   : {r2:.4f}")
        print(f"   Accuracy ±20%: {within_20:.1f}%")

    return results


def save_best_model(trained_models, results, encoders, model_type="sale"):
    """Save the best model"""
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


def run_training_pipeline(df, model_type="sale"):
    """Main training pipeline"""
    print(f"\n{'='*60}")
    print(f"🚀 START TRAINING PIPELINE: {model_type.upper()}")
    print(f"{'='*60}")

    df = remove_outliers(df)
    print(f"📦 After removing outliers: {len(df):,} records")

    X, y, encoders = prepare_features(df)
    print(f"✅ Features prepared: {X.shape[1]} columns, {len(X):,} rows")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train set: {len(X_train):,} | Test set: {len(X_test):,}")

    trained_models = train_all_models(X_train, y_train)

    print(f"\n{'='*60}")
    print("📈 EVALUATION RESULTS:")
    results = evaluate_models(trained_models, X_test, y_test)

    best_name = save_best_model(trained_models, results, encoders, model_type)
    return results, best_name