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

    # Log transform
    df['log_price'] = np.log1p(df['price_usd'])
    df['log_area'] = np.log1p(df['area_sqft'])
    df['log_lot'] = np.log1p(df['lot_area_sqft'])

    # Neighborhood median price
    neighborhood_avg = df.groupby('neighborhood')['price_usd'].median()
    df['neighborhood_median_price'] = df['neighborhood'].map(neighborhood_avg)

    # Fill missing
    df['neighborhood'] = df['neighborhood'].fillna('Unknown')
    df['building_type'] = df['building_type'].fillna('Unknown')
    df['house_style'] = df['house_style'].fillna('Unknown')
    df['kitchen_quality'] = df['kitchen_quality'].fillna('TA')
    df['exterior_quality'] = df['exterior_quality'].fillna('TA')
    df['neighborhood_median_price'] = df['neighborhood_median_price'].fillna(
        df['price_usd'].median()
    )

    # Label encode
    le_neighborhood = LabelEncoder()
    le_building = LabelEncoder()
    le_style = LabelEncoder()
    le_kitchen = LabelEncoder()
    le_exterior = LabelEncoder()

    df['neighborhood_encoded'] = le_neighborhood.fit_transform(df['neighborhood'])
    df['building_encoded'] = le_building.fit_transform(df['building_type'])
    df['style_encoded'] = le_style.fit_transform(df['house_style'])
    df['kitchen_encoded'] = le_kitchen.fit_transform(df['kitchen_quality'])
    df['exterior_encoded'] = le_exterior.fit_transform(df['exterior_quality'])

    feature_cols = [
        # Size features
        'area_sqft', 'log_area', 'total_sf', 'log_lot',
        'basement_area_sqft', 'garage_area_sqft',
        # Room features
        'bedrooms_num', 'bathrooms_num', 'total_rooms',
        'total_bath', 'fireplaces', 'garage_cars',
        # Quality features
        'overall_quality', 'overall_condition',
        'kitchen_encoded', 'exterior_encoded',
        # Age features 
        'year_built', 'property_age', 'years_since_remodel',
        # Binary features
        'has_garage', 'has_basement', 'has_fireplace', 'has_central_air',
        # Location
        'neighborhood_encoded', 'neighborhood_median_price',
        'building_encoded', 'style_encoded',
    ]

    X = df[feature_cols]
    y = df['log_price']

    encoders = {
        'le_neighborhood': le_neighborhood,
        'le_building': le_building,
        'le_style': le_style,
        'le_kitchen': le_kitchen,
        'le_exterior': le_exterior,
        'neighborhood_avg': neighborhood_avg,
        'feature_cols': feature_cols,
    }

    return X, y, encoders

def remove_outliers(df):
    Q1 = df['price_usd'].quantile(0.05)
    Q3 = df['price_usd'].quantile(0.95)
    return df[(df['price_usd'] >= Q1) & (df['price_usd'] <= Q3)]

def train_all_models(X_train, y_train):
    models = {
        'Linear Regression': LinearRegression(),
        'Random Forest': RandomForestRegressor(
            n_estimators=200, max_depth=15,
            min_samples_leaf=5, random_state=42, n_jobs=-1
        ),
        'XGBoost': XGBRegressor(
            n_estimators=500, learning_rate=0.03,
            max_depth=8, subsample=0.8,
            colsample_bytree=0.8, min_child_weight=3,
            gamma=0.1, reg_alpha=0.1, reg_lambda=1.0,
            random_state=42, verbosity=0
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
    y_test_real = np.expm1(y_test)

    for name, model in trained_models.items():
        y_pred_log = model.predict(X_test)
        y_pred_real = np.expm1(y_pred_log)

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)
        within_20 = np.mean(
            np.abs(y_pred_real - y_test_real) / y_test_real < 0.2
        ) * 100

        results[name] = {
            'MAE (USD)': round(mae, 2),
            'RMSE (USD)': round(rmse, 2),
            'R² Score': round(r2, 4),
            'Accuracy ±20%': f"{within_20:.1f}%"
        }

        print(f"\n📊 {name}:")
        print(f"   MAE  : ${mae:,.0f}")
        print(f"   RMSE : ${rmse:,.0f}")
        print(f"   R²   : {r2:.4f}")
        print(f"   Accuracy ±20%: {within_20:.1f}%")

    return results

def save_best_model(trained_models, results, encoders, model_type="ames"):
    best_name = max(results, key=lambda x: results[x]['R² Score'])
    best_model = trained_models[best_name]
    print(f"\n🏆 Best model: {best_name}")

    os.makedirs("models", exist_ok=True)
    bundle = {
        'model': best_model,
        'model_name': best_name,
        'encoders': encoders,
        'results': results
    }
    filepath = f"models/{model_type}_model.pkl"
    with open(filepath, 'wb') as f:
        pickle.dump(bundle, f)
    print(f"💾 Saved → {filepath}")
    return best_name

def run_training_pipeline(df, model_type="ames"):
    print(f"\n{'='*60}")
    print(f"🚀 START TRAINING: {model_type.upper()}")
    print(f"{'='*60}")

    df = remove_outliers(df)
    print(f"📦 After outlier removal: {len(df):,} records")

    X, y, encoders = prepare_features(df)
    print(f"✅ Features: {X.shape[1]} columns")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

    trained_models = train_all_models(X_train, y_train)

    print(f"\n{'='*60}")
    print("📈 EVALUATION RESULTS:")
    results = evaluate_models(trained_models, X_test, y_test)

    best_name = save_best_model(trained_models, results, encoders, model_type)
    return results, best_name