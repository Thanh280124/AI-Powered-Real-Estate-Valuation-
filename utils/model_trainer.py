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

    # ===== THÊM FEATURES MỚI =====
    # Giá log (giảm ảnh hưởng outliers)
    df["log_price"] = np.log1p(df["price_million"])
    df["log_area"] = np.log1p(df["area_m2"])

    # Giá trung bình theo quận
    district_avg = df.groupby("district")["price_million"].median()
    df["district_median_price"] = df["district"].map(district_avg)

    # Giá trung bình theo thành phố
    city_avg = df.groupby("city")["price_million"].median()
    df["city_median_price"] = df["city"].map(city_avg)

    # Điền giá trị thiếu
    df["bedrooms_num"] = df["bedrooms_num"].fillna(df["bedrooms_num"].median())
    df["bathrooms_num"] = df["bathrooms_num"].fillna(df["bathrooms_num"].median())
    df["district"] = df["district"].fillna("Không rõ")
    df["city"] = df["city"].fillna("Không rõ")
    df["district_median_price"] = df["district_median_price"].fillna(df["price_million"].median())
    df["city_median_price"] = df["city_median_price"].fillna(df["price_million"].median())

    # Encode cột chữ
    le_district = LabelEncoder()
    le_city = LabelEncoder()
    df["district_encoded"] = le_district.fit_transform(df["district"])
    df["city_encoded"] = le_city.fit_transform(df["city"])

    feature_cols = [
        "area_m2", "log_area",
        "bedrooms_num", "bathrooms_num",
        "district_encoded", "city_encoded",
        "district_median_price", "city_median_price"
    ]

    X = df[feature_cols]
    y = df["log_price"]  # Train trên log price

    return X, y, le_district, le_city, district_avg, city_avg


def remove_outliers(df):
    """Loại bỏ outliers theo IQR"""
    df = df.copy()
    Q1 = df["price_million"].quantile(0.05)
    Q3 = df["price_million"].quantile(0.95)
    df = df[(df["price_million"] >= Q1) & (df["price_million"] <= Q3)]
    return df


def train_all_models(X_train, y_train):
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
            n_estimators=300,
            learning_rate=0.05,
            max_depth=6,
            subsample=0.8,
            colsample_bytree=0.8,
            random_state=42,
            verbosity=0
        )
    }

    trained = {}
    for name, model in models.items():
        print(f"⏳ Đang train {name}...")
        model.fit(X_train, y_train)
        trained[name] = model
        print(f"✅ {name} xong!")

    return trained


def evaluate_models(trained_models, X_test, y_test):
    results = {}
    # y_test là log_price → chuyển về giá thật để đánh giá
    y_test_real = np.expm1(y_test)

    for name, model in trained_models.items():
        y_pred_log = model.predict(X_test)
        y_pred_real = np.expm1(y_pred_log)  # chuyển về triệu VNĐ

        mae = mean_absolute_error(y_test_real, y_pred_real)
        rmse = np.sqrt(mean_squared_error(y_test_real, y_pred_real))
        r2 = r2_score(y_test_real, y_pred_real)
        within_20 = np.mean(np.abs(y_pred_real - y_test_real) / y_test_real < 0.2) * 100

        results[name] = {
            "MAE (triệu)": round(mae, 2),
            "RMSE (triệu)": round(rmse, 2),
            "R² Score": round(r2, 4),
            "Accuracy ±20%": f"{within_20:.1f}%"
        }

        print(f"\n📊 {name}:")
        print(f"   MAE  : {mae:,.0f} triệu")
        print(f"   RMSE : {rmse:,.0f} triệu")
        print(f"   R²   : {r2:.4f}")
        print(f"   Accuracy ±20%: {within_20:.1f}%")

    return results


def save_best_model(trained_models, results, le_district, le_city,
                    district_avg, city_avg, model_type="sale"):
    best_name = max(results, key=lambda x: results[x]["R² Score"])
    best_model = trained_models[best_name]

    print(f"\n🏆 Model tốt nhất: {best_name}")

    os.makedirs("models", exist_ok=True)

    bundle = {
        "model": best_model,
        "model_name": best_name,
        "le_district": le_district,
        "le_city": le_city,
        "district_avg": district_avg,
        "city_avg": city_avg,
        "results": results
    }

    filepath = f"models/{model_type}_model.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f)

    print(f"💾 Đã lưu → {filepath}")
    return best_name


def run_training_pipeline(df, model_type="sale"):
    print(f"\n{'='*50}")
    print(f"🚀 BẮT ĐẦU TRAIN MODEL: {model_type.upper()}")
    print(f"{'='*50}")

    # Loại bỏ outliers
    df = remove_outliers(df)
    print(f"📦 Sau khi lọc outliers: {len(df):,} bản ghi")

    X, y, le_district, le_city, district_avg, city_avg = prepare_features(df)
    print(f"✅ Features: {X.shape[1]} cột, {len(X):,} dòng")

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")

    trained_models = train_all_models(X_train, y_train)

    print(f"\n{'='*50}")
    print("📈 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"{'='*50}")
    results = evaluate_models(trained_models, X_test, y_test)

    best_name = save_best_model(
        trained_models, results,
        le_district, le_city,
        district_avg, city_avg,
        model_type
    )

    return results, best_name