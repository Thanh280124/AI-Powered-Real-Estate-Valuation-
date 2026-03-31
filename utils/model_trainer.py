import pandas as pd
import numpy as np
import pickle
import os
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from xgboost import XGBRegressor

# ========== BƯỚC 1: CHUẨN BỊ FEATURES ==========
def prepare_features(df):
    """Chuyển data thô → features cho ML"""
    df = df.copy()
    
    # Chỉ lấy các cột cần thiết
    features = ["area_m2", "bedrooms_num", "bathrooms_num", "district", "city"]
    target = "price_million"
    
    df = df[features + [target]].copy()
    
    # Điền giá trị thiếu
    df["bedrooms_num"] = df["bedrooms_num"].fillna(df["bedrooms_num"].median())
    df["bathrooms_num"] = df["bathrooms_num"].fillna(df["bathrooms_num"].median())
    df["district"] = df["district"].fillna("Không rõ")
    df["city"] = df["city"].fillna("Không rõ")
    
    # Encode cột chữ → số
    le_district = LabelEncoder()
    le_city = LabelEncoder()
    df["district_encoded"] = le_district.fit_transform(df["district"])
    df["city_encoded"] = le_city.fit_transform(df["city"])
    
    X = df[["area_m2", "bedrooms_num", "bathrooms_num", 
            "district_encoded", "city_encoded"]]
    y = df[target]
    
    return X, y, le_district, le_city


# ========== BƯỚC 2: TRAIN 3 MODELS ==========
def train_all_models(X_train, y_train):
    models = {
        "Linear Regression": LinearRegression(),
        "Random Forest": RandomForestRegressor(
            n_estimators=100, 
            random_state=42,
            n_jobs=-1
        ),
        "XGBoost": XGBRegressor(
            n_estimators=100,
            learning_rate=0.1,
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


# ========== BƯỚC 3: ĐÁNH GIÁ MODELS ==========
def evaluate_models(trained_models, X_test, y_test):
    results = {}
    
    for name, model in trained_models.items():
        y_pred = model.predict(X_test)
        
        mae = mean_absolute_error(y_test, y_pred)
        rmse = np.sqrt(mean_squared_error(y_test, y_pred))
        r2 = r2_score(y_test, y_pred)
        
        # Accuracy theo kiểu: % dự đoán trong khoảng ±20% giá thật
        within_20 = np.mean(np.abs(y_pred - y_test) / y_test < 0.2) * 100
        
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


# ========== BƯỚC 4: LƯU MODEL TỐT NHẤT ==========
def save_best_model(trained_models, results, le_district, le_city, model_type="sale"):
    # Chọn model có R² cao nhất
    best_name = max(results, key=lambda x: results[x]["R² Score"])
    best_model = trained_models[best_name]
    
    print(f"\n🏆 Model tốt nhất: {best_name}")
    
    os.makedirs("models", exist_ok=True)
    
    # Lưu model + encoders vào 1 file
    bundle = {
        "model": best_model,
        "model_name": best_name,
        "le_district": le_district,
        "le_city": le_city,
        "results": results
    }
    
    filepath = f"models/{model_type}_model.pkl"
    with open(filepath, "wb") as f:
        pickle.dump(bundle, f)
    
    print(f"💾 Đã lưu → {filepath}")
    return best_name


# ========== CHẠY TOÀN BỘ PIPELINE ==========
def run_training_pipeline(df, model_type="sale"):
    print(f"\n{'='*50}")
    print(f"🚀 BẮT ĐẦU TRAIN MODEL: {model_type.upper()}")
    print(f"{'='*50}")
    print(f"📦 Tổng dữ liệu: {len(df):,} bản ghi")
    
    # Chuẩn bị features
    X, y, le_district, le_city = prepare_features(df)
    print(f"✅ Features: {X.shape[1]} cột, {len(X):,} dòng")
    
    # Chia train/test (80/20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    print(f"📊 Train: {len(X_train):,} | Test: {len(X_test):,}")
    
    # Train
    trained_models = train_all_models(X_train, y_train)
    
    # Đánh giá
    print(f"\n{'='*50}")
    print("📈 KẾT QUẢ ĐÁNH GIÁ:")
    print(f"{'='*50}")
    results = evaluate_models(trained_models, X_test, y_test)
    
    # Lưu best model
    best_name = save_best_model(
        trained_models, results, le_district, le_city, model_type
    )
    
    return results, best_name