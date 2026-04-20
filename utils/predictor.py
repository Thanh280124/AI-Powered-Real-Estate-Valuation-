import pickle
import numpy as np
import pandas as pd

# Map property type → model file
PROPERTY_MODEL_MAP = {
    "Nhà": "house",
    "Biệt thự/Nhà liền kề": "villa",
    "Căn hộ chung cư": "apartment",
    "Shophouse": "shophouse",
    "Đất": "land",
}

def load_model(model_type="sale_all"):
    filepath = f"models/{model_type}_model.pkl"
    with open(filepath, "rb") as f:
        return pickle.load(f)

def predict_price(area, bedrooms, bathrooms, district, city,
                  property_type="Nhà", floors=2, frontage_width=4,
                  road_width=6, direction="unknown",
                  street_type="unknown", legal_status="unknown",
                  ward="unknown"):
    """
    Dự đoán giá BĐS dựa trên property type
    Returns: (predicted, low, high) — đơn vị: triệu VNĐ
    """
    model_name = PROPERTY_MODEL_MAP.get(property_type, "sale_all")

    try:
        bundle = load_model(model_name)
    except FileNotFoundError:
        bundle = load_model("sale_all")

    model = bundle["model"]
    encoders = bundle["encoders"]

    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0

    district_encoded  = safe_encode(encoders["le_district"],  district)
    city_encoded      = safe_encode(encoders["le_city"],      city)
    ward_encoded      = safe_encode(encoders["le_ward"],      ward)
    property_encoded  = safe_encode(encoders["le_property"],  property_type)
    direction_encoded = safe_encode(encoders["le_direction"], direction)
    street_encoded    = safe_encode(encoders["le_street"],    street_type)
    legal_encoded     = safe_encode(encoders["le_legal"],     legal_status)
    condition_encoded = 0

    district_avg = encoders["district_avg"]
    city_avg     = encoders["city_avg"]
    ward_avg     = encoders["ward_avg"]

    district_median = district_avg.get(district, district_avg.median())
    city_median     = city_avg.get(city, city_avg.median())
    ward_median     = ward_avg.get(ward, ward_avg.median()) if ward_avg is not None else city_median

    log_area = np.log1p(area)
    X = pd.DataFrame([{
        "area_m2":              area,
        "log_area":             log_area,
        "bedrooms_num":         bedrooms,
        "bathrooms_num":        bathrooms,
        "floors":               floors,
        "frontage_width":       frontage_width,
        "road_width":           road_width,
        "district_encoded":     district_encoded,
        "city_encoded":         city_encoded,
        "ward_encoded":         ward_encoded,
        "property_encoded":     property_encoded,
        "direction_encoded":    direction_encoded,
        "street_encoded":       street_encoded,
        "legal_encoded":        legal_encoded,
        "condition_encoded":    condition_encoded,
        "district_median_price": district_median,
        "city_median_price":    city_median,
        "ward_median_price":    ward_median,
    }])

    feature_cols = encoders["feature_cols"]
    X = X[feature_cols]

    log_pred  = model.predict(X)[0]
    predicted = np.expm1(log_pred)
    low       = predicted * 0.85
    high      = predicted * 1.15

    return round(predicted, 2), round(low, 2), round(high, 2)


def get_similar_properties(df, district, city, area, price,
                           property_type=None, n=5):
    filtered = df.copy()

    # Lọc cùng loại BĐS
    if property_type and "property_type" in df.columns:
        filtered = filtered[filtered["property_type"] == property_type]

    # Lọc cùng thành phố TRƯỚC
    city_filtered = filtered[filtered["city"] == city].copy()

    # Lọc range giá ±40% và area ±50%
    strict = city_filtered[
        (city_filtered["area_m2"].between(area * 0.5, area * 1.5)) &
        (city_filtered["price_million"].between(price * 0.6, price * 1.4))
    ]

    # Nếu đủ kết quả dùng strict
    if len(strict) >= n:
        filtered = strict
    elif len(city_filtered) >= n:
        # Nới rộng range nhưng vẫn giữ cùng city
        filtered = city_filtered[
            city_filtered["area_m2"].between(area * 0.3, area * 1.7)
        ]
        if len(filtered) < n:
            filtered = city_filtered
    else:
        # Fallback: cùng property type toàn quốc
        filtered = df.copy()
        if property_type and "property_type" in df.columns:
            filtered = filtered[filtered["property_type"] == property_type]
        filtered = filtered[
            filtered["price_million"].between(price * 0.6, price * 1.4)
        ]

    if len(filtered) == 0:
        return pd.DataFrame()

    # ===== SIMILARITY SCORE =====
    score = pd.Series(0.0, index=filtered.index)
    score += 0.35 * (abs(filtered["area_m2"] - area) / (area + 1))
    score += 0.30 * (abs(filtered["price_million"] - price) / (price + 1))

    if "bedrooms_num" in filtered.columns:
        median_bed = filtered["bedrooms_num"].median()
        score += 0.15 * (
            abs(filtered["bedrooms_num"].fillna(median_bed) - median_bed)
            / (median_bed + 1)
        )

    if "district" in filtered.columns:
        score += 0.15 * (filtered["district"] != district).astype(float)

    if "street_type" in filtered.columns:
        score += 0.05 * (filtered["street_type"] == "unknown").astype(float)

    filtered = filtered.copy()
    filtered["similarity_score"] = score

    display_cols = [
        "property_type", "district", "city",
        "area_m2", "bedrooms_num", "bathrooms_num",
        "price_million", "price_per_m2"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    return filtered.nsmallest(n, "similarity_score")[display_cols].reset_index(drop=True)