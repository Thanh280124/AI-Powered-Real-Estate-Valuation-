import pickle
import numpy as np
import pandas as pd

def load_model(model_type="sale"):
    filepath = f"models/{model_type}_model.pkl"
    with open(filepath, "rb") as f:
        bundle = pickle.load(f)
    return bundle

def predict_price(model_type, area, bedrooms, bathrooms, district, city):
    """
    Dự đoán giá BĐS
    Returns: (gia_du_doan, gia_thap, gia_cao) - đơn vị: triệu VNĐ
    """
    bundle = load_model(model_type)
    model = bundle["model"]
    le_district = bundle["le_district"]
    le_city = bundle["le_city"]
    district_avg = bundle["district_avg"]
    city_avg = bundle["city_avg"]

    # Encode district
    if district in le_district.classes_:
        district_encoded = le_district.transform([district])[0]
    else:
        district_encoded = 0  # unknown

    # Encode city
    if city in le_city.classes_:
        city_encoded = le_city.transform([city])[0]
    else:
        city_encoded = 0

    # District/city median price
    district_median = district_avg.get(district, district_avg.median())
    city_median = city_avg.get(city, city_avg.median())

    # Tạo feature vector
    log_area = np.log1p(area)
    X = pd.DataFrame([{
        "area_m2": area,
        "log_area": log_area,
        "bedrooms_num": bedrooms,
        "bathrooms_num": bathrooms,
        "district_encoded": district_encoded,
        "city_encoded": city_encoded,
        "district_median_price": district_median,
        "city_median_price": city_median
    }])

    # Predict (log scale → real scale)
    log_pred = model.predict(X)[0]
    predicted = np.expm1(log_pred)

    # Khoảng giá ±15%
    low = predicted * 0.85
    high = predicted * 1.15

    return round(predicted, 2), round(low, 2), round(high, 2)


def get_similar_properties(df, district, city, area, price, n=5):
    """Tìm BĐS tương tự"""
    filtered = df[
        (df["city"] == city) &
        (df["area_m2"].between(area * 0.7, area * 1.3)) &
        (df["price_million"].between(price * 0.6, price * 1.4))
    ].copy()

    if len(filtered) < 3:
        # Nới rộng tìm kiếm nếu không đủ
        filtered = df[
            (df["city"] == city) &
            (df["area_m2"].between(area * 0.5, area * 1.5))
        ].copy()

    # Tính độ tương đồng
    filtered["similarity"] = (
        abs(filtered["area_m2"] - area) / area +
        abs(filtered["price_million"] - price) / price
    )

    return filtered.nsmallest(n, "similarity")[
        ["district", "city", "area_m2", "bedrooms_num",
         "bathrooms_num", "price_million", "price_per_m2"]
    ].reset_index(drop=True)