import pickle
import numpy as np
import pandas as pd
import os

print(">>> USING PREDICTOR FILE:", os.path.abspath(__file__))

def load_model(model_type="ames"):
    filepath = f"models/{model_type}_model.pkl"
    print(">>> LOADING MODEL FROM:", os.path.abspath(filepath))
    with open(filepath, "rb") as f:
        return pickle.load(f)


def predict_price(
    area_sqft, bedrooms, bathrooms, year_built,
    overall_quality, overall_condition,
    neighborhood, building_type, house_style,
    has_garage, garage_cars, garage_area,
    has_basement, basement_area,
    has_fireplace, fireplaces,
    has_central_air, kitchen_quality, exterior_quality,
    total_rooms, lot_area, year_sold=2010
):
    bundle = load_model("ames")
    model = bundle["model"]
    encoders = bundle["encoders"]

    # Safe encoding
    def safe_encode(le, val):
        if val in le.classes_:
            return le.transform([val])[0]
        return 0

    neighborhood_encoded = safe_encode(encoders["le_neighborhood"], neighborhood)
    building_encoded     = safe_encode(encoders["le_building"], building_type)
    style_encoded        = safe_encode(encoders["le_style"], house_style)

    # Quality mapping
    quality_map = {"Ex": 5, "Gd": 4, "TA": 3, "Fa": 2, "Po": 1}
    kitchen_quality_num  = quality_map.get(kitchen_quality, 3)
    exterior_quality_num = quality_map.get(exterior_quality, 3)

    # Neighborhood median
    neighborhood_avg = encoders["neighborhood_avg"]
    neighborhood_median = neighborhood_avg.get(neighborhood, neighborhood_avg.median())

    # Feature engineering
    property_age = max(0, year_sold - year_built)
    years_since_remodel = 0
    remodel_age = 0
    total_bath = bathrooms
    total_sf = area_sqft + basement_area
    log_area = np.log1p(area_sqft)
    log_lot = np.log1p(lot_area)

    quality_score = (
        overall_quality * 0.5 +
        kitchen_quality_num * 0.2 +
        exterior_quality_num * 0.2 +
        fireplaces * 0.1
    )

    nbr_price_per_sqft = neighborhood_median / (area_sqft + 1)

    X = pd.DataFrame([{
        "area_sqft": area_sqft,
        "log_area": log_area,
        "total_sf": total_sf,
        "log_lot": log_lot,
        "lot_area_sqft": lot_area,
        "basement_area_sqft": basement_area,
        "garage_area_sqft": garage_area,
        "bedrooms_num": bedrooms,
        "bathrooms_num": bathrooms,
        "total_rooms": total_rooms,
        "total_bath": total_bath,
        "fireplaces": fireplaces,
        "garage_cars": garage_cars,
        "overall_quality": overall_quality,
        "overall_condition": overall_condition,
        "kitchen_quality_num": kitchen_quality_num,
        "exterior_quality_num": exterior_quality_num,
        "quality_score": quality_score,
        "year_built": year_built,
        "property_age": property_age,
        "years_since_remodel": years_since_remodel,
        "remodel_age": remodel_age,
        "has_garage": int(has_garage),
        "has_basement": int(has_basement),
        "has_fireplace": int(has_fireplace),
        "has_central_air": int(has_central_air),
        "neighborhood_encoded": neighborhood_encoded,
        "neighborhood_median_price": neighborhood_median,
        "nbr_price_per_sqft": nbr_price_per_sqft,
        "building_encoded": building_encoded,
        "style_encoded": style_encoded,
        "month_sold": 6,
        "year_sold": year_sold,
        "price_per_sqft": neighborhood_median / (area_sqft + 1),
    }])

    feature_cols = encoders["feature_cols"]
    X = X[feature_cols]

    predicted = model.predict(X)[0]
    low = predicted * 0.85
    high = predicted * 1.15

    return round(predicted, 2), round(low, 2), round(high, 2)


def get_similar_properties(df, neighborhood, area_sqft, price,
                           year_built, overall_quality, n=5):
    filtered = df.copy()

    nbr_filtered = filtered[filtered["neighborhood"] == neighborhood]
    if len(nbr_filtered) >= n:
        filtered = nbr_filtered

    filtered = filtered[
        (filtered["price_usd"].between(price * 0.6, price * 1.4)) &
        (filtered["area_sqft"].between(area_sqft * 0.5, area_sqft * 1.5))
    ]

    if len(filtered) < 3:
        filtered = df[df["area_sqft"].between(area_sqft * 0.4, area_sqft * 1.6)]

    if len(filtered) == 0:
        return pd.DataFrame()

    score = pd.Series(0.0, index=filtered.index)
    score += 0.30 * abs(filtered["area_sqft"] - area_sqft) / (area_sqft + 1)
    score += 0.25 * abs(filtered["price_usd"] - price) / (price + 1)
    score += 0.20 * abs(filtered["overall_quality"] - overall_quality)
    score += 0.15 * abs(filtered["year_built"] - year_built) / 100
    score += 0.10 * (filtered["neighborhood"] != neighborhood).astype(float)

    filtered = filtered.copy()
    filtered["similarity_score"] = score

    display_cols = [
        "neighborhood", "building_type", "house_style",
        "area_sqft", "bedrooms_num", "bathrooms_num",
        "year_built", "overall_quality",
        "price_usd", "price_per_sqft"
    ]
    display_cols = [c for c in display_cols if c in filtered.columns]

    return filtered.nsmallest(n, "similarity_score")[display_cols].reset_index(drop=True)
