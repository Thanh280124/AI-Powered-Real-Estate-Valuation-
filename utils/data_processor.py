import pandas as pd
import numpy as np
import re

def parse_price_sale(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).strip().lower()
    price_str = price_str.replace(".", "").replace(",", ".")
    try:
        if "tỷ" in price_str:
            num = float(re.sub(r"[^\d.]", "", price_str))
            return num * 1000
        elif "triệu" in price_str:
            num = float(re.sub(r"[^\d.]", "", price_str))
            return num
        else:
            return np.nan
    except:
        return np.nan

def parse_price_rental(price_str):
    if pd.isna(price_str):
        return np.nan
    price_str = str(price_str).strip().lower()
    price_str = price_str.replace(".", "").replace(",", ".")
    try:
        num = float(re.sub(r"[^\d.]", "", price_str))
        return num
    except:
        return np.nan

def parse_area(area_str):
    if pd.isna(area_str):
        return np.nan
    area_str = str(area_str).strip()
    area_str = area_str.replace(".", "").replace(",", ".")
    try:
        num = float(re.sub(r"[^\d.]", "", area_str))
        return num
    except:
        return np.nan

def parse_address(address_str):
    if pd.isna(address_str):
        return "Không rõ", "Không rõ"
    cleaned = str(address_str).replace("·", "").replace("\n", "").strip()
    parts = [p.strip() for p in cleaned.split(",") if p.strip()]
    if len(parts) >= 2:
        return parts[0], parts[1]
    elif len(parts) == 1:
        return parts[0], "Không rõ"
    return "Không rõ", "Không rõ"

def load_sale_data(filepath="data/sale_real_estate.csv"):
    df = pd.read_csv(filepath)
    df["price_million"] = df["price"].apply(parse_price_sale)
    df["area_m2"] = df["area"].apply(parse_area)
    df[["district", "city"]] = df["address"].apply(
        lambda x: pd.Series(parse_address(x))
    )
    df["bedrooms_num"] = pd.to_numeric(df["bedrooms_num"], errors="coerce")
    df["bathrooms_num"] = pd.to_numeric(df["bathrooms_num"], errors="coerce")
    df["type"] = "sale"
    return df

def load_rental_data(filepath="data/rental_real_estate.csv"):
    df = pd.read_csv(filepath)
    df["price_million"] = df["price"].apply(parse_price_rental)
    df["area_m2"] = df["area"].apply(parse_area)
    df[["district", "city"]] = df["address"].apply(
        lambda x: pd.Series(parse_address(x))
    )
    df["bedrooms_num"] = pd.to_numeric(df["bedrooms_num"], errors="coerce")
    df["bathrooms_num"] = pd.to_numeric(df["bathrooms_num"], errors="coerce")
    df["type"] = "rental"
    return df

def clean_sale_data(df):
    df = df.copy()
    df = df.dropna(subset=["price_million", "area_m2"])
    df = df[df["price_million"] > 100]
    df = df[df["price_million"] < 200000]
    df = df[df["area_m2"] >= 10]
    df = df[df["area_m2"] <= 2000]
    df = df.drop_duplicates(subset=["product_id"])
    df["price_per_m2"] = df["price_million"] / df["area_m2"]
    return df.reset_index(drop=True)

def clean_rental_data(df):
    df = df.copy()
    df = df.dropna(subset=["price_million", "area_m2"])
    df = df[df["price_million"] > 1]
    df = df[df["price_million"] < 5000]
    df = df[df["area_m2"] >= 10]
    df = df[df["area_m2"] <= 2000]
    df = df.drop_duplicates(subset=["product_id"])
    df["price_per_m2"] = df["price_million"] / df["area_m2"]
    return df.reset_index(drop=True)

def get_stats(df):
    return {
        "total": len(df),
        "avg_price": df["price_million"].mean(),
        "min_price": df["price_million"].min(),
        "max_price": df["price_million"].max(),
        "avg_area": df["area_m2"].mean(),
        "cities": df["city"].nunique(),
        "districts": df["district"].nunique(),
    }