import pandas as pd
import numpy as np
import re

def parse_price(price_val):
    """Convert price to million VND"""
    if pd.isna(price_val):
        return np.nan
    try:
        val = float(price_val)
        return val / 1_000_000  
    except:
        return np.nan

def parse_area(area_val):
    """Convert area to float m²"""
    if pd.isna(area_val):
        return np.nan
    try:
        return float(area_val)
    except:
        return np.nan

def load_new_data(filepath="data/vietnam_real_estate_sampled.csv"):
    """Load new dataset from Tinix 2025"""
    df = pd.read_csv(filepath, low_memory=False)
    
    # Parse price and area
    df["price_million"] = df["price"].apply(parse_price)
    df["area_m2"] = df["area"].apply(parse_area)
    
    # Rename columns to English for consistency
    df = df.rename(columns={
        "province_name": "city",
        "district_name": "district",
        "ward_name": "ward",
        "bedroom_count": "bedrooms_num",
        "bathroom_count": "bathrooms_num",
        "floor_count": "floors",
        "property_type_name": "property_type",
        "house_direction": "direction",
        "frontage_width": "frontage_width",
        "road_width": "road_width",
    })
    
    # Convert to numeric
    numeric_cols = ["bedrooms_num", "bathrooms_num", "floors", 
                    "frontage_width", "road_width", "house_depth"]
    for col in numeric_cols:
        if col in df.columns:
            df[col] = pd.to_numeric(df[col], errors="coerce")
    
    return df


def clean_new_data(df, listing_type="sale"):
    """Clean data based on listing type"""
    df = df.copy()
    
    # Filter by listing type
    if "listing_type" in df.columns:
        if listing_type == "sale":
            df = df[df["listing_type"].str.contains("sale|bán", case=False, na=False)]
        else:
            df = df[df["listing_type"].str.contains("rent|thuê", case=False, na=False)]
    
    # Drop rows with missing price or area
    df = df.dropna(subset=["price_million", "area_m2"])
    
    # Reasonable price filter
    if listing_type == "sale":
        df = df[(df["price_million"] > 100) & (df["price_million"] < 500_000)]
    else:
        df = df[(df["price_million"] > 1) & (df["price_million"] < 5_000)]
    
    # Area filter
    df = df[(df["area_m2"] >= 10) & (df["area_m2"] <= 5000)]
    
    # Remove duplicates
    df = df.drop_duplicates()
    
    # Calculate price per m²
    df["price_per_m2"] = df["price_million"] / df["area_m2"]
    
    return df.reset_index(drop=True)


def load_and_clean_data(filepath="data/vietnam_real_estate_sampled.csv"):
    """Main function to load and clean data"""
    df = load_new_data(filepath)
    
    print("⏳ Extracting features from description...")
    combined_text = df["description"].fillna("") + " " + df["name"].fillna("")
    
    df["street_type"] = combined_text.apply(extract_street_type)
    df["legal_status"] = combined_text.apply(extract_legal)
    df["condition"] = combined_text.apply(extract_condition)
    
    print(f"✅ street_type: {df['street_type'].value_counts().to_dict()}")
    print(f"✅ legal_status: {df['legal_status'].value_counts().to_dict()}")
    
    # Filter main property types for sale
    sale_types = ["Nhà", "Biệt thự/Nhà liền kề", "Căn hộ chung cư", "Shophouse", "Đất"]
    df_sale = df[df["property_type"].isin(sale_types)].copy()
    df_sale = clean_new_data(df_sale, "sale")
    
    return df_sale


def extract_street_type(text):
    """Extract street type from text"""
    if pd.isna(text):
        return "unknown"
    text = str(text).lower()
    
    if any(w in text for w in ["mặt phố", "mặt tiền", "mặt đường", "mt "]):
        return "main_road"
    elif any(w in text for w in ["hẻm", "ngõ", "ngách", "kiệt"]):
        return "alley"
    else:
        return "unknown"


def extract_legal(text):
    """Extract legal status from text"""
    if pd.isna(text):
        return "unknown"
    text = str(text).lower()
    
    if "sổ đỏ" in text:
        return "red_book"
    elif "sổ hồng" in text:
        return "pink_book"
    elif "sổ chung" in text:
        return "shared_book"
    else:
        return "unknown"


def extract_condition(text):
    """Extract house condition"""
    if pd.isna(text):
        return "unknown"
    text = str(text).lower()
    
    if any(w in text for w in ["mới xây", "mới hoàn", "brand new"]):
        return "new"
    elif any(w in text for w in ["cần sửa", "xuống cấp", "cũ"]):
        return "old"
    elif any(w in text for w in ["nội thất", "đầy đủ nội thất"]):
        return "furnished"
    else:
        return "unknown"


def get_stats(df):
    """Return basic statistics"""
    return {
        "total": len(df),
        "avg_price": df["price_million"].mean(),
        "min_price": df["price_million"].min(),
        "max_price": df["price_million"].max(),
        "avg_area": df["area_m2"].mean(),
        "cities": df["city"].nunique(),
        "districts": df["district"].nunique(),
    }