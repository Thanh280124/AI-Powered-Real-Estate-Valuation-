import pandas as pd
import numpy as np

def load_ames_data(filepath="data/ames_housing.csv"):
    """Load Ames Housing dataset"""
    df = pd.read_csv(filepath)
    return df

def clean_ames_data(df):
    """Clean và chuẩn bị Ames Housing data"""
    df = df.copy()

    # ===== RENAME columns cho dễ đọc =====
    df = df.rename(columns={
        'SalePrice':    'price_usd',
        'GrLivArea':    'area_sqft',
        'BedroomAbvGr': 'bedrooms_num',
        'FullBath':     'bathrooms_num',
        'YearBuilt':    'year_built',
        'YearRemodAdd': 'year_remodeled',
        'OverallQual':  'overall_quality',
        'OverallCond':  'overall_condition',
        'Neighborhood': 'neighborhood',
        'BldgType':     'building_type',
        'HouseStyle':   'house_style',
        'TotalBsmtSF':  'basement_area_sqft',
        'GarageCars':   'garage_cars',
        'GarageArea':   'garage_area_sqft',
        'Fireplaces':   'fireplaces',
        'LotArea':      'lot_area_sqft',
        'TotRmsAbvGrd': 'total_rooms',
        'KitchenQual':  'kitchen_quality',
        'ExterQual':    'exterior_quality',
        'CentralAir':   'central_air',
        'MoSold':       'month_sold',
        'YrSold':       'year_sold',
    })

    # ===== DROP rows với missing price =====
    df = df.dropna(subset=['price_usd'])

    # ===== FILTER outliers =====
    df = df[df['price_usd'] > 10000]
    df = df[df['price_usd'] < 800000]
    df = df[df['area_sqft'] > 100]
    df = df[df['area_sqft'] < 10000]

    # ===== FILL missing values =====
    num_cols = ['basement_area_sqft', 'garage_area_sqft',
                'garage_cars', 'fireplaces', 'lot_area_sqft']
    for col in num_cols:
        if col in df.columns:
            df[col] = df[col].fillna(0)

    cat_cols = ['neighborhood', 'building_type', 'house_style',
                'kitchen_quality', 'exterior_quality', 'central_air']
    for col in cat_cols:
        if col in df.columns:
            df[col] = df[col].fillna('Unknown')

    # ===== THÊM features mới =====
    df['property_age'] = df['year_sold'] - df['year_built']
    df['years_since_remodel'] = df['year_sold'] - df['year_remodeled']
    df['price_per_sqft'] = df['price_usd'] / df['area_sqft']
    df['has_garage'] = (df['garage_cars'] > 0).astype(int)
    df['has_basement'] = (df['basement_area_sqft'] > 0).astype(int)
    df['has_fireplace'] = (df['fireplaces'] > 0).astype(int)
    df['has_central_air'] = (df['central_air'] == 'Y').astype(int)
    df['total_bath'] = df['bathrooms_num'] + df.get('HalfBath', 0) * 0.5
    df['total_sf'] = df['area_sqft'] + df['basement_area_sqft']

    return df.reset_index(drop=True)

def get_stats(df):
    return {
        "total": len(df),
        "avg_price": df["price_usd"].mean(),
        "min_price": df["price_usd"].min(),
        "max_price": df["price_usd"].max(),
        "avg_area": df["area_sqft"].mean(),
        "neighborhoods": df["neighborhood"].nunique(),
    }

def load_and_clean_data(filepath="data/ames_housing.csv"):
    df = load_ames_data(filepath)
    return clean_ames_data(df)