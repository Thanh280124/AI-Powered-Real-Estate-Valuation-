import sys
sys.path.append(".")

from utils.data_processor import load_and_clean_data
from utils.model_trainer import run_training_pipeline

# Load and clean data
df = load_and_clean_data("data/vietnam_real_estate_sampled.csv")

print(f"✅ Total records loaded: {len(df):,}")
print("Property types in data:")
print(df["property_type"].value_counts())

# ====================== TRAINING SEPARATE MODELS ======================
property_mapping = {
    "Nhà": "house",
    "Biệt thự/Nhà liền kề": "villa",
    "Căn hộ chung cư": "apartment",
    "Shophouse": "shophouse",
    "Đất": "land"
}

print("\n🚀 Starting training separate models...")

for viet_name, eng_name in property_mapping.items():
    df_sub = df[df["property_type"] == viet_name].copy()
    
    if len(df_sub) < 1000:
        print(f"⚠️ Skipping {viet_name} → only {len(df_sub):,} records")
        continue
        
    print(f"\n🏠 Training {viet_name} → model name: {eng_name} ({len(df_sub):,} records)")
    
    run_training_pipeline(df_sub, model_type=eng_name)

# ====================== TRAINING GENERAL MODEL ======================
print("\n🏠 Training general model for ALL property types")
run_training_pipeline(df, model_type="sale_all")