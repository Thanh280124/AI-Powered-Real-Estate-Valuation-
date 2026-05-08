import sys
sys.path.append(".")
from utils.data_processor import load_and_clean_data
from utils.model_trainer import run_training_pipeline

print("🏠 TRAINING AMES HOUSING MODEL")
df = load_and_clean_data("data/ames_housing.csv")
print(f"✅ Loaded: {len(df):,} records")
print(f"📋 Features available: {len(df.columns)} columns")
print(f"\nKey stats:")
print(f"  Price range: ${df['price_usd'].min():,.0f} - ${df['price_usd'].max():,.0f}")
print(f"  Avg price: ${df['price_usd'].mean():,.0f}")
print(f"  Year built range: {df['year_built'].min()} - {df['year_built'].max()}")
print(f"  Neighborhoods: {df['neighborhood'].nunique()}")

results, best = run_training_pipeline(df, model_type="ames")
print(f"\n🎉 DONE! Best model: {best}")