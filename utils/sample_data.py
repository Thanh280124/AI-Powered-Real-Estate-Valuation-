import pandas as pd
import os

# Đọc file gốc - thay tên file đúng của bạn
INPUT_FILE = "data/vietnam_real_estate.csv"  
OUTPUT_FILE = "data/vietnam_real_estate_sampled.csv"

print("⏳ Đang đọc file lớn...")
df = pd.read_csv(INPUT_FILE, low_memory=False)
print(f"✅ Tổng records: {len(df):,}")
print(f"📋 Columns: {df.columns.tolist()}")

# Xem phân phối
print(f"\n📊 Phân phối theo province:")
print(df["province_name"].value_counts().head(10))

print(f"\n📊 Phân phối theo property_type:")
print(df["property_type_name"].value_counts().head(10))


sampled = df.groupby("province_name", group_keys=False).apply(
    lambda x: x.sample(
        min(len(x), int(100000 * len(x) / len(df))),
        random_state=42
    )
)

# Đảm bảo đủ 100k
if len(sampled) < 100000:
    sampled = df.sample(100000, random_state=42)

print(f"\n✅ Sau khi sample: {len(sampled):,} records")
print(f"📊 Size ước tính: ~{len(sampled) * 500 / 1024 / 1024:.0f} MB")

sampled.to_csv(OUTPUT_FILE, index=False)
print(f"💾 Đã lưu → {OUTPUT_FILE}")