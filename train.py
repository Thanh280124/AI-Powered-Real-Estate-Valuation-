import sys
sys.path.append(".")
from utils.data_processor import (
    load_sale_data, load_rental_data,
    clean_sale_data, clean_rental_data
)
from utils.model_trainer import run_training_pipeline

# ===== TRAIN MODEL BÁN =====
print("\n🏠 TRAIN MODEL BÁN BĐS")
df_sale = clean_sale_data(load_sale_data())
sale_results, sale_best = run_training_pipeline(df_sale, model_type="sale")

# ===== TRAIN MODEL CHO THUÊ =====
print("\n🏠 TRAIN MODEL CHO THUÊ BĐS")
df_rental = clean_rental_data(load_rental_data())
rental_results, rental_best = run_training_pipeline(df_rental, model_type="rental")

# ===== TỔNG KẾT =====
print("\n" + "="*50)
print("🎉 HOÀN THÀNH TRAINING!")
print(f"   Sale model tốt nhất   : {sale_best}")
print(f"   Rental model tốt nhất : {rental_best}")
print("="*50)