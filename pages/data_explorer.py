import streamlit as st
import plotly.express as px
import sys

sys.path.append(".")
from utils.data_processor import (
    load_sale_data, load_rental_data,
    clean_sale_data, clean_rental_data, get_stats
)

st.title("🔍 Khám phá Dataset")

# ===== LOAD DATA =====
@st.cache_data
def get_data():
    df_sale = clean_sale_data(load_sale_data())
    df_rental = clean_rental_data(load_rental_data())
    return df_sale, df_rental

try:
    df_sale, df_rental = get_data()
except FileNotFoundError as e:
    st.error(f"❌ Không tìm thấy file: {e}")
    st.stop()

# ===== CHỌN LOẠI DATA =====
tab1, tab2 = st.tabs(["🏷️ Bán", "🏠 Cho thuê"])

for tab, df, label in [
    (tab1, df_sale, "Bán"),
    (tab2, df_rental, "Cho thuê")
]:
    with tab:
        stats = get_stats(df)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Tổng bản ghi", f"{stats['total']:,}")
        col2.metric("Giá TB", f"{stats['avg_price']:,.0f} triệu")
        col3.metric("Diện tích TB", f"{stats['avg_area']:.0f} m²")
        col4.metric("Tỉnh/Thành", stats['cities'])

        st.divider()

        # Xem dữ liệu mẫu
        st.subheader("📋 Dữ liệu mẫu")
        st.dataframe(
            df[["district", "city", "area_m2", "bedrooms_num",
                "bathrooms_num", "price_million", "price_per_m2"]].head(50),
            width="stretch"                    # ← đã sửa
        )

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            # Phân phối giá
            fig1 = px.histogram(
                df, 
                x="price_million", 
                nbins=60,
                title=f"Phân phối giá ({label})",
                labels={"price_million": "Giá (triệu VNĐ)"},
                color_discrete_sequence=["#1a1a2e"]
            )
            st.plotly_chart(fig1, width="stretch")   # ← đã sửa

        with col_b:
            # Top 10 tỉnh/thành - ĐÃ SỬA LỖI DUPLICATE 'count'
            top_cities = (
                df["city"]
                .value_counts()
                .head(10)
                .reset_index(name="số_lượng")      # Cách này an toàn nhất
                .rename(columns={"city": "city"})
            )

            fig2 = px.bar(
                top_cities, 
                x="city", 
                y="số_lượng",
                title="Top 10 tỉnh/thành",
                color_discrete_sequence=["#e94560"]
            )
            
            fig2.update_layout(
                xaxis_title="Tỉnh/Thành phố",
                yaxis_title="Số lượng bất động sản",
                xaxis_tickangle=-45
            )
            
            st.plotly_chart(fig2, width="stretch")   # ← đã sửa

        # Scatter: Diện tích vs Giá
        fig3 = px.scatter(
            df.sample(min(2000, len(df))),
            x="area_m2", 
            y="price_million",
            color="city",
            title="Diện tích vs Giá",
            labels={
                "area_m2": "Diện tích (m²)", 
                "price_million": "Giá (triệu)"
            },
            opacity=0.5
        )
        st.plotly_chart(fig3, width="stretch")       