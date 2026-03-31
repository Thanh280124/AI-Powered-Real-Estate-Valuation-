import streamlit as st
import sys
sys.path.append(".")

# ========== CẤU HÌNH TRANG ==========
st.set_page_config(
    page_title="T-Bank | AI Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title {
        font-size: 2.5rem;
        font-weight: 700;
        color: #1a1a2e;
        text-align: center;
    }
    .sub-title {
        font-size: 1.1rem;
        color: #666;
        text-align: center;
        margin-bottom: 2rem;
    }
    .result-box {
        background: #f0f4ff;
        border-radius: 12px;
        padding: 1.5rem;
        margin-top: 1rem;
    }
    .stButton > button {
        background-color: #1a1a2e;
        color: white;
        border-radius: 8px;
        width: 100%;
        padding: 0.6rem;
        font-size: 1rem;
    }
</style>
""", unsafe_allow_html=True)

# ========== LOAD DATA & MODEL ==========
@st.cache_data
def load_data():
    from utils.data_processor import (
        load_sale_data, load_rental_data,
        clean_sale_data, clean_rental_data
    )
    df_sale = clean_sale_data(load_sale_data())
    df_rental = clean_rental_data(load_rental_data())
    return df_sale, df_rental

df_sale, df_rental = load_data()

# Lấy danh sách thành phố và quận từ data thật
all_cities_sale = sorted(df_sale["city"].dropna().unique().tolist())
all_cities_rental = sorted(df_rental["city"].dropna().unique().tolist())

# ========== SIDEBAR ==========
with st.sidebar:
    st.title("🏠 T-Bank Valuation")
    st.markdown("---")

    language = st.radio("🌐 Language", ["Tiếng Việt", "English"])

    st.markdown("---")
    page = st.selectbox("📌 Menu", [
        "🏠 Định giá BĐS",
        "🔍 So sánh BĐS",
        "📊 Thị trường",
        "ℹ️ Giới thiệu"
    ])

    st.markdown("---")
    st.caption(f"📦 Sale data: {len(df_sale):,} BĐS")
    st.caption(f"📦 Rental data: {len(df_rental):,} BĐS")

# ========== HEADER ==========
st.markdown('<p class="main-title">🏠 T-Bank AI Real Estate Valuation</p>',
            unsafe_allow_html=True)
st.markdown('<p class="sub-title">Định giá bất động sản thông minh · Powered by XGBoost AI</p>',
            unsafe_allow_html=True)
st.divider()

# ========== TRANG ĐỊNH GIÁ ==========
if page == "🏠 Định giá BĐS":
    from utils.predictor import predict_price, get_similar_properties

    mode = st.radio("Loại giao dịch", ["🏷️ Mua bán", "🏠 Cho thuê"], horizontal=True)
    is_sale = mode == "🏷️ Mua bán"
    model_type = "sale" if is_sale else "rental"
    df_current = df_sale if is_sale else df_rental
    all_cities = all_cities_sale if is_sale else all_cities_rental

    st.subheader("📋 Nhập thông tin bất động sản")
    col1, col2 = st.columns(2)

    with col1:
        area = st.number_input(
            "📐 Diện tích (m²)", min_value=10, max_value=1000, value=60)
        bedrooms = st.selectbox("🛏️ Số phòng ngủ", [0, 1, 2, 3, 4, 5])
        bathrooms = st.selectbox("🚿 Số phòng tắm", [0, 1, 2, 3, 4])

    with col2:
        city = st.selectbox("🏙️ Tỉnh/Thành phố", all_cities)

        # Quận lọc theo thành phố đã chọn
        districts_in_city = sorted(
            df_current[df_current["city"] == city]["district"]
            .dropna().unique().tolist()
        )
        district = st.selectbox("📍 Quận/Huyện", districts_in_city)

    st.markdown("---")

    if st.button("🔮 Định giá ngay"):
        with st.spinner("🤖 AI đang phân tích..."):
            try:
                predicted, low, high = predict_price(
                    model_type, area, bedrooms, bathrooms, district, city
                )

                st.success("✅ Định giá hoàn tất!")

                # Hiển thị kết quả
                col1, col2, col3 = st.columns(3)

                if is_sale:
                    col1.metric("💰 Giá thấp nhất",
                                f"{low/1000:.2f} tỷ" if low >= 1000 else f"{low:.0f} triệu")
                    col2.metric("🎯 Giá ước tính",
                                f"{predicted/1000:.2f} tỷ" if predicted >= 1000 else f"{predicted:.0f} triệu",
                                delta="XGBoost AI")
                    col3.metric("💰 Giá cao nhất",
                                f"{high/1000:.2f} tỷ" if high >= 1000 else f"{high:.0f} triệu")
                else:
                    col1.metric("💰 Giá thấp nhất", f"{low:.1f} triệu/tháng")
                    col2.metric("🎯 Giá ước tính", f"{predicted:.1f} triệu/tháng",
                                delta="XGBoost AI")
                    col3.metric("💰 Giá cao nhất", f"{high:.1f} triệu/tháng")

                # Giá/m²
                st.info(f"📐 Giá/m²: **{predicted/area:.1f} triệu/m²**")

                # BĐS tương tự
                st.markdown("### 🔍 BĐS tương tự")
                similar = get_similar_properties(
                    df_current, district, city, area, predicted
                )
                if len(similar) > 0:
                    similar.columns = [
                        "Quận", "Thành phố", "Diện tích (m²)",
                        "Phòng ngủ", "Phòng tắm",
                        "Giá (triệu)", "Giá/m²"
                    ]
                    st.dataframe(similar, use_container_width=True)
                else:
                    st.info("Không tìm thấy BĐS tương tự trong khu vực này.")

            except Exception as e:
                st.error(f"❌ Lỗi: {e}")

# ========== TRANG SO SÁNH ==========
elif page == "🔍 So sánh BĐS":
    st.subheader("🔍 So sánh BĐS theo khu vực")

    col1, col2 = st.columns(2)
    with col1:
        selected_city = st.selectbox("Chọn thành phố", all_cities_sale)
    with col2:
        top_n = st.slider("Số BĐS hiển thị", 5, 50, 20)

    filtered = df_sale[df_sale["city"] == selected_city].nsmallest(top_n, "price_million")
    st.dataframe(
        filtered[["district", "area_m2", "bedrooms_num",
                  "bathrooms_num", "price_million", "price_per_m2"]],
        use_container_width=True
    )

# ========== TRANG THỊ TRƯỜNG ==========
elif page == "📊 Thị trường":
    import plotly.express as px

    st.subheader("📊 Phân tích thị trường")

    tab1, tab2 = st.tabs(["🏷️ Mua bán", "🏠 Cho thuê"])

    for tab, df, label in [(tab1, df_sale, "Bán"), (tab2, df_rental, "Cho thuê")]:
        with tab:
            col1, col2 = st.columns(2)

            with col1:
                top_cities = (df["city"].value_counts().head(10)
                              .reset_index())
                top_cities.columns = ["city", "count"]
                fig = px.bar(top_cities, x="city", y="count",
                             title=f"Top 10 tỉnh/thành ({label})",
                             color_discrete_sequence=["#1a1a2e"])
                st.plotly_chart(fig, use_container_width=True)

            with col2:
                city_price = (df.groupby("city")["price_million"]
                              .median().sort_values(ascending=False)
                              .head(10).reset_index())
                fig2 = px.bar(city_price, x="city", y="price_million",
                              title=f"Giá trung bình theo thành phố ({label})",
                              color_discrete_sequence=["#e94560"])
                st.plotly_chart(fig2, use_container_width=True)

            fig3 = px.scatter(
                df.sample(min(3000, len(df))),
                x="area_m2", y="price_million", color="city",
                title="Diện tích vs Giá",
                opacity=0.4,
                labels={"area_m2": "Diện tích (m²)",
                        "price_million": "Giá (triệu)"}
            )
            st.plotly_chart(fig3, use_container_width=True)

# ========== TRANG GIỚI THIỆU ==========
elif page == "ℹ️ Giới thiệu":
    st.subheader("ℹ️ Về dự án T-Bank Valuation")
    st.markdown("""
    **T-Bank AI Real Estate Valuation** là ứng dụng định giá bất động sản tự động
    sử dụng trí tuệ nhân tạo cho thị trường Việt Nam.

    ### 🤖 Công nghệ sử dụng
    - **ML Model**: XGBoost (R² ~0.69)
    - **Data**: 34,000+ BĐS bán · 14,000+ BĐS cho thuê
    - **Framework**: Streamlit + Python

    ### 👤 Thông tin
    - 🎓 Sinh viên: Pham Thanh
    - 🏫 Trường: VAMK - University of Applied Sciences
    - 📅 Năm: 2026
    """)