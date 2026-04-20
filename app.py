import streamlit as st
import sys
sys.path.append(".")
from utils.translations import t

st.set_page_config(
    page_title="T-Bank | AI Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

st.markdown("""
<style>
    .main-title { font-size: 3.5rem; font-weight: 700; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .stButton > button { background-color: #1a1a2e; color: white; border-radius: 8px; width: 100%; padding: 0.6rem; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    from utils.data_processor import load_and_clean_data
    return load_and_clean_data("data/vietnam_real_estate_sampled.csv")

df_sale = load_data()
all_cities_sale = sorted(df_sale["city"].dropna().unique().tolist())

with st.sidebar:
    st.title("🏠 T-Bank Valuation")
    st.markdown("---")
    lang = st.radio("🌐 Language", ["EN", "VI", "FI", "SV"])
    st.session_state["lang"] = lang
    st.markdown("---")
    page = st.selectbox(t(lang, "menu"), [
        t(lang, "page_valuation"),
        t(lang, "page_compare"),
        t(lang, "page_market"),
        t(lang, "page_about"),
    ])
    st.markdown("---")
    st.caption(f"📦 {t(lang, 'sale_data')}: {len(df_sale):,}")

st.markdown(f'<p class="main-title">{t(lang, "main_title")}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{t(lang, "sub_title")}</p>', unsafe_allow_html=True)
st.divider()

# ========== TRANG ĐỊNH GIÁ ==========
if page == t(lang, "page_valuation"):
    from utils.predictor import predict_price, get_similar_properties
    from utils.chatbot import get_system_prompt, get_context_message, chat_with_advisor

    st.subheader(t(lang, "input_title"))

    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(t(lang, "area"), min_value=10, max_value=2000, value=60)
        bedrooms = st.selectbox(t(lang, "bedrooms"), [0, 1, 2, 3, 4, 5, 6])
        bathrooms = st.selectbox(t(lang, "bathrooms"), [0, 1, 2, 3, 4, 5])
        floors = st.selectbox("🏢 Số tầng / Floors", [1, 2, 3, 4, 5, 6, 7, 8])
        frontage_width = st.number_input("📏 Mặt tiền (m) / Frontage (m)",
                                          min_value=1.0, max_value=50.0,
                                          value=4.0, step=0.5)
    with col2:
        city = st.selectbox(t(lang, "city"), all_cities_sale)
        districts_in_city = sorted(
            df_sale[df_sale["city"] == city]["district"].dropna().unique().tolist()
        )
        district = st.selectbox(t(lang, "district"), districts_in_city)
        property_type = st.selectbox("🏘️ Loại BĐS / Property Type", [
            "Nhà", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse", "Đất"
        ])
        direction = st.selectbox("🧭 Hướng / Direction", [
            "unknown", "Đông", "Tây", "Nam", "Bắc",
            "Đông Nam", "Đông Bắc", "Tây Nam", "Tây Bắc"
        ])
        street_type = st.selectbox("🛣️ Loại đường / Street Type", [
            "unknown", "main_road", "alley"
        ], format_func=lambda x: {
            "unknown": "Không rõ / Unknown",
            "main_road": "Mặt phố / Main Road",
            "alley": "Ngõ / Alley"
        }.get(x, x))
        legal_status = st.selectbox("📄 Pháp lý / Legal", [
            "unknown", "red_book", "pink_book"
        ], format_func=lambda x: {
            "unknown": "Không rõ / Unknown",
            "red_book": "Sổ đỏ / Red Book",
            "pink_book": "Sổ hồng / Pink Book"
        }.get(x, x))

    st.markdown("---")

    if st.button(t(lang, "btn_predict")):
        with st.spinner(t(lang, "predicting")):
            try:
                predicted, low, high = predict_price(
                    area=area, bedrooms=bedrooms, bathrooms=bathrooms,
                    district=district, city=city,
                    property_type=property_type, floors=floors,
                    frontage_width=frontage_width, direction=direction,
                    street_type=street_type, legal_status=legal_status,
                )
                st.session_state["result"] = {
                    "predicted": predicted, "low": low, "high": high,
                    "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "district": district, "city": city,
                    "property_type": property_type,
                    "mode": property_type,
                    "is_sale": True,
                    "price_per_m2": predicted / area
                }
                st.session_state["messages"] = []
            except Exception as e:
                st.error(f"❌ {e}")

    if "result" in st.session_state:
        r = st.session_state["result"]
        predicted = r["predicted"]
        low = r["low"]
        high = r["high"]

        st.success(t(lang, "done"))
        col1, col2, col3 = st.columns(3)
        col1.metric(t(lang, "price_low"),
                    f"{low/1000:.2f} {t(lang, 'unit_ty')}" if low >= 1000
                    else f"{low:.0f} {t(lang, 'unit_trieu')}")
        col2.metric(t(lang, "price_est"),
                    f"{predicted/1000:.2f} {t(lang, 'unit_ty')}" if predicted >= 1000
                    else f"{predicted:.0f} {t(lang, 'unit_trieu')}",
                    delta="XGBoost AI")
        col3.metric(t(lang, "price_high"),
                    f"{high/1000:.2f} {t(lang, 'unit_ty')}" if high >= 1000
                    else f"{high:.0f} {t(lang, 'unit_trieu')}")

        st.info(f"{t(lang, 'price_per_m2')}: **{r['price_per_m2']:.1f} {t(lang, 'unit_trieu')}/m²**")

        # ===== BĐS TƯƠNG TỰ =====
        st.markdown(t(lang, "similar_title"))
        similar = get_similar_properties(
            df_sale, r["district"], r["city"],
            r["area"], predicted, r["property_type"]
        )
        if len(similar) > 0:
            # Đổi tên cột đẹp
            rename_map = {
                "property_type": "Type",
                "district": "District",
                "city": "City",
                "area_m2": "Area (m²)",
                "bedrooms_num": "Beds",
                "bathrooms_num": "Baths",
                "price_million": "Price",
                "price_per_m2": "Price/m²",
            }
            similar = similar.rename(columns=rename_map)

            # Đổi price sang tỷ
            if "Price" in similar.columns:
                similar["Price"] = similar["Price"].apply(
                    lambda x: f"{x/1000:.2f} tỷ" if x >= 1000 else f"{x:.0f} tr"
                )
            if "Price/m²" in similar.columns:
                similar["Price/m²"] = similar["Price/m²"].apply(
                lambda x: f"{float(x):.1f} tr/m²" if not str(x).endswith("tr/m²") else x
                )

            st.dataframe(similar, use_container_width=True)
        else:
            st.info(t(lang, "no_similar"))

        # ===== CHATBOX =====
        st.markdown("---")
        st.markdown("### 💬 Ask the AI Expert")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        context = get_context_message(r, "EN")
        system_messages = [
            {"role": "system", "content": get_system_prompt("EN")},
            {"role": "user", "content": context},
            {"role": "assistant", "content": "I have the property details. What would you like to know?"},
        ]

        if st.button("🔄 Reset chat"):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("**💡 Suggested Questions:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        quick_questions = {
            "EN": ["Is this price reasonable?", "Should I buy?", "Is this area promising?"],
            "VI": ["Giá này có hợp lý không?", "Nên mua không?", "Khu vực này có tiềm năng không?"],
            "FI": ["Onko hinta kohtuullinen?", "Kannattaako ostaa?", "Onko alue lupaava?"],
            "SV": ["Är priset rimligt?", "Ska jag köpa?", "Är området lovande?"]
        }
        questions = quick_questions.get(lang, quick_questions["EN"])
        for col, q in zip([col_q1, col_q2, col_q3], questions):
            if col.button(q, key=f"q_{q}"):
                st.session_state["messages"].append({"role": "user", "content": q})
                with st.spinner("🤖 Analyzing..."):
                    reply = chat_with_advisor(
                        system_messages + st.session_state["messages"], "EN"
                    )
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Ask about this property...", key="chat_main"):
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing..."):
                    reply = chat_with_advisor(
                        system_messages + st.session_state["messages"], "EN"
                    )
                st.write(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()

# ========== TRANG SO SÁNH ==========
elif page == t(lang, "page_compare"):
    st.subheader(t(lang, "compare_title"))
    col1, col2 = st.columns(2)
    with col1:
        selected_city = st.selectbox(t(lang, "select_city"), all_cities_sale)
    with col2:
        top_n = st.slider(t(lang, "num_display"), 5, 50, 20)

    # Thêm filter property type
    property_filter = st.multiselect(
        "🏘️ Lọc loại BĐS / Filter Property Type",
        ["Nhà", "Căn hộ chung cư", "Biệt thự/Nhà liền kề", "Shophouse", "Đất"],
        default=["Nhà", "Căn hộ chung cư"]
    )

    filtered = df_sale[
        (df_sale["city"] == selected_city) &
        (df_sale["property_type"].isin(property_filter))
    ].nsmallest(top_n, "price_million")

    display_cols = ["property_type", "district", "area_m2",
                    "bedrooms_num", "bathrooms_num",
                    "price_million", "price_per_m2"]
    st.dataframe(filtered[display_cols], use_container_width=True)

# ========== TRANG THỊ TRƯỜNG ==========
elif page == t(lang, "page_market"):
    import plotly.express as px
    st.subheader(t(lang, "market_title"))
    st.caption(t(lang, "data_note"))

    col1, col2 = st.columns(2)
    with col1:
        top_cities = df_sale["city"].value_counts().head(10).reset_index()
        top_cities.columns = ["city", "count"]
        fig = px.bar(top_cities, x="city", y="count",
                     title=t(lang, "chart_top_cities"),
                     color_discrete_sequence=["#1a1a2e"])
        st.plotly_chart(fig, use_container_width=True)

    with col2:
        city_price = (df_sale.groupby("city")["price_million"]
                      .median().sort_values(ascending=False)
                      .head(10).reset_index())
        fig2 = px.bar(city_price, x="city", y="price_million",
                      title=t(lang, "chart_avg_price"),
                      color_discrete_sequence=["#e94560"])
        st.plotly_chart(fig2, use_container_width=True)

    # Scatter theo property type
    fig3 = px.scatter(
        df_sale.sample(min(3000, len(df_sale))),
        x="area_m2", y="price_million",
        color="property_type",
        title=t(lang, "chart_scatter"),
        opacity=0.4,
        labels={"area_m2": t(lang, "col_area"),
                "price_million": t(lang, "col_price")}
    )
    st.plotly_chart(fig3, use_container_width=True)

    # Giá TB theo loại BĐS
    type_price = (df_sale.groupby("property_type")["price_million"]
                  .median().sort_values(ascending=False).reset_index())
    fig4 = px.bar(type_price, x="property_type", y="price_million",
                  title="Median Price by Property Type",
                  color_discrete_sequence=["#f0a500"])
    st.plotly_chart(fig4, use_container_width=True)

# ========== TRANG GIỚI THIỆU ==========
elif page == t(lang, "page_about"):
    st.subheader(t(lang, "about_title"))
    st.markdown(t(lang, "about_body"))
    