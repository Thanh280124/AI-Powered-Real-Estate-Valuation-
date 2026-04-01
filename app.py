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
    from utils.data_processor import (
        load_sale_data, load_rental_data,
        clean_sale_data, clean_rental_data
    )
    return clean_sale_data(load_sale_data()), clean_rental_data(load_rental_data())

df_sale, df_rental = load_data()
all_cities_sale = sorted(df_sale["city"].dropna().unique().tolist())
all_cities_rental = sorted(df_rental["city"].dropna().unique().tolist())

with st.sidebar:
    st.title("🏠 T-Bank Valuation")
    st.markdown("---")
    lang = st.radio("🌐 Language", ["VI", "EN", "FI", "SV"])
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
    st.caption(f"📦 {t(lang, 'rental_data')}: {len(df_rental):,}")

st.markdown(f'<p class="main-title">{t(lang, "main_title")}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{t(lang, "sub_title")}</p>', unsafe_allow_html=True)
st.divider()

# ========== TRANG ĐỊNH GIÁ ==========
if page == t(lang, "page_valuation"):
    from utils.predictor import predict_price, get_similar_properties
    from utils.chatbot import get_system_prompt, get_context_message, chat_with_advisor

    mode = st.radio(t(lang, "transaction_type"),
                    [t(lang, "buy_sell"), t(lang, "rent")], horizontal=True)
    is_sale = mode == t(lang, "buy_sell")
    model_type = "sale" if is_sale else "rental"
    df_current = df_sale if is_sale else df_rental
    all_cities = all_cities_sale if is_sale else all_cities_rental

    st.subheader(t(lang, "input_title"))
    col1, col2 = st.columns(2)
    with col1:
        area = st.number_input(t(lang, "area"), min_value=10, max_value=1000, value=60)
        bedrooms = st.selectbox(t(lang, "bedrooms"), [0, 1, 2, 3, 4, 5])
        bathrooms = st.selectbox(t(lang, "bathrooms"), [0, 1, 2, 3, 4])
    with col2:
        city = st.selectbox(t(lang, "city"), all_cities)
        districts_in_city = sorted(
            df_current[df_current["city"] == city]["district"].dropna().unique().tolist()
        )
        district = st.selectbox(t(lang, "district"), districts_in_city)

    st.markdown("---")

    if st.button(t(lang, "btn_predict")):
        with st.spinner(t(lang, "predicting")):
            try:
                predicted, low, high = predict_price(
                    model_type, area, bedrooms, bathrooms, district, city
                )
                st.session_state["result"] = {
                    "predicted": predicted, "low": low, "high": high,
                    "area": area, "bedrooms": bedrooms, "bathrooms": bathrooms,
                    "district": district, "city": city, "mode": mode,
                    "is_sale": is_sale, "model_type": model_type,
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
        is_sale = r["is_sale"]

        st.success(t(lang, "done"))
        col1, col2, col3 = st.columns(3)
        if is_sale:
            col1.metric(t(lang, "price_low"),
                        f"{low/1000:.2f} {t(lang, 'unit_ty')}" if low >= 1000 else f"{low:.0f} {t(lang, 'unit_trieu')}")
            col2.metric(t(lang, "price_est"),
                        f"{predicted/1000:.2f} {t(lang, 'unit_ty')}" if predicted >= 1000 else f"{predicted:.0f} {t(lang, 'unit_trieu')}",
                        delta="XGBoost AI")
            col3.metric(t(lang, "price_high"),
                        f"{high/1000:.2f} {t(lang, 'unit_ty')}" if high >= 1000 else f"{high:.0f} {t(lang, 'unit_trieu')}")
        else:
            col1.metric(t(lang, "price_low"), f"{low:.1f} {t(lang, 'unit_thang')}")
            col2.metric(t(lang, "price_est"), f"{predicted:.1f} {t(lang, 'unit_thang')}", delta="XGBoost AI")
            col3.metric(t(lang, "price_high"), f"{high:.1f} {t(lang, 'unit_thang')}")

        st.info(f"{t(lang, 'price_per_m2')}: **{r['price_per_m2']:.1f} {t(lang, 'unit_trieu')}/m²**")

        st.markdown(t(lang, "similar_title"))
        df_current2 = df_sale if is_sale else df_rental
        similar = get_similar_properties(df_current2, r["district"], r["city"], r["area"], predicted)
        if len(similar) > 0:
            similar.columns = [
                t(lang, "col_district"), t(lang, "col_city"), t(lang, "col_area"),
                t(lang, "col_bed"), t(lang, "col_bath"), t(lang, "col_price"), t(lang, "col_price_m2")
            ]
            st.dataframe(similar, use_container_width=True)
        else:
            st.info(t(lang, "no_similar"))

        # ===== CHATBOX =====
        st.markdown("---")
        st.markdown("### 💬 Hỏi chuyên gia AI")

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        context = get_context_message(r, lang)
        system_messages = [
            {"role": "system", "content": get_system_prompt(lang)},
            {"role": "user", "content": context},
            {"role": "assistant", "content": "Tôi đã nắm đầy đủ thông tin BĐS này. Bạn muốn hỏi gì?"},
        ]

        if st.button("🔄 Reset chat"):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("**💡 Gợi ý:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        quick_questions = {
            "VI": ["Giá này có hợp lý không?", "Nên mua không?", "Khu vực này có tiềm năng không?"],
            "EN": ["Is this price reasonable?", "Should I buy?", "Is this area promising?"],
            "FI": ["Onko hinta kohtuullinen?", "Kannattaako ostaa?", "Onko alue lupaava?"],
            "SV": ["Är priset rimligt?", "Ska jag köpa?", "Är området lovande?"]
        }
        questions = quick_questions.get(lang, quick_questions["VI"])
        for col, q in zip([col_q1, col_q2, col_q3], questions):
            if col.button(q, key=f"q_{q}"):
                st.session_state["messages"].append({"role": "user", "content": q})
                with st.spinner("🤖 Đang phân tích..."):
                    reply = chat_with_advisor(system_messages + st.session_state["messages"], lang)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Hỏi gì đó về BĐS này...", key="chat_main"):
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("assistant"):
                with st.spinner("🤖 Đang phân tích..."):
                    reply = chat_with_advisor(system_messages + st.session_state["messages"], lang)
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
    filtered = df_sale[df_sale["city"] == selected_city].nsmallest(top_n, "price_million")
    st.dataframe(
        filtered[["district", "area_m2", "bedrooms_num", "bathrooms_num", "price_million", "price_per_m2"]],
        use_container_width=True
    )

# ========== TRANG THỊ TRƯỜNG ==========
elif page == t(lang, "page_market"):
    import plotly.express as px
    st.subheader(t(lang, "market_title"))
    st.caption(t(lang, "data_note"))
    tab1, tab2 = st.tabs([t(lang, "tab_sale"), t(lang, "tab_rental")])
    for tab, df, label_key in [(tab1, df_sale, "tab_sale"), (tab2, df_rental, "tab_rental")]:
        with tab:
            col1, col2 = st.columns(2)
            with col1:
                top_cities = df["city"].value_counts().head(10).reset_index()
                top_cities.columns = ["city", "count"]
                fig = px.bar(top_cities, x="city", y="count",
                             title=f"{t(lang, 'chart_top_cities')} ({t(lang, label_key)})",
                             color_discrete_sequence=["#1a1a2e"])
                st.plotly_chart(fig, use_container_width=True)
            with col2:
                city_price = (df.groupby("city")["price_million"]
                              .median().sort_values(ascending=False).head(10).reset_index())
                fig2 = px.bar(city_price, x="city", y="price_million",
                              title=f"{t(lang, 'chart_avg_price')} ({t(lang, label_key)})",
                              color_discrete_sequence=["#e94560"])
                st.plotly_chart(fig2, use_container_width=True)
            fig3 = px.scatter(
                df.sample(min(3000, len(df))),
                x="area_m2", y="price_million", color="city",
                title=t(lang, "chart_scatter"), opacity=0.4,
                labels={"area_m2": t(lang, "col_area"), "price_million": t(lang, "col_price")}
            )
            st.plotly_chart(fig3, use_container_width=True)

# ========== TRANG GIỚI THIỆU ==========
elif page == t(lang, "page_about"):
    st.subheader(t(lang, "about_title"))
    st.markdown(t(lang, "about_body"))