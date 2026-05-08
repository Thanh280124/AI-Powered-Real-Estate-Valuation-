import streamlit as st
import sys
sys.path.append(".")

# ==================== TRANSLATION ====================
from utils.translations import t

st.set_page_config(
    page_title="T-Bank | AI Real Estate Valuation",
    page_icon="🏠",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Language Selector
with st.sidebar:
    st.title("🏠 T-Bank Valuation")
    lang = st.selectbox("🌐 Language / Ngôn ngữ", ["EN", "VI", "FI", "SV"], index=0)
    st.markdown("---")

st.markdown("""
<style>
    .main-title { font-size: 3rem; font-weight: 700; text-align: center; }
    .sub-title { font-size: 1.1rem; color: #666; text-align: center; margin-bottom: 2rem; }
    .stButton > button { background-color: #1a1a2e; color: white; border-radius: 8px; width: 100%; padding: 0.6rem; font-size: 1rem; }
</style>
""", unsafe_allow_html=True)

@st.cache_data
def load_data():
    from utils.data_processor import load_and_clean_data
    return load_and_clean_data("data/ames_housing.csv")

df = load_data()
all_neighborhoods = sorted(df["neighborhood"].dropna().unique().tolist())
all_building_types = sorted(df["building_type"].dropna().unique().tolist())
all_styles = sorted(df["house_style"].dropna().unique().tolist())

QUALITY_MAP = {
    "Ex — Excellent": "Ex", "Gd — Good": "Gd", "TA — Average": "TA",
    "Fa — Fair": "Fa", "Po — Poor": "Po",
}

with st.sidebar:
    page = st.selectbox(t(lang, "menu"), [
        t(lang, "page_valuation"),
        t(lang, "page_compare"),
        t(lang, "page_market"),
        t(lang, "page_about")
    ])
    st.markdown("---")
    st.caption(f"📦 Dataset: {len(df):,} properties")
    st.caption(f"📍 Neighborhoods: {df['neighborhood'].nunique()}")
    st.caption(f"📅 Year built: {int(df['year_built'].min())}–{int(df['year_built'].max())}")

st.markdown(f'<p class="main-title">{t(lang, "main_title")}</p>', unsafe_allow_html=True)
st.markdown(f'<p class="sub-title">{t(lang, "sub_title")}</p>', unsafe_allow_html=True)
st.divider()

# ========== VALUATION PAGE ==========
if page == t(lang, "page_valuation"):
    from utils.predictor import predict_price, get_similar_properties
    from utils.chatbot import get_system_prompt, get_context_message, chat_with_advisor

    st.subheader(t(lang, "input_title"))

    col1, col2, col3 = st.columns(3)

    with col1:
        st.markdown("**📐 Size & Rooms**")
        area_sqft = st.number_input(t(lang, "area"), min_value=100, max_value=10000, value=1500)
        lot_area = st.number_input("Lot Area (sq ft)", min_value=100, max_value=100000, value=8000)
        bedrooms = st.selectbox(t(lang, "bedrooms"), [1, 2, 3, 4, 5, 6])
        bathrooms = st.selectbox(t(lang, "bathrooms"), [1, 2, 3, 4])
        total_rooms = st.selectbox("Total Rooms", [3, 4, 5, 6, 7, 8, 9, 10, 11, 12])

    with col2:
        st.markdown("**🏗️ Property Details**")
        year_built = st.number_input(t(lang, "year_built"), min_value=1872, max_value=2025, value=2000)
        overall_quality = st.slider(t(lang, "overall_quality"), 1, 10, 7,
            help="1=Very Poor, 5=Average, 10=Very Excellent")
        overall_condition = st.slider("Overall Condition (1-9)", 1, 9, 5,
            help="1=Very Poor, 5=Average, 9=Excellent")
        kitchen_qual_label = st.selectbox("Kitchen Quality", list(QUALITY_MAP.keys()), index=1)
        exterior_qual_label = st.selectbox("Exterior Quality", list(QUALITY_MAP.keys()), index=1)

    with col3:
        st.markdown("**📍 Location & Features**")
        neighborhood = st.selectbox(t(lang, "neighborhood"), all_neighborhoods)
        building_type = st.selectbox(t(lang, "building_type"), all_building_types,
            format_func=lambda x: {
                "1Fam": "Single Family (1Fam)",
                "2fmCon": "Two Family Conversion (2fmCon)",
                "Duplex": "Duplex",
                "TwnhsE": "Townhouse End Unit (TwnhsE)",
                "Twnhs": "Townhouse (Twnhs)",
            }.get(x, x))
        house_style = st.selectbox(t(lang, "house_style"), all_styles)
        has_garage = st.checkbox(t(lang, "has_garage"), value=True)
        garage_cars = st.selectbox(t(lang, "garage_capacity"), [0, 1, 2, 3, 4]) if has_garage else 0
        garage_area = st.number_input(t(lang, "garage_area"), 0, 2000, 400) if has_garage else 0
        has_basement = st.checkbox(t(lang, "has_basement"), value=True)
        basement_area = st.number_input(t(lang, "basement_area"), 0, 5000, 800) if has_basement else 0
        has_fireplace = st.checkbox(t(lang, "has_fireplace"), value=False)
        fireplaces = st.selectbox("Number of Fireplaces", [0, 1, 2, 3]) if has_fireplace else 0
        has_central_air = st.checkbox(t(lang, "central_air"), value=True)

    kitchen_quality  = QUALITY_MAP[kitchen_qual_label]
    exterior_quality = QUALITY_MAP[exterior_qual_label]

    st.markdown("---")

    if st.button(t(lang, "btn_predict")):
        with st.spinner(t(lang, "predicting")):
            try:
                predicted, low, high = predict_price(
                    area_sqft=area_sqft, bedrooms=bedrooms,
                    bathrooms=bathrooms, year_built=year_built,
                    overall_quality=overall_quality,
                    overall_condition=overall_condition,
                    neighborhood=neighborhood,
                    building_type=building_type,
                    house_style=house_style,
                    has_garage=has_garage, garage_cars=garage_cars,
                    garage_area=garage_area,
                    has_basement=has_basement, basement_area=basement_area,
                    has_fireplace=has_fireplace, fireplaces=fireplaces,
                    has_central_air=has_central_air,
                    kitchen_quality=kitchen_quality,
                    exterior_quality=exterior_quality,
                    total_rooms=total_rooms, lot_area=lot_area,
                )
                st.session_state["result"] = {
                    "predicted": predicted, "low": low, "high": high,
                    "area_sqft": area_sqft, "bedrooms": bedrooms,
                    "bathrooms": bathrooms, "year_built": year_built,
                    "overall_quality": overall_quality,
                    "neighborhood": neighborhood,
                    "building_type": building_type,
                    "house_style": house_style,
                    "has_garage": has_garage, "has_basement": has_basement,
                    "has_fireplace": has_fireplace,
                    "has_central_air": has_central_air,
                    "price_per_sqft": predicted / area_sqft,
                    "property_age": 2010 - year_built,
                }
                st.session_state["messages"] = []
                st.success(t(lang, "done"))
            except Exception as e:
                st.error(f"❌ Error: {e}")

    if "result" in st.session_state:
        r = st.session_state["result"]
        predicted = r["predicted"]
        low = r["low"]
        high = r["high"]

        st.success(t(lang, "done"))
        col1, col2, col3 = st.columns(3)
        col1.metric(t(lang, "price_low"),   f"${low:,.0f}")
        col2.metric(t(lang, "price_est"),   f"${predicted:,.0f}", delta="XGBoost AI")
        col3.metric(t(lang, "price_high"),  f"${high:,.0f}")

        col4, col5, col6 = st.columns(3)
        col4.metric(t(lang, "price_per_sqft"),   f"${r['price_per_sqft']:.1f}/sqft")
        col5.metric(t(lang, "property_age"),      f"{r['property_age']} years")
        col6.metric("⭐ Quality Score",     f"{r['overall_quality']}/10")

        st.markdown(t(lang, "similar_title"))
        similar = get_similar_properties(
            df, r["neighborhood"], r["area_sqft"],
            predicted, r["year_built"], r["overall_quality"]
        )
        if len(similar) > 0:
            rename = {
                "neighborhood": "Neighborhood",
                "building_type": "Type",
                "house_style": "Style",
                "area_sqft": "Area (sqft)",
                "bedrooms_num": "Beds",
                "bathrooms_num": "Baths",
                "year_built": "Year Built",
                "overall_quality": "Quality",
                "price_usd": "Price (USD)",
                "price_per_sqft": "$/sqft",
            }
            similar = similar.rename(columns=rename)
            if "Price (USD)" in similar.columns:
                similar["Price (USD)"] = similar["Price (USD)"].apply(lambda x: f"${x:,.0f}")
            if "$/sqft" in similar.columns:
                similar["$/sqft"] = similar["$/sqft"].apply(lambda x: f"${x:.1f}")
            st.dataframe(similar, use_container_width=True)
        else:
            st.info(t(lang, "no_similar"))

        # Chatbox
        st.markdown("---")
        st.markdown(t(lang, "ask_ai"))

        if "messages" not in st.session_state:
            st.session_state["messages"] = []

        context = get_context_message(r, lang)
        system_messages = [
            {"role": "system", "content": get_system_prompt(lang)},
            {"role": "user", "content": context},
            {"role": "assistant", "content": "I have the property details. What would you like to know?"},
        ]

        if st.button("🔄 Reset chat"):
            st.session_state["messages"] = []
            st.rerun()

        st.markdown("**💡 Suggested Questions:**")
        col_q1, col_q2, col_q3 = st.columns(3)
        questions = ["Is this price reasonable?", "Should I buy?", "Is this neighborhood promising?"]
        for col, q in zip([col_q1, col_q2, col_q3], questions):
            if col.button(q, key=f"q_{q}"):
                st.session_state["messages"].append({"role": "user", "content": q})
                with st.spinner("🤖 Analyzing..."):
                    reply = chat_with_advisor(system_messages + st.session_state["messages"], lang)
                st.session_state["messages"].append({"role": "assistant", "content": reply})
                st.rerun()

        for msg in st.session_state["messages"]:
            with st.chat_message(msg["role"]):
                st.write(msg["content"])

        if user_input := st.chat_input("Ask about this property...", key="chat_main"):
            st.session_state["messages"].append({"role": "user", "content": user_input})
            with st.chat_message("assistant"):
                with st.spinner("🤖 Analyzing..."):
                    reply = chat_with_advisor(system_messages + st.session_state["messages"], lang)
                st.write(reply)
            st.session_state["messages"].append({"role": "assistant", "content": reply})
            st.rerun()

# ========== COMPARE PAGE ==========
elif page == t(lang, "page_compare"):
    st.subheader("🔍 Compare Properties by Neighborhood")
    col1, col2 = st.columns(2)
    with col1:
        selected_nbr = st.selectbox("Select Neighborhood", all_neighborhoods)
    with col2:
        top_n = st.slider("Number of Properties", 5, 50, 20)

    quality_filter = st.slider("Minimum Quality Score", 1, 10, 5)

    filtered = df[
        (df["neighborhood"] == selected_nbr) &
        (df["overall_quality"] >= quality_filter)
    ].nsmallest(top_n, "price_usd")

    display = ["building_type", "house_style", "area_sqft", "bedrooms_num",
               "bathrooms_num", "year_built", "overall_quality",
               "price_usd", "price_per_sqft"]
    display = [c for c in display if c in filtered.columns]
    st.dataframe(filtered[display], use_container_width=True)

# ========== MARKET PAGE ==========
elif page == t(lang, "page_market"):
    import plotly.express as px
    st.subheader("📊 Market Analysis — Ames Housing Dataset")
    st.caption("📌 Data: Ames, Iowa housing market (De Cock 2011). 1,460 residential properties.")

    col1, col2 = st.columns(2)
    with col1:
        nbr_price = (df.groupby("neighborhood")["price_usd"]
                     .median().sort_values(ascending=False).reset_index())
        fig1 = px.bar(nbr_price, x="neighborhood", y="price_usd",
                      title="Median Price by Neighborhood",
                      color_discrete_sequence=["#1a1a2e"])
        fig1.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig1, use_container_width=True)

    with col2:
        fig2 = px.scatter(df.sample(min(800, len(df))),
                          x="area_sqft", y="price_usd",
                          color="overall_quality",
                          title="Area vs Price (coloured by Quality)",
                          opacity=0.6,
                          color_continuous_scale="Blues")
        st.plotly_chart(fig2, use_container_width=True)

    col3, col4 = st.columns(2)
    with col3:
        fig3 = px.scatter(df, x="year_built", y="price_usd",
                          title="Year Built vs Price",
                          opacity=0.4,
                          color_discrete_sequence=["#e94560"])
        st.plotly_chart(fig3, use_container_width=True)

    with col4:
        qual_price = (df.groupby("overall_quality")["price_usd"]
                      .median().reset_index())
        fig4 = px.bar(qual_price, x="overall_quality", y="price_usd",
                      title="Median Price by Quality Score",
                      color_discrete_sequence=["#f0a500"])
        st.plotly_chart(fig4, use_container_width=True)

# ========== ABOUT PAGE ==========
elif page == t(lang, "page_about"):
    st.subheader("ℹ️ About T-Bank AI Real Estate Valuation")
    st.markdown("""
    **T-Bank AI Real Estate Valuation** is a web-based application that uses machine learning
    to provide automated property price estimates.

    ### 🤖 Technology
    - **ML Model**: XGBoost (R² = 0.89, Accuracy ±20% = 94.7%)
    - **Dataset**: Ames Housing Dataset (De Cock 2011) — 1,460 properties, 79 features
    - **Framework**: Python + Streamlit
    - **Chatbot**: Groq llama-3.3-70b-versatile

    ### 📊 Key Features Used
    - Living area, lot area, basement area, garage area
    - Year built, property age, remodel year
    - Overall quality & condition (1-10 scale)
    - Neighborhood location
    - Building type, house style
    - Kitchen & exterior quality
    - Garage, basement, fireplace, central air

    ### 👤 Project Info
    - 🎓 Student: Pham Thanh
    - 🏫 VAMK — University of Applied Sciences
    - 📅 2026

    ### 📚 Reference
    De Cock, D. 2011. Ames, Iowa: Alternative to the Boston Housing Data Set.
    Journal of Statistics Education. Vol. 19, No. 3.
    """)