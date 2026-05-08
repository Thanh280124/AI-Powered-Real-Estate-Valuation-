import streamlit as st
import plotly.express as px
import pandas as pd
import sys

sys.path.append(".")
from utils.data_processor import load_and_clean_data, get_stats

st.set_page_config(
    page_title="🔍 Data Explorer | T-Bank Ames",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Ames Housing Data Explorer")
st.markdown("### Comprehensive Analysis of Ames, Iowa Real Estate Dataset")

# ===== LOAD DATA =====
@st.cache_data
def get_data():
    df = load_and_clean_data("data/ames_housing.csv")
    return df

try:
    df = get_data()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

st.success(f"✅ Loaded **{len(df):,}** residential properties in Ames, Iowa")

# ===== METRICS =====
stats = get_stats(df)

col1, col2, col3, col4 = st.columns(4)
col1.metric("Total Properties", f"{stats['total']:,}")
col2.metric("Average Price", f"${stats['avg_price']:,.0f}")
col3.metric("Average Area", f"{stats['avg_area']:,.0f} sq ft")
col4.metric("Neighborhoods", stats['neighborhoods'])

st.divider()

# ===== TABS =====
tab1, tab2 = st.tabs(["📊 Overview", "🔍 Detailed Analysis"])

with tab1:
    # Sample Data
    st.subheader("📋 Sample Properties")
    display_cols = [
        "neighborhood", "building_type", "house_style", 
        "area_sqft", "bedrooms_num", "bathrooms_num", 
        "overall_quality", "price_usd", "price_per_sqft"
    ]
    
    sample = df[display_cols].sample(15).copy()
    sample["price_usd"] = sample["price_usd"].apply(lambda x: f"${x:,.0f}")
    sample["price_per_sqft"] = sample["price_per_sqft"].apply(lambda x: f"${x:.1f}")
    
    sample = sample.rename(columns={
        "neighborhood": "Neighborhood",
        "building_type": "Type",
        "house_style": "Style",
        "area_sqft": "Area (sqft)",
        "bedrooms_num": "Bedrooms",
        "bathrooms_num": "Bathrooms",
        "overall_quality": "Quality",
        "price_usd": "Price (USD)",
        "price_per_sqft": "Price/sqft"
    })
    
    st.dataframe(sample, use_container_width=True)

    # Price Distribution
    col_a, col_b = st.columns(2)
    with col_a:
        fig1 = px.histogram(
            df, x="price_usd", nbins=80,
            title="Price Distribution",
            labels={"price_usd": "Sale Price (USD)"},
            color_discrete_sequence=["#1a1a2e"]
        )
        st.plotly_chart(fig1, use_container_width=True)

    with col_b:
        fig2 = px.box(
            df, x="neighborhood", y="price_usd",
            title="Price by Neighborhood",
            color_discrete_sequence=["#e94560"]
        )
        fig2.update_layout(xaxis_tickangle=-45)
        st.plotly_chart(fig2, use_container_width=True)

with tab2:
    st.subheader("📈 Advanced Analysis")
    
    col1, col2 = st.columns(2)
    
    with col1:
        # Area vs Price
        fig3 = px.scatter(
            df.sample(min(1500, len(df))),
            x="area_sqft",
            y="price_usd",
            color="overall_quality",
            title="Living Area vs Sale Price (colored by Quality)",
            labels={"area_sqft": "Area (sqft)", "price_usd": "Price (USD)"},
            opacity=0.6,
            color_continuous_scale="Viridis"
        )
        st.plotly_chart(fig3, use_container_width=True)

    with col2:
        # Quality vs Price
        fig4 = px.box(
            df, 
            x="overall_quality", 
            y="price_usd",
            title="Price Distribution by Overall Quality",
            labels={"overall_quality": "Overall Quality (1-10)"}
        )
        st.plotly_chart(fig4, use_container_width=True)

    # Top Neighborhoods
    st.subheader("🏘️ Top Neighborhoods by Median Price")
    nbr_price = df.groupby("neighborhood")["price_usd"].median().sort_values(ascending=False).head(15)
    fig5 = px.bar(
        x=nbr_price.index, 
        y=nbr_price.values,
        title="Median Price by Neighborhood",
        labels={"x": "Neighborhood", "y": "Median Price (USD)"},
        color_discrete_sequence=["#1a1a2e"]
    )
    fig5.update_layout(xaxis_tickangle=-45)
    st.plotly_chart(fig5, use_container_width=True)