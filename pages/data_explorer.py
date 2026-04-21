import streamlit as st
import plotly.express as px
import pandas as pd
import sys

sys.path.append(".")
from utils.data_processor import (
    load_new_data,
    clean_new_data,
    get_stats
)

st.set_page_config(
    page_title="🔍 Data Explorer | T-Bank",
    page_icon="📊",
    layout="wide"
)

st.title("🔍 Data Explorer")

# ===== LOAD DATA =====
@st.cache_data
def get_data():
    df_raw = load_new_data("data/vietnam_real_estate_sampled.csv")
    
    sale_types = ["Nhà", "Biệt thự/Nhà liền kề", "Căn hộ chung cư", "Shophouse", "Đất"]
    
    df_sale = df_raw[df_raw["property_type"].isin(sale_types)].copy()
    df_sale = clean_new_data(df_sale, "sale")
    
    df_rental = clean_new_data(df_raw.copy(), "rent")
    
    return df_sale, df_rental

try:
    df_sale, df_rental = get_data()
except FileNotFoundError as e:
    st.error(f"❌ File not found: {e}\n\nPlease make sure data/vietnam_real_estate_sampled.csv exists.")
    st.stop()
except Exception as e:
    st.error(f"❌ Error loading data: {e}")
    st.stop()

st.success(f"✅ Loaded {len(df_sale):,} sale records | {len(df_rental):,} rental records")

# ===== TABS =====
tab1, tab2 = st.tabs(["🏷️ Sale", "🏠 Rental"])

for tab, df, label in [
    (tab1, df_sale, "Sale"),
    (tab2, df_rental, "Rental")
]:
    with tab:
        if len(df) == 0:
            st.warning(f"⚠️ No {label.lower()} properties found.")
            continue

        stats = get_stats(df)

        # Metrics
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Total Records", f"{stats['total']:,}")
        
        # Avg Price - hiển thị billion hoặc million (English)
        avg_price = stats['avg_price']
        if avg_price >= 1000:
            col2.metric("Avg Price", f"{avg_price/1000:.2f} billion VND")
        else:
            col2.metric("Avg Price", f"{avg_price:,.0f} million VND")
            
        col3.metric("Avg Area", f"{stats['avg_area']:.0f} m²")
        col4.metric("Cities/Provinces", stats['cities'])

        st.divider()

        # Sample Data - English
        st.subheader("📋 Sample Data")
        display_cols = ["district", "city", "area_m2", "bedrooms_num",
                        "bathrooms_num", "price_million", "price_per_m2"]
        
        sample_df = df[display_cols].head(50).copy()
        
        # Format Price: billion or million
        def format_price(x):
            if x >= 1000:
                return f"{x/1000:.2f} billion"
            else:
                return f"{x:,.0f} million"
        
        # Format Price per m²: always million/m²
        sample_df["price_million"] = sample_df["price_million"].apply(format_price)
        sample_df["price_per_m2"] = sample_df["price_per_m2"].apply(lambda x: f"{x:.1f} million/m²")
        
        # Rename columns to English for better readability
        sample_df = sample_df.rename(columns={
            "district": "District",
            "city": "City",
            "area_m2": "Area (m²)",
            "bedrooms_num": "Bedrooms",
            "bathrooms_num": "Bathrooms",
            "price_million": "Price",
            "price_per_m2": "Price/m²"
        })
        
        st.dataframe(sample_df, use_container_width=True)

        st.divider()

        col_a, col_b = st.columns(2)

        with col_a:
            fig1 = px.histogram(
                df, 
                x="price_million", 
                nbins=60,
                title=f"Price Distribution ({label})",
                labels={"price_million": "Price (million VND)"},
                color_discrete_sequence=["#1a1a2e"]
            )
            st.plotly_chart(fig1, use_container_width=True)

        with col_b:
            top_cities = (
                df["city"]
                .value_counts()
                .head(10)
                .reset_index(name="count")
            )

            fig2 = px.bar(
                top_cities, 
                x="city", 
                y="count",
                title=f"Top 10 Cities/Provinces ({label})",
                color_discrete_sequence=["#e94560"]
            )
            fig2.update_layout(
                xaxis_title="City / Province",
                yaxis_title="Number of Properties",
                xaxis_tickangle=-45
            )
            st.plotly_chart(fig2, use_container_width=True)

        # Scatter plot
        fig3 = px.scatter(
            df.sample(min(2000, len(df))),
            x="area_m2", 
            y="price_million",
            color="city",
            title=f"Area vs Price ({label})",
            labels={
                "area_m2": "Area (m²)", 
                "price_million": "Price (million VND)"
            },
            opacity=0.5
        )
        st.plotly_chart(fig3, use_container_width=True)