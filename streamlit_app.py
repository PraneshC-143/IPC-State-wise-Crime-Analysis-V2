"""
CrimeScope: Interactive IPC Crime Intelligence Dashboard
Main Streamlit Application
"""

import streamlit as st
import pandas as pd
import plotly.express as px
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error, mean_absolute_error
import numpy as np
import os
from datetime import datetime

# Import custom modules
from config import PAGE_TITLE, PAGE_LAYOUT, PAGE_ICON
from data_loader import load_data, validate_data, filter_data
from visualizations import (
    plot_top_districts, plot_crime_trend, plot_distribution,
    plot_correlation_heatmap, plot_heatmap_by_district, plot_crime_hotspots
)
from analytics import predict_crime_trend, get_crime_statistics, get_crime_by_type
from utils import (
    apply_custom_styling, format_number, get_download_button,
    display_kpi_card, display_warning_message, display_info_message
)


# ==================================================
# PAGE CONFIG
# ==================================================
st.set_page_config(
    page_title=PAGE_TITLE,
    page_icon=PAGE_ICON,
    layout=PAGE_LAYOUT,
    initial_sidebar_state="expanded"
)

# Apply custom styling
apply_custom_styling()


# ==================================================
# LOAD DATA
# ==================================================
df, crime_columns = load_data()

if not validate_data(df):
    st.error("❌ Failed to load or validate data. Please check the data file.")
    st.stop()


# ==================================================
# SIDEBAR CONFIGURATION
# ==================================================
with st.sidebar:
    st.title("🔍 CrimeScope")
    st.caption("IPC Crime Intelligence System")
    st.divider()
    
    # State Selection
    state = st.selectbox(
        "📍 Select State",
        sorted(df["state_name"].unique()),
        help="Choose a state to analyze"
    )
    
    # District Selection
    districts = sorted(df[df["state_name"] == state]["district_name"].unique())
    district = st.selectbox(
        "🏘️ Select District",
        ["All Districts"] + list(districts),
        help="Choose a specific district or view all"
    )
    
    # Year Range Selection
    year_range = st.slider(
        "📅 Select Year Range",
        int(df["year"].min()),
        int(df["year"].max()),
        (int(df["year"].min()), int(df["year"].max())),
        help="Filter data by year range"
    )
    
    # Crime Types Selection
    st.divider()
    st.subheader("🚨 Crime Types")
    all_crimes = st.checkbox("Select All", value=True, help="Toggle all crime types")
    
    if all_crimes:
        crime_types = list(crime_columns)
    else:
        crime_types = st.multiselect(
            "Choose crime types",
            crime_columns.tolist(),
            default=list(crime_columns[:3])
        )
    
    if not crime_types:
        display_warning_message("Please select at least one crime type")
        crime_types = list(crime_columns[:1])


# ==================================================
# FILTER DATA
# ==================================================
filtered_df = filter_data(df, state, district, year_range, crime_types)


# ==================================================
# HEADER & CONTEXT
# ==================================================
st.markdown("# 📊 IPC Crime Analysis Dashboard")

col1, col2 = st.columns([3, 1])
with col1:
    st.markdown(f"""
    **State:** `{state}` | **District:** `{district}` | **Years:** `{year_range[0]} – {year_range[1]}`
    """
    )
with col2:
    get_download_button(filtered_df, filename=f"crime_analysis_{state}")


# ==================================================
# KPIs (CONTEXT-AWARE)
# ==================================================
st.subheader("📈 Key Performance Indicators")

stats = get_crime_statistics(filtered_df)

col1, col2, col3, col4 = st.columns(4)

with col1:
    display_kpi_card("Total Crimes", stats['total_crimes'], "🔢")

with col2:
    display_kpi_card("Peak Year", stats['peak_year'], "📍")

with col3:
    display_kpi_card("Avg/Year", stats['avg_crimes_per_year'], "📊")

with col4:
    display_kpi_card("Std Dev", stats['std_deviation'], "📉")

st.divider()


# ==================================================
# MAIN CONTENT TABS
# ==================================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Overview",
    "📈 Distribution",
    "🔗 Relationships",
    "🗺️ Hotspots",
    "🤖 Prediction",
    "📋 Data"
])


# ==================================================
# TAB 1: OVERVIEW
# ==================================================
with tab1:
    st.subheader("Crime Overview & Trends")
    
    col1, col2 = st.columns(2)
    
    with col1:
        fig = plot_top_districts(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    with col2:
        fig = plot_crime_trend(filtered_df)
        st.plotly_chart(fig, use_container_width=True)
    
    # Crime by type
    st.subheader("🚨 Crimes by Type (Top 10)")
    crime_by_type = get_crime_by_type(filtered_df, crime_types)
    
    col1, col2 = st.columns([2, 1])
    with col1:
        crime_df = pd.DataFrame({
            'Crime Type': crime_by_type.index[:10],
            'Count': crime_by_type.values[:10]
        })
        
        fig_plot = px.bar(
            crime_df,
            x='Crime Type',
            y='Count',
            color='Count',
            color_continuous_scale='Blues'
        )
        st.plotly_chart(fig_plot, use_container_width=True)
    
    with col2:
        st.metric("Most Common Crime", crime_by_type.index[0])
        st.metric("Count", format_number(crime_by_type.values[0]))


# ==================================================
# TAB 2: DISTRIBUTION
# ==================================================
with tab2:
    st.subheader("Crime Distribution Analysis")
    
    fig = plot_distribution(filtered_df)
    st.pyplot(fig)
    
    st.divider()
    
    # Additional statistics
    col1, col2, col3 = st.columns(3)
    
    with col1:
        st.metric("Mean", format_number(filtered_df['crime_sum'].mean()))
    with col2:
        st.metric("Median", format_number(filtered_df['crime_sum'].median()))
    with col3:
        st.metric("Max", format_number(filtered_df['crime_sum'].max()))


# ==================================================
# TAB 3: RELATIONSHIPS
# ==================================================
with tab3:
    st.subheader("Crime Type Correlations")
    
    fig = plot_correlation_heatmap(df, crime_types)
    st.pyplot(fig)
    
    display_info_message(
        "Shows Spearman correlation between different crime types. "
        "Values closer to 1 indicate strong positive correlation."
    )
    
    st.divider()
    
    st.subheader("Crime Intensity Heatmap")
    fig = plot_heatmap_by_district(filtered_df, crime_types)
    st.pyplot(fig)


# ==================================================
# TAB 4: HOTSPOTS
# ==================================================
with tab4:
    st.subheader("Geographic Crime Hotspots")
    
    fig = plot_crime_hotspots(filtered_df)
    st.plotly_chart(fig, use_container_width=True)
    
    display_info_message(
        "Hotspots represent aggregated crime intensity at the state level. "
        "Red areas indicate higher crime concentration."
    )


# ==================================================
# TAB 5: PREDICTION
# ==================================================
with tab5:
    st.subheader("🤖 Crime Trend Prediction")
    
    prediction_result = predict_crime_trend(filtered_df, years_ahead=5)
    
    if prediction_result:
        col1, col2, col3 = st.columns(3)
        
        with col1:
            st.metric("Model R² Score", f"{prediction_result['r2_score']:.3f}")
        with col2:
            st.metric("RMSE", f"{prediction_result['rmse']:.0f}")
        with col3:
            st.metric("MAE", f"{prediction_result['mae']:.0f}")
        
        st.divider()
        
        # Combine historical and predicted data
        historical = prediction_result['historical_data'].copy()
        historical['type'] = 'Historical'
        
        future_df = pd.DataFrame({
            'year': prediction_result['future_years'],
            'crime_sum': prediction_result['future_predictions'],
            'type': 'Predicted'
        })
        
        combined = pd.concat([historical, future_df], ignore_index=True)
        
        fig = px.line(
            combined,
            x='year',
            y='crime_sum',
            color='type',
            markers=True,
            title='Historical vs Predicted Crime Trends',
            labels={'crime_sum': 'Total Crimes', 'year': 'Year'}
        )
        
        fig.update_traces(line=dict(width=3))
        st.plotly_chart(fig, use_container_width=True)
        
        # Display predictions table
        st.subheader("📊 Predicted Values (Next 5 Years)")
        pred_table = pd.DataFrame({
            'Year': prediction_result['future_years'],
            'Predicted Crimes': [int(x) for x in prediction_result['future_predictions']]
        })
        st.dataframe(pred_table, use_container_width=True)
    
    else:
        display_warning_message(
            "Insufficient data for prediction. Need at least 5 years of historical data."
        )


# ==================================================
# TAB 6: DATA
# ==================================================
with tab6:
    st.subheader("📋 Filtered Crime Data")
    
    # Display data statistics
    col1, col2, col3 = st.columns(3)
    with col1:
        st.metric("Total Records", len(filtered_df))
    with col2:
        st.metric("States", filtered_df['state_name'].nunique())
    with col3:
        st.metric("Districts", filtered_df['district_name'].nunique())
    
    st.divider()
    
    # Data table
    st.dataframe(filtered_df, use_container_width=True, height=400)
    
    # Download button
    st.divider()
    get_download_button(filtered_df, filename=f"crime_data_{state}_{district}")


# ==================================================
# FOOTER
# ==================================================
st.divider()
st.caption(
    "**CrimeScope** | Interactive IPC Crime Intelligence Dashboard v2.0 | "
    "Designed for policy analysis and academic evaluation"
)