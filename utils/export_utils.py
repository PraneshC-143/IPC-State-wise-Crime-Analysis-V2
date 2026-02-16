"""
Export Utilities Module
Handles data and visualization exports
"""

import streamlit as st
import pandas as pd
from io import BytesIO
import plotly.graph_objects as go
import sys
import os

# Add parent directory to path to import utils
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
from utils import format_crime_name


def export_to_csv(df, filename="crime_data.csv"):
    """
    Export dataframe to CSV with download button
    
    Args:
        df: DataFrame to export
        filename: Name of the file
    """
    csv = df.to_csv(index=False)
    st.download_button(
        label="📥 Download as CSV",
        data=csv,
        file_name=filename,
        mime="text/csv",
        help="Download the filtered data as CSV file"
    )


def export_chart_as_png(fig, filename="chart.png"):
    """
    Export Plotly chart as PNG with download button
    
    Args:
        fig: Plotly figure object
        filename: Name of the file
    """
    try:
        # Convert figure to bytes
        img_bytes = fig.to_image(format="png", width=1200, height=800)
        
        st.download_button(
            label="📥 Download Chart as PNG",
            data=img_bytes,
            file_name=filename,
            mime="image/png",
            help="Download the chart as PNG image"
        )
    except Exception as e:
        st.warning(f"Chart export not available: {str(e)}")


def create_summary_report(df, crime_columns, state=None, district=None):
    """
    Create a summary report dataframe
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime columns
        state: Selected state (optional)
        district: Selected district (optional)
    
    Returns:
        DataFrame with summary statistics
    """
    summary_data = {
        'Metric': [],
        'Value': []
    }
    
    # Basic stats
    summary_data['Metric'].append('Total Records')
    summary_data['Value'].append(len(df))
    
    summary_data['Metric'].append('Year Range')
    summary_data['Value'].append(f"{df['year'].min()} - {df['year'].max()}")
    
    summary_data['Metric'].append('Total Crimes')
    summary_data['Value'].append(f"{int(df[crime_columns].sum().sum()):,}")
    
    if state:
        summary_data['Metric'].append('State')
        summary_data['Value'].append(state)
    
    if district and district != "All Districts":
        summary_data['Metric'].append('District')
        summary_data['Value'].append(district)
    
    summary_data['Metric'].append('Number of Districts')
    summary_data['Value'].append(df['district_name'].nunique())
    
    # Crime stats
    top_crime = df[crime_columns].sum().idxmax()
    summary_data['Metric'].append('Highest Crime Type')
    summary_data['Value'].append(top_crime)
    
    summary_data['Metric'].append('Highest Crime Count')
    summary_data['Value'].append(f"{int(df[crime_columns].sum().max()):,}")
    
    return pd.DataFrame(summary_data)


def prepare_export_data(df, crime_columns):
    """
    Prepare dataframe for export with calculated columns
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime columns
    
    Returns:
        DataFrame ready for export
    """
    export_df = df.copy()
    
    # Add total crimes if not present
    if 'total_crimes' not in export_df.columns:
        export_df['total_crimes'] = export_df[crime_columns].sum(axis=1)
    
    # Add rank by district
    district_crimes = export_df.groupby('district_name')['total_crimes'].sum()
    district_rank = district_crimes.rank(ascending=False, method='dense').to_dict()
    export_df['district_rank'] = export_df['district_name'].map(district_rank)
    
    # Reorder columns
    cols = ['state_name', 'district_name', 'year', 'total_crimes', 'district_rank'] + list(crime_columns)
    export_df = export_df[[col for col in cols if col in export_df.columns]]
    
    return export_df


def create_yearly_summary(df, crime_columns):
    """
    Create yearly summary statistics
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime columns
    
    Returns:
        DataFrame with yearly statistics
    """
    yearly_stats = df.groupby('year')[crime_columns].sum()
    yearly_stats['Total Crimes'] = yearly_stats.sum(axis=1)
    
    # Add year-over-year change
    yearly_stats['YoY Change'] = yearly_stats['Total Crimes'].pct_change() * 100
    
    # Add absolute change
    yearly_stats['Absolute Change'] = yearly_stats['Total Crimes'].diff()
    
    return yearly_stats


def create_state_summary(df, crime_columns):
    """
    Create state-wise summary statistics
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime columns
    
    Returns:
        DataFrame with state statistics
    """
    state_stats = df.groupby('state_name')[crime_columns].sum()
    state_stats['Total Crimes'] = state_stats.sum(axis=1)
    state_stats['Number of Districts'] = df.groupby('state_name')['district_name'].nunique()
    state_stats['Avg Crimes per District'] = (state_stats['Total Crimes'] / state_stats['Number of Districts']).round(0)
    
    # Sort by total crimes
    state_stats = state_stats.sort_values('Total Crimes', ascending=False)
    
    return state_stats


def create_crime_type_summary(df, crime_columns):
    """
    Create crime type summary statistics
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime columns
    
    Returns:
        DataFrame with crime type statistics
    """
    crime_totals = df[crime_columns].sum().sort_values(ascending=False)
    
    summary = pd.DataFrame({
        'Crime Type': [format_crime_name(name) for name in crime_totals.index],
        'Total Cases': crime_totals.values,
        'Percentage': (crime_totals.values / crime_totals.sum() * 100).round(2)
    })
    
    summary['Rank'] = range(1, len(summary) + 1)
    
    return summary[['Rank', 'Crime Type', 'Total Cases', 'Percentage']]
