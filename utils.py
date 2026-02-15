import streamlit as st
import pandas as pd


def apply_custom_styling():
    """Apply custom CSS styling to the Streamlit app"""
    st.markdown("""
        <style>
        .main {
            background-color: #fafafa;
        }
        .stMetric {
            background-color: #ffffff;
            padding: 15px;
            border-radius: 10px;
            box-shadow: 0 2px 4px rgba(0,0,0,0.1);
        }
        .stTabs [data-baseweb="tab-list"] {
            gap: 10px;
        }
        .stTabs [data-baseweb="tab"] {
            padding: 10px 20px;
            background-color: #f0f2f6;
            border-radius: 5px;
        }
        .stTabs [aria-selected="true"] {
            background-color: #1f77b4;
            color: white;
        }
        </style>
    """, unsafe_allow_html=True)


def format_number(value):
    """Format a number with thousands separator"""
    if isinstance(value, (int, float)):
        return f'{value:,}'
    return value


def get_download_button(data, filename):
    """Create a download button for given data"""
    csv = data.to_csv(index=False)
    st.download_button(
        label='📥 Download CSV',
        data=csv,
        file_name=f"{filename}.csv",
        mime='text/csv'
    )


def display_kpi_card(title, value, icon=""):
    """Display a Key Performance Indicator (KPI) card with optional icon"""
    formatted_value = format_number(value) if isinstance(value, (int, float)) else value
    st.metric(label=f"{icon} {title}" if icon else title, value=formatted_value)


def display_warning_message(message):
    """Display a warning message"""
    st.warning(f'⚠️ {message}')


def display_info_message(message):
    """Display an info message"""
    st.info(f'ℹ️ {message}')


def display_success_message(message):
    """Display a success message"""
    st.success(f'✅ {message}')


def display_error_message(message):
    """Display an error message"""
    st.error(f'❌ {message}')
