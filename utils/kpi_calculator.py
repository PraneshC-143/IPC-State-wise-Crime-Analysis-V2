"""
KPI Calculator Module
Calculates key performance indicators for crime analytics
"""

import pandas as pd
import numpy as np


def calculate_total_crimes(df, crime_columns):
    """Calculate total crimes with year-over-year change"""
    current_year = int(df['year'].max())
    previous_year = current_year - 1
    
    current_crimes = df[df['year'] == current_year][crime_columns].sum().sum()
    previous_crimes = df[df['year'] == previous_year][crime_columns].sum().sum()
    
    change = current_crimes - previous_crimes
    pct_change = (change / previous_crimes * 100) if previous_crimes > 0 else 0
    
    return {
        'value': int(current_crimes),
        'change': int(change),
        'pct_change': float(pct_change),
        'direction': 'up' if change > 0 else 'down'
    }


def calculate_crime_rate(df, crime_columns, population=100000):
    """Calculate crime rate per 100K population"""
    # Note: This is a simplified calculation
    # In real scenario, you'd need actual population data
    total_crimes = df[crime_columns].sum().sum()
    districts = df['district_name'].nunique()
    
    # Estimate: assuming average district population
    estimated_total_pop = districts * population
    crime_rate = (total_crimes / estimated_total_pop) * 100000
    
    return {
        'value': round(crime_rate, 2),
        'per': 100000
    }


def get_most_affected_state(df, crime_columns):
    """Get the state with highest crime count"""
    state_crimes = df.groupby('state_name')[crime_columns].sum().sum(axis=1)
    most_affected = state_crimes.idxmax()
    max_crimes = int(state_crimes.max())
    
    return {
        'state': most_affected,
        'crimes': max_crimes
    }


def get_highest_crime_category(df, crime_columns):
    """Get the crime category with highest count"""
    crime_totals = df[crime_columns].sum()
    highest_crime = crime_totals.idxmax()
    highest_count = int(crime_totals.max())
    
    return {
        'category': highest_crime,
        'count': highest_count
    }


def calculate_yoy_growth(df, crime_columns):
    """Calculate year-over-year growth rate"""
    yearly_crimes = df.groupby('year')[crime_columns].sum().sum(axis=1)
    
    if len(yearly_crimes) < 2:
        return {'rate': 0, 'trend': 'stable'}
    
    # Calculate average YoY growth
    growth_rates = []
    for i in range(1, len(yearly_crimes)):
        prev_val = yearly_crimes.iloc[i-1]
        curr_val = yearly_crimes.iloc[i]
        if prev_val > 0:
            growth_rate = ((curr_val - prev_val) / prev_val) * 100
            growth_rates.append(growth_rate)
    
    avg_growth = np.mean(growth_rates) if growth_rates else 0
    trend = 'increasing' if avg_growth > 1 else 'decreasing' if avg_growth < -1 else 'stable'
    
    return {
        'rate': round(avg_growth, 2),
        'trend': trend
    }


def get_top_states_ranking(df, crime_columns, top_n=5):
    """Get ranking of top N states by crime count"""
    state_crimes = df.groupby('state_name')[crime_columns].sum().sum(axis=1)
    top_states = state_crimes.nlargest(top_n).reset_index()
    top_states.columns = ['State', 'Total Crimes']
    top_states['Rank'] = range(1, len(top_states) + 1)
    top_states['Total Crimes'] = top_states['Total Crimes'].astype(int)
    
    return top_states[['Rank', 'State', 'Total Crimes']]


def get_trend_indicator(current, previous):
    """Get trend indicator with color coding"""
    if previous == 0:
        return '🟡', 'New', 'yellow'
    
    change = ((current - previous) / previous) * 100
    
    if abs(change) < 1:
        return '🟡', 'Stable', 'yellow'
    elif change > 0:
        return '🔴', f'+{change:.1f}%', 'red'
    else:
        return '🟢', f'{change:.1f}%', 'green'


def calculate_district_kpis(df, crime_columns, state=None):
    """Calculate district-level KPIs"""
    if state:
        df = df[df['state_name'] == state]
    
    district_crimes = df.groupby('district_name')[crime_columns].sum().sum(axis=1)
    
    return {
        'total_districts': len(district_crimes),
        'highest_district': district_crimes.idxmax(),
        'highest_crimes': int(district_crimes.max()),
        'lowest_district': district_crimes.idxmin(),
        'lowest_crimes': int(district_crimes.min()),
        'avg_crimes': int(district_crimes.mean()),
        'median_crimes': int(district_crimes.median())
    }


def calculate_crime_concentration(df, crime_columns):
    """Calculate crime concentration (top 10% districts)"""
    district_crimes = df.groupby('district_name')[crime_columns].sum().sum(axis=1).sort_values(ascending=False)
    
    total_crimes = district_crimes.sum()
    top_10_pct = int(len(district_crimes) * 0.1)
    top_crimes = district_crimes.head(top_10_pct).sum()
    
    concentration = (top_crimes / total_crimes * 100) if total_crimes > 0 else 0
    
    return {
        'concentration_pct': round(concentration, 1),
        'top_districts': top_10_pct,
        'total_districts': len(district_crimes)
    }
