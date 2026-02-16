"""
Map Generator Module
Creates geographic visualizations for crime analysis
"""

import plotly.express as px
import plotly.graph_objects as go
import pandas as pd


# Indian state codes mapping (ISO 3166-2:IN)
STATE_CODES = {
    'Andaman & Nicobar Islands': 'IN-AN',
    'Andhra Pradesh': 'IN-AP',
    'Arunachal Pradesh': 'IN-AR',
    'Assam': 'IN-AS',
    'Bihar': 'IN-BR',
    'Chandigarh': 'IN-CH',
    'Chhattisgarh': 'IN-CT',
    'Dadra & Nagar Haveli': 'IN-DN',
    'Daman & Diu': 'IN-DD',
    'Delhi': 'IN-DL',
    'Goa': 'IN-GA',
    'Gujarat': 'IN-GJ',
    'Haryana': 'IN-HR',
    'Himachal Pradesh': 'IN-HP',
    'Jammu & Kashmir': 'IN-JK',
    'Jharkhand': 'IN-JH',
    'Karnataka': 'IN-KA',
    'Kerala': 'IN-KL',
    'Lakshadweep': 'IN-LD',
    'Madhya Pradesh': 'IN-MP',
    'Maharashtra': 'IN-MH',
    'Manipur': 'IN-MN',
    'Meghalaya': 'IN-ML',
    'Mizoram': 'IN-MZ',
    'Nagaland': 'IN-NL',
    'Odisha': 'IN-OR',
    'Puducherry': 'IN-PY',
    'Punjab': 'IN-PB',
    'Rajasthan': 'IN-RJ',
    'Sikkim': 'IN-SK',
    'Tamil Nadu': 'IN-TN',
    'Telangana': 'IN-TG',
    'Tripura': 'IN-TR',
    'Uttar Pradesh': 'IN-UP',
    'Uttarakhand': 'IN-UT',
    'West Bengal': 'IN-WB'
}


def create_choropleth_map(df, crime_columns, title="Crime Distribution by State"):
    """
    Create a choropleth map showing crime distribution across Indian states
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime column names
        title: Title for the map
    
    Returns:
        Plotly figure object
    """
    # Aggregate crimes by state
    state_crimes = df.groupby('state_name')[crime_columns].sum().sum(axis=1).reset_index()
    state_crimes.columns = ['state_name', 'total_crimes']
    
    # Add state codes
    state_crimes['state_code'] = state_crimes['state_name'].map(STATE_CODES)
    
    # Filter out states without codes
    state_crimes = state_crimes.dropna(subset=['state_code'])
    
    # Create choropleth map
    fig = go.Figure(data=go.Choropleth(
        locations=state_crimes['state_code'],
        z=state_crimes['total_crimes'],
        text=state_crimes['state_name'],
        colorscale='Reds',
        autocolorscale=False,
        reversescale=False,
        marker_line_color='darkgray',
        marker_line_width=0.5,
        colorbar_title="Total Crimes",
    ))
    
    fig.update_geos(
        visible=False,
        showcountries=True,
        countrycolor="LightGray",
        fitbounds="locations"
    )
    
    fig.update_layout(
        title_text=title,
        geo=dict(
            scope='asia',
            center=dict(lat=20.5937, lon=78.9629),
            projection_scale=4
        ),
        height=600
    )
    
    return fig


def create_bubble_map(df, crime_columns, title="Crime Hotspots - Bubble Map"):
    """
    Create a bubble map showing district-level crime distribution
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime column names
        title: Title for the map
    
    Returns:
        Plotly figure object
    """
    # Aggregate by district
    district_crimes = df.groupby(['state_name', 'district_name'])[crime_columns].sum().sum(axis=1).reset_index()
    district_crimes.columns = ['state_name', 'district_name', 'total_crimes']
    
    # Get top districts for visualization
    top_districts = district_crimes.nlargest(50, 'total_crimes')
    
    # Create scatter plot (simplified bubble map without coordinates)
    fig = px.scatter(
        top_districts,
        x='state_name',
        y='total_crimes',
        size='total_crimes',
        color='total_crimes',
        hover_data=['district_name'],
        title=title,
        labels={'state_name': 'State', 'total_crimes': 'Total Crimes'},
        color_continuous_scale='Reds',
        size_max=50
    )
    
    fig.update_layout(
        height=600,
        xaxis_tickangle=-45
    )
    
    return fig


def create_state_heatmap(df, crime_columns, top_n=20):
    """
    Create a heatmap showing crime distribution across states and crime types
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime column names
        top_n: Number of top states to display
    
    Returns:
        Plotly figure object
    """
    # Get top N states by total crimes
    state_crimes = df.groupby('state_name')[crime_columns].sum()
    state_total = state_crimes.sum(axis=1).nlargest(top_n)
    top_states = state_crimes.loc[state_total.index]
    
    # Select top crime types for better visualization
    top_crimes = top_states.sum().nlargest(15)
    heatmap_data = top_states[top_crimes.index]
    
    fig = px.imshow(
        heatmap_data,
        labels=dict(x="Crime Type", y="State", color="Count"),
        title=f'Crime Heatmap: Top {top_n} States vs Top 15 Crime Types',
        aspect='auto',
        color_continuous_scale='YlOrRd'
    )
    
    fig.update_layout(height=700)
    fig.update_xaxes(tickangle=45)
    
    return fig


def create_treemap(df, crime_columns, title="Crime Distribution - Treemap"):
    """
    Create a treemap visualization of crime distribution
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime column names
        title: Title for the treemap
    
    Returns:
        Plotly figure object
    """
    # Aggregate by state and district
    tree_data = df.groupby(['state_name', 'district_name'])[crime_columns].sum().sum(axis=1).reset_index()
    tree_data.columns = ['State', 'District', 'Total_Crimes']
    
    # Filter to top entries for better visualization
    top_data = tree_data.nlargest(100, 'Total_Crimes')
    
    fig = px.treemap(
        top_data,
        path=['State', 'District'],
        values='Total_Crimes',
        title=title,
        color='Total_Crimes',
        color_continuous_scale='Reds',
        hover_data={'Total_Crimes': ':,.0f'}
    )
    
    fig.update_layout(height=600)
    
    return fig


def create_sunburst(df, crime_columns, title="Crime Distribution - Sunburst"):
    """
    Create a sunburst chart for hierarchical crime distribution
    
    Args:
        df: DataFrame with crime data
        crime_columns: List of crime column names
        title: Title for the chart
    
    Returns:
        Plotly figure object
    """
    # Aggregate by state and district
    sun_data = df.groupby(['state_name', 'district_name'])[crime_columns].sum().sum(axis=1).reset_index()
    sun_data.columns = ['State', 'District', 'Total_Crimes']
    
    # Filter to top entries for better visualization
    top_data = sun_data.nlargest(75, 'Total_Crimes')
    
    fig = px.sunburst(
        top_data,
        path=['State', 'District'],
        values='Total_Crimes',
        title=title,
        color='Total_Crimes',
        color_continuous_scale='Reds',
        hover_data={'Total_Crimes': ':,.0f'}
    )
    
    fig.update_layout(height=600)
    
    return fig
