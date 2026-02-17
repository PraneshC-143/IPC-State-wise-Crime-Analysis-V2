"""
Interactive crime hotspot map generator using Folium
"""

import folium
from folium import plugins
import pandas as pd
import streamlit as st
from streamlit_folium import st_folium
from utils.district_geocoding import get_district_coordinates

def create_crime_hotspot_map(data, crime_columns, selected_crimes=None, year_range=None):
    """
    Create an interactive GPS-style crime hotspot map
    
    Args:
        data: DataFrame with crime data
        crime_columns: List of crime column names
        selected_crimes: List of selected crime types (None = all)
        year_range: Tuple of (min_year, max_year) (None = all years)
    
    Returns:
        folium.Map object
    """
    
    # Filter data
    filtered_data = data.copy()
    
    if year_range:
        filtered_data = filtered_data[
            (filtered_data['year'] >= year_range[0]) & 
            (filtered_data['year'] <= year_range[1])
        ]
    
    # Calculate crime totals per district
    if selected_crimes:
        crime_cols = selected_crimes
    else:
        crime_cols = crime_columns
    
    district_summary = filtered_data.groupby(['state_name', 'district_name']).agg({
        **{col: 'sum' for col in crime_cols},
        'year': ['min', 'max']
    }).reset_index()
    
    # Flatten column names
    district_summary.columns = ['state_name', 'district_name'] + \
                                list(crime_cols) + ['year_min', 'year_max']
    
    # Calculate total crimes
    district_summary['total_crimes'] = district_summary[crime_cols].sum(axis=1)
    
    # Convert crime columns to numeric for later processing
    for col in crime_cols:
        district_summary[col] = pd.to_numeric(district_summary[col], errors='coerce').fillna(0)
    
    # Add geocoding
    district_summary = add_coordinates(district_summary)
    
    # Remove districts without coordinates
    district_summary = district_summary.dropna(subset=['latitude', 'longitude'])
    
    if district_summary.empty:
        st.warning("⚠️ No geographic data available for selected filters")
        return None
    
    # Create base map centered on India
    india_map = folium.Map(
        location=[20.5937, 78.9629],  # India center
        zoom_start=5,
        tiles=None  # We'll add custom tiles
    )
    
    # Add multiple tile layers
    folium.TileLayer('OpenStreetMap', name='Street Map').add_to(india_map)
    folium.TileLayer('CartoDB positron', name='Light Map').add_to(india_map)
    folium.TileLayer('CartoDB dark_matter', name='Dark Map').add_to(india_map)
    
    # Add heatmap layer
    heat_data = [
        [row['latitude'], row['longitude'], row['total_crimes']] 
        for _, row in district_summary.iterrows()
    ]
    
    heat_layer = plugins.HeatMap(
        heat_data,
        name='Crime Heatmap',
        radius=25,
        blur=35,
        max_zoom=13,
        gradient={
            0.0: 'green',
            0.3: 'yellow',
            0.6: 'orange',
            1.0: 'red'
        }
    )
    heat_layer.add_to(india_map)
    
    # Add marker cluster
    marker_cluster = plugins.MarkerCluster(name='District Markers').add_to(india_map)
    
    # Add markers for each district
    for _, row in district_summary.iterrows():
        # Determine marker color based on crime severity
        total = row['total_crimes']
        max_crimes = district_summary['total_crimes'].max()
        severity = total / max_crimes if max_crimes > 0 else 0
        
        if severity > 0.7:
            color = 'red'
            icon_name = 'exclamation-triangle'
        elif severity > 0.4:
            color = 'orange'
            icon_name = 'exclamation-circle'
        else:
            color = 'green'
            icon_name = 'info-sign'
        
        # Get top 3 crimes for this district
        top_crimes = row[crime_cols].nlargest(3)
        top_crimes_html = '<br>'.join([
            f"• {crime}: {int(count):,}" 
            for crime, count in top_crimes.items()
        ])
        
        # Create popup content
        popup_html = f"""
        <div style="font-family: Arial; min-width: 200px;">
            <h4 style="margin: 0 0 10px 0; color: #333;">
                📍 {row['district_name']}
            </h4>
            <p style="margin: 5px 0; color: #666;">
                <b>State:</b> {row['state_name']}<br>
                <b>Period:</b> {int(row['year_min'])}-{int(row['year_max'])}<br>
                <b>Total Crimes:</b> {int(row['total_crimes']):,}
            </p>
            <p style="margin: 10px 0 0 0;">
                <b>Top Crimes:</b><br>
                {top_crimes_html}
            </p>
        </div>
        """
        
        # Add marker
        folium.Marker(
            location=[row['latitude'], row['longitude']],
            popup=folium.Popup(popup_html, max_width=300),
            tooltip=f"{row['district_name']} - {int(row['total_crimes']):,} crimes",
            icon=folium.Icon(color=color, icon=icon_name, prefix='glyphicon')
        ).add_to(marker_cluster)
    
    # Add layer control
    folium.LayerControl(collapsed=False).add_to(india_map)
    
    # Add fullscreen button
    plugins.Fullscreen(
        position='topleft',
        title='Fullscreen',
        title_cancel='Exit Fullscreen'
    ).add_to(india_map)
    
    # Add mini map
    plugins.MiniMap(toggle_display=True).add_to(india_map)
    
    # Add measure control
    plugins.MeasureControl(position='bottomleft').add_to(india_map)
    
    return india_map


def add_coordinates(df):
    """Add latitude/longitude coordinates to district data"""
    coords = []
    
    for _, row in df.iterrows():
        coord = get_district_coordinates(
            row['state_name'], 
            row['district_name']
        )
        coords.append(coord)
    
    df['latitude'] = [c[0] if c else None for c in coords]
    df['longitude'] = [c[1] if c else None for c in coords]
    
    return df
