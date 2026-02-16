"""
Utils Package
Utility modules for crime analysis dashboard
"""

from .kpi_calculator import (
    calculate_total_crimes,
    calculate_crime_rate,
    get_most_affected_state,
    get_highest_crime_category,
    calculate_yoy_growth,
    get_top_states_ranking,
    get_trend_indicator,
    calculate_district_kpis,
    calculate_crime_concentration
)

from .map_generator import (
    create_choropleth_map,
    create_bubble_map,
    create_state_heatmap,
    create_treemap,
    create_sunburst
)

from .export_utils import (
    export_to_csv,
    export_chart_as_png,
    create_summary_report,
    prepare_export_data,
    create_yearly_summary,
    create_state_summary,
    create_crime_type_summary
)

__all__ = [
    # KPI functions
    'calculate_total_crimes',
    'calculate_crime_rate',
    'get_most_affected_state',
    'get_highest_crime_category',
    'calculate_yoy_growth',
    'get_top_states_ranking',
    'get_trend_indicator',
    'calculate_district_kpis',
    'calculate_crime_concentration',
    
    # Map functions
    'create_choropleth_map',
    'create_bubble_map',
    'create_state_heatmap',
    'create_treemap',
    'create_sunburst',
    
    # Export functions
    'export_to_csv',
    'export_chart_as_png',
    'create_summary_report',
    'prepare_export_data',
    'create_yearly_summary',
    'create_state_summary',
    'create_crime_type_summary'
]
