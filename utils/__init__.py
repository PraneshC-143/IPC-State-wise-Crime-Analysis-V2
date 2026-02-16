"""
Utils Package
Utility modules for crime analysis dashboard
"""

# Import utility functions from parent utils.py module using proper relative imports
# to avoid circular import issues
import importlib.util
import sys
import os

def _import_root_utils():
    """Import functions from root utils.py without circular import"""
    spec = importlib.util.spec_from_file_location(
        "root_utils", 
        os.path.join(os.path.dirname(os.path.dirname(__file__)), "utils.py")
    )
    root_utils = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(root_utils)
    return root_utils

# Import functions from root utils.py
_root = _import_root_utils()
apply_custom_styling = _root.apply_custom_styling
format_number = _root.format_number
get_download_button = _root.get_download_button
display_kpi_card = _root.display_kpi_card
display_warning_message = _root.display_warning_message
display_info_message = _root.display_info_message
display_success_message = _root.display_success_message
display_error_message = _root.display_error_message

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
    # Original utils functions
    'apply_custom_styling',
    'format_number',
    'get_download_button',
    'display_kpi_card',
    'display_warning_message',
    'display_info_message',
    'display_success_message',
    'display_error_message',
    
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
