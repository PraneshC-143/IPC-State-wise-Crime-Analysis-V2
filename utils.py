def apply_custom_styling():
    # Apply custom CSS styling for Streamlit application
    st.markdown('<style>...</style>', unsafe_allow_html=True)


def format_number(value):
    # Format a number with thousands separator
    return f'{value:,}'


def get_download_button(data, filename):
    # Create a download button for given data
    return st.download_button('Download', data=data, file_name=filename)


def display_kpi_card(title, value):
    # Display a Key Performance Indicator (KPI) card
    col1, col2 = st.columns([2, 1])
    with col1:
        st.subheader(title)
    with col2:
        st.write(value)


def display_warning_message(message):
    # Display a warning message
    st.warning(message)


def display_info_message(message):
    # Display an info message
    st.info(message)


def display_success_message(message):
    # Display a success message
    st.success(message)


def display_error_message(message):
    # Display an error message
    st.error(message)