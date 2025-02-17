import streamlit as st
from data_preview import data_preview
from data_overview import data_overview
from data_cleaning import data_cleaning
from data_visualization import data_visualization
from home import home
from contact import contact
from prediction import prediction
from function import load_data
# from ydata_profiling import ProfileReport


# Sidebar
with st.sidebar:
    st.header("Menu Options")
    # Checkboxes for multiple selection
    show_home = st.checkbox("Home")
    show_data_preview = st.checkbox("Data Preview")
    show_data_overview = st.checkbox("Data Overview")
    show_data_cleaning = st.checkbox("Data Cleaning")
    show_visualization = st.checkbox("Visualization")
    show_prediction = st.checkbox("Prediction Model")
    show_contact = st.checkbox("Contact")

    # File uploader in the sidebar
    uploaded_file = st.file_uploader("Upload your file (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])

# Main content
st.title("Data Analysis & Prediction Building Application")

# If a file is uploaded
if uploaded_file:
    # Load the data
    df = load_data(uploaded_file)
    if df is not None:
        # Store the DataFrame in session_state
        st.session_state['df'] = df

        # Display content dynamically based on selected checkboxes
        if show_home:
            with st.expander("Home", expanded=True):
                home()

        if show_data_preview:
            with st.expander("Data Preview", expanded=True):
                data_preview(df)

        if show_data_overview:
            with st.expander("Data Overview", expanded=True):
                data_overview(df)

        if show_data_cleaning:
            with st.expander("Data Cleaning", expanded=True):
                data_cleaning(df)

        if show_visualization:
            with st.expander("Visualization", expanded=True):
                data_visualization(df)

        if show_prediction:
            with st.expander("Prediction Model", expanded=True):
                st.write("### Prediction Model")
                st.write("This feature is under development.")

        if show_contact:
            with st.expander("Contact", expanded=True):
                contact()
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
else:
    # If no file is uploaded, check if df exists in session_state
    if 'df' in st.session_state:
        df = st.session_state['df']

        # Display content dynamically based on selected checkboxes
        if show_home:
            with st.expander("Home", expanded=True):
                home()

        if show_data_preview:
            with st.expander("Data Preview", expanded=True):
                data_preview(df)

        if show_data_overview:
            with st.expander("Data Overview", expanded=True):
                data_overview(df)

        if show_data_cleaning:
            with st.expander("Data Cleaning", expanded=True):
                data_cleaning(df)

        if show_visualization:
            with st.expander("Visualization", expanded=True):
                data_visualization(df)

        if show_prediction:
            with st.expander("Prediction Model", expanded=True):
                prediction()

        if show_contact:
            with st.expander("Contact", expanded=True):
                contact()
    else:
        # Display content dynamically based on selected checkboxes
        if show_home:
            with st.expander("Home", expanded=True):
                st.write("Welcome to the **Data Analysis & Prediction Building Application**! Use the checkboxes to navigate.")
                home()

        if show_contact:
            with st.expander("Contact", expanded=True):
                contact()

        if show_prediction:
            with st.expander("Prediction Model", expanded=True):
                prediction()

        st.warning("Please upload a file to proceed.")
