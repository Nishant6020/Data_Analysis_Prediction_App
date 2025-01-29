from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
import pandas as pd
import streamlit as st
from streamlit_option_menu import option_menu
from data_preview import data_preview
from data_overview import data_overview
from data_cleaning import data_cleaning
from data_visualization import data_visualization
from home import home
from contact import contact
from prediction import prediction
from function import load_data
from eda import eda
import os

st.set_page_config(layout="wide")

# Sidebar Menu
with st.sidebar:
    select = option_menu(
        menu_title="Menu",
        options=["Home", "Data Preview", "Data Overview", "Data Cleaning", "EDA Report", "Visualization", 
                 "Prediction Model", "Contact"],
        icons=["house", "search", "images", "arrow-repeat", "", "graph-up-arrow", "boxes", "person-lines-fill"],
        menu_icon="menu-button-wide-fill",
        default_index=0,
        orientation="vertical",
    )
    uploaded_file = st.file_uploader("Upload your file (CSV, Excel, or JSON)", type=["csv", "xlsx", "json"])
    st.write(clean_and_download_file(df))

st.title("Data Analysis & Prediction Building Application")

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        st.session_state['df'] = df
        
        if select == "Home":
            with st.expander("Home", expanded=True):
                home()

        elif select == "Data Preview":
            with st.expander("Data Preview", expanded=True):
                data_preview(df)

        elif select == "Data Overview":
            with st.expander("Data Overview", expanded=True):
                data_overview(df)

        elif select == "Data Cleaning":
            with st.expander("Data Cleaning", expanded=True):
                data_cleaning(df)

        elif select == "EDA Report":
            with st.expander("EDA Report", expanded=True):
                eda(df)
        
        elif select == "Visualization":
            with st.expander("Visualization", expanded=True):
                data_visualization(df)

        elif select == "Prediction Model":
            with st.expander("Prediction Model", expanded=True):
                st.write("### Prediction Model")
                st.write("This feature is under development.")

        elif select == "Contact":
            with st.expander("Contact", expanded=True):
                contact()
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
else:
    if 'df' in st.session_state:
        df = st.session_state['df']
        
        if select == "Home":
            with st.expander("Home", expanded=True):
                home()

        elif select == "Data Preview":
            with st.expander("Data Preview", expanded=True):
                data_preview(df)

        elif select == "Data Overview":
            with st.expander("Data Overview", expanded=True):
                data_overview(df)

        elif select == "Data Cleaning":
            with st.expander("Data Cleaning", expanded=True):
                data_cleaning(df)

        elif select == "EDA Report":
            with st.expander("EDA Report", expanded=True):
                eda(df)
            
        elif select == "Visualization":
            with st.expander("Visualization", expanded=True):
                data_visualization(df)

        elif select == "Prediction Model":
            with st.expander("Prediction Model", expanded=True):
                prediction()

        elif select == "Contact":
            with st.expander("Contact", expanded=True):
                contact()
    else:
        if select == "Home":
            with st.expander("Home", expanded=True):
                st.write("Welcome to the **Data Analysis & Prediction Building Application**! Use the expanders to navigate.")
                home()

        elif select == "Contact":
            with st.expander("Contact", expanded=True):
                contact()

        elif select == "Prediction Model":
            with st.expander("Prediction Model", expanded=True):
                prediction()

        st.warning("Please upload a file to proceed.")
