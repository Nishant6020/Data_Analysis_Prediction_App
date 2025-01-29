import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns
from function import *

def data_overview(df):
    # Data Overview Section (Initially hidden)
    overview_option = [
    "Data Shape",
    "Data Types",
    "Statistics Summary",
    "Check Missing Values",
    "Show Rows with Missing Values",
    "Check Duplicates",
    "Show Duplicate Rows",
    "Show Unique Values",
    "Show Value Counts",
    "Show Numerical or Categorical Columns",
    "Separate Numerical & Categorical Columns",
]
    overview_tabs = st.tabs(overview_option)
    # Handle Data Overview tasks
    with overview_tabs[0]:
        st.write("Shape of Data")
        st.write(get_shape(st.session_state.df))

        # st.write("Number of Rows: ",st.session_state.df.shape[0])
        # st.write("Number of columns: ",st.session_state.df.shape[1])


    with overview_tabs[1]: #"Data Types"
        st.write("### Data Types")
        st.write(st.session_state.df.dtypes)

    with overview_tabs[2]: #"Statistics Summary"
        statistics(st.session_state.df)


    with overview_tabs[3]: #"Show Rows with Missing Values":
        rows_with_missing_values = st.session_state.df[st.session_state.df.isnull().any(axis=1)]
        if len(rows_with_missing_values) > 0:
            st.write("### Rows with Missing Values")
            st.write(rows_with_missing_values)
        else:
            st.write("No rows with missing values found.")

    with overview_tabs[4]:# "Check Missing Values":
        st.write("### Missing Values Summary")
        st.write(check_missing_values(st.session_state.df))

    with overview_tabs[5]:# "Check Duplicates":
        st.write("### Duplicate Summary")
        st.write(check_duplicates(st.session_state.df))

    with overview_tabs[6]: #"Show Duplicate Rows":
        st.write("### Duplicate Rows")
        st.write(duplicate(st.session_state.df))

    with overview_tabs[7]: #"Show Unique Values":
        st.write("### Unique Values by Column")
        st.write(unique_columns(st.session_state.df))

    with overview_tabs[8]: #"Show Value Counts":
        column = st.selectbox("Select a column to show value counts:", st.session_state.df.columns)
        if st.button("Show Value Counts"):
            show_value_counts(st.session_state.df, column)

            
    with overview_tabs[9]: #"Show Numerical or Categorical Columns":
        column_type = st.radio("Select column type:", ['Numerical', 'Categorical'])
        st.write(f"### {column_type} Columns Data")
        st.write(show_column_data(st.session_state.df, column_type))

    with overview_tabs[10]: #"Separate Numerical & Categorical Columns":
        st.write("### Column Types")
        st.write(column_selection(st.session_state.df))
