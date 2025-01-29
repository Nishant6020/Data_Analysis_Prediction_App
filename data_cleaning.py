import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns
from function import *


def data_cleaning(df):
    # Data Cleaning Section (Initially hidden)
    cleaning_option = [
    "Change Data Type",
    "Fill Missing Values",
    "Remove Missing Values",
    "Remove Duplicates",
    "Drop Columns",
    "Replace Values",
    "Clean Categorical Text",
    "Encode Categorical Columns",
    "Download Clean Data",
]
    clean_tabs= st.tabs(cleaning_option)

    with clean_tabs[0]: #"Change Data Type":
        column = st.selectbox("Select a column to change data type:", st.session_state.df.columns)
        new_type = st.selectbox("Select the new data type:", ['int64', 'float64', 'object', 'datetime64[ns]', 'category'])
        
        if st.button("Change Data Type"):
            try:
                # Handle conversion based on the new type
                if new_type == 'datetime64[ns]':
                    # Convert to datetime
                    st.session_state.df[column] = pd.to_datetime(st.session_state.df[column], errors='coerce')
                elif new_type in ['float64', 'int64']:
                    # Attempt to convert to numeric, forcing errors to NaN
                    st.session_state.df[column] = pd.to_numeric(st.session_state.df[column].str.strip(), errors='coerce')
                else:
                    # For object and category types, just change the type
                    st.session_state.df[column] = st.session_state.df[column].astype(new_type)

                st.success(f"Data type for column '{column}' changed to {new_type}!")
                st.write(st.session_state.df.dtypes)
            except Exception as e:
                st.error(f"Error changing data type: {e}")

    with clean_tabs[1]: #"Fill Missing Values":
        column = st.selectbox("Select the column with missing values", st.session_state.df.columns)
        
        # Check if the selected column is numerical
        if st.session_state.df[column].dtype in ['int64', 'float64']:
            # Show statistics of the selected column (mean, median, min, max)
            mean_value = st.session_state.df[column].mean()
            median_value = st.session_state.df[column].median()
            min_value = st.session_state.df[column].min()
            max_value = st.session_state.df[column].max()
            
            st.write(f"Mean: {mean_value}")
            st.write(f"Median: {median_value}")
            st.write(f"Min: {min_value}")
            st.write(f"Max: {max_value}")
            
            method = st.selectbox(
                "Select the method to fill missing values",
                ['Mean', 'Median', 'Min', 'Max', 'Zero', 'Custom Value']
            )
            
            if method == 'Custom Value':
                custom_value = st.number_input(f"Enter a custom value to fill missing values in {column}", value=0)
                method = 'Custom'
        
        # Check if the selected column is categorical
        elif st.session_state.df[column].dtype in ['object', 'category']:
            # Show the most frequent category (mode) and its count
            mode_value = st.session_state.df[column].mode()[0]
            mode_count = st.session_state.df[column].value_counts()[mode_value]
            
            st.write(f"Most frequent in {column}: {mode_value} (Count: {mode_count})")
            
            method = st.selectbox(
                "Select the method to fill missing values",
                ['Most Frequent', 'Other', 'Custom Value']
            )
            
            if method == 'Custom Value':
                custom_value = st.text_input(f"Enter a custom value to fill missing values in {column}", value="Unknown")
                method = 'Custom'
        else:
            st.write("Selected column is neither numerical nor categorical.")
            method = None  # No filling method is available
        
        # Apply the filling method
        if method is not None and st.button("Fill Missing Values"):
            if method == 'Mean':
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mean())
            elif method == 'Median':
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].median())
            elif method == 'Min':
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].min())
            elif method == 'Max':
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].max())
            elif method == 'Zero':
                st.session_state.df[column] = st.session_state.df[column].fillna(0)
            elif method == 'Most Frequent':
                st.session_state.df[column] = st.session_state.df[column].fillna(st.session_state.df[column].mode()[0])
            elif method == 'Other':
                st.session_state.df[column] = st.session_state.df[column].fillna("Other")
            elif method == 'Custom':
                st.session_state.df[column] = st.session_state.df[column].fillna(custom_value)
            
            st.write("Missing values filled successfully!")
            st.write(st.session_state.df[column].head())  # Show a preview of the filled column
            
            # Refresh the page to reflect the changes
            st.rerun()

    with clean_tabs[2]: #"Remove Missing Values":
        if st.button("Remove Missing Values"):
            st.session_state.df, missing_values = remove_missing_values(st.session_state.df)
            st.success("Rows with missing values removed!")
            st.write(missing_values)
            st.rerun()  # Refresh the page to reflect changes

    with clean_tabs[3]: #"Remove Duplicates":
        if st.button("Remove Duplicates"):
            st.session_state.df = remove_duplicates(st.session_state.df)  # Reassign the cleaned DataFrame back to `df`
            st.success("Duplicate rows removed!")
            st.write(check_duplicates(st.session_state.df))  # Display updated DataFrame
            st.rerun()  # Refresh the page to reflect changes

    with clean_tabs[4]: #"Drop Columns":
        columns = st.multiselect("Select columns to drop:", st.session_state.df.columns)
        if st.button("Drop Columns"):
            st.session_state.df = drop_columns(st.session_state.df, columns)
            st.success("Selected columns dropped!")
            st.write(st.session_state.df.head())

    with clean_tabs[5]: #"Replace Values":
        column = st.selectbox("Select a column to replace values:", st.session_state.df.columns)
        
        # Determine the data type of the selected column
        column_dtype = st.session_state.df[column].dtype
        
        if column_dtype == 'object':
            old_value = st.text_input("Enter the value to replace:", "old_value")
            new_value = st.text_input("Enter the new value:", "new_value")
        elif column_dtype in ['int64', 'float64']:
            old_value = st.number_input("Enter the value to replace:", value=0)
            new_value = st.number_input("Enter the new value:", value=0)
        else:
            st.warning("Unsupported data type for replacement.")
            old_value = None
            new_value = None

        if st.button("Replace Values"):
            # Trim whitespace from the column values if dtype is object
            if column_dtype == 'object':
                st.session_state.df[column] = st.session_state.df[column].str.strip()
            
            # Check if the old_value exists in the column
            if old_value in st.session_state.df[column].values:
                st.session_state.df[column] = st.session_state.df[column].replace(old_value, new_value)
                st.success("Values replaced successfully!")
                st.write(st.session_state.df[column].head())
            else:
                st.warning(f"The value '{old_value}' does not exist in the selected column.")


    with clean_tabs[6]: #"Clean Categorical Text":
        categorical_columns = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
        columns = st.multiselect("Select categorical columns to clean:", categorical_columns)
        if st.button("Clean Text"):
            st.session_state.df = clean_categorical_text(st.session_state.df, columns)
            st.success("Text cleaned for selected columns!")
            st.write(st.session_state.df.head())

    with clean_tabs[7]: #"Encode Categorical Columns":
        categorical_columns = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
        columns = st.multiselect("Select categorical columns to encode:", categorical_columns)
        if st.button("Encode Columns"):
            st.session_state.df = encode_categorical(st.session_state.df, columns)
            st.success("Selected columns encoded!")
            st.write(unique_columns(st.session_state.df))

    with clean_tabs[8]: # "Download Clean Data":
        file_name = st.text_input("Enter file name to save:", "cleaned_data.csv")
        if st.button("Download Data"):
            download_clean_data(st.session_state.df, file_name)
            st.success(f"File saved as {file_name}!")
