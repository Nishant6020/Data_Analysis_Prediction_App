# Function definitions (reuse from previous implementation)
def data_view(df):
    st.write("### Data Preview")
    st.write(df.head())

def sample_data(df):
    st.write("### Sample Data")
    st.write(df.sample(5))

def statistics(df):
    st.write("### Data Statistics")
    st.write(df.describe())

def load_data(file):
    if file.name.endswith('.csv'):
        return pd.read_csv(file)
    elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
        return pd.read_excel(file)
    elif file.name.endswith('.json'):
        return pd.read_json(file)
    else:
        st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
        return None

def check_missing_values(df):
    missing = df.isnull().sum()
    missing_percentage = (missing / len(df)) * 100
    return pd.DataFrame({'Missing Values': missing, 'Percentage': missing_percentage})

def remove_missing_values(df):
    df = df.dropna()
    return df, check_missing_values(df)

def check_duplicates(df):
    duplicates_count = df.duplicated().sum()
    return pd.DataFrame({'Has Duplicates': [duplicates_count > 0], 'Duplicate Count': [duplicates_count]})

def duplicate(df):
    return df[df.duplicated()]

def remove_duplicates(df):
    df = df.drop_duplicates()
    return df  # Return only the cleaned DataFrame


def unique_columns(df):
    unique_values = {col: len(df[col].unique()) for col in df.columns}
    return pd.DataFrame({'Column': list(unique_values.keys()), 'Unique Values': list(unique_values.values())})

def drop_columns(df, columns):
    df = df.drop(columns=columns, axis=1)
    return df.head()

def get_shape(df):
    shape = df.shape
    return pd.DataFrame({'Rows': [shape[0]], 'Columns': [shape[1]]})

def column_selection(df):
    numerical = df.select_dtypes(include=['number']).columns.tolist()
    categorical = df.select_dtypes(exclude=['number']).columns.tolist()
    
    bool_columns = df.select_dtypes(include=['bool']).columns.tolist()
    categorical.extend(bool_columns)
    
    date_columns = df.select_dtypes(include=['datetime']).columns.tolist()
    
    return pd.DataFrame({
        'Type': ['Numerical', 'Categorical', 'Date'],
        'Columns': [numerical, categorical, date_columns],
        'Total Columns': [len(numerical), len(categorical), len(date_columns)]
    })

def show_column_data(df, column_type):
    if column_type == 'Numerical':
        return df.select_dtypes(include=['number'])
    elif column_type == 'Categorical':
        return df.select_dtypes(exclude=['number'])
    else:
        return pd.DataFrame()

def clean_categorical_text(df, columns):
    def clean_text(text):
        text = text.lower()
        text = re.sub(r'[^a-zA-Z0-9\s]', '', text)
        text = re.sub(r'\s+', ' ', text).strip()
        return text

    for col in columns:
        df[col] = df[col].astype(str).apply(clean_text)
    return df

def encode_categorical(df, columns):
    encoder = LabelEncoder()
    for col in columns:
        df[col] = encoder.fit_transform(df[col])
    return unique_columns(df)

def download_clean_data(df, file_name):
    df.to_csv(file_name, index=False)
    return file_name

def fill_missing_values(df, column, method=None):
    df_cleaned = df.copy()
    
    column_type = df_cleaned[column].dtype
    
    if column_type in ['int64', 'float64']:
        if method == 'Mean':
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mean())
        elif method == 'Median':
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].median())
        elif method == 'Min':
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].min())
        elif method == 'Max':
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].max())
        elif method == 'Zero':
            df_cleaned[column] = df_cleaned[column].fillna(0)
        else:
            st.error("Invalid method for filling missing values.")
    elif column_type in ['object', 'category']:
        if method == 'Most Frequent':
            df_cleaned[column] = df_cleaned[column].fillna(df_cleaned[column].mode()[0])
        elif method == 'Other':
            df_cleaned[column] = df_cleaned[column].fillna("Other")
        else:
            st.error("Invalid method for categorical columns.")
    
    return df_cleaned   

def show_value_counts(df, column):
    st.write(f"### Value Counts for Column: {column}")
    st.write(df[column].value_counts())

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns


# Function definitions (reuse from previous implementation)
# ... (keep all your function definitions as they are)

# Streamlit App
st.title("Data Cleaning and Preprocessing App")

uploaded_file = st.file_uploader("Upload your data file (CSV, Excel, JSON)", type=['csv', 'xlsx', 'xls', 'json'])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # Store the DataFrame in session state
        if 'df' not in st.session_state:
            st.session_state.df = df

        # Sidebar for options
        st.sidebar.title("Menu")

        # Data Preivew button
        if st.sidebar.button("Data Preview"):
            comparison_option = [
                "Data Preview", "First 5 Rows","Last 5 Rows","10 Rows","20 Rows",
             "50 Rows","Sample Data",]
            
            Pview_tabs = st.tabs(comparison_option)
            # Handle Data Overview tasks
            with Pview_tabs[0]:  # Data Preview
                st.write(st.session_state.df)
            with Pview_tabs[1]:  # First 5 Rows
                st.write(st.session_state.df.head())
            with Pview_tabs[2]:  # Last 5 Rows
                st.write(st.session_state.df.tail())
            with Pview_tabs[3]:  # 10 Rows
                st.write(st.session_state.df.head(10))
            with Pview_tabs[4]:  # 20 Rows
                st.write(st.session_state.df.head(20))
            with Pview_tabs[5]:  # 50 Rows
                st.write(st.session_state.df.head(50))
            with Pview_tabs[6]:  # Sample Data
                st.write(st.session_state.df.sample(5)) 
                
        # Data Overview Section (Initially hidden)
        if st.sidebar.button("Data Overview"):
            overview_option = [
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
            with overview_tabs[0]: #"Statistics Summary"
                statistics(st.session_state.df)
            
            with overview_tabs[1]: #"Data Types"
                st.write("### Data Types")
                st.write(st.session_state.df.dtypes)

            with overview_tabs[2]: #"Show Rows with Missing Values":
                rows_with_missing_values = st.session_state.df[st.session_state.df.isnull().any(axis=1)]
                if len(rows_with_missing_values) > 0:
                    st.write("### Rows with Missing Values")
                    st.write(rows_with_missing_values)
                else:
                    st.write("No rows with missing values found.")

            with overview_tabs[3]:# "Check Missing Values":
                st.write("### Missing Values Summary")
                st.write(check_missing_values(st.session_state.df))

            with overview_tabs[4]:# "Check Duplicates":
                st.write("### Duplicate Summary")
                st.write(check_duplicates(st.session_state.df))

            with overview_tabs[5]: #"Show Duplicate Rows":
                st.write("### Duplicate Rows")
                st.write(duplicate(st.session_state.df))

            with overview_tabs[6]: #"Show Unique Values":
                st.write("### Unique Values by Column")
                st.write(unique_columns(st.session_state.df))

            with overview_tabs[7]: #"Show Value Counts":
                column = st.selectbox("Select a column to show value counts:", st.session_state.df.columns)
                if st.button("Show Value Counts"):
                    show_value_counts(st.session_state.df, column)

                    
            with overview_tabs[8]: #"Show Numerical or Categorical Columns":
                column_type = st.radio("Select column type:", ['Numerical', 'Categorical'])
                st.write(f"### {column_type} Columns Data")
                st.write(show_column_data(st.session_state.df, column_type))

            with overview_tabs[9]: #"Separate Numerical & Categorical Columns":
                st.write("### Column Types")
                st.write(column_selection(st.session_state.df))
      
        # Data Cleaning Section (Initially hidden)
        if st.sidebar.button("Data Cleaning"):
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
        
        




        # Visualize Data button
        if st.sidebar.button("Visualization"):
            
            # Create tabs for different plots
            tabs = ["Pair Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Count Plot",
                    "Violin Plot", "Line Chart", "Correlation Heatmap", "Distribution Plot", "Pie Chart"]
            tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(tabs)

            with tab1:
                st.header("Pair Plot")

            with tab2:
                st.header("Bar Plot")

            with tab3:
                st.header("Scatter Plot")

            with tab4:
                st.header("Histogram")

            with tab5:
                st.header("Box Plot")

            with tab6:
                st.header("Count Plot")

            with tab7:
                st.header("Violin Plot")

            with tab8:
                st.header("Line Chart")

            with tab9:
                st.header("Correlation Heatmap")

            with tab10:
                st.header("Distribution Plot")

            with tab11:
                st.header("Pie Chart")

