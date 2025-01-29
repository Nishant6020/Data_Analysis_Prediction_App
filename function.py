import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns
from function import *
from ydata_profiling import ProfileReport


# Function definitions (reuse from previous implementation)

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

def data_view(df):
    st.write("### Data Preview")
    st.write(df)

def sample_data(df):
    st.write("### Sample Data")
    st.write(df.sample(5))

def statistics(df):
    st.write("### Data Statistics")
    st.write(df.describe())

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

def corelation(df):
    return df.select_dtypes(include=['number']).corr()

def skewness_kurtosis(df):
    return df.select_dtypes(include=['number']).agg(['skew','kurtosis']).T


# Data Visualization

# Function to create Pair Plots
def create_pairplot(df):
    st.markdown(f'<h2 class="sub-title">PairPlots</h2>', unsafe_allow_html=True)
    fig = sns.pairplot(df)
    st.pyplot(fig)

# Function to create Boxplot
def create_boxplot(df):
    st.markdown(f'<h2 class="sub-title">Boxplot</h2>', unsafe_allow_html=True)
    st.write("X-axis only contains numeric columns, same as y.")
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.boxplot(
        data=df,
        x=st.selectbox("x-axis", df.columns, key="box_x-axis"),
        y=st.selectbox("y-axis", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])], key="box_y-axis"),
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# Function to create Histogram
def create_histogram(df):
    st.markdown(f'<h2 class="sub-title">Histogram</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.histplot(
        data=df,
        x=st.selectbox("x-axis:", df.columns, key="hist_x-axis"),
        bins=st.slider("Choose bins:", 1, 1000),
        kde=True,
        ax=ax
    )
    st.pyplot(fig)

# Function to create Scatter Plot
def create_scatter_plot(df):
    st.markdown(f'<h2 class="sub-title">Scatter Plot</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 6))
    sns.scatterplot(
        data=df,
        x=st.selectbox("X-axis", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])], key="scatter_x-axis"),
        y=st.selectbox("Y-axis", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])], key="scatter_y-axis"),
        ax=ax
    )
    st.pyplot(fig)

# Function to create Count Plot
def create_count_plot(df):
    st.markdown(f'<h2 class="sub-title">Count Plot</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.countplot(data=df, x=st.selectbox("Axis", df.columns, key="count_axis"), ax=ax)
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# Function to create Violin Plot
def create_violin_plot(df):
    st.markdown(f'<h2 class="sub-title">Violin Plot</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.violinplot(
        data=df,
        x=st.selectbox("X-axis", df.columns, key="violin_x-axis"),
        y=st.selectbox("Y-axis", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])], key="violin_y-axis"),
        ax=ax
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# Function to create Bar Chart
def create_bar_chart(df):
    st.markdown(f'<h2 class="sub-title">Bar Chart</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.barplot(
        data=df,
        x=st.selectbox("Labels", df.columns, key="bar_x-axis"),
        y=st.selectbox("Values", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])], key="bar_y-axis"),
        ax=ax,
        errorbar=None
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# Function to create Line Chart
def create_line_chart(df):
    st.markdown(f'<h2 class="sub-title">Line Chart</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 5))
    sns.lineplot(
        data=df,
        x=st.selectbox("Labels", df.columns, key="line_x-axis"),
        y=st.selectbox("Values", [col for col in df.columns if pd.api.types.is_numeric_dtype(df[col])], key="line_y-axis"),
        ax=ax,
        errorbar=None
    )
    ax.set_xticklabels(ax.get_xticklabels(), rotation=90)
    plt.tight_layout()
    st.pyplot(fig)

# Function to create Heatmap (added as a missing plot type)
def create_heatmap(df):
    st.markdown(f'<h2 class="sub-title">Heatmap</h2>', unsafe_allow_html=True)
    fig, ax = plt.subplots(figsize=(10, 8))
    sns.heatmap(df.corr(), annot=True, fmt=".2f", cmap='coolwarm', ax=ax)
    st.pyplot(fig)
