# Function definitions (reuse from previous implementation)
def about():
   
    st.write(
        """
        ### Project Overview
        This application provides various tools for managing and analyzing data.
        It is designed to help users interact with datasets, clean them, perform exploratory data analysis (EDA), and visualize data. Additionally, users can work with machine learning models and predictions.

        ### Features:
        1. **Data Preview**: View and explore the uploaded dataset with options to display different parts of the data.
        2. **Data Overview**: Get an overview of the dataset, including statistics, missing values, duplicates, and column types.
        3. **Data Cleaning**: Clean the data by handling missing values, duplicates, and encoding categorical variables.
        4. **Data Visualization**: Visualize the data with various charting options, such as pair plots, bar plots, scatter plots, etc.
        5. **Prediction Model**: Build predictive models and analyze results.
        6. **Contact**: Find information on how to get in touch.

        ### How to Use:
        - Upload your data through the **"File Uploader"** in the sidebar.
        - Select any of the menu options in the sidebar to explore or clean the data.
        - You can visualize the data and apply machine learning models for predictions.
        """
    )


def contact():
    st.title("Contact Information")
    
    st.write(
        """
        **Name**: Nishant Kumar  
        **Email**: nishant575051@gmail.com  
        **Contact**: +91 9654546020  
        **GitHub**: [Nishant6020](https://github.com/Nishant6020)  
        **LinkedIn**: [Nishant Kumar - Data Analyst](https://linkedin.com/in/nishant-kumar-data-analyst)  
        **Project & Portfolio**: [Data Science Portfolio](https://www.datascienceportfol.io/nishant575051)

        ### About Me
        I am a passionate and results-driven **Data Analyst** with over 4 years of experience. I specialize in leveraging data-driven strategies to optimize business performance and drive decision-making.

        I have expertise in **e-commerce management**, utilizing tools like **Excel**, **Power BI**, and **Python** for data analysis, visualization, and reporting. My goal is to transform complex datasets into actionable insights that have a tangible business impact.

        I have worked on enhancing product visibility, improving PPC campaign performance, and providing strategic insights to e-commerce platforms. I enjoy solving problems using data and am always learning new techniques and tools to improve my skills.
        """
    )
    st.write(
        """
        **Skills**:
        - **Programming Languages**: Python
        - **Libraries**: NumPy, Pandas, Matplotlib, Seaborn, Scikit-Learn, BeautifulSoup, Streamlit
        - **Databases**: MySQL
        - **Visualization**: Power BI, Tableau, Google Looker, Microsoft Excel
        - **Skills**: ETL, Data Cleaning, Visualization, EDA, Web Scraping, Problem Solving, Critical Thinking, Statistical Analysis, MLops, Prediction Model, Random Forest, Linear Regression & Classification, GridCV, XGBoost
        """
    )
    st.write(
        """
        **Work Experience**:
        - **Sr. E-commerce Manager & Data Analyst** at **Sanctus Innovation Pvt Ltd**, Delhi (Dec 2020 - Feb 2024)
        - **Sr. E-commerce Manager** at **Adniche Adcom Private Limited**, Delhi (Sep 2020)
        - **E-commerce Executive** at **Tech2globe Web Solutions LLP**, Delhi (Aug 2019 - Jul 2020)
        """
    )
    
    st.write(
        """
        **Certifications**:
        - Data Analytics, DUCAT The IT Training School
        - SQL Fundamental Course, Scaler
        - Power BI Course, Skillcourse
        - Data Analytics Using Python, IBM
        """
    )
        
    resume_path = "resume_nishant_kumar.pdf"  # Replace with the correct path to your resume file
    st.markdown("---")
    st.write("### Click Below to Download Resume")
    with open(resume_path, "rb") as resume_file:
        st.download_button(
            label="Download Resume",
            data=resume_file,
            file_name="Nishant_Kumar_Resume.pdf",
            mime="application/pdf"
        )


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


def corelation(df):
    return df.select_dtypes(include=['number']).corr()


def skewness_kurtosis(df):
    return df.select_dtypes(include=['number']).agg(['skew','kurtosis']).T

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

def eda(df):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("report.html")


def Model_Building(df):
    st.title("Split Data")
    
    # Select independent and dependent variables
    X_columns = st.multiselect("Select Independent Variables (X)", df.columns)
    y_column = st.selectbox("Select Dependent Variable (y)", df.columns)

    if not X_columns or not y_column:
        st.warning("Please select the independent and dependent variables to proceed.")
        return

    X = df[X_columns]
    y = df[y_column]

    # Handle categorical variables with one-hot encoding
    X = pd.get_dummies(X, drop_first=True)

    # Model type selection
    model_type = st.selectbox("Select Model Type", ["Regression", "Classification"])

    # Split data into train and test sets
    test_size = st.slider("Test Size", 0.1, 0.5, 0.2)

    if model_type == "Classification":
        # For classification, ensure y is categorical
        if y.dtype != 'object' and len(np.unique(y)) > 20:
            st.warning("The target variable seems continuous. Please choose a categorical target for classification.")
            return
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, stratify=y, random_state=42
        )
    else:
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=test_size, random_state=42
        )

    # Feature scaling
    feature_scaling = st.checkbox("Apply Feature Scaling")
    if feature_scaling:
        scaler = StandardScaler()
        X_train = scaler.fit_transform(X_train)
        X_test = scaler.transform(X_test)

    # Model selection
    model = None
    if model_type == "Regression":
        regression_model = st.selectbox(
            "Select Regression Model",
            [
                "Linear Regression",
                "Ridge Regression",
                "Lasso Regression",
                "SVR",
                "Random Forest Regression",
                "Gradient Boosting Regression",
            ],
        )
        if regression_model == "Linear Regression":
            model = LinearRegression()
        elif regression_model == "Ridge Regression":
            alpha = st.slider("Ridge alpha", 0.1, 10.0, 1.0)
            model = Ridge(alpha=alpha)
        elif regression_model == "Lasso Regression":
            alpha = st.slider("Lasso alpha", 0.1, 10.0, 1.0)
            model = Lasso(alpha=alpha)
        elif regression_model == "SVR":
            C = st.slider("SVR C", 0.1, 10.0, 1.0)
            kernel = st.selectbox("SVR kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVR(C=C, kernel=kernel)
        elif regression_model == "Random Forest Regression":
            n_estimators = st.slider("Number of Estimators", 10, 100, 10)
            max_depth = st.slider("Max Depth", 1, 30, 10)
            model = RandomForestRegressor(
                n_estimators=n_estimators, max_depth=max_depth
            )
        elif regression_model == "Gradient Boosting Regression":
            n_estimators = st.slider("Number of Estimators", 10, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            model = GradientBoostingRegressor(
                n_estimators=n_estimators, learning_rate=learning_rate
            )

    elif model_type == "Classification":
        classification_model = st.selectbox(
            "Select Classification Model",
            [
                "Logistic Regression",
                "Decision Tree Classifier",
                "Random Forest Classifier",
                "SVM",
                "KNN",
                "Naive Bayes",
                "Gradient Boosting Classifier",
            ],
        )
        if classification_model == "Logistic Regression":
            C = st.slider("Inverse of Regularization Strength (C)", 0.1, 10.0, 1.0)
            model = LogisticRegression(C=C, max_iter=1000)
        elif classification_model == "Decision Tree Classifier":
            max_depth = st.slider("Max Depth", 1, 30, 10)
            model = DecisionTreeClassifier(max_depth=max_depth)
        elif classification_model == "Random Forest Classifier":
            n_estimators = st.slider("Number of Estimators", 10, 100, 10)
            max_depth = st.slider("Max Depth", 1, 30, 10)
            model = RandomForestClassifier(
                n_estimators=n_estimators, max_depth=max_depth
            )
        elif classification_model == "SVM":
            C = st.slider("SVM C", 0.1, 10.0, 1.0)
            kernel = st.selectbox("SVM kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = SVC(C=C, kernel=kernel)
        elif classification_model == "KNN":
            n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 5)
            model = KNeighborsClassifier(n_neighbors=n_neighbors)
        elif classification_model == "Naive Bayes":
            model = GaussianNB()
        elif classification_model == "Gradient Boosting Classifier":
            n_estimators = st.slider("Number of Estimators", 10, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            model = GradientBoostingClassifier(
                n_estimators=n_estimators, learning_rate=learning_rate
            )

    # Train the model
    if st.button("Train Model"):
        model.fit(X_train, y_train)
        st.success("Model Trained Successfully!")

        # Cross-validation
        st.subheader("Cross-Validation")
        if model_type == "Classification":
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'accuracy'
        else:
            cv = KFold(n_splits=5, shuffle=True, random_state=42)
            scoring = 'neg_mean_squared_error'

        cv_scores = cross_val_score(model, X_train, y_train, cv=cv, scoring=scoring)
        st.write("Cross-Validation Scores:", cv_scores)
        st.write("Mean CV Score:", cv_scores.mean())

    # Evaluate the model
    st.subheader("Model Evaluation")
    predictions = model.predict(X_test)
    if model_type == "Regression":
        mse = mean_squared_error(y_test, predictions)
        r2 = r2_score(y_test, predictions)
        st.write(f"Mean Squared Error (MSE): {mse:.4f}")
        st.write(f"R² Score: {r2:.4f}")

        # Plotting Actual vs Predicted
        fig, ax = plt.subplots()
        ax.scatter(y_test, predictions)
        ax.plot([y_test.min(), y_test.max()], [y_test.min(), y_test.max()], 'k--', lw=2)
        ax.set_xlabel('Actual')
        ax.set_ylabel('Predicted')
        ax.set_title('Actual vs Predicted')
        st.pyplot(fig)

    elif model_type == "Classification":
        accuracy = accuracy_score(y_test, predictions)
        st.write(f"Accuracy: {accuracy:.4f}")

        # Display confusion matrix
        cm = confusion_matrix(y_test, predictions)
        st.write("Confusion Matrix:")
        st.write(cm)

        # Plot Confusion Matrix
        fig, ax = plt.subplots()
        cax = ax.matshow(cm, cmap='Blues')
        fig.colorbar(cax)
        plt.xlabel('Predicted')
        plt.ylabel('Actual')
        st.pyplot(fig)

    # Option to save the model
    save_model = st.checkbox("Save the Model")
    if save_model:
        model_name = st.text_input("Enter model name to save (without extension):")
        if st.button("Save Model"):
            with open(f"{model_name}.pkl", "wb") as f:
                pickle.dump(model, f)
            st.success(f"Model saved as {model_name}.pkl")

# steamlit app

import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, StratifiedKFold, KFold, cross_val_score
from sklearn.preprocessing import StandardScaler, OneHotEncoder, LabelEncoder
from sklearn.linear_model import LinearRegression, LogisticRegression, Ridge, Lasso
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor, GradientBoostingClassifier, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import pickle



# Function definitions (reuse from previous implementation)
# ... (keep all your function definitions as they are)

# Streamlit App
st.title("Data Cleaning and Preprocessing App")

uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV, Excel, JSON)", type=['csv', 'xlsx', 'xls', 'json'])

if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # Store the DataFrame in session state
        if 'df' not in st.session_state:
            st.session_state.df = df

        # Sidebar for options
        st.sidebar.title("About")
    # about page
        if st.sidebar.button("About"):
            about()

    # Contact page
        if st.sidebar.button("Contact"):
            contact()
            st.write("If you have any questions or feedback, please reach out to us.")

# clean data download
        if st.sidebar.button("Download Clean Data"):
                file_name = st.text_input("Enter file name to save:", "cleaned_data.csv")
                if st.button("Download Data"):
                    download_clean_data(st.session_state.df, file_name)
                    st.success(f"File saved as {file_name}!")


        # Data prerview Section (Initially hidden)
        data_prerview_expander = st.sidebar.expander("Data Preview", expanded=False)
        if data_prerview_expander:
            prerview_option = st.sidebar.selectbox("Select a task for Data Preview", [
                "Select Option",
                "Data Preview",
                "First 5 Rows",
                "Last 5 Rows",
                "10 Rows",
                "20 Rows",
                "50 Rows",
                "Sample Data",
            ])         
            # Handle Data prerview tasks
            if prerview_option == "Data Preview":
                data_view(st.session_state.df)
            elif prerview_option == "First 5 Rows":
                st.write(st.session_state.df.head())
            elif prerview_option == "Last 5 Rows":
                st.write(st.session_state.df.tail())
            elif prerview_option == "10 Rows":
                st.write(st.session_state.df.head(10))
            elif prerview_option == "20 Rows":
                st.write(st.session_state.df.head(20))
            elif prerview_option == "50 Rows":
                st.write(st.session_state.df.head(50))
            elif prerview_option == "Sample Data":
                sample_data(st.session_state.df)
        
        # Data Overview Section (Initially hidden)
        data_overview_expander = st.sidebar.expander("Data Overview", expanded=False)
        if data_overview_expander:
            overview_option = st.sidebar.selectbox("Select a task for Data Overview", [
                "Select Option",
                "Data Types",
                "Statistics Summary",
                "Show Missing Values & Rows",
                "Show Duplicate Valuess & Rows",
                "Show Unique Values",
                "Show Value Counts",
                "Correlation",
                "Skewness & Kurtosis",
                "Show Numerical or Categorical Columns",
                "Separate Numerical & Categorical Columns",
            ])
            
            # Handle Data Overview tasks
            if overview_option == "Statistics Summary":
                statistics(st.session_state.df)
            
            elif overview_option == "Data Types":
                st.write("### Data Types")
                st.write(st.session_state.df.dtypes)

            elif overview_option == "Show Missing Values & Rows":
                rows_with_missing_values = st.session_state.df[st.session_state.df.isnull().any(axis=1)]
                if len(rows_with_missing_values) > 0:
                    st.write("### Missing Values Summary")
                    st.write(check_missing_values(st.session_state.df))
                    st.write("### Rows with Missing Values")
                    st.write(rows_with_missing_values)
                else:
                    st.write("No rows with missing values found.")


            elif overview_option == "Show Duplicate Valuess & Rows":
                st.write("### Duplicate Summary")
                st.write(check_duplicates(st.session_state.df))
                st.write("### Duplicate Rows")
                st.write(duplicate(st.session_state.df))

            elif overview_option == "Show Unique Values":
                st.write("### Unique Values by Column")
                st.write(unique_columns(st.session_state.df))

            elif overview_option == "Show Value Counts":
                column = st.selectbox("Select a column to show value counts:", st.session_state.df.columns)
                if st.button("Show Value Counts"):
                    show_value_counts(st.session_state.df, column)
                    
            elif overview_option == "Show Numerical or Categorical Columns":
                column_type = st.radio("Select column type:", ['Numerical', 'Categorical'])
                st.write(f"### {column_type} Columns Data")
                st.write(show_column_data(st.session_state.df, column_type))
                st.write(column_selection(st.session_state.df))

            elif overview_option == "Separate Numerical & Categorical Columns":
                st.write("### Column Types")
                st.write(column_selection(st.session_state.df))

            elif overview_option == "Correlation":
                st.write("### Correlation")
                st.write(corelation(st.session_state.df))

            elif overview_option == "Skewness & Kurtosis":
                st.write("### Skewness & Kurtosis")
                st.write(skewness_kurtosis(st.session_state.df))

        # Data Cleaning Section (Initially hidden)
        data_cleaning_expander = st.sidebar.expander("Data Cleaning", expanded=False)
        if data_cleaning_expander:
            cleaning_option = st.sidebar.selectbox("Select a task for Data Cleaning", [
                "Select Option",
                "Change Data Type",
                "Fill Missing Values",
                "Remove Missing Values",
                "Remove Duplicates",
                "Drop Columns",
                "Replace Values",
                "Clean Categorical Text",
                "Encode Categorical Columns",
            ])  
            
            # Handle Data Cleaning tasks
            if cleaning_option == "Check Missing Values":
                st.write("### Missing Values Summary")
                st.write(check_missing_values(st.session_state.df))

            elif cleaning_option == "Change Data Type":
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


            elif cleaning_option == "Replace Values":
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



            elif cleaning_option == "Drop Columns":
                columns = st.multiselect("Select columns to drop:", st.session_state.df.columns)
                if st.button("Drop Columns"):
                    st.session_state.df = drop_columns(st.session_state.df, columns)
                    st.success("Selected columns dropped!")
                    st.write(st.session_state.df.head())


            elif cleaning_option == "Clean Categorical Text":
                categorical_columns = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
                columns = st.multiselect("Select categorical columns to clean:", categorical_columns)
                if st.button("Clean Text"):
                    st.session_state.df = clean_categorical_text(st.session_state.df, columns)
                    st.success("Text cleaned for selected columns!")
                    st.write(st.session_state.df.head())

            elif cleaning_option == "Encode Categorical Columns":
                categorical_columns = st.session_state.df.select_dtypes(exclude=['number']).columns.tolist()
                columns = st.multiselect("Select categorical columns to encode:", categorical_columns)
                if st.button("Encode Columns"):
                    st.session_state.df = encode_categorical(st.session_state.df, columns)
                    st.success("Selected columns encoded!")
                    st.write(unique_columns(st.session_state.df))
            
            elif cleaning_option == "Remove Missing Values":
                if st.button("Remove Missing Values"):
                    st.session_state.df, missing_values = remove_missing_values(st.session_state.df)
                    st.success("Rows with missing values removed!")
                    st.write(missing_values)
                    st.rerun()  # Refresh the page to reflect changes

            elif cleaning_option == "Remove Duplicates":
                if st.button("Remove Duplicates"):
                    st.session_state.df = remove_duplicates(st.session_state.df)  # Reassign the cleaned DataFrame back to `df`
                    st.success("Duplicate rows removed!")
                    st.write(check_duplicates(st.session_state.df))  # Display updated DataFrame
                    st.rerun()  # Refresh the page to reflect changes

            elif cleaning_option == "Fill Missing Values":
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

        visual_expander = st.sidebar.expander("Visualization", expanded=False)
        if visual_expander:
            Visual_option = st.sidebar.selectbox("Select a task for Visualization", [
                "Select Option","Pair Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Count Plot",
                    "Violin Plot", "Line Chart", "Correlation Heatmap", "Distribution Plot", "Pie Chart"
            ])
            if Visual_option == "Pair Plot" :
                st.header("Pair Plot")
                st.write(create_pairplot(st.session_state.df))

            elif Visual_option == "Bar Plot":
                st.header("Bar Plot")
                st.write(create_bar_chart(st.session_state.df))

            elif Visual_option == "Scatter Plot":
                st.header("Scatter Plot")
                st.write(create_scatter_plot(st.session_state.df))

            elif Visual_option == "Histogram":
                st.header("Histogram")
                st.write(create_histogram(st.session_state.df))

            elif Visual_option == "Box Plot":
                st.header("Box Plot")
                st.write(create_boxplot(st.session_state.df))

            elif Visual_option == "Count Plot":
                st.header("Count Plot")
                st.write(create_count_plot(st.session_state.df))

            elif Visual_option == "Violin Plot":
                st.header("Violin Plot")
                st.write(create_violin_plot(st.session_state.df))

            elif Visual_option == "Line Chart":
                st.header("Line Chart")
                st.write(create_line_chart(st.session_state.df))

            elif Visual_option == "Correlation Heatmap":
                st.header("Correlation Heatmap")
                st.write(create_heatmap(st.session_state.df))

            elif Visual_option == "Distribution Plot":
                st.header("Distribution Plot")

            elif Visual_option == "Pie Chart":
                st.header("Pie Chart")

        Model_Building_expander = st.sidebar.expander("Model Building", expanded=False)
        if Model_Building_expander:
            Model_Building_option = st.sidebar.selectbox("Select a task for Model Building", [
                "Select Option","Model Building"
            ])  
            
            # Handle Data Cleaning tasks
            if Model_Building_option == "Model Building":
                st.write("### Model Building")
                st.write(Model_Building(st.session_state.df))





