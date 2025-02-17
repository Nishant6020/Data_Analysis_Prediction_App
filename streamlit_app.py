# my details functions
def about():
   
    st.write(
        """
        ### Project Overview
        This application provides various tools for managing and analyzing data.
        It is designed to help users interact with datasets, clean them, perform exploratory data analysis (EDA), and visualize data. Additionally, users can work with machine learning models and predictions.

        ### Features:
        1. **Data Preview**: View and explore the uploaded dataset with options to display different parts of the data.
        2. **Data Overview**: Get an overviewimp of the dataset, including statistics, missing values, duplicates, and column types.
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

# cleaing and overview functions
def data_view(df):
    st.write("### Data Preview")
    st.write(df)

def sample_data(df):
    st.write("### Sample Data")
    st.write(df.sample(5))

def statistics(df):
    st.write("### Data Statistics")
    st.write(df.describe())

def load_data(file):
    try:
        if file.name.endswith('.csv'):
            encodings = ['utf-8', 'ISO-8859-1', 'latin1']
            for encoding in encodings:
                try:
                    return pd.read_csv(file, encoding=encoding)
                except UnicodeDecodeError:
                    continue
                except pd.errors.EmptyDataError:
                    st.error("The CSV file is empty.")
                    return None
                except pd.errors.ParserError as e:
                    st.error(f"Error parsing CSV file: {e}")
                    return None
                except Exception as e:
                    st.error(f"An unexpected error occurred with {encoding} encoding: {e}")
                    return None
            st.error("All encoding attempts failed. Please check the file format and contents.")
            return None

        elif file.name.endswith('.xlsx') or file.name.endswith('.xls'):
            try:
                return pd.read_excel(file)
            except Exception as e:
                st.error(f"Error loading Excel file: {e}")
                return None

        elif file.name.endswith('.json'):
            try:
                return pd.read_json(file)
            except ValueError as e:
                st.error(f"Error loading JSON file: {e}")
                return None

        else:
            st.error("Unsupported file format. Please upload a CSV, Excel, or JSON file.")
            return None

    except Exception as e:
        st.error(f"An unexpected error occurred: {e}")
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
    return df

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
    value_counts = df[column].value_counts().reset_index()
    value_counts.columns = [column, 'Count']  # Rename the columns for clarity
    st.write(value_counts)
    dtype = df[column].dtype
    if dtype == 'object' or pd.api.types.is_categorical_dtype(df[column]):
        if len(value_counts) < 15:
            fig = px.bar(value_counts, x=column, y='Count', text='Count')
            fig.update_traces(texttemplate='%{text:.s}', textposition='inside')
            avg_value = value_counts['Count'].mean()
            fig.add_hline(y=avg_value, line_dash="dash", annotation_text=f"Mean: {avg_value:.2f}")
            st.plotly_chart(fig)
            generate_observations(value_counts, column)
        else:
            fig = px.bar(value_counts, x=column, y='Count')
            st.plotly_chart(fig)
            generate_observations(value_counts, column)
    else:
        min_val = int(df[column].min())
        max_val = int(df[column].max())
        range_values = st.slider(
            'Select range',
            min_val,
            max_val,
            (min_val, max_val)
        )
        bin_size = st.slider(
            'Select bin size',
            1,
            50,
            10
        )
        filtered_data = df[column][(df[column] >= range_values[0]) & (df[column] <= range_values[1])]
        avg_value = filtered_data.mean()
        fig = go.Figure()
        fig.add_trace(go.Histogram(x=filtered_data, nbinsx=bin_size))
        fig.add_vline(x=avg_value, line_dash="dash", annotation_text=f"Mean: {avg_value:.2f}")
        st.plotly_chart(fig)
        generate_observations(filtered_data, column, numerical=True)

def generate_observations(data, column, numerical=False):
    st.write(f"### Observations for Column: {column}")
    if numerical:
        mean_value = data.mean()
        median_value = data.median()
        mode_value = data.mode()[0]
        skewness = data.skew()
        kurtosis = data.kurtosis()
        st.write(f"Mean value: {mean_value:.2f}")
        st.write(f"Median value: {median_value:.2f}")
        st.write(f"Mode value: {mode_value}")
        st.write(f"Skewness: {skewness:.2f}")
        st.write(f"Kurtosis: {kurtosis:.2f}")
        st.write(f"Range: from {data.min()} to {data.max()}")
        st.write(f"Sum: {data.sum()}")
        st.write(f"Count: {data.count()}")
        st.write(f"Variance: {data.var():.2f}")
        st.write(f"Standard Deviation: {data.std():.2f}")
        st.write(f"Quantiles: {data.quantile([0.25, 0.5, 0.75]).to_dict()}")
    else:
        top_values = data.nlargest(3, 'Count')
        st.write("Top values:")
        for i, row in top_values.iterrows():
            st.write(f"{i+1}. {row[column]}: {row['Count']} counts")
        most_common = data[column].mode()[0]
        st.write(f"Most common value: {most_common}")
        st.write(f"Number of unique values: {data[column].nunique()}")
        st.write(f"Diversity Index (Shannon Entropy): {scipy.stats.entropy(data[column].value_counts(normalize=True)):.2f}")

def correlation(df):
    return df.select_dtypes(include=['number']).corr()

def create_heatmap(df):
    palette = st.selectbox("Select Color", ['Viridis', 'Cividis', 'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys','magenta','solar', 'spectral', 'speed', 'sunset','darkmint', 'deep', 'delta', 'dense', 'earth', 'edge', 'electric', 'emrld', 'fall', 'geyser','cool'])
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale=palette)
    st.plotly_chart(fig)

def correlation_observations(corr_matrix):
    observations = []
    for col in corr_matrix.columns:
        for index in corr_matrix.index:
            # Only consider pairs where index comes before column to avoid duplicates
            if index < col:
                value = corr_matrix.loc[index, col]
                if value > 0.7:
                    observations.append(f"Strong Positive Correlation between {index} and {col}: {value:.2f}")
                elif value < -0.7:
                    observations.append(f"Strong Negative Correlation between {index} and {col}: {value:.2f}")
                elif 0.3 < value <= 0.7:
                    observations.append(f"Weak Positive Correlation between {index} and {col}: {value:.2f}")
                elif -0.7 < value < -0.2:
                    observations.append(f"Weak Negative Correlation between {index} and {col}: {value:.2f}")
                # elif -0.3 <= value <= 0.3:
                #     observations.append(f"No Correlation between {index} and {col}: {value:.2f}")
    return observations

def skewness_kurtosis(df):
    return df.select_dtypes(include=['number']).agg(['skew','kurtosis'])

def plot_skewness_kurtosis(df):
    numeric_df = df.select_dtypes(include=['number'])
    skewness = numeric_df.skew()
    kurtosis = numeric_df.kurtosis()

    skewness_fig = go.Figure([go.Bar(x=skewness.index, y=skewness.values)])
    skewness_fig.update_layout(title="Skewness", xaxis_title="Features", yaxis_title="Skewness")
    
    kurtosis_fig = go.Figure([go.Bar(x=kurtosis.index, y=kurtosis.values)])
    kurtosis_fig.update_layout(title="Kurtosis", xaxis_title="Features", yaxis_title="Kurtosis")
    
    st.plotly_chart(skewness_fig)
    st.plotly_chart(kurtosis_fig)

def get_data_info(df):
    # Create the data info summary DataFrame
    data_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non Null Value Count': df.notnull().sum(),
        'Missing Values': df.isnull().sum(), 
        'Duplicate Values': df.duplicated().sum(),
        'Unique Values': df.nunique()
    })

    # Create the summary info DataFrame
    dtype_counts = df.dtypes.value_counts()
    dtype_summary = {str(dtype): count for dtype, count in dtype_counts.items()}
    total_columns = df.shape[1]
    range_index = len(df)

    summary_info = pd.DataFrame({
        'Total Rows': [range_index],
        'Total Columns': [total_columns],
        **dtype_summary
    })

    return summary_info, data_info
# Function for cross-tabulation
def cross_tabulation(df, col1, col2):
    st.write(f"### Cross-Tabulation between {col1} and {col2}")
    crosstab = pd.crosstab(df[col1], df[col2])
    st.write(crosstab)


    try:
        if name == 'iris':
            data = datasets.load_iris()
        elif name == 'wine':
            data = datasets.load_wine()
        elif name == 'digits':
            data = datasets.load_digits()
        elif name == 'boston':
            data = datasets.load_boston()
        elif name == 'diabetes':
            data = datasets.load_diabetes()
        else:
            st.error("Unsupported Scikit-Learn dataset.")
            return None

        return pd.DataFrame(data.data, columns=data.feature_names)
    except Exception as e:
        st.error(f"Error loading Scikit-Learn dataset: {e}")
        return None

# Clear previous data only when a new dataset is selected
def clear_data():
    if 'df' in st.session_state:
        del st.session_state.df

def fetch_data_from_url(url):
    # Check if the URL is from GitHub
    if "github.com" in url and "blob" in url:
        url = url.replace("github.com", "raw.githubusercontent.com").replace("/blob", "")
    if url.endswith('.csv'):
        return pd.read_csv(url, on_bad_lines='warn')
    elif url.endswith('.xlsx') or url.endswith('.xls'):
        return pd.read_excel(url)
    elif url.endswith('.json'):
        return pd.read_json(url)
    else:
        st.error("Unsupported file format. Please use CSV, Excel, or JSON.")
        return None

# visualization function

def create_pairplot(df):
    kind = st.selectbox("Select type of kind", ['scatter', 'hist', 'reg'])
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    palette = st.selectbox("Select Color Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    
    if kind == 'scatter':
        fig = px.scatter_matrix(df, color=hue, color_discrete_sequence=px.colors.qualitative.Plotly)
    elif kind == 'hist':
        fig = px.histogram(df, x=df.columns, color=hue, color_discrete_sequence=px.colors.qualitative.Plotly)
    elif kind == 'reg':
        fig = px.scatter_matrix(df, color=hue, color_discrete_sequence=px.colors.qualitative.Plotly, trendline='ols')
    
    st.plotly_chart(fig)

def create_bar_plot(df):
    if df.empty:
        st.write("The DataFrame is empty. Please provide data to plot.")
        return None
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    x_column = st.selectbox('Select the X-axis column', df.columns)
    y_column = st.selectbox('Select the Y-axis column', df.columns)
    hue = st.selectbox('Select Hue', [None] + cat_columns)
    add_avg_line = st.checkbox('Add Average Line', value=False)

    title = st.text_input("Enter title for the Bar Plot", value=f"{x_column} vs {y_column} Bar Plot") 
    # Create the bar plot
    if hue:
        fig = px.bar(df, x=x_column, y=y_column, color=hue,)
    else:
        fig = px.bar(df, x=x_column, y=y_column)
    # Add average line if checked
    if add_avg_line and y_column in num_columns:
        avg_y = df[y_column].mean()
        fig.add_hline(y=avg_y, line_dash="dash", line_color="red", annotation_text="Average", annotation_position="bottom right")

    # Add average lines if checked
    if add_avg_line:
        avg_x = df[x_column].mean()
        avg_y = df[y_column].mean()

        # Add horizontal average line
        fig.add_hline(y=avg_y, line_dash="dash", line_color="gold", annotation_text="Avg Y", annotation_position="bottom right")
        
        # Add vertical average line
        fig.add_vline(x=avg_x, line_dash="dash", line_color="gold", annotation_text="Avg X", annotation_position="top right")
        
        # Add annotation for the average values
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.05, y=avg_y,
            text=f"Avg Y: {avg_y:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="yellow")
        )
        
        fig.add_annotation(
            xref="x", yref="paper",
            x=avg_x, y=1.05,
            text=f"Avg X: {avg_x:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=0,
            font=dict(color="yellow")
        )

    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_heatmap(df):
    title = st.text_input("Enter title for the heatmap", value="Correlation Heatmap")
    palette = st.selectbox("Select Color Palette", ['Viridis', 'Cividis', 'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'PuBuGn', 'PuRd', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'Reds', 'YlGnBu', 'YlGn', 'Blues', 'Purples', 'Greens', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'Reds', 'Greys', 'YlGnBu', 'YlGn', 'Blues', 'Purples', 'Greens'])
    
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale=palette)
    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_scatter(df):
    x_col = st.selectbox('X-axis (Only Numeric Columns are Valid) :', df.select_dtypes(include=['number']).columns.tolist())
    y_col = st.selectbox('Y-axis (Only Numeric Columns are Valid) :', df.select_dtypes(include=['number']).columns.tolist())
    hue_col = st.selectbox('Hue:', [None] + list(df.columns))
    style_col = st.selectbox('Style:', [None] + list(df.columns))
    size_col = st.selectbox('Size:', [None] + list(df.select_dtypes(include=['number']).columns.tolist()))
    title = st.text_input("Enter title for the scatter plot", value=f"{x_col} vs {y_col} scatter Plot")
    add_reg_line = st.checkbox('Add Regression Line', value=False)
    add_avg_line = st.checkbox('Add Average Line', value=False)
    # Create the scatter plot
    if hue_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, hover_name=style_col, size=size_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)
    # Add regression line if checked
    if add_reg_line:
        # Fit a linear regression model
        slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
        x_vals = np.array([df[x_col].min(), df[x_col].max()])
        y_vals = slope * x_vals + intercept
        
        # Determine the color of the regression line
        if slope > 0:
            reg_color = "green"
        elif slope < 0:
            reg_color = "red"
        else:
            reg_color = "white"    
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=reg_color, width=2), name='Regression Line'))

    # Add average lines if checked
    if add_avg_line:
        avg_x = df[x_col].mean()
        avg_y = df[y_col].mean()
        # Add horizontal average line
        fig.add_hline(y=avg_y, line_dash="dash", line_color="gold", annotation_text="Avg Y", annotation_position="bottom right")  
        # Add vertical average line
        fig.add_vline(x=avg_x, line_dash="dash", line_color="gold", annotation_text="Avg X", annotation_position="top right")   
        # Add annotation for the average values
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.05, y=avg_y,
            text=f"Avg Y: {avg_y:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="gold")
        )
        
        fig.add_annotation(
            xref="x", yref="paper",
            x=avg_x, y=1.05,
            text=f"Avg X: {avg_x:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=0,
            font=dict(color="gold")
        )
    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_histogram(df):
    x = st.selectbox("Select Numerical Variable", df.select_dtypes(include=['number']).columns.tolist())   
    hue = st.selectbox("Select Column for Color (Optional)", ['None'] + df.columns.tolist())
    stat = st.selectbox("Select Stat", ['count', 'percent', 'probability', 'density', 'probability density'])
    barmode = st.selectbox("Select Bar Mode", ['stack', 'group', 'overlay', 'relative'])
    nbins = st.slider("Number of Bins", min_value=1, max_value=100, value=10)
    cumulative = st.checkbox("Cumulative", value=False)
    title = st.text_input("Enter title for the histogram plot", value=f"{x} Histogram Plot")
    # Create the histogram
    if hue == 'None':
        fig = px.histogram(df, x=x, 
                           histnorm=stat if stat != 'count' else None,
                           nbins=nbins, 
                           barmode=barmode,
                           text_auto=True)
    else:
        fig = px.histogram(df, x=x, color=hue,
                           histnorm=stat if stat != 'count' else None,
                           nbins=nbins, 
                           barmode=barmode,
                           text_auto=True)
    if cumulative:
        fig.update_traces(cumulative_enabled=True)

    avg_value = df[x].mean()
    fig.add_shape(type='line', x0=avg_value, x1=avg_value, 
                   y0=0, y1=1, yref='paper', 
                   line=dict(color='yellow', width=2, dash='dash'),
                   name='Average')
    # Add annotation for average value
    fig.add_annotation(x=avg_value, y=1, yref='paper', 
                       text=f'Average: {avg_value:.2f}', 
                       showarrow=True, 
                       arrowhead=2, ax=0, ay=-40,
                       font=dict(color='red'))
    fig.update_layout(title=title,
                      xaxis_title=x,
                      yaxis_title='Count' if stat == 'count' else 'Density',
                      showlegend=True)
    return fig

def create_line_plot(df):
    # Select X and Y variables
    x = st.selectbox("Select X (numerical) column", df.select_dtypes(include=['number']).columns.tolist())
    y = st.selectbox("Select Y (numerical) column", df.select_dtypes(include=['number']).columns.tolist())
    
    # Select optional parameters
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    
    # Line and marker options
    markers = st.checkbox("Show Markers", value=False)
    
    # Average line option
    add_avg_line = st.checkbox("Add Average Line", value=False)
    
    # Title input
    title = st.text_input("Enter title for the line plot", value=f"{x} vs {y} Line Plot")
    
    # Create the line plot
    if hue:
        fig = px.line(df, x=x, y=y, color=hue, markers=markers)
    else:
        fig = px.line(df, x=x, y=y, markers=markers)
    
    # Add average lines if checked
    if add_avg_line:
        avg_y = df[y].mean()
        avg_x = df[x].mean()
        
        # Add average line for Y
        fig.add_hline(y=avg_y, line_dash="dash", line_color="gold", annotation_text="Avg Y", annotation_position="bottom right")
        
        # Add average line for X
        fig.add_vline(x=avg_x, line_dash="dash", line_color="gold", annotation_text="Avg X", annotation_position="top right")
        
        # Add annotation for the average values
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.05, y=avg_y,
            text=f"Avg Y: {avg_y:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="gold")
        )
        
        fig.add_annotation(
            xref="x", yref="paper",
            x=avg_x, y=1.05,
            text=f"Avg X: {avg_x:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=0,
            font=dict(color="gold")
        )

    # Update layout
    fig.update_layout(title=title)
    
    # Display the plot
    st.plotly_chart(fig)

def create_pie_chart(df):
    # Select categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_columns:
        st.warning("No categorical columns available in the DataFrame.")
        return
    category_col = st.selectbox("Select Category Variable", cat_columns)
    if category_col:
        # Select the category to highlight
        highlight_category = st.selectbox("Select Category to Highlight", df[category_col].unique())
        # Count the occurrences of each category
        data_counts = df[category_col].value_counts(ascending = False)
        title = st.text_input("Enter title for the pie chart", value=f"{category_col} Pie Chart")
        # Create a pull list to highlight the selected category
        pull = [0.1 if name == highlight_category else 0 for name in data_counts.index]
        # Create the pie chart using go.Figure
        fig = go.Figure(data=[go.Pie(labels=data_counts.index, values=data_counts.values, pull=pull)])
        # Update layout with the title
        fig.update_layout(title=title)
        # Display the plot
        st.plotly_chart(fig)
    else:
        st.warning("Please select a valid categorical variable.")

def create_boxplot(df):
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['number']).columns.tolist()

    x_col = st.selectbox("Select X (categorical) variable", num_columns + cat_columns)
    y_col = st.selectbox("Select Y (numerical) variable", [None]+ num_columns + cat_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    title = st.text_input("Enter title for the box plot", value=f"{x_col} vs {y_col} box plot")
    
    if hue:
        fig = px.box(df, x=x_col, y=y_col, color=hue,)
    else:
        fig = px.box(df, x=x_col, y=y_col)

    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_count_plot(df):
    # Select categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_columns:
        st.warning("No categorical columns available in the DataFrame.")
        return

    x = st.selectbox("Select X (categorical) variable", cat_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    stat = st.selectbox("Select Stat", ['count', 'percent', 'proportion', 'probability'])
    add_avg_line = st.checkbox('Add Average Line', value=False)
    title = st.text_input("Enter title for the count plot", value=f"{x} count plot")
    
    if x:
        fig = px.histogram(
            df,
            x=x,
            color=hue,
            barmode='group',
            histnorm=None if stat == 'count' else stat,
        )
        # Add average line if checked
        if add_avg_line:
            avg_x = df[x].value_counts().mean()
            fig.add_hline(y=avg_x, line_dash="dash", line_color="red", annotation_text="Avg", annotation_position="bottom right")

            
        fig.update_layout(title=title)
        st.plotly_chart(fig)
    else:
        st.warning("Please select a valid categorical variable.")

def create_kde_plot(df):
    # Identify numeric and categorical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Streamlit widgets for user input
    x = st.selectbox("Select X variable", [None] + num_columns )
    y = st.selectbox("Select Y variable", [None] + num_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    add_avg_line = st.checkbox('Add Average Line', value=False)
    fill = st.checkbox('Fill KDE', value=False)

    fig, ax = plt.subplots()
    plot = sns.kdeplot(data=df, x=x, y=y, hue=hue,fill=fill, palette=palette, ax=ax)
    plt.xticks(rotation=90)

    # Add average lines if selected and variables are numeric
    if add_avg_line:
        if x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.1f}')
            ax.text(avg_x, ax.get_ylim()[1], f'{avg_x:.1f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=10)
        
        if y in num_columns:
            avg_y = df[y].mean()
            ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.1f}')
            ax.text(0, avg_y, f'{avg_y:.1f}', color='red', ha='right', va='center', 
                    transform=ax.get_yaxis_transform(), fontsize=10)
    
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    st.pyplot(fig)

# seaborn graphs
def mat_create_pairplot(df):
    cat_column = df.select_dtypes(include=['object'])
    columns = cat_column.columns.tolist()
    kind = st.selectbox("Select type of kind", ['scatter', 'kde', 'hist', 'reg'])
    hue = st.selectbox("Select Hue (categorical variable)", [None] + columns)
    palette = st.selectbox("Select Color Palette", ['bright', 'tab10','rocket', 'viridis','icefire','Paired',"Set2"])
    sns.pairplot(df, hue=hue, palette=palette, kind=kind)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def mat_create_bar_plot(df):
    if df.empty:
        st.write("The DataFrame is empty. Please provide data to plot.")
        return None
    # Select numeric and date columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime']).columns.tolist()

    # Convert year columns in date_columns to numerical
    for col in date_columns:
        if (df[col].dt.year == df[col].dt.year).all():
            df[col] = df[col].dt.year
            num_columns.append(col)
            date_columns.remove(col)

    # Select x and y columns
    x_column = st.selectbox('Select the X-axis column', df.columns)
    y_column = st.selectbox('Select the Y-axis column', df.columns)
    plot_type = st.selectbox('Select plot type', ['Vertical', 'Horizontal'])
    title = st.text_input("Enter title for the Bar Plot", value=f"{x_column} vs {y_column} Bar Plot")
    hue = st.selectbox('Select Hue', [None] + list(df.columns))
    palette = st.selectbox("Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    add_avg_line = st.checkbox('Add Average Line', value=False)

    # Validate selected columns
    if x_column not in df.columns or y_column not in df.columns:
        st.write("Please select valid columns for X and Y axes.")
        return None

    if plot_type == 'Vertical':
        fig, ax = plt.subplots()
        vbar_plot = sns.barplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette)
        plt.title(title)
        plt.tight_layout()
        plt.xticks(rotation=90)

        if df[x_column].nunique() < 10:
            for i in vbar_plot.containers:
                vbar_plot.bar_label(i, label_type="center", rotation=90,padding=3)

        if add_avg_line and y_column in num_columns:
            avg_value = df[y_column].mean()
            ax.axhline(avg_value, color='red', linewidth=1, linestyle='-', label=f'Average: {avg_value:.2f}')
            ax.legend()
            ax.text(0.95, avg_value, f'{avg_value:.2f}', color='red', ha='right', va='center', transform=ax.get_yaxis_transform())
        return fig

    elif plot_type == 'Horizontal':
        fig, ax = plt.subplots()
        hbar_plot = sns.barplot(x=y_column, y=x_column, data=df, hue=hue, ax=ax, palette=palette, orient='h')
        plt.title(title)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if df[x_column].nunique() < 40:
            for i in hbar_plot.containers:
                hbar_plot.bar_label(i, label_type="center", rotation=0,padding=3)

        if add_avg_line and x_column in num_columns:
            avg_value = df[x_column].mean()
            ax.axvline(avg_value, color='red', linewidth=1, linestyle='-', label=f'Average: {avg_value:.2f}')
            ax.legend()
            ax.text(avg_value, 0.95, f'{avg_value:.2f}', color='red', ha='center', va='top', transform=ax.get_xaxis_transform())
        return fig

    else:
        st.write("Unsupported plot type. Please choose from 'Vertical' or 'Horizontal'.")
        return None

def mat_create_heatmap(df):
    title = st.text_input("Enter title for the heatmap", value="Correlation Heatmap")
    # Filter the DataFrame to include only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    # Create the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    return fig

def mat_create_scatter(df):   
    # Define options for user input
    def get_unique_values(column):
        return sorted(df[column].unique().tolist())
    x_col = st.selectbox('X-axis:', df.columns)
    y_col = st.selectbox('Y-axis:', df.columns)
    hue_col = st.selectbox('Hue:', [None] + list(df.columns), format_func=lambda x: 'None' if x is None else x)
    style_col = st.selectbox('Style:', [None] + list(df.columns), format_func=lambda x: 'None' if x is None else x)
    size_col = st.selectbox('Size:', [None] + list(df.columns), format_func=lambda x: 'None' if x is None else x)
    title = st.text_input("Enter title for the scatter plot", value=f"{x_col} vs {y_col} scatter Plot")
    
    # Check if both x and y columns are numerical
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
    
    # Checkboxes for regression and average line
    add_reg_line = st.checkbox('Add Regression Line', disabled=not (x_is_numeric and y_is_numeric))
    add_avg_line = st.checkbox('Add Average Line')
    
    # Sliders for X and Y axes if columns are numerical
    if x_is_numeric:
        x_min, x_max = int(df[x_col].min()), int(df[x_col].max())
        x_range = st.slider('X-axis range', min_value=x_min, max_value=x_max, value=(x_min, x_max))
    else:
        x_range = (None, None)
    
    if y_is_numeric:
        y_min, y_max = int(df[y_col].min()), int(df[y_col].max())
        y_range = st.slider('Y-axis range', min_value=y_min, max_value=y_max, value=(y_min, y_max))
    else:
        y_range = (None, None)
    
    # Filter data based on slider range if columns are numerical
    if x_is_numeric:
        df = df[(df[x_col] >= x_range[0]) & (df[x_col] <= x_range[1])]
    if y_is_numeric:
        df = df[(df[y_col] >= y_range[0]) & (df[y_col] <= y_range[1])]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        x=x_col, y=y_col, data=df,
        hue=hue_col if hue_col is not None else None,
        style=style_col if style_col is not None else None,
        size=size_col if size_col is not None else None
    )
    
    # Add regression line if checked and columns are numerical
    if add_reg_line and x_is_numeric and y_is_numeric:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, ax=scatter)
    
    # Add average line if checked
    if add_avg_line and y_is_numeric:
        mean_y = df[y_col].mean()
        plt.axhline(y=mean_y, color='red', linestyle='--', label=f'Average {y_col}')
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add colorbar if color column is selected and position it at the bottom
    if hue_col:
        colorbar = plt.colorbar(scatter.collections[0], ax=scatter, orientation='vertical', pad=0.21)
        colorbar.set_label(hue_col, labelpad=10)
    
    plt.grid(True)
    plt.tight_layout()  # Automatically adjust subplot parameters for a better fit
    st.pyplot(plt)

def mat_create_histplot(df):
    # Select numerical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()   
    x = st.selectbox("Select Numerical Variable", num_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    palette = st.selectbox("Select Color Palette", ['deep', 'muted', 'bright', 'dark', 'colorblind'])
    stat = st.selectbox("Select Stat", ['count', 'frequency', 'density', 'probability'])
    multiple = st.selectbox("Multiple", ['layer', 'dodge', 'stack', 'fill'])
    element = st.selectbox("Element", ['bars', 'step'])
    palette = st.selectbox("Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    bins = st.slider("Number of Bins", min_value=1, max_value=100, value=10)
    shrink = st.slider("Adjust Bar Size", min_value=0.1, max_value=1.0, step=0.1, value=1.0)
    fill = st.checkbox("Fill", value=True)
    cumulative = st.checkbox("Cumulative")
    kde = st.checkbox("KDE")
    add_avg_line = st.checkbox('Add Average Line', value=False)
    # Plot the histogram
    if x:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=x, hue=hue, stat=stat, bins=bins, multiple=multiple, cumulative=cumulative, element=element, fill=fill, shrink=shrink, kde=kde, palette=palette, ax=ax)      
        if add_avg_line and x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.1f}')
            ax.text(avg_x, 1, f'{avg_x:.1f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=10)
        plt.show()
        st.pyplot(fig)

    else:
        st.warning("Please select a valid numerical variable.")

def mat_create_line_plot(df):
    for col in df.select_dtypes(include=['datetime']):
        if df[col].dt.year.nunique() > 1:
            df[col] = df[col].dt.year
        elif df[col].dt.month.nunique() > 1:
            df[col] = df[col].dt.month
        elif df[col].dt.day.nunique() > 1:
            df[col] = df[col].dt.day
    
    # Identify numeric and categorical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Streamlit widgets for user input
    x = st.selectbox("Select X variable", [None] + num_columns + cat_columns)
    y = st.selectbox("Select Y variable", [None] + cat_columns)
    hue = st.selectbox("Select Hue (categorical)", [None] + cat_columns)
    size = st.selectbox("Select Size (numeric)", [None] + num_columns)
    style = st.selectbox("Select Style (categorical)", [None] + cat_columns)
    markers = st.checkbox("Show Markers", value=False)
    dashes = st.checkbox("Show Dashes", value=True)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    estimator = st.selectbox("Select Estimator", ['mean', 'median', 'sum', 'min', 'max'])
    add_avg_line = st.checkbox("Add Average Line", value=False)
    markers = st.selectbox("Select Marker points",['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])
    marker_size = st.slider('Marker Size', min_value=1, max_value=100, value=14)
    title = st.text_input("Enter title for the line plot", value=f"{x} vs {y} line Plot")
    # Validate selected inputs
    if not x or not y:
        st.warning("Please select valid X and Y variables.")
        return
    # Create the figure
    fig, ax = plt.subplots(figsize=(25, 15))
    try:
        sns.lineplot(data=df, x=x, y=y, hue=hue, size=size, style=style,
                     markers=markers, dashes=dashes, palette=palette, 
                     estimator=estimator,marker=markers,markersize=marker_size, ax=ax)
        ax.set_xlabel(x, fontsize=40)
        ax.set_ylabel(y, fontsize=40)
        ax.tick_params(axis='x', labelrotation=90, labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        plt.tight_layout()

        # Add average lines if requested
        if add_avg_line and y in num_columns:
            avg_y = df[y].mean()
            ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.2f}')
            ax.text(1, avg_y, f'{avg_y:.2f}', color='red', ha='left', va='center', 
                    transform=ax.get_yaxis_transform(), fontsize=40)
        
        if add_avg_line and x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.2f}')
            ax.text(avg_x, 1, f'{avg_x:.2f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=40)
        
        # Handle legend positioning if hue or avg line exists
        if hue or add_avg_line:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=40)
            plt.subplots_adjust(right=0.8)
        
        plt.tight_layout()
        plt.title(title,fontsize=50,fontweight='bold')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating line plot: {e}")

def mat_create_pie_chart(df):
    # Select categorical column
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    category_col = st.selectbox("Select Category Variable", cat_columns)
    
    # Color palette selection
    palette = st.selectbox("Select Color Palette", 
                           ['Accent', 'Accent_r','Dark2', 'Dark2_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r'])

    if category_col:
        data_counts = df[category_col].value_counts()
        
        # Plot the pie chart
        fig, ax = plt.subplots()
        ax.pie(
            data_counts, 
            labels=data_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=plt.get_cmap(palette).colors[:len(data_counts)]
        )
        
        plt.show()
        st.pyplot(fig)
    else:
        st.warning("Please select a valid category variable.")

def mat_create_boxplot(df):
    # Select categorical and numerical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['number']).columns.tolist()

    x_col = st.selectbox("Select X (categorical) variable", num_columns + cat_columns)
    y_col = st.selectbox("Select Y (numerical) variable", [None]+ num_columns + cat_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])

    # Plot the box plot
    if x_col or y_col:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
        plt.xticks(rotation=45)
        plt.show()
        st.pyplot(fig)
    else:
        st.warning("Please select valid X and Y variables.")

def mat_create_count_plot(df):
    x = st.selectbox("Select X (categorical) variable", df.select_dtypes(include=['object', 'category']).columns.tolist())
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    palette = st.selectbox("Select Color Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    stat = st.selectbox("Select Stat", ['count', 'percent', 'proportion', 'probability'])
    add_avg_line = st.checkbox('Average Line', value=False)
    fig, ax = plt.subplots()
    plt.figure(figsize=(20,20))
    plot = sns.countplot(data=df, x=x, hue=hue, stat=stat, palette=palette, ax=ax)
    plt.xticks(rotation=90)
    if add_avg_line:
        if stat == 'count':
            avg_y = df[x].value_counts().mean()
        elif stat == 'percent':
            avg_y = (df[x].value_counts() / len(df)).mean() * 100
        elif stat == 'proportion':
            avg_y = (df[x].value_counts() / len(df)).mean()
        elif stat == 'probability':
            avg_y = (df[x].value_counts() / len(df)).mean()

        ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.1f}')
        ax.text(0, avg_y, f'{avg_y:.1f}', color='red', ha='right', va='center', transform=ax.get_yaxis_transform(), fontsize=10)

    # Add bar labels if hue variable has less than or equal to 10 unique values
    if hue is None or df[hue].nunique() <= 15:
        for container in plot.containers:
            plot.bar_label(container, label_type="edge", rotation=90,padding=3)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    st.pyplot(fig)

def mat_create_kde_plot(df):
    # Identify numeric and categorical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Streamlit widgets for user input
    x = st.selectbox("Select X variable", [None] + num_columns )
    y = st.selectbox("Select Y variable", [None] + num_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    add_avg_line = st.checkbox('Add Average Line', value=False)

    fig, ax = plt.subplots()
    plot = sns.kdeplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
    plt.xticks(rotation=90)

    # Add average lines if selected and variables are numeric
    if add_avg_line:
        if x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.1f}')
            ax.text(avg_x, ax.get_ylim()[1], f'{avg_x:.1f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=10)
        
        if y in num_columns:
            avg_y = df[y].mean()
            ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.1f}')
            ax.text(0, avg_y, f'{avg_y:.1f}', color='red', ha='right', va='center', 
                    transform=ax.get_yaxis_transform(), fontsize=10)
    
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    st.pyplot(fig)

def eda(df):
    profile = ProfileReport(df, title="Profiling Report")
    profile.to_file("report.html")


# model functions 
# Function to split data
def split_data(df, model_type):
    try:
        split_option = st.radio("Choose Splitting Option", ["Select X and y", "Select y only and auto-detect X"])

        if split_option == "Select X and y":
            X_columns = st.multiselect("Select Independent Variables (X)", df.columns)
            y_column = st.selectbox("Select Dependent Variable (y)", df.columns)
            if not X_columns or not y_column:
                st.error("Please select both independent and dependent variables.")
                return None, None, None, None
            X = df[X_columns]
            y = df[y_column]
        else:
            y_column = st.selectbox("Select Dependent Variable (y)", df.columns)
            if not y_column:
                st.error("Please select a dependent variable.")
                return None, None, None, None
            X = df.drop(columns=[y_column])
            y = df[y_column]

        test_size = st.slider("Test Size", 0.1, 0.5, 0.2)
        random_state = st.number_input("Enter random state value", value=0, step=1)

        if model_type == "Classification":
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, stratify=y, random_state=random_state)
        else:
            X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state)

        return X_train, X_test, y_train, y_test
    except Exception as e:
        st.error(f"Error in data splitting: {e}")
        return None, None, None, None

# Function for column transformation
def column_transformation(X):
    try:
        numerical_features = X.select_dtypes(include=['int64', 'float64']).columns
        categorical_features = X.select_dtypes(include=['object']).columns

        col_trans = ColumnTransformer([
            ('num', Pipeline([
                ('imputer', SimpleImputer(strategy='mean')),
                ('scaler', StandardScaler())
            ]), numerical_features),
            ('cat', OneHotEncoder(handle_unknown='ignore'), categorical_features)
        ])
        return col_trans
    except Exception as e:
        st.error(f"Error in column transformation: {e}")
        return None

# Function for regression models
def regression_model(X_train, X_test, y_train, y_test, col_trans):
    try:
        regression_model_name = st.selectbox(
            "Select Regression Model",
            ["Linear Regression", "Ridge Regression", "Lasso Regression", "SVR", "KNN","Decision Tree","Random Forest Regression", "Gradient Boosting Regression","AdaBoost Regressor","ExtraTrees Regressor","GridSearchCV"]
        )
        if regression_model_name == "Linear Regression":
            model = make_pipeline(col_trans, LinearRegression())
            params = None
        
        elif regression_model_name == "Ridge Regression":
            alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
            model = make_pipeline(col_trans, Ridge(alpha=alpha))    
            params = {"alpha": alpha}
        
        elif regression_model_name == "Lasso Regression":
            alpha = st.slider("Alpha", 0.1, 10.0, 1.0)
            model = make_pipeline(col_trans, Lasso(alpha=alpha))
            params = {"alpha": alpha}
        
        elif regression_model_name == "SVR":
            C = st.slider("C", 0.1, 10.0, 1.0)
            epsilon = st.slider("Epsilon", 0.01, 1.0, 0.1)
            kernel = st.selectbox("Kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = make_pipeline(col_trans, SVR(C=C, epsilon=epsilon, kernel=kernel))
            params = {"C": C, "epsilon": epsilon, "kernel": kernel}


        elif regression_model_name == "Random Forest Regression":
            n_estimators = st.slider("Number of Estimators", 50, 200, 100)
            max_depth = st.slider("Max Depth",min_value= 1, max_value=30, value=10)
            model = make_pipeline(col_trans, RandomForestRegressor(n_estimators=n_estimators, max_depth=max_depth))
            params = {"n_estimators": n_estimators, "max_depth": max_depth}

        elif regression_model_name == "Gradient Boosting Regression":
            n_estimators = st.slider("Number of Estimators", 50, 200, 100)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            model = make_pipeline(col_trans, GradientBoostingRegressor(n_estimators=n_estimators, learning_rate=learning_rate))
        
            params = {"n_estimators": n_estimators, "learning_rate": learning_rate}

        elif regression_model_name == "KNN":
            n_neighbors = st.slider("Number of Neighbors", 1, 20, 3)
            KNN = KNeighborsRegressor(n_neighbors=n_neighbors)
            scaler = StandardScaler(with_mean=False)
            model = make_pipeline(col_trans, scaler, KNN)
            params = {"n_neighbors": n_neighbors}
        
        elif regression_model_name == "Decision Tree":
            Max_depth = st.slider('Max Depth', min_value=1, max_value=20, value=8)
            DTR = DecisionTreeRegressor(max_depth=Max_depth)
            scaler = StandardScaler(with_mean=False)
            model = make_pipeline(col_trans,scaler,DTR)
            params = {"max_depth": Max_depth}
        
        elif regression_model_name == "AdaBoost Regressor":
            n_estimator = st.slider('n_estimators', min_value=10, max_value=100, value=15)
            learning_rates = st.slider('learning_rate', min_value=0.01, max_value=2.0, value=1.0, step=0.01)
            ADA = AdaBoostRegressor(n_estimators=n_estimator,learning_rate=learning_rates)
            scaler = StandardScaler(with_mean=False)
            model = make_pipeline(col_trans,scaler,ADA)
            params = {"n_estimator":n_estimator,"learning_rates":learning_rates}
        
        elif regression_model_name == "ExtraTrees Regressor":
            # Sliders for ExtraTreesRegressor parameters
            n_estimators = st.slider('n_estimators', min_value=10, max_value=200, value=100)
            max_samples = st.slider('max_samples', min_value=0.1, max_value=1.0, value=0.5, step=0.1)
            random_state = st.slider('random_state', min_value=1, max_value=100, value=3)
            max_features = st.slider('max_features', min_value=0.1, max_value=1.0, value=0.75, step=0.1)
            max_depth = st.slider('max_depth', min_value=1, max_value=30, value=15)
            ERT = ExtraTreesRegressor(n_estimators=n_estimators,
                                    random_state=random_state,
                                    max_samples=max_samples,
                                    max_features=max_features,
                                    max_depth=max_depth,
                                    bootstrap=True)
            scaler = StandardScaler(with_mean=False)  # Set with_mean=False
            model = make_pipeline(col_trans, scaler, ERT)
            params = {"n_estimators": n_estimators, "max_samples": max_samples, "random_state": random_state, "max_features": max_features, "max_depth": max_depth}

        elif regression_model_name == "GridSearchCV":
            # Sliders for GridSearchCV parameters
            n_estimators_values = st.multiselect('n_estimators', options=[50, 100, 200], default=[50, 100, 200])
            max_depth_values = st.multiselect('max_depth', options=[None, 10, 20], default=[None, 10, 20])
            min_samples_split_values = st.multiselect('min_samples_split', options=[2, 5, 10], default=[2, 5, 10])
            # Construct the parameter grid from the selected values
            param_grid = {
                'n_estimators': n_estimators_values,
                'max_depth': max_depth_values,
                'min_samples_split': min_samples_split_values}
            # Create the GridSearchCV object
            grid_search = GridSearchCV(RandomForestRegressor(), param_grid, cv=3, scoring='r2')
            # Create the model pipeline
            scaler = StandardScaler(with_mean=False)
            model = make_pipeline(col_trans, scaler, grid_search)
 
        # if st.button("Evalute Model"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)
        st.write("Model Score:", model.score(X_test, y_test))
        st.write('R² Score:', r2_score(y_test, y_pred))
        st.write('MSE:', mean_squared_error(y_test, y_pred))
        st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Option to save the model
        model_name = st.text_input("Enter model name to save (without extension)", key="model_name_input")
        if st.button("Save Train Model", key="save_model_button"):
            if model_name:
                file_name = f"{model_name}.pkl"
                try:
                    with open(file_name, "wb") as f:
                        pickle.dump(model, f)
                    st.success(f"Model saved successfully as `{file_name}`")
                except Exception as e:
                    st.error(f"Error saving model: {e}")
            else:
                st.error("Please enter a valid model name before saving.")

        # Input form for prediction
        st.subheader("Test Current Model")
        input_data = {}
        for idx, col in enumerate(X_train.columns):
            if X_train[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select value for {col}", options=X_train[col].unique(), key=f"select_{idx}")
            else:
                input_data[col] = st.number_input(f"Enter value for {col}", key=f"num_{idx}")

        if st.button("Predict", key="predict_button"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")


    except Exception as e:
        st.error(f"Error in regression model: {e}")

# Function for classification models
def classification_models(X_train, X_test, y_train, y_test, col_trans):
    try:
        classification_model = st.selectbox(
            "Select Classification Model",
            [
                "Logistic Regression",
                "Decision Tree Classifier",
                "Random Forest Classifier",
                "Support Vector Machine",
                "KNN",
                "Naive Bayes",
                "Gradient Boosting Classifier"
            ]
        )
        if classification_model == "Logistic Regression":
            C = st.slider("Inverse of Regularization Strength (C)", 0.1, 10.0, 1.0)
            model = make_pipeline(col_trans, LogisticRegression(C=C, max_iter=1000))
            params = {"C": C}
        elif classification_model == "Decision Tree Classifier":
            max_depth = st.slider("Max Depth", 1, 30, 10)
            model = make_pipeline(col_trans, DecisionTreeClassifier(max_depth=max_depth))
            params = {"max_depth": max_depth}
        elif classification_model == "Random Forest Classifier":
            n_estimators = st.slider("Number of Estimators", 10, 100, 10)
            max_depth = st.slider("Max Depth", 1, 30, 10)
            model = make_pipeline(col_trans, RandomForestClassifier(n_estimators=n_estimators, max_depth=max_depth))
            params = {"n_estimators": n_estimators, "max_depth": max_depth}
        elif classification_model == "Support Vector Machine":
            C = st.slider("C", 0.1, 10.0, 1.0)
            kernel = st.selectbox("kernel", ["linear", "poly", "rbf", "sigmoid"])
            model = make_pipeline(col_trans, SVC(C=C, kernel=kernel))
            params = {"C": C, "kernel": kernel}
        elif classification_model == "KNN":
            n_neighbors = st.slider("Number of Neighbors (k)", 1, 20, 5)
            model = make_pipeline(col_trans, KNeighborsClassifier(n_neighbors=n_neighbors))
            params = {"n_neighbors": n_neighbors}
        elif classification_model == "Naive Bayes":
            model = make_pipeline(col_trans, GaussianNB())
            params = None
        elif classification_model == "Gradient Boosting Classifier":
            n_estimators = st.slider("Number of Estimators", 10, 100, 10)
            learning_rate = st.slider("Learning Rate", 0.01, 1.0, 0.1)
            model = make_pipeline(col_trans, GradientBoostingClassifier(n_estimators=n_estimators, learning_rate=learning_rate))

        if st.button("Evalute Model"):
            model.fit(X_train, y_train)
            y_pred = model.predict(X_test)
            st.write("Model Score:", model.score(X_test, y_test))
            st.write('R² Score:', r2_score(y_test, y_pred))
            st.write('MSE:', mean_squared_error(y_test, y_pred))
            st.write('MAE:', mean_absolute_error(y_test, y_pred))

        # Input form for prediction
        st.subheader("Make a Prediction")
        input_data = {}
        for idx, col in enumerate(X_train.columns):
            if X_train[col].dtype == 'object':
                input_data[col] = st.selectbox(f"Select value for {col}", options=X_train[col].unique(), key=f"select_{idx}")
            else:
                input_data[col] = st.number_input(f"Enter value for {col}", key=f"num_{idx}")

        if st.button("Predict", key="predict_button"):
            input_df = pd.DataFrame([input_data])
            prediction = model.predict(input_df)
            st.write(f"Prediction: {prediction[0]}")

        # Option to save the model
        model_name = st.text_input("Enter model name to save (without extension)", key="model_name_input")

        if st.button("Save Train Model", key="save_model_button"):
            if model_name:
                file_name = f"{model_name}.pkl"
                try:
                    with open(file_name, "wb") as f:
                        pickle.dump(model, f)
                    st.success(f"Model saved successfully as `{file_name}`")
                except Exception as e:
                    st.error(f"Error saving model: {e}")
            else:
                st.error("Please enter a valid model name before saving.")


    except Exception as e:
        st.error(f"Error in Classification model: {e}")

# Data profiling
def eda(df):
    profile = ProfileReport(df, title="Profiling Report")
    if st.button("Generate Report"):
        # Generate and cache the report
        profile.to_file("report.html")
        st.session_state['report_generated'] = True
        st.success("Report generated! You can view or download it below.")

    if st.session_state.get('report_generated'):
        # Display a download link for the report
        with open("report.html", "rb") as file:
            st.download_button(
                label="Download Report",
                data=file,
                file_name="Profiling_Report.html",
                mime="text/html"
            )

        if st.button("View Report"):
            with open("report.html", "r", encoding="utf-8") as f:
                report_html = f.read()
                components.html(report_html, height=800, scrolling=True)



# steamlit app
import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from ydata_profiling import ProfileReport
import streamlit.components.v1 as components
from sklearn.model_selection import train_test_split, GridSearchCV
# from xgboost import XGBRegressor
from sklearn.preprocessing import OneHotEncoder, StandardScaler, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import make_pipeline
from sklearn.metrics import r2_score, mean_absolute_error, mean_squared_error, accuracy_score, f1_score
from sklearn.linear_model import LinearRegression, Ridge, Lasso, LogisticRegression
from sklearn.tree import DecisionTreeClassifier, DecisionTreeRegressor
from sklearn.ensemble import RandomForestClassifier,ExtraTreesRegressor,AdaBoostRegressor, VotingRegressor, StackingRegressor,GradientBoostingClassifier, RandomForestRegressor, GradientBoostingRegressor
from sklearn.svm import SVR, SVC
from sklearn.neighbors import KNeighborsClassifier, KNeighborsRegressor
from sklearn.naive_bayes import GaussianNB
from sklearn.impute import SimpleImputer
from sklearn.pipeline import Pipeline
import os
import joblib
import plotly.express as px
import plotly.graph_objects as go
import scipy.stats
# Streamlit App
st.title("Data Analysis & Model Building App")

st.sidebar.title("Menu")
st.sidebar.markdown("________________________")
if st.sidebar.button("clear old cache"):
    clear_data()

uploaded_file = st.sidebar.file_uploader("Upload your data file (CSV, Excel, JSON)", type=['csv', 'xlsx', 'xls', 'json'])
if uploaded_file:
    df = load_data(uploaded_file)
    if df is not None:
        # Store the DataFrame in session state
        if 'df' not in st.session_state:
            st.session_state.df = df

# External link data fetching
external_link = st.sidebar.text_input("Enter github dataset URL to fetch data:")
if external_link and st.sidebar.button("Fetch Data"):
    df = fetch_data_from_url(external_link)
    if df is not None:
        st.session_state.df = df

if 'df' in st.session_state:
# clean data download
        down = st.sidebar.toggle("Download Clean Data")
        if down:
            file_name = st.sidebar.text_input("Enter file name to save:", "cleaned_data.csv")
            if st.sidebar.button("Download Data"):
                download_clean_data(st.session_state.df, file_name)
                st.success(f"File saved as {file_name}!")
        st.sidebar.markdown("________________________")

        # Data prerview Section (Initially hidden)
        data_prerview_expander = st.sidebar.expander("Data Preview", expanded=False)
        if data_prerview_expander:
            prerview_option = st.sidebar.selectbox("Data Preview", [
                "Select Option",
                "Data Preview",
                "First 5 Rows",
                "Last 5 Rows",
                "10 Rows",
                "20 Rows",
                "50 Rows",
                "Sample Data",
               "EDA Report",
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
            elif prerview_option == "EDA Report":
               eda(st.session_state.df)
        

        # Data Overview Section (Initially hidden)
        data_overview_expander = st.sidebar.expander("Data Overview", expanded=False)
        if data_overview_expander:
            overview_option = st.sidebar.selectbox("Data Overview", [
                "Select Option",
                "Data info",
                "Statistics Summary",
                "Show Value Counts",
                "Cross Tabulation",
                "Correlation",
                "Skewness & Kurtosis",
                "Numerical & Categorical Columns",
            ])
            
            if overview_option == "Data info":
                st.write(st.session_state.df.head(2))
                summary_info, data_info = get_data_info(st.session_state.df)
                st.write("**DataFrame Summary**")
                st.write(summary_info)
                st.write("**Detailed Column Info**")
                st.write(data_info)
                # Display rows with missing values if any
                rows_with_missing_values = st.session_state.df[st.session_state.df.isnull().any(axis=1)]
                if not rows_with_missing_values.empty:
                    st.write("**Rows with Missing Values**")
                    st.write(rows_with_missing_values)
                    if st.button("Remove Missing Values"):
                        st.session_state.df, missing_values = remove_missing_values(st.session_state.df)
                        st.success("Rows with missing values removed!")
                        st.write(missing_values)
                        st.rerun()  # Refresh the page to reflect changes
                else:
                    st.write("**No rows with missing values found.**")

                # Display duplicate rows if any
                duplicate_rows = st.session_state.df[st.session_state.df.duplicated()]
                if not duplicate_rows.empty:
                    st.write("**Duplicate Rows**")
                    st.write(duplicate_rows)
                    if st.button("Remove Duplicates"):
                        st.session_state.df = remove_duplicates(st.session_state.df)  # Reassign the cleaned DataFrame back to `df`
                        st.success("Duplicate rows removed!")
                        st.write(check_duplicates(st.session_state.df))  # Display updated DataFrame
                        st.rerun()  # Refresh the page to reflect changes
                else:
                    st.write("**No duplicate rows found.**")

            # Handle Data Overview tasks
            elif overview_option == "Statistics Summary":
                statistics(st.session_state.df)
            
            elif overview_option == "Show Value Counts":
                column = st.selectbox("Select a column to show value counts:", st.session_state.df.columns)
                show_value_counts(st.session_state.df, column)
                    
            elif overview_option == "Numerical & Categorical Columns":
                column_type = st.radio("Select column type:", ['Numerical', 'Categorical'])
                st.write(f"### {column_type} Columns Data")
                st.write(show_column_data(st.session_state.df, column_type))
                st.write(column_selection(st.session_state.df))

            elif overview_option == "Cross Tabulation":
                col1 = st.selectbox("Select the first column for cross-tabulation:", st.session_state.df.columns)
                col2 = st.selectbox("Select the second column for cross-tabulation:", st.session_state.df.columns)
                cross_tabulation(st.session_state.df, col1, col2)

            elif overview_option == "Correlation":
                st.write("### Correlation Matrix")
                corr_matrix = correlation(st.session_state.df)
                st.write(corr_matrix)
                st.write("### Heatmap")
                create_heatmap(st.session_state.df)
                st.write("### Observations")
                observations = correlation_observations(corr_matrix)
                for observation in observations:
                    st.write(observation)

            elif overview_option == "Skewness & Kurtosis":
                st.write("### Skewness & Kurtosis")
                st.write(skewness_kurtosis(st.session_state.df))
                plot_skewness_kurtosis(st.session_state.df)

        # Data Cleaning Section (Initially hidden)
        data_cleaning_expander = st.sidebar.expander("Data Cleaning", expanded=False)
        if data_cleaning_expander:
            cleaning_option = st.sidebar.selectbox("Data Cleaning", [
                "Select Option",
                "Change Data Type",
                "Fill Missing Values",
                "Drop Columns",
                "Replace Values",
                "Clean Categorical Text",
                "Encode Categorical Columns",
            ])  
            
            # Handle Data Cleaning tasks
            if cleaning_option == "Change Data Type":
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

                replace_option = st.radio("Do you want to:", ("Select values to replace", "Select values not to replace", "Use custom text input"))

                if replace_option == "Select values to replace":
                    all_values = list(st.session_state.df[column].unique())
                    selected_values = st.multiselect("Select the values you want to replace:", all_values, [])
                    values_to_replace = selected_values
                elif replace_option == "Select values not to replace":
                    all_values = list(st.session_state.df[column].unique())
                    not_selected_values = st.multiselect("Select the values you do not want to replace:", all_values, [])
                    values_to_replace = list(set(all_values) - set(not_selected_values))
                elif replace_option == "Use custom text input":
                    if column_dtype == 'object':
                        old_value = st.text_input("Enter the value to replace:", "old_value")
                        values_to_replace = [old_value]
                    elif column_dtype in ['int64', 'float64']:
                        old_value = st.number_input("Enter the value to replace:", value=0)
                        values_to_replace = [old_value]
                    else:
                        st.warning("Unsupported data type for replacement.")
                        values_to_replace = None

                if column_dtype == 'object':
                    new_value = st.text_input("Enter the new value:", "new_value")
                elif column_dtype in ['int64', 'float64']:
                    new_value = st.number_input("Enter the new value:", value=0)
                else:
                    st.warning("Unsupported data type for replacement.")
                    new_value = None

                if st.button("Replace Values") and values_to_replace is not None and new_value is not None:
                    # Trim whitespace from the column values if dtype is object
                    if column_dtype == 'object':
                        st.session_state.df[column] = st.session_state.df[column].str.strip()

                    # Replace selected values
                    if any(val in st.session_state.df[column].values for val in values_to_replace):
                        st.session_state.df[column] = st.session_state.df[column].replace(dict.fromkeys(values_to_replace, new_value))
                        st.success("Values replaced successfully!")
                        st.write(st.session_state.df[column].head())
                    else:
                        st.warning(f"The values {values_to_replace} do not exist in the selected column.")


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

# visualisation
        graph_option = st.sidebar.toggle("Switch to Plotly")
        if graph_option:
            visual_expander = st.sidebar.expander("Visualization", expanded=False)       
            Visual_option = st.sidebar.selectbox("Select a task for Visualization", [
                "Select Option","Pair Plot", "Bar Plot", "Correlation Heatmap", "Scatter Plot", "Histogram","Line Chart",
                "Pie Chart","Box Plot","Count Plot","KDE Plot","Skewness & Kurtosis",
            ])
            if Visual_option == "Pair Plot":
                st.header("Pair Plot")
                create_pairplot(st.session_state.df)

            elif Visual_option == "Bar Plot":
                st.header("Bar Plot")
                create_bar_plot(st.session_state.df)

            elif Visual_option == "Correlation Heatmap":
                st.header("Correlation Heatmap")
                create_heatmap(st.session_state.df)

            elif Visual_option == "Scatter Plot":
                st.header("Scatter Plot")
                create_scatter(st.session_state.df)

            elif Visual_option == "Histogram":
                st.header("Histogram")
                st.write(create_histogram(st.session_state.df))

            elif Visual_option == "Line Chart":
                st.header("Line Chart")
                create_line_plot(st.session_state.df)

            elif Visual_option == "Pie Chart":
                st.header("Pie Chart")
                create_pie_chart(st.session_state.df)

            elif Visual_option == "Box Plot":
                st.header("Box Plot")
                create_boxplot(st.session_state.df)

            elif Visual_option == "Count Plot":
                st.header("Count Plot")
                create_count_plot(st.session_state.df)

            elif Visual_option == "KDE Plot":
                st.header("KDE Plot")
                create_kde_plot(st.session_state.df)

        else:


            Visual_option = st.sidebar.selectbox("Select a task for Visualization", [
                    "Select Option","Pair Plot", "Bar Plot", "Correlation Heatmap", "Scatter Plot", "Histogram","Line Chart",
                    "Pie Chart","Box Plot", "Count Plot", "Distribution Plot", 
                ])
            if Visual_option == "Pair Plot":
                st.header("Pair Plot")
                st.write(mat_create_pairplot(st.session_state.df))

            elif Visual_option == "Bar Plot":
                st.header("Bar Plot")
                fig = mat_create_bar_plot(st.session_state.df)
                st.pyplot(fig)

            elif Visual_option == "Correlation Heatmap":
                fig = mat_create_heatmap(st.session_state.df)
                if fig is not None:
                    st.pyplot(fig)

            elif Visual_option == "Scatter Plot":
                st.write(mat_create_scatter(st.session_state.df))

            elif Visual_option == "Histogram":
                st.header("Histogram")
                st.write(mat_create_histplot(st.session_state.df))

            elif Visual_option == "Line Chart":
                st.header("Line Chart")
                st.write(mat_create_line_plot(st.session_state.df))

            elif Visual_option == "Pie Chart":
                st.header("Pie Chart")
                st.write(mat_create_pie_chart(st.session_state.df))

            elif Visual_option == "Box Plot":
                st.header("Box Plot")
                st.write(mat_create_boxplot(st.session_state.df))

            elif Visual_option == "Count Plot":
                st.header("Count Plot")
                st.write(mat_create_count_plot(st.session_state.df))

            elif Visual_option == "Distribution Plot":
                st.header("Distribution Plot")
                st.write(mat_create_kde_plot(st.session_state.df))


# model building

        Model_Building_expander = st.sidebar.expander("Model Building", expanded=False)
        if Model_Building_expander:
            Model_Building_option = st.sidebar.selectbox("Machine Learning & Testing", [
                "Select Option","Model Building", "Test Model"
            ])  
            
            # Handle Data Cleaning tasks
            if Model_Building_option == "Model Building":
                model_type = st.selectbox("Choose Model Type", ["Regression", "Classification"])
                X_train, X_test, y_train, y_test = split_data(st.session_state.df, model_type)
    
                if X_train is not None:
                    col_trans = column_transformation(X_train)

                    if col_trans is not None:
                        st.success("Column transformation successful. Proceed to model selection.")
                        
                        if model_type == "Regression":
                            regression_model(X_train, X_test, y_train, y_test, col_trans)
                        else:
                            classification_models(X_train, X_test, y_train, y_test, col_trans)
            

            if Model_Building_option == "Test Model":
                st.title("Test Your Existing Model Here")

                # Upload model file
                st.subheader("Upload Model File")
                model_file = st.file_uploader("Upload a trained model (Pickle or Joblib)", type=["pkl", "joblib"])

                pipe = None  # Initialize model variable

                if model_file is not None:
                    try:
                        # Load the model from Pickle or Joblib
                        if model_file.name.endswith(".pkl"):
                            pipe  = pickle.load(model_file)
                        else:
                            pipe  = joblib.load(model_file)

                        # Ensure it's a valid model with a predict method
                        if not hasattr(pipe ,"predict"):
                            st.error("Invalid model! The uploaded file does not have a 'predict' method.")
                            st.stop()

                        st.success("Model uploaded successfully! Now, upload a dataset for testing.")
                    except Exception as e:
                        st.error(f"Error loading model: {e}")
                        pipe = None  # Reset model

                # Upload dataset file
                st.subheader("Upload Dataset File")
                dataset_file = st.file_uploader("Upload a dataset file (CSV, Excel)", type=["csv", "xlsx"])

                if dataset_file is not None and pipe is not None:
                    try:
                        # Load dataset
                        if dataset_file.name.endswith(".csv"):
                            df = pd.read_csv(dataset_file)
                        else:
                            df = pd.read_excel(dataset_file)

                        st.success("Dataset uploaded successfully!")
                        st.dataframe(df.head())

                        # Select Target Column
                        target_column = st.selectbox("Select the Target Column (y) to Remove", df.columns)
                        X = df.drop(columns=[target_column])  # Features only
                        st.write("Feature Columns:")
                        st.dataframe(X.head())

                        # Ensure correct input format
                        st.subheader("Make a Prediction")
                        input_data = {}

                        for idx, col in enumerate(X.columns):
                            if X[col].dtype == 'object':  # Categorical input
                                input_data[col] = st.selectbox(f"Select value for {col}", options=X[col].unique(), key=f"select_{idx}")
                            else:  # Numerical input
                                input_data[col] = st.number_input(f"Enter value for {col}", value=float(X[col].mean()), key=f"num_{idx}")

                        # Convert input data to DataFrame
                        query = pd.DataFrame([input_data])

                        # Ensure feature count matches the model's expectation
                        if query.shape[1] != X.shape[1]:
                            st.error(f"Feature mismatch: Model expects {X.shape[1]} features but received {query.shape[1]}")
                            st.stop()

                        if st.button("Predict"):
                            try:
                                predicted = pipe.predict(query)
                                st.success(f"The predicted value is: {predicted[0]}")
                            except Exception as e:
                                st.error(f"Prediction error: {e}")

                    except Exception as e:
                        st.error(f"Error processing dataset: {e}")

st.sidebar.markdown("________________________")

# about page
if st.sidebar.button("About"):
    about()
# Contact page
if st.sidebar.button("Contact"):
    contact()
    st.write("If you have any questions or feedback, please reach out to us.")




