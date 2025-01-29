import streamlit as st

def home():
   
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
