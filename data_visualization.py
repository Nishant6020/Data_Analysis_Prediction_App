import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from function import *
import pygwalker as pyg
from pygwalker.api.streamlit import StreamlitRenderer


# Function to display PyGWalker visualization
def pygwalker_visualization(df):
    # Adjust the width of the Streamlit page
    st.set_page_config(
        page_title="Use Pygwalker In Streamlit",
        layout="wide")
    st.subheader("Interactive Visualization with PyGWalker")
    pyg_html = pyg.walk(st.session_state.df, return_html=True)
    st.components.v1.html(pyg_html, height=800, scrolling=True)


# def data_visualization(df):
    # Visualize Data button
    # tabs = ["Pair Plot", "Bar Plot", "Scatter Plot", "Histogram", "Box Plot", "Count Plot",
    #         "Violin Plot", "Line Chart", "Correlation Heatmap", "Distribution Plot", "Pie Chart"]
    
    # # Create tabs
    # tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10, tab11 = st.tabs(tabs)

    # # Tab content for Pair Plot
    # with tab1:
    #     st.header("Pair Plot")
    #     create_pairplot(df)

    # # Tab content for Bar Plot
    # with tab2:
    #     st.header("Bar Plot")
    #     create_bar_chart(df)

    # # Tab content for Scatter Plot
    # with tab3:
    #     st.header("Scatter Plot")
    #     create_scatter_plot(df)

    # # Tab content for Histogram
    # with tab4:
    #     st.header("Histogram")
    #     create_histogram(df)

    # # Tab content for Box Plot
    # with tab5:
    #     st.header("Box Plot")
    #     create_boxplot(df)

    # # Tab content for Count Plot
    # with tab6:
    #     st.header("Count Plot")
    #     create_count_plot(df)

    # # Tab content for Violin Plot
    # with tab7:
    #     st.header("Violin Plot")
    #     create_violin_plot(df)

    # # Tab content for Line Chart
    # with tab8:
    #     st.header("Line Chart")
    #     create_line_chart(df)

    # # Tab content for Heatmap (Correlation)
    # with tab9:
    #     st.header("Correlation Heatmap")
    #     create_heatmap(df)

    # # Tab content for Distribution Plot (optional)
    # with tab10:
    #     st.header("Distribution Plot")
    #     st.write("Distribution plot functionality can be implemented here!")

    # # Tab content for Pie Chart (optional)
    # with tab11:
    #     st.header("Pie Chart")
    #     st.write("Pie chart functionality can be implemented here!")
