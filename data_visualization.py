from function import *
import streamlit as st
import pygwalker as pyg

# Function to display PyGWalker visualization
def pygwalker_visualization(df):
    st.subheader("Interactive Visualization with PyGWalker")
    
    # Check if the DataFrame is not empty
    if df is not None and not df.empty:
        pyg_html = pyg.walk(df, return_html=True)
        st.components.v1.html(pyg_html, height=800, scrolling=True)
    else:
        st.error("The DataFrame is empty or not valid for visualization.")
