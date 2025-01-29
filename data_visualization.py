import streamlit as st
import pygwalker as pyg

def pygwalker_visualization(df):
    st.subheader("Interactive Visualization with PyGWalker")
    
    if df is not None and not df.empty:
        st.write("DataFrame shape:", df.shape)
        st.write("DataFrame types:", df.dtypes)
        st.write("NaN values in DataFrame:", df.isnull().sum())
        
        try:
            pyg_html = pyg.walk(df, return_html=True)
            st.components.v1.html(pyg_html, height=800, scrolling=True)
        except Exception as e:
            st.error(f"An error occurred while generating the visualization: {e}")
    else:
        st.error("The DataFrame is empty or not valid for visualization.")
