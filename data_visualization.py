import streamlit as st
import pygwalker as pyg

# Function to display PyGWalker visualization
def pygwalker_visualization(df):
    st.subheader("Interactive Visualization with PyGWalker")
    
    # Check if the DataFrame is not empty
    if df is not None and not df.empty:
        st.write("DataFrame shape:", df.shape)  # Debugging line
        st.write("DataFrame sample:", df.head())  # Debugging line
        
        try:
            pyg_html = pyg.walk(df, return_html=True)
            st.components.v1.html(pyg_html, height=800, scrolling=True)
        except Exception as e:
            st.error(f"An error occurred while generating the visualization: {e}")
    else:
        st.error("The DataFrame is empty or not valid for visualization.")
