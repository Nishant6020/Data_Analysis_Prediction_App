import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from function import *
import pygwalker as pyg


# Function to display PyGWalker visualization
def pygwalker_visualization(df):
    st.subheader("Interactive Visualization with PyGWalker")
    pyg_html = pyg.walk(st.session_state.df, return_html=True)
    st.components.v1.html(pyg_html, height=800, scrolling=True)
