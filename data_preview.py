import streamlit as st
import pandas as pd
from sklearn.preprocessing import LabelEncoder
import re
from skimpy import skim
import matplotlib.pyplot as plt
import seaborn as sns
from function import *

def data_preview(df):
    # Data Preview Section
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
