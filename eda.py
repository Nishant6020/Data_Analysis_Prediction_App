from ydata_profiling import ProfileReport
import streamlit as st
import streamlit.components.v1 as components

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
