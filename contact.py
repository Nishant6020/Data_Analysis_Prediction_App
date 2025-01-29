import streamlit as st

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
