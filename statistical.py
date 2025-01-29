# Import necessary libraries
import scipy.stats as stats
import numpy as np

# Add a function for statistical analysis
def statistical_analysis(df):
    st.subheader("Statistical Analysis Tests")

    st.write("Choose the test you want to perform:")

    test_options = [
        "ANOVA Test",
        "t-Test (Independent Samples)",
        "Chi-Square Test",
        "Hypothesis Test (Z-Test, T-Test, F-Test)",
        "Non-Parametric Test (Mann-Whitney U, Kruskal-Wallis)"
    ]

    selected_test = st.selectbox("Select a Test", test_options)

    if selected_test == "ANOVA Test":
        st.write("ANOVA Test: Compare means of multiple groups.")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_column = st.selectbox("Select the column to analyze:", numeric_columns)
        groups = st.text_area("Enter group names and values (e.g., Group1: 10,12,15; Group2: 11,13,14)").strip()

        if st.button("Perform ANOVA"):
            try:
                # Parse groups input
                group_data = {
                    group.split(":")[0].strip(): list(map(float, group.split(":")[1].split(",")))
                    for group in groups.split(";")
                }
                anova_result = stats.f_oneway(*group_data.values())
                st.write(f"ANOVA Result: F={anova_result.statistic}, p-value={anova_result.pvalue}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif selected_test == "t-Test (Independent Samples)":
        st.write("t-Test: Compare means of two groups.")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_column = st.selectbox("Select the column to analyze:", numeric_columns)
        group_column = st.selectbox("Select the column defining groups:", df.columns.tolist())

        if st.button("Perform t-Test"):
            try:
                group1, group2 = [df[df[group_column] == g][selected_column] for g in df[group_column].unique()[:2]]
                t_test_result = stats.ttest_ind(group1, group2)
                st.write(f"t-Test Result: t={t_test_result.statistic}, p-value={t_test_result.pvalue}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif selected_test == "Chi-Square Test":
        st.write("Chi-Square Test: Test for independence between two categorical variables.")
        categorical_columns = df.select_dtypes(include='object').columns.tolist()
        col1 = st.selectbox("Select the first column:", categorical_columns)
        col2 = st.selectbox("Select the second column:", categorical_columns)

        if st.button("Perform Chi-Square Test"):
            try:
                contingency_table = pd.crosstab(df[col1], df[col2])
                chi2_result = stats.chi2_contingency(contingency_table)
                st.write(f"Chi-Square Result: χ²={chi2_result[0]}, p-value={chi2_result[1]}")
            except Exception as e:
                st.error(f"Error: {e}")

    elif selected_test == "Hypothesis Test (Z-Test, T-Test, F-Test)":
        st.write("Hypothesis Testing: Perform Z-Test, T-Test, or F-Test based on the data.")
        st.write("This feature is currently being extended. Use the t-Test or ANOVA options.")

    elif selected_test == "Non-Parametric Test (Mann-Whitney U, Kruskal-Wallis)":
        st.write("Non-Parametric Tests: Use Mann-Whitney U or Kruskal-Wallis tests.")
        numeric_columns = df.select_dtypes(include=np.number).columns.tolist()
        selected_column = st.selectbox("Select the column to analyze:", numeric_columns)
        group_column = st.selectbox("Select the column defining groups:", df.columns.tolist())

        if st.button("Perform Non-Parametric Test"):
            try:
                group1, group2 = [df[df[group_column] == g][selected_column] for g in df[group_column].unique()[:2]]
                non_parametric_result = stats.mannwhitneyu(group1, group2)
                st.write(f"Mann-Whitney U Test Result: U={non_parametric_result.statistic}, p-value={non_parametric_result.pvalue}")
            except Exception as e:
                st.error(f"Error: {e}")

# Updated Streamlit code with new option
if select == "Statistical Analysis":
    with st.expander("Statistical Analysis", expanded=True):
        if "df" in st.session_state:
            statistical_analysis(st.session_state['df'])
        else:
            st.error("Please upload a file to perform statistical analysis.")
