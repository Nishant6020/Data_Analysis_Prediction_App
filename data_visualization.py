import streamlit as st
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go

# visualization function

def create_pairplot(df):
    kind = st.selectbox("Select type of kind", ['scatter', 'hist', 'reg'])
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    palette = st.selectbox("Select Color Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    
    if kind == 'scatter':
        fig = px.scatter_matrix(df, color=hue, color_discrete_sequence=px.colors.qualitative.Plotly)
    elif kind == 'hist':
        fig = px.histogram(df, x=df.columns, color=hue, color_discrete_sequence=px.colors.qualitative.Plotly)
    elif kind == 'reg':
        fig = px.scatter_matrix(df, color=hue, color_discrete_sequence=px.colors.qualitative.Plotly, trendline='ols')
    
    st.plotly_chart(fig)

def create_bar_plot(df):
    if df.empty:
        st.write("The DataFrame is empty. Please provide data to plot.")
        return None
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    x_column = st.selectbox('Select the X-axis column', df.columns)
    y_column = st.selectbox('Select the Y-axis column', df.columns)
    hue = st.selectbox('Select Hue', [None] + cat_columns)
    add_avg_line = st.checkbox('Add Average Line', value=False)

    title = st.text_input("Enter title for the Bar Plot", value=f"{x_column} vs {y_column} Bar Plot") 
    # Create the bar plot
    if hue:
        fig = px.bar(df, x=x_column, y=y_column, color=hue,)
    else:
        fig = px.bar(df, x=x_column, y=y_column)
    # Add average line if checked
    if add_avg_line and y_column in num_columns:
        avg_y = df[y_column].mean()
        fig.add_hline(y=avg_y, line_dash="dash", line_color="red", annotation_text="Average", annotation_position="bottom right")

    # Add average lines if checked
    if add_avg_line:
        avg_x = df[x_column].mean()
        avg_y = df[y_column].mean()

        # Add horizontal average line
        fig.add_hline(y=avg_y, line_dash="dash", line_color="gold", annotation_text="Avg Y", annotation_position="bottom right")
        
        # Add vertical average line
        fig.add_vline(x=avg_x, line_dash="dash", line_color="gold", annotation_text="Avg X", annotation_position="top right")
        
        # Add annotation for the average values
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.05, y=avg_y,
            text=f"Avg Y: {avg_y:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="yellow")
        )
        
        fig.add_annotation(
            xref="x", yref="paper",
            x=avg_x, y=1.05,
            text=f"Avg X: {avg_x:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=0,
            font=dict(color="yellow")
        )

    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_heatmap(df):
    title = st.text_input("Enter title for the heatmap", value="Correlation Heatmap")
    palette = st.selectbox("Select Color Palette", ['Viridis', 'Cividis', 'Blues', 'Reds', 'Greens', 'Oranges', 'Purples', 'Greys', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'BuPu', 'GnBu', 'PuBu', 'PuBuGn', 'PuRd', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'Reds', 'YlGnBu', 'YlGn', 'Blues', 'Purples', 'Greens', 'Oranges', 'YlOrBr', 'YlOrRd', 'OrRd', 'PuRd', 'RdPu', 'Reds', 'Greys', 'YlGnBu', 'YlGn', 'Blues', 'Purples', 'Greens'])
    
    numeric_df = df.select_dtypes(include=['number'])
    correlation_matrix = numeric_df.corr()
    fig = px.imshow(correlation_matrix, text_auto=True, color_continuous_scale=palette)
    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_scatter(df):
    x_col = st.selectbox('X-axis:', df.select_dtypes(include=['number']).columns.tolist())
    y_col = st.selectbox('Y-axis:', df.select_dtypes(include=['number']).columns.tolist())
    hue_col = st.selectbox('Hue:', [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    style_col = st.selectbox('Style:', [None] + list(df.columns))
    size_col = st.selectbox('Size:', [None] + list(df.select_dtypes(include=['number']).columns.tolist()))
    title = st.text_input("Enter title for the scatter plot", value=f"{x_col} vs {y_col} scatter Plot")
    add_reg_line = st.checkbox('Add Regression Line', value=False)
    add_avg_line = st.checkbox('Add Average Line', value=False)
    # Create the scatter plot
    if hue_col:
        fig = px.scatter(df, x=x_col, y=y_col, color=hue_col, hover_name=style_col, size=size_col)
    else:
        fig = px.scatter(df, x=x_col, y=y_col)
    # Add regression line if checked
    if add_reg_line:
        # Fit a linear regression model
        slope, intercept = np.polyfit(df[x_col], df[y_col], 1)
        x_vals = np.array([df[x_col].min(), df[x_col].max()])
        y_vals = slope * x_vals + intercept
        
        # Determine the color of the regression line
        if slope > 0:
            reg_color = "green"
        elif slope < 0:
            reg_color = "red"
        else:
            reg_color = "white"    
        fig.add_trace(go.Scatter(x=x_vals, y=y_vals, mode='lines', line=dict(color=reg_color, width=2), name='Regression Line'))

    # Add average lines if checked
    if add_avg_line:
        avg_x = df[x_col].mean()
        avg_y = df[y_col].mean()
        # Add horizontal average line
        fig.add_hline(y=avg_y, line_dash="dash", line_color="gold", annotation_text="Avg Y", annotation_position="bottom right")  
        # Add vertical average line
        fig.add_vline(x=avg_x, line_dash="dash", line_color="gold", annotation_text="Avg X", annotation_position="top right")   
        # Add annotation for the average values
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.05, y=avg_y,
            text=f"Avg Y: {avg_y:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="gold")
        )
        
        fig.add_annotation(
            xref="x", yref="paper",
            x=avg_x, y=1.05,
            text=f"Avg X: {avg_x:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=0,
            font=dict(color="gold")
        )
    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_histogram(df):
    x = st.selectbox("Select columns", df.select_dtypes(include=['number']).columns.tolist())   
    hue = st.selectbox("Select Column for Color (Optional)", ['None'] + df.columns.tolist())
    stat = st.selectbox("Select Stat", ['count', 'percent', 'probability', 'density', 'probability density'])
    barmode = st.selectbox("Select Bar Mode", ['stack', 'group', 'overlay', 'relative'])
    nbins = st.slider("Number of Bins", min_value=1, max_value=100, value=10)
    cumulative = st.checkbox("Cumulative", value=False)
    title = st.text_input("Enter title for the histogram plot", value=f"{x} Histogram Plot")
    # Create the histogram
    if hue == 'None':
        fig = px.histogram(df, x=x, 
                           histnorm=stat if stat != 'count' else None,
                           nbins=nbins, 
                           barmode=barmode,
                           text_auto=True)
    else:
        fig = px.histogram(df, x=x, color=hue,
                           histnorm=stat if stat != 'count' else None,
                           nbins=nbins, 
                           barmode=barmode,
                           text_auto=True)
    if cumulative:
        fig.update_traces(cumulative_enabled=True)

    avg_value = df[x].mean()
    fig.add_shape(type='line', x0=avg_value, x1=avg_value, 
                   y0=0, y1=1, yref='paper', 
                   line=dict(color='yellow', width=2, dash='dash'),
                   name='Average')
    # Add annotation for average value
    fig.add_annotation(x=avg_value, y=1, yref='paper', 
                       text=f'Average: {avg_value:.2f}', 
                       showarrow=True, 
                       arrowhead=2, ax=0, ay=-40,
                       font=dict(color='red'))
    fig.update_layout(title=title,
                      xaxis_title=x,
                      yaxis_title='Count' if stat == 'count' else 'Density',
                      showlegend=True)
    return fig

def create_line_plot(df):
    # Select X and Y variables
    x = st.selectbox("Select X column", df.select_dtypes(include=['number']).columns.tolist())
    y = st.selectbox("Select Y column", df.select_dtypes(include=['number']).columns.tolist())
    
    # Select optional parameters
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    
    # Line and marker options
    markers = st.checkbox("Show Markers", value=False)
    
    # Average line option
    add_avg_line = st.checkbox("Add Average Line", value=False)
    
    # Title input
    title = st.text_input("Enter title for the line plot", value=f"{x} vs {y} Line Plot")
    
    # Create the line plot
    if hue:
        fig = px.line(df, x=x, y=y, color=hue, markers=markers)
    else:
        fig = px.line(df, x=x, y=y, markers=markers)
    
    # Add average lines if checked
    if add_avg_line:
        avg_y = df[y].mean()
        avg_x = df[x].mean()
        
        # Add average line for Y
        fig.add_hline(y=avg_y, line_dash="dash", line_color="gold", annotation_text="Avg Y", annotation_position="bottom right")
        
        # Add average line for X
        fig.add_vline(x=avg_x, line_dash="dash", line_color="gold", annotation_text="Avg X", annotation_position="top right")
        
        # Add annotation for the average values
        fig.add_annotation(
            xref="paper", yref="y",
            x=1.05, y=avg_y,
            text=f"Avg Y: {avg_y:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=-40,
            font=dict(color="gold")
        )
        
        fig.add_annotation(
            xref="x", yref="paper",
            x=avg_x, y=1.05,
            text=f"Avg X: {avg_x:.2f}",
            showarrow=True,
            arrowhead=2,
            ax=0,
            ay=0,
            font=dict(color="gold")
        )

    # Update layout
    fig.update_layout(title=title)
    
    # Display the plot
    st.plotly_chart(fig)

def create_pie_chart(df):
    # Select categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_columns:
        st.warning("No categorical columns available in the DataFrame.")
        return
    category_col = st.selectbox("Select Category Variable", cat_columns)
    if category_col:
        # Select the category to highlight
        highlight_category = st.selectbox("Select Category to Highlight", df[category_col].unique())
        # Count the occurrences of each category
        data_counts = df[category_col].value_counts(ascending = False)
        title = st.text_input("Enter title for the pie chart", value=f"{category_col} Pie Chart")
        # Create a pull list to highlight the selected category
        pull = [0.1 if name == highlight_category else 0 for name in data_counts.index]
        # Create the pie chart using go.Figure
        fig = go.Figure(data=[go.Pie(labels=data_counts.index, values=data_counts.values, pull=pull)])
        # Update layout with the title
        fig.update_layout(title=title)
        # Display the plot
        st.plotly_chart(fig)
    else:
        st.warning("Please select a valid categorical variable.")

def create_boxplot(df):
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['number']).columns.tolist()

    x_col = st.selectbox("Select X column", num_columns + cat_columns)
    y_col = st.selectbox("Select Y column", [None]+ num_columns + cat_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    title = st.text_input("Enter title for the box plot", value=f"{x_col} vs {y_col} box plot")
    
    if hue:
        fig = px.box(df, x=x_col, y=y_col, color=hue,)
    else:
        fig = px.box(df, x=x_col, y=y_col)

    fig.update_layout(title=title)
    st.plotly_chart(fig)

def create_count_plot(df):
    # Select categorical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    if not cat_columns:
        st.warning("No categorical columns available in the DataFrame.")
        return

    x = st.selectbox("Select X column", cat_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    stat = st.selectbox("Select Stat", ['count', 'percent', 'proportion', 'probability'])
    add_avg_line = st.checkbox('Add Average Line', value=False)
    title = st.text_input("Enter title for the count plot", value=f"{x} count plot")
    
    if x:
        fig = px.histogram(
            df,
            x=x,
            color=hue,
            barmode='group',
            histnorm=None if stat == 'count' else stat,
        )
        # Add average line if checked
        if add_avg_line:
            avg_x = df[x].value_counts().mean()
            fig.add_hline(y=avg_x, line_dash="dash", line_color="red", annotation_text="Avg", annotation_position="bottom right")

            
        fig.update_layout(title=title)
        st.plotly_chart(fig)
    else:
        st.warning("Please select a valid categorical variable.")

def create_kde_plot(df):
    # Identify numeric and categorical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Streamlit widgets for user input
    x = st.selectbox("Select X variable", [None] + num_columns )
    y = st.selectbox("Select Y variable", [None] + num_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    add_avg_line = st.checkbox('Add Average Line', value=False)
    fill = st.checkbox('Fill KDE', value=False)

    fig, ax = plt.subplots()
    plot = sns.kdeplot(data=df, x=x, y=y, hue=hue,fill=fill, palette=palette, ax=ax)
    plt.xticks(rotation=90)

    # Add average lines if selected and variables are numeric
    if add_avg_line:
        if x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.1f}')
            ax.text(avg_x, ax.get_ylim()[1], f'{avg_x:.1f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=10)
        
        if y in num_columns:
            avg_y = df[y].mean()
            ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.1f}')
            ax.text(0, avg_y, f'{avg_y:.1f}', color='red', ha='right', va='center', 
                    transform=ax.get_yaxis_transform(), fontsize=10)
    
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    st.pyplot(fig)

# seaborn graphs
def mat_create_pairplot(df):
    cat_column = df.select_dtypes(include=['object'])
    columns = cat_column.columns.tolist()
    kind = st.selectbox("Select type of kind", ['scatter', 'kde', 'hist', 'reg'])
    hue = st.selectbox("Select Hue (categorical variable)", [None] + columns)
    palette = st.selectbox("Select Color Palette", ['bright', 'tab10','rocket', 'viridis','icefire','Paired',"Set2"])
    sns.pairplot(df, hue=hue, palette=palette, kind=kind)
    plt.legend(bbox_to_anchor=(1.05, 0.5), loc='upper left')
    plt.grid(True)
    plt.tight_layout()
    st.pyplot(plt)

def mat_create_bar_plot(df):
    if df.empty:
        st.write("The DataFrame is empty. Please provide data to plot.")
        return None
    # Select numeric and date columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    date_columns = df.select_dtypes(include=['datetime']).columns.tolist()

    # Convert year columns in date_columns to numerical
    for col in date_columns:
        if (df[col].dt.year == df[col].dt.year).all():
            df[col] = df[col].dt.year
            num_columns.append(col)
            date_columns.remove(col)

    # Select x and y columns
    x_column = st.selectbox('Select X column', df.columns)
    y_column = st.selectbox('Select Y column', df.columns)
    plot_type = st.selectbox('Select plot type', ['Vertical', 'Horizontal'])
    title = st.text_input("Enter title for the Bar Plot", value=f"{x_column} vs {y_column} Bar Plot")
    hue = st.selectbox('Select Hue', [None] + list(df.columns))
    palette = st.selectbox("Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    add_avg_line = st.checkbox('Add Average Line', value=False)

    # Validate selected columns
    if x_column not in df.columns or y_column not in df.columns:
        st.write("Please select valid columns for X and Y axes.")
        return None

    if plot_type == 'Vertical':
        fig, ax = plt.subplots()
        vbar_plot = sns.barplot(x=x_column, y=y_column, data=df, hue=hue, ax=ax, palette=palette)
        plt.title(title)
        plt.tight_layout()
        plt.xticks(rotation=90)

        if df[x_column].nunique() < 10:
            for i in vbar_plot.containers:
                vbar_plot.bar_label(i, label_type="center", rotation=90,padding=3)

        if add_avg_line and y_column in num_columns:
            avg_value = df[y_column].mean()
            ax.axhline(avg_value, color='red', linewidth=1, linestyle='-', label=f'Average: {avg_value:.2f}')
            ax.legend()
            ax.text(0.95, avg_value, f'{avg_value:.2f}', color='red', ha='right', va='center', transform=ax.get_yaxis_transform())
        return fig

    elif plot_type == 'Horizontal':
        fig, ax = plt.subplots()
        hbar_plot = sns.barplot(x=y_column, y=x_column, data=df, hue=hue, ax=ax, palette=palette, orient='h')
        plt.title(title)
        plt.tight_layout()
        plt.xticks(rotation=90)
        plt.yticks(rotation=0)
        if df[x_column].nunique() < 40:
            for i in hbar_plot.containers:
                hbar_plot.bar_label(i, label_type="center", rotation=0,padding=3)

        if add_avg_line and x_column in num_columns:
            avg_value = df[x_column].mean()
            ax.axvline(avg_value, color='red', linewidth=1, linestyle='-', label=f'Average: {avg_value:.2f}')
            ax.legend()
            ax.text(avg_value, 0.95, f'{avg_value:.2f}', color='red', ha='center', va='top', transform=ax.get_xaxis_transform())
        return fig

    else:
        st.write("Unsupported plot type. Please choose from 'Vertical' or 'Horizontal'.")
        return None

def mat_create_heatmap(df):
    title = st.text_input("Enter title for the heatmap", value="Correlation Heatmap")
    # Filter the DataFrame to include only numeric columns
    numeric_df = df.select_dtypes(include=['number'])
    # Calculate the correlation matrix
    correlation_matrix = numeric_df.corr()
    # Create the heatmap
    fig, ax = plt.subplots()
    sns.heatmap(correlation_matrix, annot=True, fmt=".2f", cmap='coolwarm', ax=ax, square=True, cbar_kws={"shrink": .8})
    plt.title(title)
    plt.tight_layout()
    return fig

def mat_create_scatter(df):   
    # Define options for user input
    def get_unique_values(column):
        return sorted(df[column].unique().tolist())
    x_col = st.selectbox('X-axis:', df.select_dtypes(include=['number']))
    y_col = st.selectbox('Y-axis:', df.select_dtypes(include=['number']))
    hue_col = st.selectbox('Hue:', [None] + df.select_dtypes(include=['object', 'category']).columns.tolist(), format_func=lambda x: 'None' if x is None else x)
    style_col = st.selectbox('Style:', [None] + list(df.columns), format_func=lambda x: 'None' if x is None else x)
    size_col = st.selectbox('Size:', [None] + list(df.columns), format_func=lambda x: 'None' if x is None else x)
    title = st.text_input("Enter title for the scatter plot", value=f"{x_col} vs {y_col} scatter Plot")
    
    # Check if both x and y columns are numerical
    x_is_numeric = pd.api.types.is_numeric_dtype(df[x_col])
    y_is_numeric = pd.api.types.is_numeric_dtype(df[y_col])
    
    # Checkboxes for regression and average line
    add_reg_line = st.checkbox('Add Regression Line', disabled=not (x_is_numeric and y_is_numeric))
    add_avg_line = st.checkbox('Add Average Line')
    
    # Sliders for X and Y axes if columns are numerical
    if x_is_numeric:
        x_min, x_max = int(df[x_col].min()), int(df[x_col].max())
        x_range = st.slider('X-axis range', min_value=x_min, max_value=x_max, value=(x_min, x_max))
    else:
        x_range = (None, None)
    
    if y_is_numeric:
        y_min, y_max = int(df[y_col].min()), int(df[y_col].max())
        y_range = st.slider('Y-axis range', min_value=y_min, max_value=y_max, value=(y_min, y_max))
    else:
        y_range = (None, None)
    
    # Filter data based on slider range if columns are numerical
    if x_is_numeric:
        df = df[(df[x_col] >= x_range[0]) & (df[x_col] <= x_range[1])]
    if y_is_numeric:
        df = df[(df[y_col] >= y_range[0]) & (df[y_col] <= y_range[1])]
    
    # Create scatter plot
    plt.figure(figsize=(10, 6))
    scatter = sns.scatterplot(
        x=x_col, y=y_col, data=df,
        hue=hue_col if hue_col is not None else None,
        style=style_col if style_col is not None else None,
        size=size_col if size_col is not None else None
    )
    
    # Add regression line if checked and columns are numerical
    if add_reg_line and x_is_numeric and y_is_numeric:
        sns.regplot(x=x_col, y=y_col, data=df, scatter=False, ax=scatter)
    
    # Add average line if checked
    if add_avg_line and y_is_numeric:
        mean_y = df[y_col].mean()
        plt.axhline(y=mean_y, color='red', linestyle='--', label=f'Average {y_col}')
    
    # Add labels and title
    plt.title(title)
    plt.xlabel(x_col)
    plt.ylabel(y_col)
    plt.xticks(rotation=90)
    
    # Adjust layout to prevent overlap
    plt.subplots_adjust(left=0.1, right=0.9, top=0.85, bottom=0.2)
    
    # Add legend
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    
    # Add colorbar if color column is selected and position it at the bottom
    if hue_col:
        colorbar = plt.colorbar(scatter.collections[0], ax=scatter, orientation='vertical', pad=0.21)
        colorbar.set_label(hue_col, labelpad=10)
    
    plt.grid(True)
    plt.tight_layout()  # Automatically adjust subplot parameters for a better fit
    st.pyplot(plt)

def mat_create_histplot(df):
    # Select numerical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()   
    x = st.selectbox("Select Numerical Variable", num_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    palette = st.selectbox("Select Color Palette", ['deep', 'muted', 'bright', 'dark', 'colorblind'])
    stat = st.selectbox("Select Stat", ['count', 'frequency', 'density', 'probability'])
    multiple = st.selectbox("Multiple", ['layer', 'dodge', 'stack', 'fill'])
    element = st.selectbox("Element", ['bars', 'step'])
    palette = st.selectbox("Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    bins = st.slider("Number of Bins", min_value=1, max_value=100, value=10)
    shrink = st.slider("Adjust Bar Size", min_value=0.1, max_value=1.0, step=0.1, value=1.0)
    fill = st.checkbox("Fill", value=True)
    cumulative = st.checkbox("Cumulative")
    kde = st.checkbox("KDE")
    add_avg_line = st.checkbox('Add Average Line', value=False)
    # Plot the histogram
    if x:
        fig, ax = plt.subplots()
        sns.histplot(data=df, x=x, hue=hue, stat=stat, bins=bins, multiple=multiple, cumulative=cumulative, element=element, fill=fill, shrink=shrink, kde=kde, palette=palette, ax=ax)      
        if add_avg_line and x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.1f}')
            ax.text(avg_x, 1, f'{avg_x:.1f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=10)
        plt.show()
        st.pyplot(fig)

    else:
        st.warning("Please select a valid numerical variable.")

def mat_create_line_plot(df):
    for col in df.select_dtypes(include=['datetime']):
        if df[col].dt.year.nunique() > 1:
            df[col] = df[col].dt.year
        elif df[col].dt.month.nunique() > 1:
            df[col] = df[col].dt.month
        elif df[col].dt.day.nunique() > 1:
            df[col] = df[col].dt.day
    
    # Identify numeric and categorical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    # Streamlit widgets for user input
    x = st.selectbox("Select X variable", [None] + num_columns + cat_columns)
    y = st.selectbox("Select Y variable", [None] + cat_columns)
    hue = st.selectbox("Select Hue (categorical)", [None] + cat_columns)
    size = st.selectbox("Select Size (numeric)", [None] + num_columns)
    style = st.selectbox("Select Style (categorical)", [None] + cat_columns)
    markers = st.checkbox("Show Markers", value=False)
    dashes = st.checkbox("Show Dashes", value=True)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    estimator = st.selectbox("Select Estimator", ['mean', 'median', 'sum', 'min', 'max'])
    add_avg_line = st.checkbox("Add Average Line", value=False)
    markers = st.selectbox("Select Marker points",['.', ',', 'o', 'v', '^', '<', '>', '1', '2', '3', '4', 's', 'p', '*', 'h', 'H', '+', 'x', 'D', 'd', '|', '_'])
    marker_size = st.slider('Marker Size', min_value=1, max_value=100, value=14)
    title = st.text_input("Enter title for the line plot", value=f"{x} vs {y} line Plot")
    # Validate selected inputs
    if not x or not y:
        st.warning("Please select valid X and Y variables.")
        return
    # Create the figure
    fig, ax = plt.subplots(figsize=(25, 15))
    try:
        sns.lineplot(data=df, x=x, y=y, hue=hue, size=size, style=style,
                     markers=markers, dashes=dashes, palette=palette, 
                     estimator=estimator,marker=markers,markersize=marker_size, ax=ax)
        ax.set_xlabel(x, fontsize=40)
        ax.set_ylabel(y, fontsize=40)
        ax.tick_params(axis='x', labelrotation=90, labelsize=40)
        ax.tick_params(axis='y', labelsize=40)
        plt.tight_layout()

        # Add average lines if requested
        if add_avg_line and y in num_columns:
            avg_y = df[y].mean()
            ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.2f}')
            ax.text(1, avg_y, f'{avg_y:.2f}', color='red', ha='left', va='center', 
                    transform=ax.get_yaxis_transform(), fontsize=40)
        
        if add_avg_line and x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.2f}')
            ax.text(avg_x, 1, f'{avg_x:.2f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=40)
        
        # Handle legend positioning if hue or avg line exists
        if hue or add_avg_line:
            ax.legend(bbox_to_anchor=(1.05, 1), loc='upper left', fontsize=40)
            plt.subplots_adjust(right=0.8)
        
        plt.tight_layout()
        plt.title(title,fontsize=50,fontweight='bold')
        st.pyplot(fig)
        
    except Exception as e:
        st.error(f"Error creating line plot: {e}")

def mat_create_pie_chart(df):
    # Select categorical column
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    category_col = st.selectbox("Select Category Variable", cat_columns)
    
    # Color palette selection
    palette = st.selectbox("Select Color Palette", 
                           ['Accent', 'Accent_r','Dark2', 'Dark2_r', 'Paired', 'Paired_r', 'Pastel1', 'Pastel1_r', 'Pastel2', 'Pastel2_r', 'Set1', 'Set1_r', 'Set2', 'Set2_r', 'Set3', 'Set3_r', 'tab10', 'tab10_r', 'tab20', 'tab20_r', 'tab20b', 'tab20b_r', 'tab20c', 'tab20c_r'])

    if category_col:
        data_counts = df[category_col].value_counts()
        
        # Plot the pie chart
        fig, ax = plt.subplots()
        ax.pie(
            data_counts, 
            labels=data_counts.index, 
            autopct='%1.1f%%', 
            startangle=140, 
            colors=plt.get_cmap(palette).colors[:len(data_counts)]
        )
        
        plt.show()
        st.pyplot(fig)
    else:
        st.warning("Please select a valid category variable.")

def mat_create_boxplot(df):
    # Select categorical and numerical columns
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['number']).columns.tolist()

    x_col = st.selectbox("Select X column", num_columns + cat_columns)
    y_col = st.selectbox("Select Y column", [None]+ num_columns + cat_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])

    # Plot the box plot
    if x_col or y_col:
        fig, ax = plt.subplots()
        sns.boxplot(data=df, x=x_col, y=y_col, hue=hue, palette=palette, ax=ax)
        plt.xticks(rotation=45)
        plt.show()
        st.pyplot(fig)
    else:
        st.warning("Please select valid X and Y variables.")

def mat_create_count_plot(df):
    x = st.selectbox("Select X column", df.select_dtypes(include=['object', 'category']).columns.tolist())
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    hue = st.selectbox("Select Hue (categorical variable)", [None] + df.select_dtypes(include=['object', 'category']).columns.tolist())
    palette = st.selectbox("Select Color Palette", ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    stat = st.selectbox("Select Stat", ['count', 'percent', 'proportion', 'probability'])
    add_avg_line = st.checkbox('Average Line', value=False)
    fig, ax = plt.subplots()
    plt.figure(figsize=(20,20))
    plot = sns.countplot(data=df, x=x, hue=hue, stat=stat, palette=palette, ax=ax)
    plt.xticks(rotation=90)
    if add_avg_line:
        if stat == 'count':
            avg_y = df[x].value_counts().mean()
        elif stat == 'percent':
            avg_y = (df[x].value_counts() / len(df)).mean() * 100
        elif stat == 'proportion':
            avg_y = (df[x].value_counts() / len(df)).mean()
        elif stat == 'probability':
            avg_y = (df[x].value_counts() / len(df)).mean()

        ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.1f}')
        ax.text(0, avg_y, f'{avg_y:.1f}', color='red', ha='right', va='center', transform=ax.get_yaxis_transform(), fontsize=10)

    # Add bar labels if hue variable has less than or equal to 10 unique values
    if hue is None or df[hue].nunique() <= 15:
        for container in plot.containers:
            plot.bar_label(container, label_type="edge", rotation=90,padding=3)
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    st.pyplot(fig)

def mat_create_kde_plot(df):
    # Identify numeric and categorical columns
    num_columns = df.select_dtypes(include=['number']).columns.tolist()
    cat_columns = df.select_dtypes(include=['object', 'category']).columns.tolist()

    # Streamlit widgets for user input
    x = st.selectbox("Select X variable", [None] + num_columns )
    y = st.selectbox("Select Y variable", [None] + num_columns)
    hue = st.selectbox("Select Hue (categorical variable)", [None] + cat_columns)
    palette = st.selectbox("Select Color Palette", 
                           ['bright', 'tab10', 'deep', 'muted', 'dark', 'Paired', 'Set2', 
                            'colorblind', 'rocket', 'viridis', 'icefire', 'Spectral'])
    add_avg_line = st.checkbox('Add Average Line', value=False)

    fig, ax = plt.subplots()
    plot = sns.kdeplot(data=df, x=x, y=y, hue=hue, palette=palette, ax=ax)
    plt.xticks(rotation=90)

    # Add average lines if selected and variables are numeric
    if add_avg_line:
        if x in num_columns:
            avg_x = df[x].mean()
            ax.axvline(avg_x, color='blue', linestyle='--', linewidth=1, label=f'Avg X: {avg_x:.1f}')
            ax.text(avg_x, ax.get_ylim()[1], f'{avg_x:.1f}', color='blue', ha='center', va='top', 
                    transform=ax.get_xaxis_transform(), fontsize=10)
        
        if y in num_columns:
            avg_y = df[y].mean()
            ax.axhline(avg_y, color='red', linestyle='--', linewidth=1, label=f'Avg Y: {avg_y:.1f}')
            ax.text(0, avg_y, f'{avg_y:.1f}', color='red', ha='right', va='center', 
                    transform=ax.get_yaxis_transform(), fontsize=10)
    
    plt.tight_layout()
    plt.legend(loc='upper right')
    plt.show()
    st.pyplot(fig)
