"""
DATA CLEANING AND VISUALIZATION SYSTEM
========================================

INSTALLATION INSTRUCTIONS:
--------------------------
1. Install required packages:
   pip install streamlit pandas openpyxl matplotlib plotly

2. Run the application:
   streamlit run app.py

3. The app will open in your browser automatically

FEATURES:
---------
- Upload CSV or Excel files
- Automatic column type detection
- Interactive data cleaning
- Smart visualization suggestions
- Real-time data preview
"""

import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import io
import re

ILLEGAL_CHARACTERS_RE = re.compile(r'[\x00-\x08\x0B\x0C\x0E-\x1F]')

def remove_illegal_excel_chars(df):
    """
    Remove illegal control characters that break Excel (openpyxl).
    Applied only to object (string) columns.
    """
    df = df.copy()
    for col in df.select_dtypes(include=["object"]).columns:
        df[col] = (
            df[col]
            .astype(str)
            .apply(lambda x: ILLEGAL_CHARACTERS_RE.sub("", x))
        )
    return df


# ============================================================================
# HELPER FUNCTIONS - DATA TYPE DETECTION
# ============================================================================

def detect_column_types(df):
    """
    Automatically detect the type of each column in the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        
    Returns:
        dict: Dictionary mapping column names to their detected types
    """
    column_types = {}
    
    for col in df.columns:
        # Skip if all values are missing
        if df[col].isna().all():
            column_types[col] = 'empty'
            continue
        
        # Get non-null values for analysis
        non_null = df[col].dropna()
        
        # Check if numeric
        if pd.api.types.is_numeric_dtype(df[col]):
            column_types[col] = 'numeric'
        
        # Check if datetime
        elif pd.api.types.is_datetime64_any_dtype(df[col]):
            column_types[col] = 'datetime'
        
        # Try to parse as datetime if it's object type
        elif df[col].dtype == 'object':
            # Sample first few non-null values
            sample = non_null.head(min(100, len(non_null)))
            
            # Try datetime conversion
            try:
                pd.to_datetime(sample, errors='raise', infer_datetime_format=True)
                column_types[col] = 'datetime'
                continue
            except:
                pass
            
            # Check if categorical (low cardinality relative to size)
            unique_ratio = len(non_null.unique()) / len(non_null)
            
            if unique_ratio < 0.05 or len(non_null.unique()) < 20:
                column_types[col] = 'categorical'
            else:
                column_types[col] = 'text'
        else:
            column_types[col] = 'text'
    
    return column_types

# ============================================================================
# HELPER FUNCTIONS - DATA CLEANING
# ============================================================================

def clean_numeric(series):
    """
    Clean numeric column by filling missing values with median.
    
    Args:
        series (pd.Series): Input numeric series
        
    Returns:
        pd.Series: Cleaned series
    """
    if series.isna().all():
        return series
    
    median_value = series.median()
    cleaned = series.fillna(median_value)
    
    return cleaned

def clean_categorical(series):
    """
    Clean categorical/text column by:
    - Trimming whitespace
    - Standardizing case
    - Filling missing values with mode
    
    Args:
        series (pd.Series): Input categorical series
        
    Returns:
        pd.Series: Cleaned series
    """
    # Convert to string and strip whitespace
    cleaned = series.astype(str).str.strip()
    
    # Replace 'nan' string with actual NaN
    cleaned = cleaned.replace(['nan', 'NaN', 'None', ''], np.nan)
    
    # Standardize case (title case)
    cleaned = cleaned.str.title()
    
    # Fill missing values with mode if available
    if not cleaned.isna().all() and len(cleaned.dropna()) > 0:
        mode_value = cleaned.mode()[0] if len(cleaned.mode()) > 0 else 'Unknown'
        cleaned = cleaned.fillna(mode_value)
    
    return cleaned

def clean_datetime(series):
    """
    Clean datetime column by:
    - Parsing dates safely
    - Dropping invalid entries (replace with NaT)
    
    Args:
        series (pd.Series): Input datetime series
        
    Returns:
        pd.Series: Cleaned series with parsed dates
    """
    # Try to convert to datetime
    cleaned = pd.to_datetime(series, errors='coerce')
    
    return cleaned

def clean_text(series):
    """
    Clean text column by:
    - Trimming whitespace
    - Filling missing values with 'Unknown'
    
    Args:
        series (pd.Series): Input text series
        
    Returns:
        pd.Series: Cleaned series
    """
    # Convert to string and strip whitespace
    cleaned = series.astype(str).str.strip()
    
    # Replace 'nan' string with actual NaN
    cleaned = cleaned.replace(['nan', 'NaN', 'None', ''], np.nan)
    
    # Fill missing values
    cleaned = cleaned.fillna('Unknown')
    
    return cleaned

# ============================================================================
# HELPER FUNCTIONS - DATA ANALYSIS
# ============================================================================

def get_data_summary(df, column_types):
    """
    Generate a comprehensive summary of the dataframe.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_types (dict): Detected column types
        
    Returns:
        pd.DataFrame: Summary dataframe
    """
    summary_data = []
    
    for col in df.columns:
        summary_data.append({
            'Column Name': col,
            'Data Type': column_types.get(col, 'unknown'),
            'Missing Values': df[col].isna().sum(),
            'Missing %': f"{(df[col].isna().sum() / len(df) * 100):.1f}%",
            'Unique Values': df[col].nunique(),
            'Sample Value': str(df[col].dropna().iloc[0]) if len(df[col].dropna()) > 0 else 'N/A'
        })
    
    return pd.DataFrame(summary_data)

# ============================================================================
# HELPER FUNCTIONS - VISUALIZATION
# ============================================================================

def suggest_visualizations(df, column_types):
    """
    Suggest appropriate visualizations based on column types.
    
    Args:
        df (pd.DataFrame): Input dataframe
        column_types (dict): Detected column types
        
    Returns:
        dict: Dictionary of visualization suggestions
    """
    suggestions = {}
    
    numeric_cols = [col for col, dtype in column_types.items() if dtype == 'numeric']
    categorical_cols = [col for col, dtype in column_types.items() if dtype == 'categorical']
    datetime_cols = [col for col, dtype in column_types.items() if dtype == 'datetime']
    
    # Single numeric column visualizations
    for col in numeric_cols:
        suggestions[f"{col} - Histogram"] = {
            'type': 'histogram',
            'columns': [col],
            'description': f'Distribution of {col}'
        }
        suggestions[f"{col} - Box Plot"] = {
            'type': 'boxplot',
            'columns': [col],
            'description': f'Box plot showing outliers in {col}'
        }
    
    # Single categorical column visualizations
    for col in categorical_cols:
        suggestions[f"{col} - Bar Chart"] = {
            'type': 'bar',
            'columns': [col],
            'description': f'Frequency of categories in {col}'
        }
    
    # Numeric vs Numeric (scatter plots)
    if len(numeric_cols) >= 2:
        for i, col1 in enumerate(numeric_cols):
            for col2 in numeric_cols[i+1:]:
                suggestions[f"{col1} vs {col2} - Scatter"] = {
                    'type': 'scatter',
                    'columns': [col1, col2],
                    'description': f'Relationship between {col1} and {col2}'
                }
    
    # Date vs Numeric (line charts)
    if datetime_cols and numeric_cols:
        for date_col in datetime_cols:
            for num_col in numeric_cols:
                suggestions[f"{date_col} vs {num_col} - Line"] = {
                    'type': 'line',
                    'columns': [date_col, num_col],
                    'description': f'{num_col} over time'
                }
    
    return suggestions

def create_visualization(df, viz_type, columns):
    """
    Create a visualization based on type and columns.
    
    Args:
        df (pd.DataFrame): Input dataframe
        viz_type (str): Type of visualization
        columns (list): Columns to visualize
        
    Returns:
        plotly figure or matplotlib figure
    """
    if viz_type == 'histogram':
        fig = px.histogram(df, x=columns[0], 
                          title=f'Distribution of {columns[0]}',
                          labels={columns[0]: columns[0]},
                          color_discrete_sequence=['#1f77b4'])
        fig.update_layout(showlegend=False)
        return fig
    
    elif viz_type == 'boxplot':
        fig = px.box(df, y=columns[0], 
                     title=f'Box Plot of {columns[0]}',
                     labels={columns[0]: columns[0]},
                     color_discrete_sequence=['#ff7f0e'])
        return fig
    
    elif viz_type == 'bar':
        value_counts = df[columns[0]].value_counts().head(20)
        fig = px.bar(x=value_counts.index, y=value_counts.values,
                     title=f'Frequency of {columns[0]}',
                     labels={'x': columns[0], 'y': 'Count'},
                     color_discrete_sequence=['#2ca02c'])
        return fig
    
    elif viz_type == 'scatter':
        fig = px.scatter(df, x=columns[0], y=columns[1],
                        title=f'{columns[0]} vs {columns[1]}',
                        labels={columns[0]: columns[0], columns[1]: columns[1]},
                        color_discrete_sequence=['#d62728'])
        return fig
    
    elif viz_type == 'line':
        # Group by date and aggregate numeric column
        date_col, num_col = columns[0], columns[1]
        grouped = df.groupby(date_col)[num_col].mean().reset_index()
        grouped = grouped.sort_values(date_col)
        
        fig = px.line(grouped, x=date_col, y=num_col,
                     title=f'{num_col} over {date_col}',
                     labels={date_col: date_col, num_col: num_col},
                     color_discrete_sequence=['#9467bd'])
        return fig
    
    return None

# ============================================================================
# MAIN APPLICATION
# ============================================================================

def main():
    """Main application function"""
    
    # Page configuration
    st.set_page_config(
        page_title="Richard Data Cleaning & Visualization System",
        page_icon="📊",
        layout="wide"
    )
    
    # Title and description
    st.title("📊 Richard Data Cleaning & Visualization System")
    st.markdown("""
    Upload your dataset and let the system automatically detect column types, 
    clean your data, and suggest intelligent visualizations.
    """)
    
    # Initialize session state
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_cleaned' not in st.session_state:
        st.session_state.df_cleaned = None
    if 'column_types' not in st.session_state:
        st.session_state.column_types = None
    
    # Sidebar - File Upload
    with st.sidebar:
        st.header("Upload Dataset")
        uploaded_file = st.file_uploader(
            "Upload only CSV or Excel file",
            type=['csv', 'xlsx', 'xls']
        )
        
        if uploaded_file is not None:
            try:
                # Read file based on extension
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                st.session_state.df_original = df.copy()
                st.session_state.df_cleaned = df.copy()
                st.session_state.column_types = detect_column_types(df)
                
                st.success(f"I'ts Loaded with {len(df)} rows and {len(df.columns)} columns")
                
            except Exception as e:
                st.error(f"❌ Error loading file: {str(e)}")
    
    # Main content area
    if st.session_state.df_original is not None:
        df_original = st.session_state.df_original
        df_cleaned = st.session_state.df_cleaned
        column_types = st.session_state.column_types
        
        # Create tabs
        tab1, tab2, tab3, tab4 = st.tabs([
            "📋 Data Summary", 
            "🧹 Data Cleaning", 
            "📊 Visualizations",
            "💾 Export"
        ])
        
        # ====================================================================
        # TAB 1: DATA SUMMARY
        # ====================================================================
        with tab1:
            st.header("Data Summary")
            
            # Display summary statistics
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("Total Rows", len(df_original))
            with col2:
                st.metric("Total Columns", len(df_original.columns))
            with col3:
                total_missing = df_original.isna().sum().sum()
                st.metric("Total Missing Values", total_missing)
            with col4:
                missing_pct = (total_missing / (len(df_original) * len(df_original.columns)) * 100)
                st.metric("Missing %", f"{missing_pct:.1f}%")
            
            st.subheader("Column Information")
            summary_df = get_data_summary(df_original, column_types)
            st.dataframe(summary_df, use_container_width=True)
            
            st.subheader("Data Preview")
            st.dataframe(df_original.head(10), use_container_width=True)
        
        # ====================================================================
        # TAB 2: DATA CLEANING
        # ====================================================================
        with tab2:
            st.header("Data Cleaning")
            
            st.markdown("""
            Select columns to clean. The system will apply appropriate cleaning rules 
            based on the detected data type.
            """)
            
            # Column selection for cleaning
            columns_to_clean = st.multiselect(
                "Select columns to clean",
                options=df_original.columns.tolist(),
                default=[]
            )
            
            if columns_to_clean:
                st.subheader("Cleaning Preview")
                
                # Show before/after for selected columns
                for col in columns_to_clean:
                    with st.expander(f"🔍 {col} ({column_types[col]})"):
                        col_left, col_right = st.columns(2)
                        
                        with col_left:
                            st.write("**Before Cleaning**")
                            st.write(f"Missing values: {df_original[col].isna().sum()}")
                            st.write(df_original[col].head(10))
                        
                        with col_right:
                            st.write("**After Cleaning**")
                            
                            # Apply appropriate cleaning
                            if column_types[col] == 'numeric':
                                cleaned_col = clean_numeric(df_original[col])
                            elif column_types[col] == 'categorical':
                                cleaned_col = clean_categorical(df_original[col])
                            elif column_types[col] == 'datetime':
                                cleaned_col = clean_datetime(df_original[col])
                            else:
                                cleaned_col = clean_text(df_original[col])
                            
                            st.write(f"Missing values: {cleaned_col.isna().sum()}")
                            st.write(cleaned_col.head(10))
                
                # Apply cleaning button
                if st.button("✨ Apply Cleaning", type="primary"):
                    for col in columns_to_clean:
                        if column_types[col] == 'numeric':
                            st.session_state.df_cleaned[col] = clean_numeric(df_original[col])
                        elif column_types[col] == 'categorical':
                            st.session_state.df_cleaned[col] = clean_categorical(df_original[col])
                        elif column_types[col] == 'datetime':
                            st.session_state.df_cleaned[col] = clean_datetime(df_original[col])
                        else:
                            st.session_state.df_cleaned[col] = clean_text(df_original[col])
                    
                    st.success("✅ Cleaning applied successfully!")
                    st.rerun()
            
            # Show cleaned data summary
            if not df_cleaned.equals(df_original):
                st.subheader("Cleaned Data Summary")
                
                col1, col2 = st.columns(2)
                with col1:
                    st.metric(
                        "Missing Values (Before)", 
                        df_original.isna().sum().sum()
                    )
                with col2:
                    st.metric(
                        "Missing Values (After)", 
                        df_cleaned.isna().sum().sum()
                    )
                
                st.dataframe(df_cleaned.head(10), use_container_width=True)
        
        # ====================================================================
        # TAB 3: VISUALIZATIONS
        # ====================================================================
        with tab3:
            st.header("Data Visualizations")
            
            # Get visualization suggestions
            viz_suggestions = suggest_visualizations(df_cleaned, column_types)
            
            if not viz_suggestions:
                st.info("No visualization suggestions available for this dataset.")
            else:
                st.markdown(f"**{len(viz_suggestions)} visualizations available**")
                
                # Visualization selection
                selected_viz = st.selectbox(
                    "Choose a visualization",
                    options=list(viz_suggestions.keys())
                )
                
                if selected_viz:
                    viz_info = viz_suggestions[selected_viz]
                    st.markdown(f"*{viz_info['description']}*")
                    
                    # Create and display visualization
                    try:
                        fig = create_visualization(
                            df_cleaned, 
                            viz_info['type'], 
                            viz_info['columns']
                        )
                        
                        if fig:
                            st.plotly_chart(fig, use_container_width=True)
                        else:
                            st.error("Unable to create visualization")
                    
                    except Exception as e:
                        st.error(f"Error creating visualization: {str(e)}")
                
                # Show multiple visualizations
                if st.checkbox("Show all visualizations"):
                    st.subheader("All Available Visualizations")
                    
                    for viz_name, viz_info in list(viz_suggestions.items())[:6]:
                        st.markdown(f"### {viz_name}")
                        try:
                            fig = create_visualization(
                                df_cleaned, 
                                viz_info['type'], 
                                viz_info['columns']
                            )
                            if fig:
                                st.plotly_chart(fig, use_container_width=True)
                        except:
                            st.warning(f"Could not create {viz_name}")
        
        # ====================================================================
        # TAB 4: EXPORT
        # ====================================================================
        with tab4:
            st.header("Export Cleaned Data")
            
            st.markdown("""
            Download your cleaned dataset in CSV or Excel format.
            """)
            
            col1, col2 = st.columns(2)
            
            with col1:
                # CSV download
                csv = df_cleaned.to_csv(index=False)
                st.download_button(
                    label="📥 Download as CSV",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )
            
            with col2:
                # Excel download
                buffer = io.BytesIO()
                safe_df = remove_illegal_excel_chars(df_cleaned)
                with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
                    safe_df.to_excel(writer, index=False, sheet_name='Cleaned Data')

                
                st.download_button(
                    label="📥 Download as Excel",
                    data=buffer.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
            
            st.subheader("Data Cleaning Report")
            
            # Generate report
            report = f"""
            ## Data Cleaning Report
            
            **Original Dataset:**
            - Rows: {len(df_original)}
            - Columns: {len(df_original.columns)}
            - Missing Values: {df_original.isna().sum().sum()}
            
            **Cleaned Dataset:**
            - Rows: {len(df_cleaned)}
            - Columns: {len(df_cleaned.columns)}
            - Missing Values: {df_cleaned.isna().sum().sum()}
            
            **Cleaning Operations:**
            - Numeric columns: Filled with median
            - Categorical columns: Standardized and filled with mode
            - Datetime columns: Parsed and invalid entries removed
            - Text columns: Trimmed and filled with 'Unknown'
            """
            
            st.markdown(report)
    
    else:
        # Welcome screen
        st.info("👈 Please upload a dataset using the sidebar to get started!")
        
        st.markdown("""
        ### Features:
        
        - **Automatic Type Detection**: Intelligently identifies numeric, categorical, datetime, and text columns
        - **Smart Data Cleaning**: Applies appropriate cleaning rules based on column type
        - **Interactive Visualizations**: Suggests and creates visualizations based on your data
        - **Export Options**: Download cleaned data in CSV or Excel format
        
        ### Supported File Formats:
        - CSV (.csv)
        - Excel (.xlsx, .xls)
        
        ### Getting Started:
        1. Upload your dataset using the file uploader in the sidebar
        2. Review the data summary and column types
        3. Select columns to clean and apply cleaning rules
        4. Explore suggested visualizations
        5. Export your cleaned dataset
        """)

if __name__ == "__main__":
    main()