import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime
import re

# ==================== PAGE CONFIGURATION ====================
st.set_page_config(
    page_title="Richard Data Cleaning System",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ==================== CUSTOM CSS STYLING ====================
def load_custom_css():
    st.markdown("""
    <style>
    /* Main container styling */
    .main {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        padding: 0rem 1rem;
    }
    
    /* Content background */
    .block-container {
        background: rgba(255, 255, 255, 0.95);
        border-radius: 20px;
        padding: 2rem 2rem 3rem 2rem;
        box-shadow: 0 20px 60px rgba(0, 0, 0, 0.3);
        margin-top: 2rem;
        margin-bottom: 2rem;
    }
    
    /* Headers styling */
    h1 {
        color: #667eea !important;
        font-weight: 800 !important;
        font-size: 2.5rem !important;
        margin-bottom: 0.5rem !important;
        text-shadow: 2px 2px 4px rgba(0,0,0,0.1);
    }
    
    h2 {
        color: #764ba2 !important;
        font-weight: 700 !important;
        font-size: 1.8rem !important;
        margin-top: 1.5rem !important;
        margin-bottom: 1rem !important;
        border-bottom: 3px solid #667eea;
        padding-bottom: 0.5rem;
    }
    
    h3 {
        color: #5a67d8 !important;
        font-weight: 600 !important;
        font-size: 1.3rem !important;
    }
    
    /* Metric cards */
    [data-testid="stMetricValue"] {
        font-size: 2rem !important;
        font-weight: 700 !important;
        color: #667eea !important;
    }
    
    [data-testid="stMetricLabel"] {
        font-size: 1rem !important;
        font-weight: 600 !important;
        color: #4a5568 !important;
    }
    
    /* Buttons */
    .stButton > button {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        font-size: 1rem !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(102, 126, 234, 0.4) !important;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(102, 126, 234, 0.6) !important;
    }
    
    /* Download buttons */
    .stDownloadButton > button {
        background: linear-gradient(135deg, #48bb78 0%, #38a169 100%) !important;
        color: white !important;
        border: none !important;
        border-radius: 10px !important;
        padding: 0.6rem 1.5rem !important;
        font-weight: 600 !important;
        transition: all 0.3s ease !important;
        box-shadow: 0 4px 15px rgba(72, 187, 120, 0.4) !important;
    }
    
    .stDownloadButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 6px 20px rgba(72, 187, 120, 0.6) !important;
    }
    
    /* Tabs styling */
    .stTabs [data-baseweb="tab-list"] {
        gap: 8px;
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        padding: 10px;
        border-radius: 15px;
        box-shadow: inset 0 2px 4px rgba(0,0,0,0.1);
    }
    
    .stTabs [data-baseweb="tab"] {
        background: white;
        border-radius: 10px;
        padding: 12px 24px;
        font-weight: 600;
        color: #4a5568;
        border: 2px solid transparent;
        transition: all 0.3s ease;
    }
    
    .stTabs [aria-selected="true"] {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
        color: white !important;
        border-color: #667eea;
        box-shadow: 0 4px 12px rgba(102, 126, 234, 0.4);
    }
    
    /* Select boxes */
    .stSelectbox > div > div {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stSelectbox > div > div:focus-within {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* Text inputs */
    .stTextInput > div > div > input {
        border-radius: 10px !important;
        border: 2px solid #e2e8f0 !important;
        transition: all 0.3s ease !important;
    }
    
    .stTextInput > div > div > input:focus {
        border-color: #667eea !important;
        box-shadow: 0 0 0 3px rgba(102, 126, 234, 0.1) !important;
    }
    
    /* File uploader */
    [data-testid="stFileUploader"] {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%);
        border: 2px dashed #cbd5e0;
        border-radius: 15px;
        padding: 2rem;
        transition: all 0.3s ease;
    }
    
    [data-testid="stFileUploader"]:hover {
        border-color: #667eea;
        background: linear-gradient(135deg, #edf2f7 0%, #e2e8f0 100%);
    }
    
    /* Dataframe styling */
    .dataframe {
        border-radius: 10px !important;
        overflow: hidden !important;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.1) !important;
    }
    
    /* Expander */
    .streamlit-expanderHeader {
        background: linear-gradient(135deg, #f7fafc 0%, #edf2f7 100%) !important;
        border-radius: 10px !important;
        font-weight: 600 !important;
        color: #2d3748 !important;
        border: 2px solid #e2e8f0 !important;
    }
    
    .streamlit-expanderHeader:hover {
        border-color: #667eea !important;
    }
    
    /* Success/Info/Warning boxes */
    .stSuccess, .stInfo, .stWarning, .stError {
        border-radius: 10px !important;
        padding: 1rem 1.5rem !important;
        font-weight: 500 !important;
    }
    
    /* Sidebar styling */
    [data-testid="stSidebar"] {
        background: linear-gradient(180deg, #2d3748 0%, #1a202c 100%);
        box-shadow: 4px 0 15px rgba(0, 0, 0, 0.2);
    }
    
    [data-testid="stSidebar"] h1, 
    [data-testid="stSidebar"] h2, 
    [data-testid="stSidebar"] h3 {
        color: white !important;
    }
    
    [data-testid="stSidebar"] p,
    [data-testid="stSidebar"] div {
        color: #e2e8f0 !important;
    }
    
    /* Sidebar button */
    [data-testid="stSidebar"] .stButton > button {
        background: linear-gradient(135deg, #fc8181 0%, #f56565 100%) !important;
        width: 100%;
    }
    
    /* Cards effect */
    .element-container {
        animation: fadeIn 0.5s ease-in;
    }
    
    @keyframes fadeIn {
        from { opacity: 0; transform: translateY(10px); }
        to { opacity: 1; transform: translateY(0); }
    }
    
    /* Slider styling */
    .stSlider > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Radio buttons */
    .stRadio > label {
        font-weight: 600 !important;
        color: #2d3748 !important;
    }
    
    /* Checkbox */
    .stCheckbox > label {
        font-weight: 500 !important;
        color: #2d3748 !important;
    }
    
    /* Progress bar */
    .stProgress > div > div > div {
        background: linear-gradient(135deg, #667eea 0%, #764ba2 100%) !important;
    }
    
    /* Custom card styling */
    .custom-card {
        background: white;
        padding: 1.5rem;
        border-radius: 15px;
        box-shadow: 0 4px 12px rgba(0, 0, 0, 0.08);
        border: 2px solid #e2e8f0;
        margin: 1rem 0;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        box-shadow: 0 8px 24px rgba(102, 126, 234, 0.2);
        border-color: #667eea;
        transform: translateY(-2px);
    }
    </style>
    """, unsafe_allow_html=True)

# ==================== UTILITY FUNCTIONS ====================

def initialize_session_state():
    """Initialize all session state variables"""
    if 'df_original' not in st.session_state:
        st.session_state.df_original = None
    if 'df_working' not in st.session_state:
        st.session_state.df_working = None
    if 'cleaning_log' not in st.session_state:
        st.session_state.cleaning_log = []
    if 'column_types' not in st.session_state:
        st.session_state.column_types = {}

def log_action(action):
    """Add action to cleaning log with timestamp"""
    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    st.session_state.cleaning_log.append(f"[{timestamp}] {action}")

def detect_column_type(series):
    """Intelligently detect column type based on content analysis"""
    non_null = series.dropna()
    if len(non_null) == 0:
        return {'type': 'empty', 'confidence': 'high'}
    
    str_series = non_null.astype(str)
    
    # Check for boolean
    unique_vals = set(str_series.str.lower().unique())
    bool_patterns = {'true', 'false', 'yes', 'no', '1', '0', 't', 'f', 'y', 'n'}
    if len(unique_vals) <= 2 and unique_vals.issubset(bool_patterns):
        return {'type': 'boolean', 'confidence': 'high'}
    
    # Check for currency/monetary
    currency_pattern = r'[$€£¥₹₽₦₨₪₫₩₴₸₵₲₱₡₪₺₼₾₿]|RWF|USD|EUR|GBP'
    if str_series.str.contains(currency_pattern, regex=True, na=False).any():
        return {'type': 'currency', 'confidence': 'high'}
    
    # Check for datetime
    try:
        pd.to_datetime(non_null, errors='raise')
        return {'type': 'datetime', 'confidence': 'high'}
    except:
        date_patterns = [r'\d{4}-\d{2}-\d{2}', r'\d{2}/\d{2}/\d{4}', r'\d{2}-\d{2}-\d{4}']
        if any(str_series.str.contains(pat, regex=True, na=False).any() for pat in date_patterns):
            return {'type': 'datetime', 'confidence': 'medium'}
    
    # Check for numerical
    numeric_pattern = r'^-?\d+\.?\d*$'
    numeric_match = str_series.str.match(numeric_pattern, na=False).sum() / len(str_series)
    if numeric_match > 0.8:
        return {'type': 'numerical', 'confidence': 'high'}
    
    if pd.api.types.is_numeric_dtype(series):
        return {'type': 'numerical', 'confidence': 'high'}
    
    # Check for categorical
    unique_ratio = len(non_null.unique()) / len(non_null)
    if unique_ratio < 0.05 or len(non_null.unique()) < 20:
        return {'type': 'categorical', 'confidence': 'high'}
    
    return {'type': 'text', 'confidence': 'medium'}

def detect_missing_patterns(series):
    """Detect various forms of missing data"""
    missing_patterns = ['', ' ', 'NA', 'N/A', 'na', 'n/a', 'NaN', 'nan', 
                       'NULL', 'null', 'None', 'none', '?', '-', '--', 'n.a.']
    
    missing_mask = series.isna() | series.isin(missing_patterns)
    if series.dtype == 'object':
        missing_mask |= series.astype(str).str.strip() == ''
    
    return missing_mask

def clean_currency_column(series, currency_symbol='$', decimal_sep='.', thousand_sep=','):
    """Clean currency column and convert to numeric"""
    cleaned = series.astype(str)
    
    # Remove currency symbols
    for symbol in ['$', '€', '£', '¥', '₹', '₽', '₦', '₨', 'RWF', 'USD', 'EUR', 'GBP']:
        cleaned = cleaned.str.replace(symbol, '', regex=False)
    
    # Handle different formats
    if decimal_sep == ',' and thousand_sep == '.':
        cleaned = cleaned.str.replace('.', '', regex=False)
        cleaned = cleaned.str.replace(',', '.', regex=False)
    elif thousand_sep == ' ':
        cleaned = cleaned.str.replace(' ', '', regex=False)
        if decimal_sep == ',':
            cleaned = cleaned.str.replace(',', '.', regex=False)
    else:
        cleaned = cleaned.str.replace(',', '', regex=False)
    
    cleaned = cleaned.str.replace(r'[^\d.-]', '', regex=True)
    return pd.to_numeric(cleaned, errors='coerce')

# ==================== TAB 1: DATA UPLOAD ====================

def tab_data_upload():
    """Tab for dataset upload and initial display"""
    st.markdown("### 📁 Dataset Upload & Overview")
    
    uploaded_file = st.file_uploader(
        "Upload your dataset (CSV or Excel)",
        type=['csv', 'xlsx', 'xls'],
        help="Maximum file size: 200MB"
    )
    
    if uploaded_file is not None:
        try:
            with st.spinner("Loading dataset..."):
                if uploaded_file.name.endswith('.csv'):
                    df = pd.read_csv(uploaded_file)
                else:
                    df = pd.read_excel(uploaded_file)
                
                if st.session_state.df_original is None:
                    st.session_state.df_original = df.copy()
                    st.session_state.df_working = df.copy()
                    log_action(f"Dataset loaded: {uploaded_file.name}")
                    
                    # Auto-detect column types
                    st.session_state.column_types = {}
                    progress_bar = st.progress(0)
                    for idx, col in enumerate(df.columns):
                        st.session_state.column_types[col] = detect_column_type(df[col])
                        progress_bar.progress((idx + 1) / len(df.columns))
                    progress_bar.empty()
                    log_action("Column types auto-detected")
            
            st.success(f"✅ Dataset '{uploaded_file.name}' loaded successfully!")
            
            # Display metrics
            st.markdown("#### 📊 Dataset Statistics")
            col1, col2, col3, col4 = st.columns(4)
            
            with col1:
                st.metric("📋 Total Rows", f"{df.shape[0]:,}")
            with col2:
                st.metric("📊 Total Columns", f"{df.shape[1]:,}")
            with col3:
                memory_mb = df.memory_usage(deep=True).sum() / 1024**2
                st.metric("💾 Memory Usage", f"{memory_mb:.2f} MB")
            with col4:
                missing_total = df.isna().sum().sum()
                st.metric("❓ Missing Values", f"{missing_total:,}")
            
            # Dataset preview
            st.markdown("#### 👀 Dataset Preview")
            st.dataframe(df.head(10), use_container_width=True, height=300)
            
            # Column types overview
            with st.expander("🔍 Detected Column Types & Statistics", expanded=True):
                type_data = []
                for col in df.columns:
                    col_type = st.session_state.column_types[col]
                    type_data.append({
                        'Column': col,
                        'Detected Type': col_type['type'],
                        'Confidence': col_type['confidence'],
                        'Data Type': str(df[col].dtype),
                        'Non-Null': f"{df[col].notna().sum():,}",
                        'Null': f"{df[col].isna().sum():,}",
                        'Unique': f"{df[col].nunique():,}"
                    })
                
                type_df = pd.DataFrame(type_data)
                st.dataframe(type_df, use_container_width=True, height=400)
            
        except Exception as e:
            st.error(f"❌ Error loading file: {str(e)}")
    else:
        st.info("👆 Please upload a dataset to get started")

# ==================== TAB 2: MISSING VALUES ====================

def tab_missing_values():
    """Tab for missing value detection and handling"""
    if st.session_state.df_working is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' tab")
        return
    
    st.markdown("### 🔍 Missing Value Analysis & Treatment")
    
    df = st.session_state.df_working
    
    # Calculate missing data summary
    missing_summary = []
    for col in df.columns:
        missing_mask = detect_missing_patterns(df[col])
        missing_count = missing_mask.sum()
        if missing_count > 0:
            missing_summary.append({
                'Column': col,
                'Missing Count': missing_count,
                'Missing %': round(missing_count / len(df) * 100, 2),
                'Type': st.session_state.column_types[col]['type']
            })
    
    if missing_summary:
        # Display summary
        st.markdown("#### 📊 Missing Data Overview")
        summary_df = pd.DataFrame(missing_summary)
        
        col1, col2 = st.columns([2, 1])
        with col1:
            st.dataframe(summary_df, use_container_width=True, height=300)
        
        with col2:
            # Visual representation
            fig = px.bar(summary_df, x='Column', y='Missing %', 
                        title='Missing Data by Column',
                        color='Missing %',
                        color_continuous_scale='Reds')
            fig.update_layout(height=300)
            st.plotly_chart(fig, use_container_width=True)
        
        st.markdown("---")
        
        # Treatment section
        st.markdown("#### 🛠️ Apply Missing Value Treatment")
        
        col1, col2 = st.columns([1, 1])
        
        with col1:
            selected_col = st.selectbox(
                "Select column to treat",
                [item['Column'] for item in missing_summary],
                help="Choose the column you want to clean"
            )
        
        with col2:
            col_type = st.session_state.column_types[selected_col]['type']
            st.info(f"**Column Type:** {col_type.upper()}")
        
        # Show sample data
        with st.expander("👁️ Preview Column Data", expanded=False):
            preview_col1, preview_col2 = st.columns(2)
            with preview_col1:
                st.write("**Non-Missing Values:**")
                st.write(df[selected_col].dropna().head(5).tolist())
            with preview_col2:
                st.write("**Missing Values Count:**")
                st.write(detect_missing_patterns(df[selected_col]).sum())
        
        # Method selection based on type
        method_options = {
            'numerical': ['Mean', 'Median', 'Mode', 'Constant Value', 'Forward Fill', 
                         'Backward Fill', 'Drop Rows', 'Drop Column'],
            'categorical': ['Mode', 'Replace with "Unknown"', 'Drop Rows', 'Drop Column'],
            'datetime': ['Forward Fill', 'Backward Fill', 'Drop Rows', 'Drop Column'],
            'text': ['Replace with Empty String', 'Replace with Placeholder', 
                    'Drop Rows', 'Drop Column'],
            'currency': ['Mean', 'Median', 'Mode', 'Constant Value', 'Drop Rows', 'Drop Column'],
            'boolean': ['Mode', 'Constant Value', 'Drop Rows', 'Drop Column']
        }
        
        methods = method_options.get(col_type, ['Drop Rows', 'Drop Column'])
        selected_method = st.selectbox("Select treatment method", methods)
        
        # Additional parameters
        constant_value = None
        if 'Constant Value' in selected_method:
            constant_value = st.text_input("Enter constant value", key="const_val")
        elif 'Placeholder' in selected_method:
            constant_value = st.text_input("Enter placeholder text", value="N/A", key="placeholder")
        
        col1, col2, col3 = st.columns([1, 1, 2])
        with col1:
            if st.button("✅ Apply Treatment", type="primary", use_container_width=True):
                try:
                    missing_mask = detect_missing_patterns(df[selected_col])
                    rows_before = len(df)
                    
                    if selected_method == 'Mean':
                        df.loc[missing_mask, selected_col] = df[selected_col].mean()
                    elif selected_method == 'Median':
                        df.loc[missing_mask, selected_col] = df[selected_col].median()
                    elif selected_method == 'Mode':
                        mode_val = df[selected_col].mode()
                        if len(mode_val) > 0:
                            df.loc[missing_mask, selected_col] = mode_val[0]
                    elif selected_method == 'Constant Value':
                        df.loc[missing_mask, selected_col] = constant_value
                    elif selected_method == 'Forward Fill':
                        df[selected_col] = df[selected_col].fillna(method='ffill')
                    elif selected_method == 'Backward Fill':
                        df[selected_col] = df[selected_col].fillna(method='bfill')
                    elif selected_method == 'Replace with "Unknown"':
                        df.loc[missing_mask, selected_col] = 'Unknown'
                    elif selected_method == 'Replace with Empty String':
                        df.loc[missing_mask, selected_col] = ''
                    elif 'Placeholder' in selected_method:
                        df.loc[missing_mask, selected_col] = constant_value
                    elif selected_method == 'Drop Rows':
                        df = df[~missing_mask]
                    elif selected_method == 'Drop Column':
                        df = df.drop(columns=[selected_col])
                    
                    st.session_state.df_working = df
                    log_action(f"Applied {selected_method} to '{selected_col}' - Affected rows: {rows_before - len(df)}")
                    st.success(f"✅ Successfully applied {selected_method}!")
                    st.rerun()
                    
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
        
        with col2:
            if st.button("🔄 Refresh Analysis", use_container_width=True):
                st.rerun()
    else:
        st.success(" There is no missing values detected in your dataset!")
        st.balloons()

# ==================== TAB 3: CURRENCY CLEANING ====================

def tab_currency_cleaning():
    """Tab for currency column detection and cleaning"""
    if st.session_state.df_working is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' tab")
        return
    
    st.markdown("### 💰 Currency & Monetary Column Cleaning")
    
    df = st.session_state.df_working
    
    # Detect currency columns
    currency_cols = [col for col, info in st.session_state.column_types.items() 
                     if info['type'] == 'currency']
    
    if currency_cols:
        st.markdown(f"#### 🔍 Detected {len(currency_cols)} Currency Column(s)")
        
        st.info("💡 **Tip:** Currency columns contain symbols like $, €, £, or patterns like 1,000.50")
        
        selected_col = st.selectbox(
            "Select currency column to clean",
            currency_cols,
            help="Choose which currency column to process"
        )
        
        # Show sample data in a nice card
        with st.expander("👁️ Sample Values from Column", expanded=True):
            samples = df[selected_col].dropna().head(10).tolist()
            for i, sample in enumerate(samples, 1):
                st.markdown(f"`{i}.` **{sample}**")
        
        st.markdown("---")
        st.markdown("#### ⚙️ Configure Currency Format")
        
        col1, col2, col3 = st.columns(3)
        
        with col1:
            currency_symbol = st.text_input(
                "Currency Symbol",
                value="$",
                help="e.g., $, €, £, RWF"
            )
        
        with col2:
            decimal_sep = st.selectbox(
                "Decimal Separator",
                ['.', ','],
                help="Character used for decimals"
            )
        
        with col3:
            thousand_sep = st.selectbox(
                "Thousand Separator",
                [',', '.', ' ', 'None'],
                help="Character used for thousands"
            )
        
        if thousand_sep == 'None':
            thousand_sep = ''
        
        # Format examples
        st.markdown("#### 📝 Format Examples")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.info(f"**US Format**\n\n${currency_symbol}1,000.50")
        with col2:
            st.info(f"**European Format**\n\n{currency_symbol}1.000,50")
        with col3:
            st.info(f"**Space Format**\n\n{currency_symbol}1 000.50")
        
        col1, col2 = st.columns([1, 3])
        with col1:
            if st.button("🧹 Clean Currency Column", type="primary", use_container_width=True):
                try:
                    with st.spinner("Cleaning currency data..."):
                        original_values = df[selected_col].copy()
                        df[selected_col] = clean_currency_column(
                            df[selected_col],
                            currency_symbol,
                            decimal_sep,
                            thousand_sep
                        )
                        
                        st.session_state.df_working = df
                        st.session_state.column_types[selected_col] = {
                            'type': 'numerical',
                            'confidence': 'high'
                        }
                        log_action(f"Cleaned currency column '{selected_col}'")
                    
                    st.success(f"✅ Successfully cleaned '{selected_col}'!")
                    
                    # Show before/after comparison
                    st.markdown("#### 📊 Before & After Comparison")
                    comparison_df = pd.DataFrame({
                        'Before': original_values.head(10),
                        'After': df[selected_col].head(10)
                    })
                    st.dataframe(comparison_df, use_container_width=True)
                    
                    # Show statistics
                    col1, col2, col3 = st.columns(3)
                    with col1:
                        st.metric("Min Value", f"{df[selected_col].min():.2f}")
                    with col2:
                        st.metric("Max Value", f"{df[selected_col].max():.2f}")
                    with col3:
                        st.metric("Mean Value", f"{df[selected_col].mean():.2f}")
                    
                except Exception as e:
                    st.error(f"❌ Error cleaning currency: {str(e)}")
    else:
        st.info("ℹ️ No currency columns detected in the dataset")
        st.markdown("""
        **Currency columns typically contain:**
        - Currency symbols ($, €, £, etc.)
        - Formatted numbers (1,000.50)
        - Mixed text and numbers
        """)

# ==================== TAB 4: GENERAL CLEANING ====================

def tab_general_cleaning():
    """Tab for general data cleaning operations"""
    if st.session_state.df_working is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' tab")
        return
    
    st.markdown("### 🧹 General Data Cleaning Operations")
    
    df = st.session_state.df_working
    
    # Section 1: Remove Duplicates
    st.markdown("#### 🔄 Remove Duplicate Rows")
    col1, col2, col3 = st.columns([1, 1, 2])
    
    with col1:
        duplicate_count = df.duplicated().sum()
        st.metric("Duplicate Rows Found", f"{duplicate_count:,}")
    
    with col2:
        if duplicate_count > 0:
            if st.button("🗑️ Remove Duplicates", type="primary", use_container_width=True):
                original_len = len(df)
                df = df.drop_duplicates()
                st.session_state.df_working = df
                log_action(f"Removed {duplicate_count} duplicate rows")
                st.success(f"✅ Removed {duplicate_count} duplicates!")
                st.rerun()
        else:
            st.success("No duplicates found!")
    
    st.markdown("---")
    
    # Section 2: Trim Whitespace
    st.markdown("#### ✂️ Trim Whitespace from Text Columns")
    text_cols = df.select_dtypes(include=['object']).columns.tolist()
    
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        st.metric("Text Columns", len(text_cols))
    
    with col2:
        if text_cols:
            if st.button("✂️ Trim Whitespace", use_container_width=True):
                for col in text_cols:
                    df[col] = df[col].astype(str).str.strip()
                st.session_state.df_working = df
                log_action(f"Trimmed whitespace from {len(text_cols)} columns")
                st.success(f"✅ Trimmed {len(text_cols)} columns!")
                st.rerun()
    
    st.markdown("---")
    
    # Section 3: Rename Columns
    st.markdown("#### 🏷️ Rename Column")
    with st.expander("Rename a Column", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            old_name = st.selectbox("Select column", df.columns, key='rename_old')
        
        with col2:
            new_name = st.text_input("New name", value=old_name, key='rename_new')
        
        with col3:
            st.write("")
            st.write("")
            if st.button("Rename", use_container_width=True):
                if new_name and new_name != old_name:
                    if new_name not in df.columns:
                        df = df.rename(columns={old_name: new_name})
                        st.session_state.df_working = df
                        if old_name in st.session_state.column_types:
                            st.session_state.column_types[new_name] = st.session_state.column_types.pop(old_name)
                        log_action(f"Renamed '{old_name}' to '{new_name}'")
                        st.success("✅ Column renamed!")
                        st.rerun()
                    else:
                        st.error("❌ Column name already exists")
    
    # Section 4: Convert Data Types
    st.markdown("#### 🔄 Convert Data Type")
    with st.expander("Convert Column Type", expanded=False):
        col1, col2, col3 = st.columns([2, 2, 1])
        
        with col1:
            col_to_convert = st.selectbox("Select column", df.columns, key='convert_col')
        
        with col2:
            current_type = str(df[col_to_convert].dtype)
            st.info(f"Current: **{current_type}**")
            new_dtype = st.selectbox(
                "New data type",
                ['int', 'float', 'str', 'datetime', 'category']
            )
        
        with col3:
            st.write("")
            st.write("")
            if st.button("Convert", use_container_width=True):
                try:
                    if new_dtype == 'datetime':
                        df[col_to_convert] = pd.to_datetime(df[col_to_convert])
                    else:
                        df[col_to_convert] = df[col_to_convert].astype(new_dtype)
                    
                    st.session_state.df_working = df
                    log_action(f"Converted '{col_to_convert}' to {new_dtype}")
                    st.success("✅ Type converted!")
                    st.rerun()
                except Exception as e:
                    st.error(f"❌ Error: {str(e)}")
    
    # Section 5: Drop Column
    st.markdown("#### 🗑️ Drop Column")
    with st.expander("Remove a Column", expanded=False):
        col1, col2 = st.columns([3, 1])
        
        with col1:
            col_to_drop = st.selectbox("Select column to drop", df.columns, key='drop_col')
            st.warning(f"⚠️ This will permanently remove the '{col_to_drop}' column")
        
        with col2:
            st.write("")
            st.write("")
            if st.button("🗑️ Drop Column", type="primary", use_container_width=True):
                df = df.drop(columns=[col_to_drop])
                st.session_state.df_working = df
                if col_to_drop in st.session_state.column_types:
                    del st.session_state.column_types[col_to_drop]
                log_action(f"Dropped column '{col_to_drop}'")
                st.success("✅ Column dropped!")
                st.rerun()
    
    # Current dataset info
    st.markdown("---")
    st.markdown("#### 📊 Current Dataset Information")
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Rows", f"{len(df):,}")
    with col2:
        st.metric("Columns", f"{len(df.columns):,}")
    with col3:
        st.metric("Memory", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    with col4:
        st.metric("Actions", len(st.session_state.cleaning_log))

# ==================== TAB 5: VISUALIZATION ====================

def tab_visualization():
    """Tab for data visualization"""
    if st.session_state.df_working is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' tab")
        return
    
    st.markdown("### 📊 Data Visualization")
    
    df = st.session_state.df_working
    
    # Chart type selection
    col1, col2 = st.columns([2, 1])
    
    with col1:
        chart_type = st.selectbox(
            "📈 Select Chart Type",
            ['Histogram', 'Box Plot', 'Bar Chart', 'Scatter Plot', 
             'Line Plot', 'Correlation Heatmap', 'Pie Chart'],
            help="Choose the type of visualization"
        )
    
    with col2:
        st.info(f"**Chart:** {chart_type}")
    
    st.markdown("---")
    
    # Visualization logic
    if chart_type == 'Histogram':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col1, col2 = st.columns([3, 1])
            with col1:
                col = st.selectbox("Select numerical column", num_cols)
            with col2:
                bins = st.slider("Number of bins", 10, 100, 30)
            
            fig = px.histogram(
                df, x=col, nbins=bins,
                title=f'Distribution of {col}',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Statistics
            col1, col2, col3, col4 = st.columns(4)
            with col1:
                st.metric("Mean", f"{df[col].mean():.2f}")
            with col2:
                st.metric("Median", f"{df[col].median():.2f}")
            with col3:
                st.metric("Std Dev", f"{df[col].std():.2f}")
            with col4:
                st.metric("Count", f"{df[col].count():,}")
        else:
            st.warning("⚠️ No numerical columns available for histogram")
    
    elif chart_type == 'Box Plot':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col = st.selectbox("Select numerical column", num_cols)
            
            fig = px.box(
                df, y=col,
                title=f'Box Plot of {col}',
                color_discrete_sequence=['#764ba2']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Outlier statistics
            Q1 = df[col].quantile(0.25)
            Q3 = df[col].quantile(0.75)
            IQR = Q3 - Q1
            outliers = df[(df[col] < Q1 - 1.5*IQR) | (df[col] > Q3 + 1.5*IQR)]
            
            col1, col2, col3 = st.columns(3)
            with col1:
                st.metric("Q1 (25%)", f"{Q1:.2f}")
            with col2:
                st.metric("Q3 (75%)", f"{Q3:.2f}")
            with col3:
                st.metric("Outliers", len(outliers))
        else:
            st.warning("⚠️ No numerical columns available for box plot")
    
    elif chart_type == 'Bar Chart':
        cat_cols = [col for col in df.columns 
                   if st.session_state.column_types[col]['type'] == 'categorical']
        if cat_cols:
            col1, col2 = st.columns([3, 1])
            with col1:
                col = st.selectbox("Select categorical column", cat_cols)
            with col2:
                top_n = st.slider("Show top N", 5, 50, 20)
            
            value_counts = df[col].value_counts().head(top_n)
            fig = px.bar(
                x=value_counts.index, y=value_counts.values,
                labels={'x': col, 'y': 'Count'},
                title=f'Top {top_n} Categories in {col}',
                color=value_counts.values,
                color_continuous_scale='Purples'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No categorical columns available for bar chart")
    
    elif chart_type == 'Scatter Plot':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            col1, col2 = st.columns(2)
            with col1:
                x_col = st.selectbox("X-axis", num_cols)
            with col2:
                y_col = st.selectbox("Y-axis", [c for c in num_cols if c != x_col])
            
            # Optional color by categorical
            cat_cols = [col for col in df.columns 
                       if st.session_state.column_types[col]['type'] == 'categorical']
            color_col = None
            if cat_cols:
                use_color = st.checkbox("Color by category")
                if use_color:
                    color_col = st.selectbox("Select category", cat_cols)
            
            fig = px.scatter(
                df, x=x_col, y=y_col, color=color_col,
                title=f'{x_col} vs {y_col}',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
            
            # Correlation
            corr = df[[x_col, y_col]].corr().iloc[0, 1]
            st.metric("Correlation", f"{corr:.3f}")
        else:
            st.warning("⚠️ Need at least 2 numerical columns for scatter plot")
    
    elif chart_type == 'Line Plot':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if num_cols:
            col = st.selectbox("Select Y-axis column", num_cols)
            
            fig = px.line(
                df.reset_index(), x='index', y=col,
                title=f'Line Plot of {col}',
                color_discrete_sequence=['#667eea']
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No numerical columns available for line plot")
    
    elif chart_type == 'Correlation Heatmap':
        num_cols = df.select_dtypes(include=[np.number]).columns.tolist()
        if len(num_cols) >= 2:
            corr = df[num_cols].corr()
            fig = px.imshow(
                corr,
                text_auto='.2f',
                aspect="auto",
                title='Correlation Heatmap',
                color_continuous_scale='RdBu_r'
            )
            fig.update_layout(height=600)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ Need at least 2 numerical columns for correlation heatmap")
    
    elif chart_type == 'Pie Chart':
        cat_cols = [col for col in df.columns 
                   if st.session_state.column_types[col]['type'] == 'categorical']
        if cat_cols:
            col1, col2 = st.columns([3, 1])
            with col1:
                col = st.selectbox("Select categorical column", cat_cols)
            with col2:
                top_n = st.slider("Show top N", 3, 15, 8)
            
            value_counts = df[col].value_counts().head(top_n)
            fig = px.pie(
                values=value_counts.values,
                names=value_counts.index,
                title=f'Distribution of {col} (Top {top_n})'
            )
            fig.update_layout(height=500)
            st.plotly_chart(fig, use_container_width=True)
        else:
            st.warning("⚠️ No categorical columns available for pie chart")

# ==================== TAB 6: EXPORT ====================

def tab_export():
    """Tab for exporting cleaned data"""
    if st.session_state.df_working is None:
        st.warning("⚠️ Please upload a dataset first in the 'Data Upload' tab")
        return
    
    st.markdown("### 💾 Export Cleaned Data")
    
    df = st.session_state.df_working
    
    # Summary metrics
    st.markdown("#### 📊 Cleaning Summary")
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.metric(
            "Original Rows",
            f"{len(st.session_state.df_original):,}",
            delta=None
        )
    
    with col2:
        row_diff = len(st.session_state.df_original) - len(df)
        st.metric(
            "Current Rows",
            f"{len(df):,}",
            delta=f"-{row_diff:,}" if row_diff > 0 else "0"
        )
    
    with col3:
        col_diff = len(st.session_state.df_original.columns) - len(df.columns)
        st.metric(
            "Current Columns",
            f"{len(df.columns):,}",
            delta=f"-{col_diff:,}" if col_diff > 0 else "0"
        )
    
    with col4:
        st.metric("Cleaning Actions", len(st.session_state.cleaning_log))
    
    st.markdown("---")
    
    # Preview cleaned data
    st.markdown("#### 👀 Cleaned Dataset Preview")
    st.dataframe(df.head(20), use_container_width=True, height=400)
    
    st.markdown("---")
    
    # Download options
    st.markdown("#### 📥 Download Options")
    
    col1, col2, col3 = st.columns(3)
    
    with col1:
        # CSV download
        csv = df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📄 Download as CSV",
            data=csv,
            file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
            use_container_width=True
        )
    
    with col2:
        # Excel download
        buffer = BytesIO()
        with pd.ExcelWriter(buffer, engine='openpyxl') as writer:
            df.to_excel(writer, index=False, sheet_name='Cleaned Data')
        
        st.download_button(
            label="📊 Download as Excel",
            data=buffer.getvalue(),
            file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
            use_container_width=True
        )
    
    with col3:
        # JSON download
        json_str = df.to_json(orient='records', indent=2)
        st.download_button(
            label="🔧 Download as JSON",
            data=json_str,
            file_name=f"cleaned_data_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json",
            mime="application/json",
            use_container_width=True
        )
    
    st.markdown("---")
    
    # Cleaning log
    st.markdown("#### 📋 Cleaning Log History")
    if st.session_state.cleaning_log:
        log_df = pd.DataFrame({
            'Action': st.session_state.cleaning_log
        })
        st.dataframe(log_df, use_container_width=True, height=300)
        
        log_csv = log_df.to_csv(index=False).encode('utf-8')
        st.download_button(
            label="📥 Download Cleaning Log",
            data=log_csv,
            file_name=f"cleaning_log_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv"
        )
    else:
        st.info("ℹ️ No cleaning actions performed yet")

# ==================== MAIN APPLICATION ====================

def main():
    """Main application logic"""
    load_custom_css()
    initialize_session_state()
    
    # Sidebar
    with st.sidebar:
        st.markdown("<h1 style='text-align: center;'>IDC System</h1>", unsafe_allow_html=True)
        st.markdown("---")
        
        st.markdown("""
        <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px; margin-bottom: 1rem;'>
        <p style='margin: 0; font-size: 0.9rem;'><strong>👨‍💻 Created by:</strong><br>Richard Usengimana</p>
        </div>
        """, unsafe_allow_html=True)
        
        st.markdown("### ✨ Features")
        st.markdown("""
        - 🔍 Auto column type detection
        - 🎯 Smart missing value handling
        - 💰 Currency cleaning
        - 📊 Data visualization
        - 💾 Multiple export formats
        - 📋 Complete audit trail
        """)
        
        st.markdown("---")
        
        if st.session_state.df_working is not None:
            st.markdown("### 📊 Dataset Status")
            st.markdown(f"""
            <div style='background: rgba(255,255,255,0.1); padding: 1rem; border-radius: 10px;'>
            <p style='margin: 0.3rem 0;'><strong>Rows:</strong> {len(st.session_state.df_working):,}</p>
            <p style='margin: 0.3rem 0;'><strong>Columns:</strong> {len(st.session_state.df_working.columns):,}</p>
            <p style='margin: 0.3rem 0;'><strong>Actions:</strong> {len(st.session_state.cleaning_log)}</p>
            </div>
            """, unsafe_allow_html=True)
            
            st.markdown("---")
        
        if st.button("🔄 Reset Everything", use_container_width=True):
            for key in list(st.session_state.keys()):
                del st.session_state[key]
            st.rerun()
        
        st.markdown("---")
        st.markdown("""
        <div style='text-align: center; font-size: 0.8rem; color: #a0aec0;'>
        <p>Version 1.0.0</p>
        <p>© 2024 Richard Usengimana</p>
        </div>
        """, unsafe_allow_html=True)
    
    # Main content
    st.markdown("<h1 style='text-align: center;'>Richard Data Cleaning & Visualization System</h1>", unsafe_allow_html=True)
    st.markdown("<p style='text-align: center; font-size: 1.2rem; color: #4a5568; margin-bottom: 2rem;'>Professional tool for automated data cleaning and exploration</p>", unsafe_allow_html=True)
    
    # Tabs
    tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
        "📁 Data Upload",
        "🔍 Missing Values",
        "💰 Currency Cleaning",
        "🧹 General Cleaning",
        "📊 Visualization",
        "💾 Export"
    ])
    
    with tab1:
        tab_data_upload()
    
    with tab2:
        tab_missing_values()
    
    with tab3:
        tab_currency_cleaning()
    
    with tab4:
        tab_general_cleaning()
    
    with tab5:
        tab_visualization()
    
    with tab6:
        tab_export()

if __name__ == "__main__":
    main()