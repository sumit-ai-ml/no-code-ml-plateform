import streamlit as st
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.cluster import KMeans
from sklearn.metrics import accuracy_score, mean_squared_error, r2_score
import numpy as np
import os
import io
import tempfile
import traceback
import base64

st.set_page_config(page_title="No-Code ML Platform", layout="wide")

# Initialize session state for dataframe if it doesn't exist
if 'df' not in st.session_state:
    st.session_state.df = None

def load_data(file):
    try:
        # Get file extension
        file_extension = os.path.splitext(file.name)[1].lower()
        
        # Read file content
        file_content = file.read()
        
        if file_extension == '.csv':
            df = pd.read_csv(io.BytesIO(file_content))
        elif file_extension in ['.xlsx', '.xls']:
            try:
                # Try different Excel engines
                try:
                    df = pd.read_excel(io.BytesIO(file_content), engine='openpyxl')
                except Exception as e1:
                    st.warning(f"Failed with openpyxl: {str(e1)}. Trying xlrd...")
                    df = pd.read_excel(io.BytesIO(file_content), engine='xlrd')
            except Exception as e:
                st.error(f"Error reading Excel file: {str(e)}")
                st.info("Try saving the file as CSV or ensure it's not password-protected.")
                return None
        else:
            st.error("Please upload a CSV or Excel file.")
            return None
        
        return df
    except Exception as e:
        st.error(f"Error loading file: {str(e)}")
        st.error(f"Traceback: {traceback.format_exc()}")
        return None

def get_file_download_link(file_path, link_text):
    """Generate a link to download a file"""
    with open(file_path, 'rb') as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    href = f'<a href="data:file/csv;base64,{b64}" download="{os.path.basename(file_path)}">{link_text}</a>'
    return href

def perform_eda(df):
    st.subheader("Basic Information")
    col1, col2 = st.columns(2)
    
    with col1:
        st.write("Dataset Shape:", df.shape)
        st.write("Number of rows:", df.shape[0])
        st.write("Number of columns:", df.shape[1])
    
    with col2:
        st.write("Memory Usage:", f"{df.memory_usage(deep=True).sum() / 1024**2:.2f} MB")
    
    # Data Types and Missing Values
    st.subheader("Data Types and Missing Values")
    data_info = pd.DataFrame({
        'Data Type': df.dtypes,
        'Non-Null Count': df.count(),
        'Null Count': df.isnull().sum(),
        'Null Percentage': (df.isnull().sum() / len(df) * 100).round(2)
    })
    st.dataframe(data_info)
    
    # Numerical Columns Statistics
    st.subheader("Numerical Columns Statistics")
    numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
    if len(numerical_cols) > 0:
        st.dataframe(df[numerical_cols].describe())
        
        # Correlation Matrix
        st.subheader("Correlation Matrix")
        corr_matrix = df[numerical_cols].corr()
        fig = px.imshow(corr_matrix, 
                       labels=dict(color="Correlation"),
                       x=corr_matrix.columns,
                       y=corr_matrix.columns)
        st.plotly_chart(fig)

def plot_data(df, plot_type, x_col, y_col=None):
    if plot_type == "Box Plot":
        fig = px.box(df, y=x_col)
    elif plot_type == "Line Plot":
        fig = px.line(df, x=x_col, y=y_col)
    elif plot_type == "Scatter Plot":
        fig = px.scatter(df, x=x_col, y=y_col)
    elif plot_type == "Histogram":
        fig = px.histogram(df, x=x_col)
    elif plot_type == "Bar Plot":
        fig = px.bar(df, x=x_col, y=y_col)
    
    st.plotly_chart(fig)

def main():
    st.title("No-Code Machine Learning Platform")
    
    # File Upload
    st.header("1. Upload Your Data")
    
    # Single upload option for CSV and Excel files
    uploaded_file = st.file_uploader("Choose a CSV or Excel file", type=['csv', 'xlsx', 'xls'])
    
    if uploaded_file is not None:
        # Load the data and store it in session state
        df = load_data(uploaded_file)
        if df is not None:
            st.session_state.df = df
            st.success("Data loaded successfully!")
    
    # Use the dataframe from session state
    df = st.session_state.df
    
    # Continue with the rest of the application if data is loaded
    if df is not None:
        # Perform EDA
        perform_eda(df)
        
        # Data Visualization
        st.header("2. Data Visualization")
        plot_types = ["Box Plot", "Line Plot", "Scatter Plot", "Histogram", "Bar Plot"]
        selected_plot = st.selectbox("Select plot type", plot_types)
        
        numerical_cols = df.select_dtypes(include=['int64', 'float64']).columns
        categorical_cols = df.select_dtypes(include=['object', 'category']).columns
        
        if selected_plot in ["Box Plot", "Histogram"]:
            x_col = st.selectbox("Select column", numerical_cols)
            plot_data(df, selected_plot, x_col)
        else:
            x_col = st.selectbox("Select X column", df.columns)
            y_col = st.selectbox("Select Y column", numerical_cols)
            plot_data(df, selected_plot, x_col, y_col)
    else:
        st.info("Please upload a dataset to begin analysis.")

if __name__ == "__main__":
    main() 