import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder, OneHotEncoder, StandardScaler, MinMaxScaler
import plotly.express as px
import plotly.graph_objects as go

st.set_page_config(page_title="Data Preprocessing - No-Code ML Platform", layout="wide")

# Check if data is available in session state
if 'df' not in st.session_state or st.session_state.df is None:
    st.warning("Please upload a dataset in the main page first.")
    st.stop()

def handle_missing_values(df, column, method):
    if method == "Drop rows":
        return df.dropna(subset=[column])
    elif method == "Fill with mean":
        if pd.api.types.is_numeric_dtype(df[column]):
            return df.fillna({column: df[column].mean()})
        else:
            return df.fillna({column: df[column].mode()[0]})
    elif method == "Fill with median":
        if pd.api.types.is_numeric_dtype(df[column]):
            return df.fillna({column: df[column].median()})
        else:
            return df.fillna({column: df[column].mode()[0]})
    elif method == "Fill with mode":
        return df.fillna({column: df[column].mode()[0]})
    elif method == "Fill with value":
        return df.fillna({column: 0})  # Default value, can be customized
    return df

def encode_categorical(df, column, method):
    if method == "Label Encoding":
        le = LabelEncoder()
        return pd.Series(le.fit_transform(df[column]), name=column)
    elif method == "One-Hot Encoding":
        return pd.get_dummies(df[column], prefix=column)
    return df[column]

def scale_numerical(df, column, method):
    if method == "Standard Scaler":
        scaler = StandardScaler()
        return pd.Series(scaler.fit_transform(df[[column]]).flatten(), name=column)
    elif method == "Min-Max Scaler":
        scaler = MinMaxScaler()
        return pd.Series(scaler.fit_transform(df[[column]]).flatten(), name=column)
    return df[column]

def main():
    st.title("Data Preprocessing")
    
    # Get the dataframe from session state
    df = st.session_state.df.copy()
    
    # Display original data info
    st.subheader("Original Data Information")
    st.write("Shape:", df.shape)
    st.write("Columns:", df.columns.tolist())
    st.write("Data Types:", df.dtypes)
    
    # Column selection for preprocessing
    st.subheader("Select Columns to Preprocess")
    selected_columns = st.multiselect("Choose columns", df.columns)
    
    if selected_columns:
        for column in selected_columns:
            st.write(f"\nProcessing column: {column}")
            
            # Display column statistics
            col1, col2 = st.columns(2)
            with col1:
                st.write("Data Type:", df[column].dtype)
                st.write("Missing Values:", df[column].isnull().sum())
            with col2:
                if pd.api.types.is_numeric_dtype(df[column]):
                    st.write("Mean:", df[column].mean())
                    st.write("Median:", df[column].median())
                else:
                    st.write("Unique Values:", df[column].nunique())
                    st.write("Most Common:", df[column].mode()[0])
            
            # Handle missing values
            if df[column].isnull().sum() > 0:
                st.write("Handle Missing Values")
                missing_method = st.selectbox(
                    f"Select method for {column}",
                    ["Drop rows", "Fill with mean", "Fill with median", "Fill with mode", "Fill with value"],
                    key=f"missing_{column}"
                )
                df = handle_missing_values(df, column, missing_method)
            
            # Encode categorical variables
            if pd.api.types.is_object_dtype(df[column]):
                st.write("Encode Categorical Values")
                encode_method = st.selectbox(
                    f"Select encoding method for {column}",
                    ["Label Encoding", "One-Hot Encoding"],
                    key=f"encode_{column}"
                )
                encoded_data = encode_categorical(df, column, encode_method)
                if isinstance(encoded_data, pd.DataFrame):
                    df = pd.concat([df.drop(column, axis=1), encoded_data], axis=1)
                else:
                    df[column] = encoded_data
            
            # Scale numerical variables
            elif pd.api.types.is_numeric_dtype(df[column]):
                st.write("Scale Numerical Values")
                scale_method = st.selectbox(
                    f"Select scaling method for {column}",
                    ["Standard Scaler", "Min-Max Scaler"],
                    key=f"scale_{column}"
                )
                df[column] = scale_numerical(df, column, scale_method)
            
            # Show preview of processed column
            st.write("Preview of processed data:")
            st.write(df[column].head())
    
    # Update session state with processed dataframe
    if st.button("Apply Preprocessing"):
        st.session_state.df = df
        st.success("Data preprocessing completed!")
        
        # Show final data info
        st.subheader("Processed Data Information")
        st.write("Shape:", df.shape)
        st.write("Columns:", df.columns.tolist())
        st.write("Data Types:", df.dtypes)
        
        # Show sample of processed data
        st.subheader("Sample of Processed Data")
        st.write(df.head())

if __name__ == "__main__":
    main() 