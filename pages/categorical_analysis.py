import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Categorical Analysis - No-Code ML Platform", layout="wide")

def perform_categorical_analysis(df):
    st.title("Categorical Columns Analysis")
    
    # Get categorical columns
    categorical_cols = df.select_dtypes(include=['object', 'category']).columns
    
    if len(categorical_cols) > 0:
        for col in categorical_cols:
            st.write(f"\nValue counts for {col}:")
            value_counts = df[col].value_counts()
            
            # Create bar plot
            fig = px.bar(x=value_counts.index, y=value_counts.values,
                        title=f"Distribution of {col}")
            st.plotly_chart(fig)
            
            # Display value counts
            st.write(value_counts)
    else:
        st.info("No categorical columns found in the dataset.")

# Check if data is available in session state
if 'df' in st.session_state:
    perform_categorical_analysis(st.session_state.df)
else:
    st.warning("Please upload a dataset in the main page first.") 