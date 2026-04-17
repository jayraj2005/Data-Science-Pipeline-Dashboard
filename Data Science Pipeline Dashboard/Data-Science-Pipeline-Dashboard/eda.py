import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO

st.set_page_config(page_title="Universal Data Cleaner", layout="wide")

st.title("📊 Universal Data Cleaning Web App")
st.write("Upload CSV or Excel file and clean missing values automatically.")

# Upload Dataset
uploaded_file = st.file_uploader(
    "Upload Dataset",
    type=["csv", "xlsx", "xls"]
)

if uploaded_file is not None:
    
    file_type = uploaded_file.type
    file_name = uploaded_file.name
    
    # ----------- READ FILE AUTOMATICALLY -----------
    try:
        if "csv" in file_type:
            df = pd.read_csv(uploaded_file)
        
        elif "spreadsheetml" in file_type or file_name.endswith(".xlsx"):
            df = pd.read_excel(uploaded_file, engine="openpyxl")
        
        elif file_name.endswith(".xls"):
            df = pd.read_excel(uploaded_file)
        
        else:
            st.error("Unsupported file format.")
            st.stop()
    
    except Exception as e:
        st.error(f"Error reading file: {e}")
        st.stop()

    # ----------- DATA PREVIEW -----------
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # ----------- DATA INFORMATION -----------
    st.subheader("📌 Dataset Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())

    st.write("### Column Data Types")
    st.write(df.dtypes)

    # ----------- MISSING VALUES -----------
    st.subheader("❓ Missing Values Analysis")

    missing_values = df.isnull().sum()
    st.write(missing_values)

    total_missing = missing_values.sum()

    # ----------- IF NO MISSING VALUES -----------
    if total_missing == 0:
        st.success("✅ Your dataset is already clean. No missing values found.")
    
    # ----------- CLEAN DATA OPTION -----------
    else:
        st.warning(f"⚠ Total Missing Values: {total_missing}")

        if st.button("🧹 Clean Data"):

            df_cleaned = df.copy()

            for column in df_cleaned.columns:

                # Categorical columns
                if df_cleaned[column].dtype == 'object':
                    if df_cleaned[column].mode().empty:
                        df_cleaned[column].fillna("Unknown", inplace=True)
                    else:
                        mode_value = df_cleaned[column].mode()[0]
                        df_cleaned[column].fillna(mode_value, inplace=True)

                # Numeric columns
                else:
                    mean_value = df_cleaned[column].mean()

                    if not np.isnan(mean_value):
                        mean_value = int(mean_value)
                        df_cleaned[column].fillna(mean_value, inplace=True)
                        df_cleaned[column] = df_cleaned[column].astype(int)

            # Remove duplicates
            df_cleaned.drop_duplicates(inplace=True)

            st.success("Data Cleaned Successfully ✅")

            # ----------- SHOW CLEANED DATA -----------
            st.subheader("📊 Cleaned Data Preview")
            st.dataframe(df_cleaned.head())

            # ----------- DOWNLOAD CLEANED FILE -----------
            st.subheader("⬇ Download Cleaned Data")

            if file_name.endswith(".csv"):
                csv = df_cleaned.to_csv(index=False).encode("utf-8")
                st.download_button(
                    label="Download CSV File",
                    data=csv,
                    file_name="cleaned_data.csv",
                    mime="text/csv"
                )

            else:
                output = BytesIO()
                df_cleaned.to_excel(output, index=False, engine='openpyxl')
                st.download_button(
                    label="Download Excel File",
                    data=output.getvalue(),
                    file_name="cleaned_data.xlsx",
                    mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                )
                
class Destop:
    def __init__(self):
        self.__max_price = 25000
    
    def sell(self):
        return f"Selling price:{self.__max_price}"
    
    def set_mac_price(self, price):
        if price > self.__max_price:
            self.__max_price = price
            
d = Destop()
print(d.sell())
d.set_mac_price(30000)
print(d.sell())
        
