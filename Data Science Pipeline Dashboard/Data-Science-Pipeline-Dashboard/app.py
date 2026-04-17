import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from io import BytesIO

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="Data Science Dashboard", layout="wide")

# =====================================================
# SIDEBAR NAVIGATION
# =====================================================
st.sidebar.title("📂 Navigation")
page = st.sidebar.radio(
    "Choose a Module",
    ["🚀 End-to-End ML Pipeline", "📊 Universal Data Cleaner",]
)
st.sidebar.markdown("---")
st.sidebar.markdown("###  Design And Developed by🫱🏻‍🫲🏼: Jayraj")
# =====================================================
# PAGE 1: END-TO-END ML PIPELINE (clean.py)
# =====================================================
if page == "🚀 End-to-End ML Pipeline":

    st.title("🚀 End-to-End Data Science Pipeline Dashboard")

    uploaded_file = st.file_uploader("Upload CSV File", type=["csv"], key="pipeline_upload")

    if uploaded_file is not None:

        df = pd.read_csv(uploaded_file)

        # Save raw df and filename to session so page 2 can use it
        st.session_state["shared_df"] = df
        st.session_state["shared_file_name"] = uploaded_file.name

        # 1️⃣ DATA OVERVIEW
        st.header("📊 Data Overview")

        st.subheader("Dataset Preview")
        st.dataframe(df.head())

        col1, col2, col3 = st.columns(3)
        col1.metric("Rows", df.shape[0])
        col2.metric("Columns", df.shape[1])
        col3.metric("Duplicate Rows", df.duplicated().sum())

        st.subheader("Data Types")
        st.write(df.dtypes)

        st.subheader("Missing Values Per Column")
        st.write(df.isnull().sum())

        # 2️⃣ EDA
        st.header("🔍 Exploratory Data Analysis")

        st.subheader("📊 Statistical Summary")
        st.write(df.describe())

        st.subheader("📝 Categorical Summary")
        st.write(df.describe(include="object"))

        missing_df = pd.DataFrame({
            "Column": df.columns,
            "Missing Values": df.isnull().sum(),
            "Missing %": (df.isnull().sum() / len(df)) * 100
        })
        st.dataframe(missing_df)

        # 3️⃣ DATA PREPROCESSING
        st.header("🧹 Data Preprocessing")

        df_clean = df.copy()

        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
            else:
                df_clean[col].fillna(df_clean[col].mean(), inplace=True)

        df_clean.drop_duplicates(inplace=True)

        le = LabelEncoder()
        for col in df_clean.columns:
            if df_clean[col].dtype == "object":
                df_clean[col] = le.fit_transform(df_clean[col])

        st.success("✅ Data Cleaned & Encoded")

        # 4️⃣ FEATURE ENGINEERING
        st.header("⚙ Feature Engineering")

        numeric_cols = df_clean.select_dtypes(include=np.number).columns

        scaler = StandardScaler()
        df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

        st.success("✅ Features Scaled")

        # 5️⃣ VISUALIZATION
        st.header("📈 Visualization")

        cols = st.columns(3)
        for i, col in enumerate(numeric_cols[:6]):
            with cols[i % 3]:
                fig, ax = plt.subplots()
                sns.histplot(df_clean[col], kde=True, ax=ax)
                ax.set_title(col)
                st.pyplot(fig)

        # 6️⃣ MACHINE LEARNING
        st.header("🤖 Machine Learning")

        target = st.selectbox("Select Target Column", df_clean.columns)

        X = df_clean.drop(columns=[target])
        y = df_clean[target]

        if y.dtype == "float64":
            problem_type = "Regression"
        else:
            problem_type = "Classification"

        st.info(f"Detected Problem Type: {problem_type}")

        if len(df_clean) < 5:
            st.error("Dataset too small!")
            st.stop()

        if X.shape[1] == 0:
            st.error("No feature columns available!")
            st.stop()

        if y.isnull().sum() > 0:
            st.error("Target contains missing values!")
            st.stop()

        X = X.apply(pd.to_numeric, errors='coerce')
        X.fillna(0, inplace=True)

        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=0.2, random_state=42
        )

        if st.button("🚀 Train Model"):

            try:
                if problem_type == "Regression":
                    model = LinearRegression()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    score = r2_score(y_test, y_pred)
                    st.success(f"📈 R2 Score: {score:.2f}")

                else:
                    model = RandomForestClassifier()
                    model.fit(X_train, y_train)
                    y_pred = model.predict(X_test)
                    acc = accuracy_score(y_test, y_pred)
                    st.success(f"🎯 Accuracy: {acc:.2f}")

                model_bytes = pickle.dumps(model)
                st.download_button(
                    "⬇ Download Model",
                    model_bytes,
                    "model.pkl"
                )

                csv = df_clean.to_csv(index=False).encode()
                st.download_button(
                    "⬇ Download Processed Data",
                    csv,
                    "data.csv"
                )

            except Exception as e:
                st.error(f"❌ Error: {e}")

# =====================================================
# PAGE 2: UNIVERSAL DATA CLEANER (eda.py)
# =====================================================
elif page == "📊 Universal Data Cleaner":

    st.title("📊 Universal Data Cleaning Web App")
    st.write("Upload CSV or Excel file and clean missing values automatically.")

    uploaded_file = st.file_uploader(
        "Upload Dataset",
        type=["csv", "xlsx", "xls"],
        key="cleaner_upload"
    )

    # Auto-load from Page 1 if no file uploaded here
    if uploaded_file is None and "shared_df" in st.session_state:
        df = st.session_state["shared_df"]
        file_name = st.session_state.get("shared_file_name", "dataset.csv")
        st.info(f"✅ Dataset auto-loaded from ML Pipeline: **{file_name}**")
        auto_loaded = True
    elif uploaded_file is not None:
        file_name = uploaded_file.name
        file_type = uploaded_file.type
        auto_loaded = False
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
    else:
        st.info("👈 Upload a dataset here, or go to the ML Pipeline page and upload a CSV to auto-load it here.")
        st.stop()

    # DATA PREVIEW
    st.subheader("🔍 Data Preview")
    st.dataframe(df.head())

    # DATA INFORMATION
    st.subheader("📌 Dataset Information")

    col1, col2, col3 = st.columns(3)
    col1.metric("Total Rows", df.shape[0])
    col2.metric("Total Columns", df.shape[1])
    col3.metric("Duplicate Rows", df.duplicated().sum())

    st.write("### Column Data Types")
    st.write(df.dtypes)

    # MISSING VALUES
    st.subheader("❓ Missing Values Analysis")

    missing_values = df.isnull().sum()
    st.write(missing_values)

    total_missing = missing_values.sum()

    if total_missing == 0:
        st.success("✅ Your dataset is already clean. No missing values found.")
        df_cleaned = df.copy()
        df_cleaned.drop_duplicates(inplace=True)
        st.session_state["df_cleaned"] = df_cleaned
        st.session_state["file_name"] = file_name
        run_model = True

    else:
        st.warning(f"⚠ Total Missing Values: {total_missing}")
        run_model = False

        if st.button("🧹 Clean Data"):

            df_cleaned = df.copy()

            for column in df_cleaned.columns:
                if df_cleaned[column].dtype == 'object':
                    if df_cleaned[column].mode().empty:
                        df_cleaned[column].fillna("Unknown", inplace=True)
                    else:
                        mode_value = df_cleaned[column].mode()[0]
                        df_cleaned[column].fillna(mode_value, inplace=True)
                else:
                    mean_value = df_cleaned[column].mean()
                    if not np.isnan(mean_value):
                        mean_value = int(mean_value)
                        df_cleaned[column].fillna(mean_value, inplace=True)
                        df_cleaned[column] = df_cleaned[column].astype(int)

            df_cleaned.drop_duplicates(inplace=True)

            st.success("Data Cleaned Successfully ✅")

            st.subheader("📊 Cleaned Data Preview")
            st.dataframe(df_cleaned.head())

            st.session_state["df_cleaned"] = df_cleaned
            st.session_state["file_name"] = file_name
            run_model = True

    # Pull cleaned df from session if available
    if "df_cleaned" in st.session_state:
        df_cleaned = st.session_state["df_cleaned"]
        file_name = st.session_state.get("file_name", file_name)
        run_model = True

    # =====================================================
    # TRAIN MODEL SECTION (after cleaning)
    # =====================================================
    if run_model and "df_cleaned" in st.session_state:

        df_cleaned = st.session_state["df_cleaned"]

        st.header("🤖 Train Model on Cleaned Data")

        # Encode for ML
        df_ml = df_cleaned.copy()
        le = LabelEncoder()
        for col in df_ml.columns:
            if df_ml[col].dtype == "object":
                df_ml[col] = le.fit_transform(df_ml[col].astype(str))

        numeric_cols = df_ml.select_dtypes(include=np.number).columns
        scaler = StandardScaler()
        df_ml[numeric_cols] = scaler.fit_transform(df_ml[numeric_cols])

        target = st.selectbox("Select Target Column", df_ml.columns, key="eda_target")

        X = df_ml.drop(columns=[target])
        y = df_ml[target]

        if y.dtype == "float64":
            problem_type = "Regression"
        else:
            problem_type = "Classification"

        st.info(f"Detected Problem Type: {problem_type}")

        if len(df_ml) < 5:
            st.error("Dataset too small to train!")
        elif X.shape[1] == 0:
            st.error("No feature columns available!")
        elif y.isnull().sum() > 0:
            st.error("Target contains missing values!")
        else:
            X = X.apply(pd.to_numeric, errors='coerce')
            X.fillna(0, inplace=True)

            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.2, random_state=42
            )

            if st.button("🚀 Train Model", key="eda_train_btn"):

                try:
                    if problem_type == "Regression":
                        model = LinearRegression()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        score = r2_score(y_test, y_pred)
                        st.success(f"📈 R2 Score: {score:.2f}")

                    else:
                        model = RandomForestClassifier()
                        model.fit(X_train, y_train)
                        y_pred = model.predict(X_test)
                        acc = accuracy_score(y_test, y_pred)
                        st.success(f"🎯 Accuracy: {acc:.2f}")

                    st.subheader("⬇ Downloads")

                    # Download trained model
                    model_bytes = pickle.dumps(model)
                    st.download_button(
                        label="⬇ Download Trained Model",
                        data=model_bytes,
                        file_name="trained_model.pkl"
                    )

                    # Download cleaned dataset
                    if file_name.endswith(".csv"):
                        csv = df_cleaned.to_csv(index=False).encode("utf-8")
                        st.download_button(
                            label="⬇ Download Cleaned Dataset (CSV)",
                            data=csv,
                            file_name="cleaned_data.csv",
                            mime="text/csv"
                        )
                    else:
                        output = BytesIO()
                        df_cleaned.to_excel(output, index=False, engine='openpyxl')
                        st.download_button(
                            label="⬇ Download Cleaned Dataset (Excel)",
                            data=output.getvalue(),
                            file_name="cleaned_data.xlsx",
                            mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet"
                        )

                except Exception as e:
                    st.error(f"❌ Error during training: {e}")
