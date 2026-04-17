import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.linear_model import LinearRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, r2_score

st.set_page_config(page_title="Complete Data Science Dashboard", layout="wide")

st.title("🚀 End-to-End Data Science Pipeline Dashboard")

uploaded_file = st.file_uploader("Upload CSV File", type=["csv"])

if uploaded_file is not None:

    df = pd.read_csv(uploaded_file)

    # =====================================================
    # 1️⃣ DATA OVERVIEW
    # =====================================================
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

    # =====================================================
    # 2️⃣ EDA
    # =====================================================
    st.header("🔍 Exploratory Data Analysis")

    st.subheader("📊 Statistical Summary")
    st.write(df.describe())

    st.subheader("📝 Categorical Summary")
    st.write(df.describe(include="object"))

    # Missing Values Table
    missing_df = pd.DataFrame({
        "Column": df.columns,
        "Missing Values": df.isnull().sum(),
        "Missing %": (df.isnull().sum() / len(df)) * 100
    })
    st.dataframe(missing_df)

    # =====================================================
    # 3️⃣ DATA PREPROCESSING
    # =====================================================
    st.header("🧹 Data Preprocessing")

    df_clean = df.copy()

    # Handle Missing Values
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col].fillna(df_clean[col].mode()[0], inplace=True)
        else:
            df_clean[col].fillna(df_clean[col].mean(), inplace=True)

    # Remove duplicates
    df_clean.drop_duplicates(inplace=True)

    # Encode categorical
    le = LabelEncoder()
    for col in df_clean.columns:
        if df_clean[col].dtype == "object":
            df_clean[col] = le.fit_transform(df_clean[col])

    st.success("✅ Data Cleaned & Encoded")

    # =====================================================
    # 4️⃣ FEATURE ENGINEERING
    # =====================================================
    st.header("⚙ Feature Engineering")

    numeric_cols = df_clean.select_dtypes(include=np.number).columns

    scaler = StandardScaler()
    df_clean[numeric_cols] = scaler.fit_transform(df_clean[numeric_cols])

    st.success("✅ Features Scaled")

    # =====================================================
    # 5️⃣ VISUALIZATION
    # =====================================================
    st.header("📈 Visualization")

    cols = st.columns(3)
    for i, col in enumerate(numeric_cols[:6]):
        with cols[i % 3]:
            fig, ax = plt.subplots()
            sns.histplot(df_clean[col], kde=True, ax=ax)
            ax.set_title(col)
            st.pyplot(fig)

    # =====================================================
    # 6️⃣ MACHINE LEARNING
    # =====================================================
    st.header("🤖 Machine Learning")

    target = st.selectbox("Select Target Column", df_clean.columns)

    X = df_clean.drop(columns=[target])
    y = df_clean[target]

    # Detect problem type
    if y.dtype == "float64":
        problem_type = "Regression"
    else:
        problem_type = "Classification"

    st.info(f"Detected Problem Type: {problem_type}")

    # Safety checks
    if len(df_clean) < 5:
        st.error("Dataset too small!")
        st.stop()

    if X.shape[1] == 0:
        st.error("No feature columns available!")
        st.stop()

    if y.isnull().sum() > 0:
        st.error("Target contains missing values!")
        st.stop()

    # Convert to numeric safely
    X = X.apply(pd.to_numeric, errors='coerce')
    X.fillna(0, inplace=True)

    # Split
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )

    # Train button
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

            # Download model
            model_bytes = pickle.dumps(model)
            st.download_button(
                "⬇ Download Model",
                model_bytes,
                "model.pkl"
            )

            # Download dataset
            csv = df_clean.to_csv(index=False).encode()
            st.download_button(
                "⬇ Download Processed Data",
                csv,
                "data.csv"
            )

        except Exception as e:
            st.error(f"❌ Error: {e}")