import streamlit as st
import pandas as pd
import joblib

from sklearn.metrics import (
    accuracy_score,
    roc_auc_score,
    precision_score,
    recall_score,
    f1_score,
    matthews_corrcoef,
    confusion_matrix,
    classification_report
)
st.set_page_config(
    page_title="Breast Cancer Classification",
    page_icon="🔬",
    layout="wide"
)

st.title("Breast Cancer Classification")
st.write(
    "Compare machine learning classification models "
    "using the Wisconsin Diagnostic Breast Cancer dataset."
)
models = {
    "Logistic Regression": joblib.load("model/logistic_regression.pkl"),
    "Decision Tree": joblib.load("model/decision_tree.pkl"),
    "KNN": joblib.load("model/knn.pkl"),
    "Naive Bayes": joblib.load("model/naive_bayes.pkl"),
    "Random Forest": joblib.load("model/random_forest.pkl")
}
scaler = joblib.load("model/scaler.pkl")


st.sidebar.header("Model Selection")

selected_model_name = st.sidebar.selectbox(
    "Choose a classification model:",
    list(models.keys())
)
selected_model = models[selected_model_name]


uploaded_file = st.file_uploader(
    "Upload Test Dataset",
    type=["csv"]
)

if uploaded_file is not None:

    test_df = pd.read_csv(uploaded_file)

    st.subheader("Uploaded Test Data")
    st.dataframe(test_df.head())

    if "diagnosis" not in test_df.columns:
        st.error(
            "The uploaded CSV must contain the 'diagnosis' target column."
        )
        st.stop()

    X_uploaded = test_df.drop(columns=["diagnosis"])
    y_uploaded = test_df["diagnosis"]

    models_requiring_scaling = [
        "Logistic Regression",
        "KNN",
        "Naive Bayes"
    ]

    if selected_model_name in models_requiring_scaling:
        X_model = scaler.transform(X_uploaded)
    else:
        X_model = X_uploaded

    y_pred = selected_model.predict(X_model)

    y_prob = selected_model.predict_proba(X_model)[:, 1]

    accuracy = accuracy_score(y_uploaded, y_pred)
    auc = roc_auc_score(y_uploaded, y_prob)
    precision = precision_score(y_uploaded, y_pred)
    recall = recall_score(y_uploaded, y_pred)
    f1 = f1_score(y_uploaded, y_pred)
    mcc = matthews_corrcoef(y_uploaded, y_pred)

    st.subheader(f"Evaluation Results — {selected_model_name}")

    col1, col2, col3 = st.columns(3)

    col1.metric("Accuracy", f"{accuracy:.4f}")
    col2.metric("AUC", f"{auc:.4f}")
    col3.metric("Precision", f"{precision:.4f}")

    col4, col5, col6 = st.columns(3)

    col4.metric("Recall", f"{recall:.4f}")
    col5.metric("F1 Score", f"{f1:.4f}")
    col6.metric("MCC", f"{mcc:.4f}")

import matplotlib.pyplot as plt
from sklearn.metrics import ConfusionMatrixDisplay

if uploaded_file is not None:
    st.subheader("Confusion Matrix")

    cm = confusion_matrix(y_uploaded, y_pred)

    fig, ax = plt.subplots()

    disp = ConfusionMatrixDisplay(
        confusion_matrix=cm,
        display_labels=["Benign", "Malignant"]
    )

    disp.plot(ax=ax)

    st.pyplot(fig)
    
    st.subheader("Classification Report")

    report = classification_report(
        y_uploaded,
        y_pred,
        target_names=["Benign", "Malignant"],
        output_dict=True
    )

    report_df = pd.DataFrame(report).transpose()

    st.dataframe(report_df)