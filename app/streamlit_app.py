import streamlit as st
import pandas as pd
import numpy as np
from pathlib import Path

from utils import (
    load_model,
    load_feature_list,
    load_label_mapping,
    prepare_input_dataframe,
    decode_predictions,
    get_top_class_probabilities,
)


st.set_page_config(
    page_title="IoMT Cybersecurity Detection App",
    page_icon="🛡️",
    layout="wide"
)

st.title("IoMT Cybersecurity Detection App")
st.markdown(
    """
This app uses the best-performing Random Forest models from the CICIoMT2024 project to classify IoMT network traffic.

Available modes:
- Binary Detection: Benign vs Attack
- Grouped Multiclass: Benign / DDoS / DoS / MQTT Attack / Recon / Spoofing
- Full Multiclass: Detailed attack-type classification
"""
)

st.sidebar.header("Configuration")

prediction_mode = st.sidebar.selectbox(
    "Select prediction mode",
    [
        "Binary Detection",
        "Grouped Multiclass",
        "Full Multiclass"
    ],
    index=1
)

st.sidebar.markdown("---")
st.sidebar.markdown("Recommended default: **Grouped Multiclass**")

# -----------------------------
# Load artifacts based on mode
# -----------------------------
if prediction_mode == "Binary Detection":
    model = load_model("binary_random_forest.joblib")
    feature_list = load_feature_list("binary_rf_features.csv")
    label_mapping = {0: "Benign", 1: "Attack"}
    class_names = ["Benign", "Attack"]
    task_type = "binary"

elif prediction_mode == "Grouped Multiclass":
    model = load_model("grouped_random_forest.joblib")
    feature_list = load_feature_list("grouped_rf_features.csv")
    label_mapping = load_label_mapping(
        "grouped_label_mapping.csv",
        code_col="grouped_class_code",
        name_col="grouped_class_name"
    )
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    task_type = "multiclass"

else:
    model = load_model("full_random_forest.joblib")
    feature_list = load_feature_list("full_rf_features.csv")
    label_mapping = load_label_mapping(
        "full_label_mapping.csv",
        code_col="full_class_code",
        name_col="full_class_name"
    )
    class_names = [label_mapping[i] for i in sorted(label_mapping.keys())]
    task_type = "multiclass"

# -----------------------------
# Show expected input info
# -----------------------------
with st.expander("Expected input format"):
    st.write(f"Required feature count: **{len(feature_list)}**")
    st.write("Uploaded CSV files should contain network-flow feature columns.")
    st.write("If some required columns are missing, the app will add them with zeros.")
    st.write("Extra columns will be ignored.")

    st.write("First 20 required features:")
    st.code("\n".join(feature_list[:20]))

# -----------------------------
# File upload
# -----------------------------
uploaded_file = st.file_uploader("Upload a CSV file for prediction", type=["csv"])

if uploaded_file is not None:
    try:
        raw_df = pd.read_csv(uploaded_file)
        st.success("File uploaded successfully.")
    except Exception as e:
        st.error(f"Error reading CSV: {e}")
        st.stop()

    st.subheader("Uploaded data preview")
    st.dataframe(raw_df.head())

    prepared_df, missing_cols, extra_cols = prepare_input_dataframe(raw_df, feature_list)

    col1, col2 = st.columns(2)

    with col1:
        st.metric("Uploaded rows", len(raw_df))
        st.metric("Uploaded columns", raw_df.shape[1])

    with col2:
        st.metric("Missing required columns added", len(missing_cols))
        st.metric("Extra columns ignored", len(extra_cols))

    if missing_cols:
        with st.expander("Missing columns that were added with zeros"):
            st.write(missing_cols)

    if extra_cols:
        with st.expander("Extra uploaded columns that were ignored"):
            st.write(extra_cols)

    st.subheader("Prepared input preview")
    st.dataframe(prepared_df.head())

    if prepared_df.shape[0] == 0:
        st.error("The uploaded CSV contains no data rows. Please upload a file with at least one sample.")
        st.stop()

    if st.button("Run prediction"):
        with st.spinner("Generating predictions..."):
            preds = model.predict(prepared_df)
            pred_labels = decode_predictions(preds, label_mapping)

            results_df = raw_df.copy()
            results_df["predicted_code"] = preds
            results_df["predicted_label"] = pred_labels

            if hasattr(model, "predict_proba"):
                prob_array = model.predict_proba(prepared_df)

                if task_type == "binary":
                    results_df["prob_benign"] = prob_array[:, 0]
                    results_df["prob_attack"] = prob_array[:, 1]
                    results_df["top_prediction_summary"] = [
                        f"Benign: {b:.4f}; Attack: {a:.4f}"
                        for b, a in zip(prob_array[:, 0], prob_array[:, 1])
                    ]
                else:
                    top_prob_strings = get_top_class_probabilities(
                        prob_array,
                        class_names,
                        top_n=3
                    )
                    results_df["top_prediction_summary"] = top_prob_strings

        st.success("Prediction completed.")

        st.subheader("Prediction results")
        st.dataframe(results_df.head(20))

        st.subheader("Prediction distribution")
        pred_counts = results_df["predicted_label"].value_counts()
        st.bar_chart(pred_counts)

        st.subheader("Prediction summary table")
        summary_df = pred_counts.rename_axis("predicted_label").reset_index(name="count")
        summary_df["percentage"] = 100 * summary_df["count"] / summary_df["count"].sum()
        st.dataframe(summary_df)

        csv_output = results_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            label="Download predictions as CSV",
            data=csv_output,
            file_name=f"predictions_{prediction_mode.lower().replace(' ', '_')}.csv",
            mime="text/csv"
        )

# -----------------------------
# Manual info section
# -----------------------------
st.markdown("---")
st.subheader("About this app")
st.markdown(
    """
This app is part of an end-to-end IoMT cybersecurity machine learning project based on the CICIoMT2024 dataset.

Model deployment strategy:
- Binary Detection: primary intrusion-detection layer
- Grouped Multiclass: main operational attack-family classifier
- Full Multiclass: advanced forensic analysis mode
"""
)