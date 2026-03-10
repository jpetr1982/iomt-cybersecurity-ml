from pathlib import Path
import pandas as pd
import joblib


BASE_DIR = Path(__file__).resolve().parents[1]
MODEL_DIR = BASE_DIR / "models"
REPORT_DIR = BASE_DIR / "reports" / "results"


def load_feature_list(filename: str) -> list:
    path = REPORT_DIR / filename
    df = pd.read_csv(path)
    return df["feature"].tolist()


def load_label_mapping(filename: str, code_col: str, name_col: str) -> dict:
    path = REPORT_DIR / filename
    df = pd.read_csv(path)
    return dict(zip(df[code_col], df[name_col]))


def load_model(filename: str):
    path = MODEL_DIR / filename
    return joblib.load(path)


def prepare_input_dataframe(uploaded_df: pd.DataFrame, required_features: list):
    df = uploaded_df.copy()

    missing_cols = [col for col in required_features if col not in df.columns]
    extra_cols = [col for col in df.columns if col not in required_features]

    for col in missing_cols:
        df[col] = 0

    df = df[required_features].copy()

    return df, missing_cols, extra_cols


def decode_predictions(preds, mapping: dict):
    return [mapping.get(int(p), str(p)) for p in preds]


def get_top_class_probabilities(prob_array, class_names, top_n=3):
    prob_df = pd.DataFrame(prob_array, columns=class_names)
    top_rows = []

    for _, row in prob_df.iterrows():
        top = row.sort_values(ascending=False).head(top_n)
        top_rows.append(
            "; ".join([f"{cls}: {prob:.4f}" for cls, prob in top.items()])
        )

    return top_rows