# test_analysis_app.py
"""
Advanced Pre/Post Test Analyzer (single-file)
- Accepts wide (scored 0/1) and long (answers) formats
- Automatic cleaning, preprocessing, feature engineering
- Detailed analysis: per-item stats, reliability, paired tests, effect sizes
- Visualizations, Excel and PDF export (PDF optional)
- Uses caching for heavy operations
"""
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
from datetime import datetime
import sys
import math
import warnings
import matplotlib.pyplot as plt
warnings.filterwarnings("ignore")
# Optional libs for PDF/chart/stats/clustering
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False
try:
    from scipy import stats
    SCIPY_AVAILABLE = True
except Exception:
    SCIPY_AVAILABLE = False
try:
    from sklearn.cluster import KMeans
    KMEANS_AVAILABLE = True
except Exception:
    KMEANS_AVAILABLE = False
st.set_page_config(page_title="Advanced Pre/Post Test Analyzer", layout="wide")
st.title("Advanced Pre/Post Test Analyzer")
st.markdown("""
### Features
- Accepts both **wide** (scored 0/1) and **long** (answers) formats
- Automatically cleans and preprocesses data
- Comprehensive analysis: descriptive stats, reliability (Cronbach's alpha), paired tests, effect sizes
- Visualizations: histograms, boxplots, correlation, etc.
- Excel and PDF reports
""")
# ======================================
# Configuration
# ======================================
CONFIG = {
    "wrong_answers": ["0", ".", "-", "x", "wrong", "incorrect"],
    "right_answers": ["1", "v", "✓", "correct"],
    "missing_answers": ["", "nan", "none", "null", "?"],
    "max_missing_percent": 50,
}
# ======================================
# Helper Functions
# ======================================
@st.cache_data(show_spinner=False)
def detect_format(df):
    """
    Returns 'wide' (cols = items) or 'long' (rows = answers).
    If >60% of columns are numeric => likely wide scored
    If one col name is 'Item' or 'Question' => long
    """
    if "Item" in df.columns or "Question" in df.columns:
        return "long"
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) / len(df.columns) > 0.6:
        return "wide"
    return "long"
@st.cache_data(show_spinner=False)
def clean_data(df, fmt="wide"):
    """
    If wide: convert cols to 0/1, remove high missing students
    If long: pivot to wide if needed.
    """
    if fmt == "long":
        df_clean = _convert_long_to_wide(df)
    else:
        df_clean = df.copy()
    # Convert to 0/1
    df_clean = _convert_to_binary(df_clean)
    # Remove students with too many missing
    df_clean = _remove_high_missing(df_clean)
    return df_clean
def _convert_long_to_wide(df):
    """
    If long format, pivot so each row=student, each col=item.
    Assumes col names: Student_ID, Item, Answer.
    """
    df = df.copy()
    required = ["Student_ID", "Item", "Answer"]
    if all(c in df.columns for c in required):
        df_pivot = df.pivot_table(index="Student_ID", columns="Item", values="Answer", aggfunc="first")
        df_pivot.reset_index(inplace=True)
        return df_pivot
    return df
def _convert_to_binary(df):
    """
    Replace known right/wrong answers with 1/0, map unknowns to NaN.
    """
    df = df.copy()
    for col in df.columns:
        if col.lower() in ["student_id", "name", "id"]:
            continue
        df[col] = df[col].astype(str).str.lower().str.strip()
        df[col] = df[col].replace(CONFIG["missing_answers"], np.nan)
        df[col] = df[col].replace(CONFIG["right_answers"], "1")
        df[col] = df[col].replace(CONFIG["wrong_answers"], "0")
        df[col] = pd.to_numeric(df[col], errors="coerce")
    return df
def _remove_high_missing(df):
    """
    Remove students (rows) with >max_missing_percent% missing answers.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    missing_pct = df[numeric_cols].isnull().sum(axis=1) / len(numeric_cols) * 100
    df = df[missing_pct <= CONFIG["max_missing_percent"]]
    return df
@st.cache_data(show_spinner=False)
def engineer_features(df):
    """
    Add total_score, percent_correct, performance_level, etc.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if len(numeric_cols) == 0:
        return df
    df["total_score"] = df[numeric_cols].sum(axis=1, skipna=True)
    df["percent_correct"] = (df["total_score"] / len(numeric_cols)) * 100
    df["performance_level"] = pd.cut(
        df["percent_correct"],
        bins=[0, 40, 60, 80, 100],
        labels=["Low", "Medium", "High", "Excellent"],
    )
    return df
@st.cache_data(show_spinner=False)
def compute_descriptive(df):
    """
    Compute mean, median, std, skew, kurtosis for total_score.
    """
    if "total_score" not in df.columns:
        return {}
    scores = df["total_score"].dropna()
    if len(scores) == 0:
        return {}
    desc = {
        "count": len(scores),
        "mean": scores.mean(),
        "median": scores.median(),
        "std": scores.std(),
        "min": scores.min(),
        "max": scores.max(),
        "skewness": scores.skew(),
        "kurtosis": scores.kurtosis(),
    }
    return desc
@st.cache_data(show_spinner=False)
def compute_item_analysis(df):
    """
    For each item (column), compute:
    - mean (item_mean)
    - difficulty (item_mean)
    - discrimination (point-biserial corr)
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "total_score" in numeric_cols:
        numeric_cols = numeric_cols.drop("total_score")
    results = []
    for col in numeric_cols:
        item_scores = df[col].dropna()
        if len(item_scores) == 0:
            continue
        item_mean = item_scores.mean()
        difficulty = item_mean
        # discrimination: correlation with total_score
        if "total_score" in df.columns:
            valid = df[[col, "total_score"]].dropna()
            if len(valid) > 1:
                corr = valid[col].corr(valid["total_score"])
            else:
                corr = np.nan
        else:
            corr = np.nan
        results.append(
            {
                "Item": col,
                "Mean": round(item_mean, 2),
                "Difficulty": round(difficulty, 2),
                "Discrimination": round(corr, 2) if not pd.isna(corr) else np.nan,
            }
        )
    return pd.DataFrame(results)
@st.cache_data(show_spinner=False)
def compute_reliability(df):
    """
    Cronbach's alpha for internal consistency.
    """
    numeric_cols = df.select_dtypes(include=[np.number]).columns
    if "total_score" in numeric_cols:
        numeric_cols = numeric_cols.drop("total_score")
    if len(numeric_cols) < 2:
        return None
    item_data = df[numeric_cols].dropna()
    if item_data.shape[0] < 2:
        return None
    n_items = item_data.shape[1]
    item_vars = item_data.var(axis=0, ddof=1)
    total_var = item_data.sum(axis=1).var(ddof=1)
    if total_var == 0:
        return 0
    alpha = (n_items / (n_items - 1)) * (1 - (item_vars.sum() / total_var))
