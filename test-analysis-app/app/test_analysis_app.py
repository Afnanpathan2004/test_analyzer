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
    SKLEARN_AVAILABLE = True
except Exception:
    SKLEARN_AVAILABLE = False
st.set_page_config(page_title="Pre/Post Test Analyzer (Advanced)", layout="wide")
st.title("📊 Pre-test / Post-test Analyzer — Advanced edition")
st.markdown(
    """
Upload **pre-test** and **post-test** files in either **wide** or **long** formats.
The app will automatically clean, preprocess, engineer features, and run a thorough analysis:
- Per-employee scores, improvement, top/bottom performers
- Per-question difficulty & discrimination
- Cronbach's alpha (test reliability)
- Paired t-test / Wilcoxon and Cohen's d effect size
- Visualizations & export (Excel/PDF)
If your file is from Microsoft Forms and already contains scores (0/1), upload as **wide**:
`employee_name, Q1, Q2, Q3, ...` (cells 0/1 or 'Correct'/'Wrong' etc.)
If your file is long (one row per answer), upload:
`employee_id (or name), question_id, answer` and optionally an answer key.
"""
)
# -------------------------
# File system paths
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)
# -------------------------
# Utilities / Helpers
# -------------------------
def read_uploaded_file(uploaded):
    if uploaded is None:
        return None
    try:
        name = uploaded.name.lower()
        if name.endswith((".xls", ".xlsx")):
            return pd.read_excel(uploaded)
        else:
            return pd.read_csv(uploaded)
    except Exception as e:
        st.error(f"Failed to read {uploaded.name}: {e}")
        return None

def save_uploaded_file(uploaded, folder=UPLOADS_DIR):
    if uploaded is None:
        return None
    os.makedirs(folder, exist_ok=True)
    path = os.path.join(folder, uploaded.name)
    with open(path, "wb") as f:
        f.write(uploaded.getbuffer())
    return path

def _find_col_ignore_case(df, candidate):
    if candidate is None:
        return None
    cand = str(candidate).strip().lower()
    for c in df.columns:
        if c.strip().lower() == cand:
            return c
    for c in df.columns:
        if cand in c.strip().lower() or c.strip().lower() in cand:
            return c
    return None

def detect_format_and_columns(df):
    """
    Heuristic detection for 'wide' vs 'long'.
    Returns dict with keys: format, employee_col, question_col, answer_col, question_cols.
    """
    if df is None:
        return {"format":"unknown"}
    cols = list(df.columns)
    low = [c.strip().lower() for c in cols]
    # name-like column
    name_col = None
    for c in cols:
        lc = c.strip().lower()
        if any(k in lc for k in ("name","employee","emp","participant","user")):
            name_col = c
            break
    # wide-style question columns: Q1, q_1, question1 etc.
    question_cols = [c for c in cols if c.strip().lower().startswith("q") and any(ch.isdigit() for ch in c)]
    if not question_cols:
        for c in cols:
            lc = c.strip().lower()
            if lc.startswith("question") and any(ch.isdigit() for ch in lc):
                question_cols.append(c)
    # consider wide when name detected and >=2 question columns
    if name_col is not None and len(question_cols) >= 2:
        return {"format":"wide","employee_col":name_col,"question_cols":question_cols}
    # long detection
    question_col = None
    answer_col = None
    emp_col = None
    for c in cols:
        lc = c.strip().lower()
        if question_col is None and ("question" in lc or lc in ("q","question_id","questionid")):
            question_col = c
        if answer_col is None and ("answer" in lc or "response" in lc):
            answer_col = c
        if emp_col is None and any(k in lc for k in ("employee","emp","name","user")):
            emp_col = c
    if question_col and answer_col:
        return {"format":"long","employee_col":emp_col or "employee_id","question_col":question_col,"answer_col":answer_col}
    # fallback-wide: if name exists and many other cols
    if name_col is not None:
        other = [c for c in cols if c != name_col]
        if len(other) >= 2:
            return {"format":"wide","employee_col":name_col,"question_cols":other}
    return {"format":"unknown","employee_col":name_col,"question_col":question_col,"answer_col":answer_col,"question_cols":question_cols}

# Clean string-like columns: strip, unify spaces

def clean_string_columns(df):
    for c in df.select_dtypes(include=["object"]).columns:
        df[c] = df[c].astype(str).map(lambda s: " ".join(s.strip().split()))
    return df

def normalize_name(name):
    if pd.isna(name):
        return ""
    s = str(name).strip()
    # optionally lower-case? Keep capitalization but unify spacing
    s = " ".join(s.split())
    return s

def safe_int(x, default=0):
    try:
        return int(x)
    except Exception:
        return default

# wide cell -> binary (0/1)

def wide_cell_to_binary(x):
    if pd.isna(x):
        return np.nan
    if isinstance(x, (int, np.integer)):
        return 1 if int(x) == 1 else 0
    if isinstance(x, (float, np.floating)):
        if math.isfinite(x):
            # treat 1.0 as correct; percentages: >=0.5 -> correct
            return 1 if float(x) >= 0.5 else 0
        return 0
    s = str(x).strip().lower()
    if s in ("1","true","t","yes","y","correct","right","pass"):
        return 1
    if s in ("0","false","f","no","n","incorrect","wrong","fail"):
        return 0
    # try numeric
    try:
        v = float(s)
        return 1 if v >= 0.5 else 0
    except Exception:
        return 0

# convert wide scored df -> long format (employee_name, question_id, correct)

def wide_to_long_scored(df, employee_col, question_cols=None):
    if df is None or df.empty:
        return pd.DataFrame(columns=["employee_name","question_id","correct"])
    # normalize column names: keep as-is but find actual employee col
    emp_col_actual = _find_col_ignore_case(df, employee_col) or employee_col
    if emp_col_actual not in df.columns:
        # try first column as employee
        emp_col_actual = df.columns[0]
    if question_cols:
        qcols_actual = [c for c in question_cols if c in df.columns]
    else:
        qcols_actual = [c for c in df.columns if c != emp_col_actual]
    rows = []
    for _, r in df.iterrows():
        emp = normalize_name(r.get(emp_col_actual, ""))
        if emp == "":
            continue
        for q in qcols_actual:
            val = wide_cell_to_binary(r[q])
            # treat np.nan as missing: skip? we will count as unanswered
            if pd.isna(val):
                continue
            rows.append({"employee_name": emp, "question_id": str(q).strip(), "correct": int(val)})
    if not rows:
        return pd.DataFrame(columns=["employee_name","question_id","correct"])
    return pd.DataFrame(rows)

# map key df to standard

def map_key_df_to_standard(key_df):
    if key_df is None:
        return None
    q = None; corr = None
    for col in key_df.columns:
        lc = col.strip().lower()
        if q is None and ("question" in lc or lc.startswith("q") or lc.endswith("id")):
            q = col
        if corr is None and ("correct" in lc or "answer" in lc or "key" in lc):
            corr = col
    if q is None and len(key_df.columns) >= 1:
        q = key_df.columns[0]
    if corr is None and len(key_df.columns) >= 2:
        for c in key_df.columns:
            if c != q:
                corr = c; break
    if q and corr:
        return key_df.rename(columns={q:"question_id", corr:"correct_answer"})
    return key_df

# compute per-employee and per-question for already-scored long df: columns employee_name, question_id, correct (0/1)

def compute_scores_from_long_scored(df_long):
    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=["employee_name","score","num_answered"]), pd.DataFrame(columns=["question_id","pct_correct"]), pd.DataFrame()
    df = df_long.copy()
    if "employee_name" not in df.columns and "employee_id" in df.columns:
        df = df.rename(columns={"employee_id":"employee_name"})
    df["employee_name"] = df["employee_name"].apply(lambda x: normalize_name(x))
    df = df.dropna(subset=["employee_name","question_id"])
    df["correct"] = pd.to_numeric(df["correct"], errors="coerce").fillna(0).astype(int)
    per_emp = df.groupby("employee_name")["correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"score","count":"num_answered"})
    per_emp["score"] = (per_emp["score"]*100).round(2)
    per_q = df.groupby("question_id")["correct"].mean().reset_index().rename(columns={"correct":"pct_correct"})
    per_q["pct_correct"] = (per_q["pct_correct"]*100).round(2)
    return per_emp, per_q, df

# compute scores from long answers (need key or modal)

def compute_scores_from_long_answers(test_df, key_df=None, use_modal=False,
                   employee_col="employee_id", question_col="question_id", answer_col="answer"):
    df = test_df.copy()
    # flexible column locate
    emp_actual = _find_col_ignore_case(df, employee_col) or employee_col
    q_actual = _find_col_ignore_case(df, question_col) or question_col
    a_actual = _find_col_ignore_case(df, answer_col) or answer_col
    if emp_actual not in df.columns or q_actual not in df.columns or a_actual not in df.columns:
        raise ValueError("Long-format file must contain employee, question and answer columns (use manual mapping).")
    df = df.rename(columns={emp_actual:"employee_id", q_actual:"question_id", a_actual:"answer"})
    df = df.dropna(subset=["employee_id","question_id"])
    df["employee_id"] = df["employee_id"].apply(lambda x: normalize_name(x))
    df["question_id"] = df["question_id"].astype(str)
    df["answer"] = df["answer"].astype(str)
    if key_df is not None:
        k = key_df.copy()
        if "question_id" not in k.columns or "correct_answer" not in k.columns:
            k = map_key_df_to_standard(k)
        if "question_id" not in k.columns or "correct_answer" not in k.columns:
            raise ValueError("Answer key missing question_id / correct_answer columns.")
        k["question_id"] = k["question_id"].astype(str)
        k["correct_answer"] = k["correct_answer"].astype(str)
        merged = df.merge(k[["question_id","correct_answer"]], on="question_id", how="left")
        merged["correct"] = (merged["answer"].str.strip() == merged["correct_answer"].str.strip()).astype(int)
        per_emp = merged.groupby("employee_id")["correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"score","count":"num_answered"})
        per_emp["score"] = (per_emp["score"]*100).round(2)
        per_q = merged.groupby("question_id")["correct"].mean().reset_index().rename(columns={"correct":"pct_correct"})
        per_q["pct_correct"] = (per_q["pct_correct"]*100).round(2)
        merged = merged.rename(columns={"employee_id":"employee_name"})
        merged = merged[["employee_name","question_id","correct","answer","correct_answer"]]
        return per_emp.rename(columns={"employee_id":"employee_name"}), per_q, merged
    if use_modal:
        modal = df.groupby("question_id")["answer"].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan).reset_index().rename(columns={"answer":"modal_answer"})
        merged = df.merge(modal, on="question_id", how="left")
        merged["correct"] = (merged["answer"].astype(str) == merged["modal_answer"].astype(str)).astype(int)
        per_emp = merged.groupby("employee_id")["correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"score","count":"num_answered"})
        per_emp["score"] = (per_emp["score"]*100).round(2)
        per_q = merged.groupby("question_id")["correct"].mean().reset_index().rename(columns={"correct":"pct_match_modal"})
        per_q["pct_match_modal"] = (per_q["pct_match_modal"]*100).round(2)
        merged = merged.rename(columns={"employee_id":"employee_name"})
        merged = merged[["employee_name","question_id","correct","answer","modal_answer"]]
        return per_emp.rename(columns={"employee_id":"employee_name"}), per_q, merged
    raise ValueError("No scoring method available for long answers (provide key or enable modal).")

# Cronbach's alpha

def cronbach_alpha(itemscores_df):
    """
    itemscores_df: rows = respondents, cols = items (binary 0/1 or scores)
    returns Cronbach's alpha
    """
    if itemscores_df is None or itemscores_df.shape[1] < 2:
        return np.nan
    # convert to numeric
    data = itemscores_df.apply(pd.to_numeric, errors="coerce").fillna(0)
    item_vars = data.var(axis=0, ddof=1)
    total_var = data.sum(axis=1).var(ddof=1)
    n_items = data.shape[1]
    if total_var == 0:
        return np.nan
    alpha = (n_items / (n_items - 1
