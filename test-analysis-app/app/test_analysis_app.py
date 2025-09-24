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

warnings.filterwarnings("ignore")

# Optional libs for PDF/chart/stats/clustering
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    import matplotlib.pyplot as plt
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
            # treat 1.0 as correct; percentages: >0.5 -> correct
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
    alpha = (n_items / (n_items - 1.0)) * (1.0 - item_vars.sum() / total_var)
    return round(float(alpha), 4)

# point-biserial for discrimination
def point_biserial(grouped_long_df, question_id):
    """
    grouped_long_df: long scored df with columns employee_name, question_id, correct (0/1)
    compute point-biserial correlation for specific question vs total score (excluding that item)
    returns r, pval
    """
    if not SCIPY_AVAILABLE:
        return np.nan, np.nan
    df = grouped_long_df.copy()
    # pivot to get item matrix
    pivot = df.pivot_table(index="employee_name", columns="question_id", values="correct", fill_value=0)
    if question_id not in pivot.columns:
        return np.nan, np.nan
    y = pivot[question_id].values
    # total without that item
    total = pivot.sum(axis=1) - pivot[question_id]
    # if variance zero, cannot compute
    if np.var(y) == 0 or np.var(total) == 0:
        return np.nan, np.nan
    try:
        r, p = stats.pointbiserialr(y, total)
        return float(r), float(p)
    except Exception:
        return np.nan, np.nan

# Cohen's d for paired samples
def cohens_d_paired(a, b):
    a = np.asarray(a); b = np.asarray(b)
    d = a - b
    # paired Cohen's d = mean(diff) / sd(diff)
    md = np.mean(d)
    sd = np.std(d, ddof=1)
    if sd == 0:
        return np.nan
    return round(md / sd, 4)

# paired tests
def paired_tests(pre_scores, post_scores, alpha=0.05):
    """
    pre_scores/post_scores: pandas Series for the employees present in both, indexed same order
    returns dict with t-test and wilcoxon results and effect size
    """
    res = {}
    pre = np.asarray(pre_scores); post = np.asarray(post_scores)
    if len(pre) == 0:
        return None
    # paired t-test
    if SCIPY_AVAILABLE:
        try:
            tstat, tp = stats.ttest_rel(post, pre, nan_policy="omit")
            res["paired_t_stat"] = float(tstat); res["paired_t_p"] = float(tp)
        except Exception:
            res["paired_t_stat"] = np.nan; res["paired_t_p"] = np.nan
        # wilcoxon
        try:
            wstat, wp = stats.wilcoxon(post, pre)
            res["wilcoxon_stat"] = float(wstat); res["wilcoxon_p"] = float(wp)
        except Exception:
            res["wilcoxon_stat"] = np.nan; res["wilcoxon_p"] = np.nan
    else:
        res["paired_t_stat"] = res["paired_t_p"] = res["wilcoxon_stat"] = res["wilcoxon_p"] = np.nan

    # effect size (Cohen's d)
    res["cohens_d_paired"] = cohens_d_paired(post, pre)
    return res

# clustering helper (optional)
def cluster_learners(feature_df, n_clusters=3, random_state=42):
    if feature_df is None or feature_df.shape[0] < n_clusters:
        return None
    if not SKLEARN_AVAILABLE:
        return None
    km = KMeans(n_clusters=n_clusters, random_state=random_state, n_init=10)
    labels = km.fit_predict(feature_df.fillna(0))
    return labels

# caching heavy operations
@st.cache_data
def compute_item_matrix(long_df):
    """Return pivot table employees x questions with 0/1 values"""
    if long_df is None or long_df.empty:
        return pd.DataFrame()
    pivot = long_df.pivot_table(index="employee_name", columns="question_id", values="correct", aggfunc="max", fill_value=0)
    return pivot

# -------------------------
# Sidebar: Uploads & options
# -------------------------
st.sidebar.header("Files & Options")
pre_file = st.sidebar.file_uploader("Upload PRE-test (Excel/CSV)", type=["csv","xls","xlsx"])
post_file = st.sidebar.file_uploader("Upload POST-test (Excel/CSV)", type=["csv","xls","xlsx"])
key_file = st.sidebar.file_uploader("Optional: Answer Key (Excel/CSV) — for long format only", type=["csv","xls","xlsx"])
st.sidebar.markdown("---")
use_modal = st.sidebar.checkbox("If no key (long format): use modal answer as proxy", value=True)
save_uploads = st.sidebar.checkbox("Save uploaded files to disk (data/uploads/)", value=False)
pdf_include_chart = st.sidebar.checkbox("Include chart in PDF", value=True)
pdf_top_n = st.sidebar.number_input("Top N rows in PDF", min_value=5, max_value=200, value=20, step=5)
alpha = st.sidebar.number_input("Significance level (alpha) for tests", min_value=0.001, max_value=0.2, value=0.05, step=0.01)
do_clustering = st.sidebar.checkbox("Cluster learners (optional)", value=False)
cluster_k = st.sidebar.number_input("K clusters (if clustering)", min_value=2, max_value=10, value=3, step=1)
save_reports_to_outputs = st.sidebar.checkbox("Save generated reports to outputs/ folder", value=False)

if not REPORTLAB_AVAILABLE:
    st.sidebar.info("PDF export disabled: install reportlab & matplotlib to enable it.")
if not SCIPY_AVAILABLE:
    st.sidebar.info("Statistical tests disabled: install scipy to enable them.")
if do_clustering and not SKLEARN_AVAILABLE:
    st.sidebar.info("Clustering disabled: install scikit-learn to enable clustering.")

# -------------------------
# Load files
# -------------------------
pre_df = read_uploaded_file(pre_file) if pre_file else None
post_df = read_uploaded_file(post_file) if post_file else None
key_df = read_uploaded_file(key_file) if key_file else None

if save_uploads:
    if pre_file: save_uploaded_file(pre_file)
    if post_file: save_uploaded_file(post_file)
    if key_file: save_uploaded_file(key_file)

# demo generator
if st.sidebar.button("Generate demo (wide & long)"):
    employees = ["Afnan","Bina","Carlos","Devi","Eshan","Fatima"]
    questions = [f"Q{i}" for i in range(1,11)]
    # wide pre/post with improvement
    rng = np.random.default_rng(42)
    pre_wide = pd.DataFrame([[e] + list(rng.integers(0,2,size=len(questions))) for e in employees], columns=["employee_name"]+questions)
    # post slightly better
    post_wide = pre_wide.copy()
    for q in questions:
        flip = rng.choice(range(len(employees)), size=1)
        post_wide[q] = (pre_wide[q] | rng.integers(0,2,size=len(employees))).astype(int)
    # long format demos
    pre_long_rows = []
    post_long_rows = []
    choices = ["A","B","C","D"]
    key_demo = pd.DataFrame({"question_id":questions, "correct_answer":[rng.choice(choices) for _ in questions]})
    for e in employees:
        for q in questions:
            pre_long_rows.append({"employee_id":e, "question_id":q, "answer":rng.choice(choices)})
            post_long_rows.append({"employee_id":e, "question_id":q, "answer":rng.choice(choices)})
    pre_long = pd.DataFrame(pre_long_rows)
    post_long = pd.DataFrame(post_long_rows)
    st.session_state["_demo_pre_wide"] = pre_wide
    st.session_state["_demo_post_wide"] = post_wide
    st.session_state["_demo_pre_long"] = pre_long
    st.session_state["_demo_post_long"] = post_long
    st.session_state["_demo_key"] = key_demo
    st.success("Demo generated. Use 'Use demo (wide)' or 'Use demo (long)' buttons.")

if st.button("Use demo (wide)"):
    if "_demo_pre_wide" in st.session_state:
        pre_df = st.session_state["_demo_pre_wide"]
        post_df = st.session_state["_demo_post_wide"]
        key_df = None
    else:
        st.warning("Generate demo first.")

if st.button("Use demo (long)"):
    if "_demo_pre_long" in st.session_state:
        pre_df = st.session_state["_demo_pre_long"]
        post_df = st.session_state["_demo_post_long"]
        key_df = st.session_state.get("_demo_key")
    else:
        st.warning("Generate demo first.")

# -------------------------
# Previews and detection
# -------------------------
def show_preview_and_detect(df, label):
    st.subheader(f"Preview: {label}")
    st.dataframe(df.head(10))
    det = detect_format_and_columns(df)
    st.write("Auto-detected:", det)
    return det

p_det = s_det = None
if pre_df is not None:
    p_det = show_preview_and_detect(pre_df, "Pre-test")
if post_df is not None:
    s_det = show_preview_and_detect(post_df, "Post-test")
if key_df is not None:
    st.subheader("Preview: Answer Key (if provided)")
    st.dataframe(key_df.head(20))

# Manual mapping UI
st.markdown("---")
st.header("Manual mapping (override auto-detection if needed)")
col1, col2, col3 = st.columns(3)
with col1:
    pre_emp_col = st.text_input("Pre: employee/name column", value=(p_det or {}).get("employee_col") or "employee_name")
    post_emp_col = st.text_input("Post: employee/name column", value=(s_det or {}).get("employee_col") or "employee_name")
with col2:
    pre_q_col = st.text_input("Pre (long): question column", value=(p_det or {}).get("question_col") or "question_id")
    post_q_col = st.text_input("Post (long): question column", value=(s_det or {}).get("question_col") or "question_id")
with col3:
    pre_ans_col = st.text_input("Pre (long): answer column", value=(p_det or {}).get("answer_col") or "answer")
    post_ans_col = st.text_input("Post (long): answer column", value=(s_det or {}).get("answer_col") or "answer")

st.markdown("**Optional (wide):** If auto-detection misses question columns, enter them comma-separated (e.g. `Q1,Q2,Q3`)")
wide_pre_qcols_text = st.text_input("Pre: question columns (wide)", value=",".join((p_det or {}).get("question_cols",[]) or []))
wide_post_qcols_text = st.text_input("Post: question columns (wide)", value=",".join((s_det or {}).get("question_cols",[]) or []))
def parse_qcols(text):
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]
wide_pre_qcols = parse_qcols(wide_pre_qcols_text)
wide_post_qcols = parse_qcols(wide_post_qcols_text)

# -------------------------
# Processing pipeline
# -------------------------
def process_uploaded_test(df, det,
                          emp_col_override=None, question_col_override=None, answer_col_override=None,
                          wide_qcols_override=None, key_df_local=None, use_modal_local=True):
    """
    Process an uploaded test file (either wide or long), return:
    per_emp_df, per_q_df, merged_long_df, details dict
    """
    if df is None:
        return None, None, None, det
    fmt = det.get("format","unknown") if isinstance(det, dict) else "unknown"
    # if unknown, use overrides
    if fmt == "unknown":
        if wide_qcols_override:
            fmt = "wide"
            det = {"format":"wide","employee_col":emp_col_override or det.get("employee_col"), "question_cols": wide_qcols_override}
        elif question_col_override and answer_col_override:
            fmt = "long"
            det = {"format":"long","employee_col":emp_col_override or det.get("employee_col"), "question_col":question_col_override, "answer_col":answer_col_override}
        else:
            det = detect_format_and_columns(df)
            fmt = det.get("format","unknown")

    if fmt == "wide":
        emp_col = emp_col_override or det.get("employee_col")
        qcols = wide_qcols_override or det.get("question_cols") or []
        if not qcols:
            qcols = [c for c in df.columns if c != (_find_col_ignore_case(df, emp_col) or emp_col)]
        long = wide_to_long_scored(df, emp_col, qcols)
        per_emp, per_q, merged = compute_scores_from_long_scored(long)
        details = {"format":"wide","employee_col":emp_col,"question_cols":qcols}
        return per_emp, per_q, merged, details

    if fmt == "long":
        try:
            per_emp, per_q, merged = compute_scores_from_long_answers(df,
                                                                      key_df=key_df_local,
                                                                      use_modal=use_modal_local,
                                                                      employee_col=emp_col_override or det.get("employee_col"),
                                                                      question_col=question_col_override or det.get("question_col"),
                                                                      answer_col=answer_col_override or det.get("answer_col"))
            details = {"format":"long","employee_col":emp_col_override or det.get("employee_col"),
                       "question_col":question_col_override or det.get("question_col"),
                       "answer_col": answer_col_override or det.get("answer_col")}
            return per_emp, per_q, merged, details
        except Exception as e:
            raise

    raise ValueError("Unable to determine file format. Use manual mapping fields to help the app.")

# run processing for pre and post
pre_per_emp = pre_per_q = pre_merged = None
post_per_emp = post_per_q = post_merged = None
errors = []

if pre_df is None and post_df is None:
    st.info("Upload at least one pre or post file (wide or long).")
else:
    # pre processing
    if pre_df is not None:
        try:
            pre_det_local = p_det or detect_format_and_columns(pre_df)
            pre_per_emp, pre_per_q, pre_merged, pre_det_local = process_uploaded_test(
                pre_df, pre_det_local,
                emp_col_override=pre_emp_col,
                question_col_override=pre_q_col,
                answer_col_override=pre_ans_col,
                wide_qcols_override=wide_pre_qcols,
                key_df_local=key_df,
                use_modal_local=use_modal
            )
        except Exception as e:
            errors.append(f"Pre-test error: {e}")
            pre_per_emp = pre_per_q = pre_merged = None

    # post processing
    if post_df is not None:
        try:
            post_det_local = s_det or detect_format_and_columns(post_df)
            post_per_emp, post_per_q, post_merged, post_det_local = process_uploaded_test(
                post_df, post_det_local,
                emp_col_override=post_emp_col,
                question_col_override=post_q_col,
                answer_col_override=post_ans_col,
                wide_qcols_override=wide_post_qcols,
                key_df_local=key_df,
                use_modal_local=use_modal
            )
        except Exception as e:
            errors.append(f"Post-test error: {e}")
            post_per_emp = post_per_q = post_merged = None

# show errors
if errors:
    for e in errors:
        st.error(e)

# -------------------------
# Analysis & Visualization
# -------------------------
if pre_per_emp is None and post_per_emp is None:
    st.info("No processed scoring data available to analyze.")
else:
    # compute employee lists
    pre_names = set(pre_per_emp["employee_name"].astype(str).unique()) if pre_per_emp is not None else set()
    post_names = set(post_per_emp["employee_name"].astype(str).unique()) if post_per_emp is not None else set()
    both = sorted(list(pre_names.intersection(post_names)))
    pre_only = sorted(list(pre_names - post_names))
    post_only = sorted(list(post_names - pre_names))

    c1,c2,c3 = st.columns(3)
    c1.metric("Pre only", len(pre_only))
    c2.metric("Post only", len(post_only))
    c3.metric("Both", len(both))

    with st.expander("See employee lists"):
        st.subheader("Pre only")
        st.write(pre_only)
        st.subheader("Post only")
        st.write(post_only)
        st.subheader("Both")
        st.write(both)

    # Merge per-employee scores (employee_name)
    if pre_per_emp is None:
        pre_per_emp = pd.DataFrame(columns=["employee_name","score","num_answered"])
    if post_per_emp is None:
        post_per_emp = pd.DataFrame(columns=["employee_name","score","num_answered"])

    pre_per_emp = pre_per_emp.rename(columns={"score":"pre_score","num_answered":"pre_answers"})
    post_per_emp = post_per_emp.rename(columns={"score":"post_score","num_answered":"post_answers"})

    merged_scores = pd.merge(pre_per_emp, post_per_emp, on="employee_name", how="outer")
    merged_scores["pre_score"] = merged_scores["pre_score"].fillna(0)
    merged_scores["post_score"] = merged_scores["post_score"].fillna(0)
    merged_scores["pre_answers"] = merged_scores["pre_answers"].fillna(0).astype(int)
    merged_scores["post_answers"] = merged_scores["post_answers"].fillna(0).astype(int)
    merged_scores["improvement"] = (merged_scores["post_score"] - merged_scores["pre_score"]).round(2)

    def status_row(r):
        en = r["employee_name"]
        if en in both: return "both"
        if en in pre_only: return "pre_only"
        if en in post_only: return "post_only"
        return "unknown"
    merged_scores["status"] = merged_scores.apply(status_row, axis=1)

    st.subheader("Employee scores & improvement")
    st.dataframe(merged_scores.sort_values("improvement", ascending=False).reset_index(drop=True))

    # KPIs
    avg_pre = round(float(merged_scores["pre_score"].mean()), 2) if not merged_scores.empty else 0.0
    avg_post = round(float(merged_scores["post_score"].mean()), 2) if not merged_scores.empty else 0.0
    avg_impr = round((avg_post - avg_pre), 2)
    k1,k2,k3 = st.columns(3)
    k1.metric("Average pre score (%)", avg_pre)
    k2.metric("Average post score (%)", avg_post)
    k3.metric("Average improvement (pp)", avg_impr)

    # Charts
    st.subheader("Score distributions and comparisons")
    hist_df = merged_scores[["employee_name","pre_score","post_score"]].melt(id_vars="employee_name", var_name="test", value_name="score")
    st.bar_chart(hist_df.groupby("test")["score"].mean())

    # Individual distributions
    st.write("Pre / Post score histogram")
    st.pyplot(plt.figure(figsize=(6,3)))
    try:
        fig, ax = plt.subplots(1,1,figsize=(8,4))
        ax.hist(merged_scores["pre_score"], bins=10, alpha=0.6, label="Pre")
        ax.hist(merged_scores["post_score"], bins=10, alpha=0.6, label="Post")
        ax.set_xlabel("Score (%)")
        ax.set_ylabel("Count")
        ax.legend()
        st.pyplot(fig)
    except Exception:
        pass

    # Per-question analysis
    # Build per-question tables depending on availability
    if pre_per_q is not None and post_per_q is not None:
        pre_q = pre_per_q.rename(columns={pre_per_q.columns[0]:"question_id", pre_per_q.columns[1]:"pre_pct"}).set_index("question_id")
        post_q = post_per_q.rename(columns={post_per_q.columns[0]:"question_id", post_per_q.columns[1]:"post_pct"}).set_index("question_id")
        per_q = pre_q.join(post_q, how="outer").fillna(0)
        per_q["delta_pp"] = (per_q["post_pct"] - per_q["pre_pct"]).round(2)
        st.subheader("Per-question improvement (pre vs post)")
        st.dataframe(per_q.reset_index().sort_values("delta_pp", ascending=False))
        per_q_export = per_q.reset_index()
    else:
        # show whichever exists
        if pre_per_q is not None:
            st.subheader("Pre per-question stats")
            st.dataframe(pre_per_q)
            per_q_export = pre_per_q
        elif post_per_q is not None:
            st.subheader("Post per-question stats")
            st.dataframe(post_per_q)
            per_q_export = post_per_q
        else:
            per_q_export = pd.DataFrame(columns=["question_id","pre_pct","post_pct","delta_pp"])

    # Item-level analysis: point-biserial, item-total, cronbach
    st.subheader("Item-level statistics & reliability")

    # build combined item matrix for pre and post separately when available
    pre_item_matrix = compute_item_matrix(pre_merged) if pre_merged is not None else pd.DataFrame()
    post_item_matrix = compute_item_matrix(post_merged) if post_merged is not None else pd.DataFrame()

    # Cronbach's alpha for pre & post (if enough items)
    pre_alpha = cronbach_alpha(pre_item_matrix) if not pre_item_matrix.empty else np.nan
    post_alpha = cronbach_alpha(post_item_matrix) if not post_item_matrix.empty else np.nan
    st.write(f"Cronbach's alpha — Pre: {pre_alpha}   Post: {post_alpha}")

    # item discrimination (point-biserial) for pre (if scipy available)
    item_stats_rows = []
    if not pre_item_matrix.empty:
        # build long df from pre_merged to compute pb for each q
        for q in pre_item_matrix.columns:
            if SCIPY_AVAILABLE:
                try:
                    r, p = point_biserial(pre_merged, q)
                except Exception:
                    r, p = (np.nan, np.nan)
            else:
                r, p = (np.nan, np.nan)
            pct = (pre_item_matrix[q].mean()*100).round(2)
            var = pre_item_matrix[q].var(ddof=1)
            item_stats_rows.append({"question_id":q, "pre_pct_correct":pct, "pre_var":var, "pre_pointbiserial":r, "pre_pval":p})
    if not post_item_matrix.empty:
        for q in post_item_matrix.columns:
            pct = (post_item_matrix[q].mean()*100).round(2)
            var = post_item_matrix[q].var(ddof=1)
            # add or update existing row
            found = next((r for r in item_stats_rows if r["question_id"]==q), None)
            if found:
                found["post_pct_correct"] = pct
                found["post_var"] = var
            else:
                item_stats_rows.append({"question_id":q, "post_pct_correct":pct, "post_var":var})

    item_stats_df = pd.DataFrame(item_stats_rows).fillna(np.nan)
    if not item_stats_df.empty:
        st.dataframe(item_stats_df)
    else:
        st.info("Not enough item-level data for item statistics.")

    # Paired tests for employees in both
    if len(both) >= 2:
        paired = merged_scores[merged_scores["employee_name"].isin(both)].set_index("employee_name")
        pre_vals = paired["pre_score"]
        post_vals = paired["post_score"]
        st.subheader("Paired statistical tests (employees who took both)")
        if SCIPY_AVAILABLE:
            stats_res = paired_tests(pre_vals, post_vals, alpha=alpha)
            st.write("Paired t-test:", {"t_stat": stats_res.get("paired_t_stat"), "p": stats_res.get("paired_t_p")})
            st.write("Wilcoxon:", {"stat": stats_res.get("wilcoxon_stat"), "p": stats_res.get("wilcoxon_p")})
            st.write("Cohen's d (paired):", stats_res.get("cohens_d_paired"))
        else:
            st.info("Install scipy to enable paired statistical tests.")

        # show scatter of pre vs post
        try:
            fig, ax = plt.subplots(figsize=(6,6))
            ax.scatter(pre_vals, post_vals)
            ax.plot([0,100],[0,100], color='red', linestyle='--')
            ax.set_xlabel("Pre score (%)")
            ax.set_ylabel("Post score (%)")
            ax.set_title("Pre vs Post scatter (both)")
            st.pyplot(fig)
        except Exception:
            pass
    else:
        st.info("Not enough paired data to run paired tests (need >=2 employees with both).")

    # clustering (optional)
    if do_clustering:
        st.subheader("Clustering learners (optional)")
        # cluster by improvement or by item response patterns (pref: by improvement)
        cluster_feat = merged_scores[["pre_score","post_score","improvement"]].fillna(0)
        if cluster_feat.shape[0] >= cluster_k and SKLEARN_AVAILABLE:
            labels = cluster_learners(cluster_feat, n_clusters=int(cluster_k))
            merged_scores["cluster"] = labels
            st.dataframe(merged_scores.sort_values("cluster").head(50))
            st.write("Cluster counts:", merged_scores["cluster"].value_counts().to_dict())
        else:
            st.info("Not enough rows to cluster or scikit-learn not installed.")

    # top/bottom performers
    st.subheader("Top / Bottom performers")
    st.write("Top improvements")
    st.dataframe(merged_scores.sort_values("improvement", ascending=False).head(20))
    st.write("Largest declines")
    st.dataframe(merged_scores.sort_values("improvement", ascending=True).head(20))

    # question heatmap (if items present)
    if not pre_item_matrix.empty or not post_item_matrix.empty:
        st.subheader("Question heatmap (pre)")
        try:
            fig, ax = plt.subplots(figsize=(10, max(3, 0.4*pre_item_matrix.shape[0])))
            import seaborn as sns
            sns.heatmap(pre_item_matrix, cmap="YlGnBu", cbar=True, ax=ax)
            ax.set_title("Pre item correctness heatmap (rows=employees, cols=questions)")
            st.pyplot(fig)
        except Exception:
            # fallback simple table
            st.dataframe(pre_item_matrix.head(50))

    # -------------------------
    # Downloads & Reporting
    # -------------------------
    out_dfs = {
        "employee_scores": merged_scores,
        "pre_details": pre_merged if pre_merged is not None else pd.DataFrame(),
        "post_details": post_merged if post_merged is not None else pd.DataFrame(),
        "per_question": per_q_export if 'per_q_export' in locals() else pd.DataFrame(),
        "item_stats": item_stats_df
    }

    # Excel
    def make_downloadable_excel(dfs_dict):
        out = BytesIO()
        with pd.ExcelWriter(out, engine="openpyxl") as writer:
            for name, df in dfs_dict.items():
                safe_name = str(name)[:31]
                try:
                    df.to_excel(writer, sheet_name=safe_name, index=False)
                except Exception:
                    df.astype(str).to_excel(writer, sheet_name=safe_name, index=False)
        return out.getvalue()

    excel_bytes = make_downloadable_excel(out_dfs)
    st.markdown("---")
    st.subheader("Download / Save Reports")
    st.download_button("Download full report (Excel)", data=excel_bytes,
                       file_name=f"prepost_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if save_reports_to_outputs:
        pth = os.path.join(OUTPUTS_DIR, f"prepost_full_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx")
        with open(pth, "wb") as f:
            f.write(excel_bytes)
        st.write(f"Saved Excel report to: {pth}")

    # PDF summary
    def make_downloadable_pdf(merged_scores_local, per_q_local, include_chart=True, top_n=20):
        if not REPORTLAB_AVAILABLE:
            raise RuntimeError("reportlab/matplotlib not available in environment.")
        buf = BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=20,leftMargin=20, topMargin=20,bottomMargin=20)
        styles = getSampleStyleSheet()
        elements = []

        # Title and KPIs
        elements.append(Paragraph("Pre / Post Test Analysis — Summary", styles["Title"]))
        elements.append(Spacer(1,6))
        avg_pre_local = round(float(merged_scores_local["pre_score"].mean()), 2) if not merged_scores_local.empty else 0.0
        avg_post_local = round(float(merged_scores_local["post_score"].mean()), 2) if not merged_scores_local.empty else 0.0
        avg_impr_local = round(avg_post_local - avg_pre_local, 2)
        elements.append(Paragraph(f"Average Pre: {avg_pre_local}%    Average Post: {avg_post_local}%    Average Improvement: {avg_impr_local} pp", styles["Normal"]))
        elements.append(Spacer(1,8))

        # Chart
        if include_chart:
            try:
                fig, ax = plt.subplots(figsize=(6,3))
                ax.bar(["Pre (avg)","Post (avg)"], [avg_pre_local, avg_post_local])
                ax.set_ylim(0,100)
                ax.set_ylabel("Score (%)")
                ax.set_title("Average Pre vs Post")
                plt.tight_layout()
                img_buf = BytesIO()
                fig.savefig(img_buf, format="png", dpi=150)
                plt.close(fig)
                img_buf.seek(0)
                elements.append(RLImage(img_buf, width=400, height=200))
                elements.append(Spacer(1,8))
            except Exception:
                pass

        # Top improvements table
        elements.append(Paragraph(f"Top {top_n} employees by improvement", styles["Heading3"]))
        top_table = merged_scores_local.sort_values("improvement", ascending=False).head(top_n)
        tbl_data = [list(top_table.columns)] + top_table.astype(str).values.tolist()
        tbl = Table(tbl_data, repeatRows=1)
        tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#dbe9ff")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),7)]))
        elements.append(tbl)
        elements.append(Spacer(1,8))

        # Per-question summary
        elements.append(Paragraph("Per-question summary (excerpt)", styles["Heading3"]))
        if per_q_local is None or per_q_local.empty:
            elements.append(Paragraph("No per-question data available", styles["Normal"]))
        else:
            per_q_small = per_q_local.head(100)
            tbl2_data = [list(per_q_small.columns)] + per_q_small.astype(str).values.tolist()
            tbl2 = Table(tbl2_data, repeatRows=1)
            tbl2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#d7ffd9")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),7)]))
            elements.append(tbl2)

        doc.build(elements)
        buf.seek(0)
        return buf.getvalue()

    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = make_downloadable_pdf(merged_scores, per_q_export, include_chart=pdf_include_chart, top_n=int(pdf_top_n))
            st.download_button("Download summary report (PDF)", data=pdf_bytes,
                               file_name=f"prepost_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf")
            if save_reports_to_outputs:
                pth = os.path.join(OUTPUTS_DIR, f"prepost_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf")
                with open(pth, "wb") as f:
                    f.write(pdf_bytes)
                st.write(f"Saved PDF summary to: {pth}")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    else:
        st.info("PDF export disabled (reportlab/matplotlib not installed).")

st.markdown("---")
st.caption("Advanced Pre/Post Test Analyzer — automatic cleaning, preprocessing, item analysis, reliability, paired tests, and exports.")
