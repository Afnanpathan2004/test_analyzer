# test_analysis_app.py
"""
Streamlit app: Pre/Post Test Analyzer
Supports:
 - Long format (rows): employee_id / question_id / answer (string)
 - Wide format (rows): employee_name / Q1 / Q2 / Q3 ... where each Q cell is 0/1 or True/False or "correct"
   -> wide format is treated as *already scored* (1 = correct), so no answer key needed.
Other features:
 - Demo generator (wide + long)
 - Manual column mapping
 - Excel + PDF (reportlab) export
 - Save uploaded files
"""
import streamlit as st
import pandas as pd
import numpy as np
from io import BytesIO
import os
from datetime import datetime
import sys

# Optional PDF/chart libs
try:
    from reportlab.lib.pagesizes import A4, landscape
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, Image as RLImage
    from reportlab.lib import colors
    from reportlab.lib.styles import getSampleStyleSheet
    import matplotlib.pyplot as plt
    REPORTLAB_AVAILABLE = True
except Exception:
    REPORTLAB_AVAILABLE = False

st.set_page_config(page_title="Pre/Post Test Analyzer", layout="wide")
st.title("📊 Pre-test / Post-test Analysis App (supports wide & long formats)")
st.markdown(
    """
Upload pre-test and post-test files in *either* of two layouts:

**Long format (each row = one employee answer)**  
- Columns: `employee_id` (or `employee_name`), `question_id`, `answer`  
- App will score using an optional answer key or modal-proxy (as before).

**Wide format (each row = an employee, columns = questions with 0/1 correctness)**  
- Columns: `employee_name` (or similar), `Q1`, `Q2`, ... (values: `0/1`, `True/False`, or `correct/incorrect`)  
- Wide files are treated as *already-scored* — no answer key required.

This change lets you upload Microsoft Forms / MS Excel outputs where responses are pre-scored.
"""
)

# -------------------------
# Paths
# -------------------------
BASE_DIR = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
SAMPLES_DIR = os.path.join(BASE_DIR, "data", "samples")
UPLOADS_DIR = os.path.join(BASE_DIR, "data", "uploads")
OUTPUTS_DIR = os.path.join(BASE_DIR, "outputs")
os.makedirs(SAMPLES_DIR, exist_ok=True)
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(OUTPUTS_DIR, exist_ok=True)

# -------------------------
# Helpers
# -------------------------
def load_file(uploaded):
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
    Heuristic to detect whether df is:
    - 'long' format (has question_id-like and answer-like columns)
    - 'wide' format (has many columns like Q1,Q2,... and a name column)
    Returns dict with keys: format ('long'|'wide'), employee_col, question_col, answer_col, question_cols (list)
    """
    cols = list(df.columns)
    low = [c.strip().lower() for c in cols]

    # find name-like column
    name_col = None
    for c in cols:
        lc = c.strip().lower()
        if any(k in lc for k in ("name","employee","emp","participant","user")):
            name_col = c
            break

    # find question-like columns (wide)
    question_cols = [c for c in cols if c.strip().lower().startswith("q") and any(ch.isdigit() for ch in c)]
    # also accept "q1" "q_1" "question1"
    if not question_cols:
        for c in cols:
            lc = c.strip().lower()
            if lc.startswith("question") and any(ch.isdigit() for ch in lc):
                question_cols.append(c)

    # treat as wide if many question columns found (>=2) and name_col present
    if name_col is not None and len(question_cols) >= 2:
        return {"format": "wide", "employee_col": name_col, "question_cols": question_cols}

    # otherwise try long detection
    question_col = None
    answer_col = None
    emp_col = None
    for c in cols:
        lc = c.strip().lower()
        if question_col is None and ("question" in lc or lc in ("q","question_id","questionid")):
            question_col = c
        if answer_col is None and ("answer" in lc or "response" in lc):
            answer_col = c
        if emp_col is None and any(k in lc for k in ("employee","emp","name","user")) and ("id" in lc or "name" in lc or True):
            emp_col = c
    # fallback: if we find question_col and answer_col it's long
    if question_col and answer_col:
        return {"format": "long", "employee_col": emp_col or "employee_id", "question_col": question_col, "answer_col": answer_col}

    # final fallback: if many columns and name exists, try wide using all non-name columns as question columns
    if name_col is not None:
        other_qs = [c for c in cols if c != name_col]
        if len(other_qs) >= 2:
            return {"format": "wide", "employee_col": name_col, "question_cols": other_qs}

    # nothing detected -> return unknown and allow manual mapping
    return {"format": "unknown", "employee_col": name_col, "question_col": question_col, "answer_col": answer_col, "question_cols":[]}

def wide_to_long_scored(df, employee_col, question_cols):
    """
    Convert a wide scored dataframe (1/0 or True/False) into a 'long' scored dataframe with columns:
    employee_name, question_id, correct (1/0)
    """
    if df is None:
        return pd.DataFrame(columns=["employee_name","question_id","correct"])
    # map typical truthy strings to 1/0
    def cell_to_binary(x):
        if pd.isna(x):
            return np.nan
        if isinstance(x, (int, float, np.integer, np.floating)):
            # treat numeric 1 as correct, anything else 0
            if np.isfinite(x) and int(x) == 1:
                return 1
            if np.isfinite(x) and int(x) == 0:
                return 0
            # sometimes scores can be percentages; we interpret >0.5 as correct
            try:
                return 1 if float(x) > 0.5 else 0
            except Exception:
                return 0
        s = str(x).strip().lower()
        if s in ("1","true","t","yes","y","correct","right"):
            return 1
        if s in ("0","false","f","no","n","incorrect","wrong"):
            return 0
        # try numeric parse
        try:
            v = float(s)
            return 1 if v > 0.5 else 0
        except Exception:
            return 0

    # select employee name col normalized
    emp_col_actual = _find_col_ignore_case(df, employee_col) or employee_col
    # ensure question_cols exist
    qcols_actual = [c for c in question_cols if c in df.columns]
    if not qcols_actual:
        # fallback: use all except employee col
        qcols_actual = [c for c in df.columns if c != emp_col_actual]

    rows = []
    for _, r in df.iterrows():
        emp = r.get(emp_col_actual, None)
        if pd.isna(emp):
            continue
        for q in qcols_actual:
            val = cell_to_binary(r[q])
            rows.append({"employee_name": str(emp).strip(), "question_id": str(q).strip(), "correct": int(val)})
    long = pd.DataFrame(rows)
    return long

def compute_scores_from_long_scored(df_long):
    """
    df_long expected columns: employee_name / question_id / correct (0/1)
    returns per_emp (employee_name, score, num_answered), per_q (question_id, pct_correct), merged_long (with correct)
    """
    if df_long is None or df_long.empty:
        return pd.DataFrame(columns=["employee_name","score","num_answered"]), pd.DataFrame(), pd.DataFrame()
    # normalize names
    df = df_long.copy()
    if "employee_name" not in df.columns:
        # if employee_id present, rename
        if "employee_id" in df.columns:
            df = df.rename(columns={"employee_id":"employee_name"})
        else:
            df = df.rename(columns={df.columns[0]:"employee_name"})
    df["correct"] = df["correct"].astype(float)
    per_emp = df.groupby("employee_name")["correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"score","count":"num_answered"})
    per_emp["score"] = (per_emp["score"]*100).round(2)
    per_q = df.groupby("question_id")["correct"].mean().reset_index().rename(columns={"correct":"pct_correct"})
    per_q["pct_correct"] = (per_q["pct_correct"]*100).round(2)
    return per_emp, per_q, df

def compute_scores_from_long_answers(test_df, key_df=None, use_modal=False,
                   employee_col="employee_id", question_col="question_id", answer_col="answer"):
    """
    The original scoring path: test_df has employee_id/question_id/answer and we score using key_df or modal proxy.
    Same behavior as earlier app.
    """
    # This reuses previous logic but uses column names passed in.
    df = test_df.copy()
    for c in (employee_col, question_col, answer_col):
        if c not in df.columns:
            raise ValueError(f"Column '{c}' not present in the test file.")
    df = df[[employee_col, question_col, answer_col]].dropna(subset=[employee_col, question_col])
    df[question_col] = df[question_col].astype(str)
    df[answer_col] = df[answer_col].astype(str)

    if key_df is not None:
        k = key_df.copy()
        # robust key mapping
        if "question_id" not in k.columns or "correct_answer" not in k.columns:
            k = map_key_df_to_standard(k)
        if "question_id" not in k.columns or "correct_answer" not in k.columns:
            raise ValueError("Answer key must contain question id and correct answer columns.")
        k["question_id"] = k["question_id"].astype(str)
        k["correct_answer"] = k["correct_answer"].astype(str)
        merged = df.merge(k[["question_id","correct_answer"]], left_on=question_col, right_on="question_id", how="left")
        merged["is_correct"] = merged[answer_col].astype(str).fillna("").str.strip() == merged["correct_answer"].astype(str).fillna("").str.strip()
        per_emp = merged.groupby(employee_col)["is_correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"score","count":"num_answered"})
        per_emp["score"] = (per_emp["score"]*100).round(2)
        per_q = merged.groupby(question_col)["is_correct"].mean().reset_index().rename(columns={"is_correct":"pct_correct"})
        per_q["pct_correct"] = (per_q["pct_correct"]*100).round(2)
        # rename columns for unified shape
        per_emp = per_emp.rename(columns={employee_col:"employee_name"})
        merged = merged.rename(columns={employee_col:"employee_name", question_col:"question_id", "is_correct":"correct"})
        merged["correct"] = merged["correct"].astype(int)
        return per_emp, per_q, merged

    elif use_modal:
        modal = df.groupby(question_col)[answer_col].agg(lambda s: s.mode().iat[0] if not s.mode().empty else np.nan).reset_index().rename(columns={answer_col:"modal_answer"})
        merged = df.merge(modal, left_on=question_col, right_on=question_col, how="left")
        merged["is_correct"] = merged[answer_col].astype(str) == merged["modal_answer"].astype(str)
        per_emp = merged.groupby(employee_col)["is_correct"].agg(["mean","count"]).reset_index().rename(columns={"mean":"score","count":"num_answered"})
        per_emp["score"] = (per_emp["score"]*100).round(2)
        per_q = merged.groupby(question_col)["is_correct"].mean().reset_index().rename(columns={"is_correct":"pct_match_modal"})
        per_q["pct_match_modal"] = (per_q["pct_match_modal"]*100).round(2)
        per_emp = per_emp.rename(columns={employee_col:"employee_name"})
        merged = merged.rename(columns={employee_col:"employee_name", question_col:"question_id", "is_correct":"correct"})
        merged["correct"] = merged["correct"].astype(int)
        return per_emp, per_q, merged

    else:
        raise ValueError("No scoring method available (no key and modal disabled).")

def map_key_df_to_standard(key_df):
    """Attempt to map key columns to 'question_id' and 'correct_answer'"""
    if key_df is None:
        return None
    q = None; corr = None
    for col in key_df.columns:
        lc = col.strip().lower()
        if q is None and ("question" in lc or lc.startswith("q") or lc.endswith("id")):
            q = col
        if corr is None and ("correct" in lc or "answer" in lc or "key" in lc):
            corr = col
    if q is None and len(key_df.columns)>=1:
        q = key_df.columns[0]
    if corr is None and len(key_df.columns)>=2:
        for c in key_df.columns:
            if c != q:
                corr = c; break
    if q and corr:
        return key_df.rename(columns={q:"question_id", corr:"correct_answer"})
    return key_df

# Excel writer
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

# PDF report (same structure as before; uses merged_scores with pre_score/post_score columns)
def make_downloadable_pdf(merged_scores, per_q, include_chart=True, top_n=20):
    if not REPORTLAB_AVAILABLE:
        raise RuntimeError("reportlab or matplotlib not installed in this environment.")
    buf = BytesIO()
    doc = SimpleDocTemplate(buf, pagesize=landscape(A4), rightMargin=20,leftMargin=20, topMargin=20,bottomMargin=20)
    styles = getSampleStyleSheet()
    elements = []

    elements.append(Paragraph("Pre / Post Test Analysis Report", styles["Title"]))
    elements.append(Spacer(1,8))

    if merged_scores is None or merged_scores.empty:
        elements.append(Paragraph("No scored data available", styles["Normal"]))
        doc.build(elements)
        buf.seek(0)
        return buf.getvalue()

    avg_pre = round(float(merged_scores["pre_score"].mean()), 2)
    avg_post = round(float(merged_scores["post_score"].mean()), 2)
    avg_impr = round((avg_post - avg_pre), 2)
    elements.append(Paragraph(f"Average Pre Score: {avg_pre}%    Average Post Score: {avg_post}%    Average Improvement: {avg_impr} pp", styles["Normal"]))
    elements.append(Spacer(1,12))

    if include_chart:
        try:
            fig, ax = plt.subplots(figsize=(6,3))
            ax.bar(["Pre (avg)","Post (avg)"], [avg_pre, avg_post])
            ax.set_ylim(0,100)
            ax.set_ylabel("Score (%)")
            ax.set_title("Average Pre vs Post Scores")
            plt.tight_layout()
            img_buf = BytesIO()
            fig.savefig(img_buf, format="png", dpi=150)
            plt.close(fig)
            img_buf.seek(0)
            elements.append(RLImage(img_buf, width=400, height=200))
            elements.append(Spacer(1,12))
        except Exception:
            pass

    elements.append(Paragraph(f"Top {top_n} employees by improvement", styles["Heading3"]))
    top_table = merged_scores.sort_values("improvement", ascending=False).head(top_n)
    tbl_data = [list(top_table.columns)] + top_table.astype(str).values.tolist()
    tbl = Table(tbl_data, repeatRows=1)
    tbl.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#dbe9ff")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),7)]))
    elements.append(tbl)
    elements.append(Spacer(1,12))

    elements.append(Paragraph("Per-question improvement (pre vs post)", styles["Heading3"]))
    if per_q is None or per_q.empty:
        elements.append(Paragraph("No per-question data available", styles["Normal"]))
    else:
        per_q_small = per_q.head(200)
        tbl2_data = [list(per_q_small.columns)] + per_q_small.astype(str).values.tolist()
        tbl2 = Table(tbl2_data, repeatRows=1)
        tbl2.setStyle(TableStyle([('BACKGROUND',(0,0),(-1,0),colors.HexColor("#d7ffd9")),('GRID',(0,0),(-1,-1),0.5,colors.grey),('FONTSIZE',(0,0),(-1,-1),7)]))
        elements.append(tbl2)

    doc.build(elements)
    buf.seek(0)
    return buf.getvalue()

# -------------------------
# Sidebar & uploads
# -------------------------
st.sidebar.header("Files & Options")
pre_file = st.sidebar.file_uploader("Upload PRE-test (Excel/CSV)", type=["csv","xls","xlsx"])
post_file = st.sidebar.file_uploader("Upload POST-test (Excel/CSV)", type=["csv","xls","xlsx"])
key_file = st.sidebar.file_uploader("Optional: Answer Key (Excel/CSV) — only for long format", type=["csv","xls","xlsx"])
st.sidebar.markdown("---")
use_modal = st.sidebar.checkbox("If no key (long format): use class modal answer (proxy scoring)", value=True)
save_uploads = st.sidebar.checkbox("Save uploaded files to disk (data/uploads/)", value=False)
pdf_include_chart = st.sidebar.checkbox("Include chart in PDF", value=True)
pdf_top_n = st.sidebar.number_input("Top N rows in PDF", min_value=5, max_value=200, value=20, step=5)
st.sidebar.markdown("---")
if not REPORTLAB_AVAILABLE:
    st.sidebar.error("PDF generation requires 'reportlab' and 'matplotlib' installed in this environment.")

# Load files
pre_df = load_file(pre_file) if pre_file else None
post_df = load_file(post_file) if post_file else None
key_df = load_file(key_file) if key_file else None

if save_uploads:
    if pre_file: save_uploaded_file(pre_file)
    if post_file: save_uploaded_file(post_file)
    if key_file: save_uploaded_file(key_file)

# Demo generation (wide-format by default)
if st.sidebar.button("Generate demo files (wide & long)"):
    # wide demo (already scored)
    employees = ["Afnan","Bina","Carlos","Devi"]
    questions = ["Q1","Q2","Q3","Q4","Q5"]
    wide_rows_pre = [
        [1,0,0,1,0],
        [0,1,0,1,1],
        [1,1,1,1,1],
        [0,0,1,0,1],
    ]
    wide_rows_post = [
        [1,1,0,1,1],
        [1,1,1,1,1],
        [1,1,1,1,1],
        [0,1,1,1,1],
    ]
    pre_wide = pd.DataFrame([ [e]+r for e,r in zip(employees, wide_rows_pre) ], columns=["employee_name"]+questions)
    post_wide = pd.DataFrame([ [e]+r for e,r in zip(employees, wide_rows_post) ], columns=["employee_name"]+questions)
    # Long format demo (answers instead of correctness)
    # create dummy answer key and produce long format answers for demo
    key = pd.DataFrame({"question_id":questions, "correct_answer":["A","B","C","D","A"]})
    # We'll craft simple long-format answers for demo (randomized)
    long_rows_pre = []
    long_rows_post = []
    choices = ["A","B","C","D"]
    for e in employees:
        for q in questions:
            long_rows_pre.append({"employee_id":e, "question_id":q, "answer":np.random.choice(choices)})
            long_rows_post.append({"employee_id":e, "question_id":q, "answer":np.random.choice(choices)})
    pre_long = pd.DataFrame(long_rows_pre)
    post_long = pd.DataFrame(long_rows_post)
    # Save to session_state for Use demo data button
    st.session_state["_demo_pre_wide"] = pre_wide
    st.session_state["_demo_post_wide"] = post_wide
    st.session_state["_demo_pre_long"] = pre_long
    st.session_state["_demo_post_long"] = post_long
    st.session_state["_demo_key"] = key
    st.success("Demo data generated. Click `Use demo (wide)` or `Use demo (long)` below.")

if st.button("Use demo (wide format)"):
    if "_demo_pre_wide" in st.session_state:
        pre_df = st.session_state["_demo_pre_wide"]
        post_df = st.session_state["_demo_post_wide"]
        key_df = None
    else:
        st.warning("Generate demo first.")

if st.button("Use demo (long format)"):
    if "_demo_pre_long" in st.session_state:
        pre_df = st.session_state["_demo_pre_long"]
        post_df = st.session_state["_demo_post_long"]
        key_df = st.session_state.get("_demo_key", None)
    else:
        st.warning("Generate demo first.")

# -------------------------
# Previews and auto-detection
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

# Manual mapping UI (for both wide and long)
st.markdown("---")
st.header("Manual mapping (override auto-detection if needed)")
c1,c2,c3 = st.columns(3)
with c1:
    pre_emp_col = st.text_input("Pre: employee/name column", value=(p_det or {}).get("employee_col") or "employee_name")
    post_emp_col = st.text_input("Post: employee/name column", value=(s_det or {}).get("employee_col") or "employee_name")
with c2:
    pre_q_col = st.text_input("Pre (long): question column", value=(p_det or {}).get("question_col") or "question_id")
    post_q_col = st.text_input("Post (long): question column", value=(s_det or {}).get("question_col") or "question_id")
with c3:
    pre_ans_col = st.text_input("Pre (long): answer column", value=(p_det or {}).get("answer_col") or "answer")
    post_ans_col = st.text_input("Post (long): answer column", value=(s_det or {}).get("answer_col") or "answer")

# For wide: allow overrides of question columns via a comma-separated field (optional)
st.markdown("**Optional (wide format):** If auto-detection misses question columns, enter them comma-separated (e.g. `Q1,Q2,Q3`)")
wide_pre_qcols_text = st.text_input("Pre: question columns (wide)", value=",".join((p_det or {}).get("question_cols",[]) or []))
wide_post_qcols_text = st.text_input("Post: question columns (wide)", value=",".join((s_det or {}).get("question_cols",[]) or []))
def parse_qcols(text):
    if not text:
        return []
    return [t.strip() for t in text.split(",") if t.strip()]
wide_pre_qcols = parse_qcols(wide_pre_qcols_text)
wide_post_qcols = parse_qcols(wide_post_qcols_text)

# -------------------------
# Normalize / convert inputs to standard internal format
# We'll transform any input (wide or long) into a standard 'long scored' dataframe:
# columns: employee_name, question_id, correct (0/1)
# And also produce per-question DataFrame and per-employee DataFrame.
# -------------------------
def process_uploaded_test(df, det, emp_col_override=None, question_col_override=None, answer_col_override=None, wide_qcols_override=None):
    """
    Returns per_emp_df, per_q_df, merged_long_df (employee_name, question_id, correct)
    """
    if df is None:
        return None, None, None, det
    # If detection unknown, use overrides to try to determine format
    fmt = det.get("format", "unknown") if isinstance(det, dict) else "unknown"

    # If user has provided overrides, use them to decide format
    if fmt == "unknown":
        # if wide override qcols provided, treat as wide
        if wide_qcols_override:
            fmt = "wide"
            det = {"format":"wide", "employee_col": emp_col_override or det.get("employee_col"), "question_cols": wide_qcols_override}
        else:
            # if question_col_override and answer_col_override present -> long
            if question_col_override and answer_col_override:
                fmt = "long"
                det = {"format":"long", "employee_col": emp_col_override or det.get("employee_col"), "question_col": question_col_override, "answer_col": answer_col_override}
            else:
                # last resort: try to detect format again
                det = detect_format_and_columns(df)
                fmt = det.get("format","unknown")

    # Process wide
    if fmt == "wide":
        emp_col = emp_col_override or det.get("employee_col")
        qcols = wide_qcols_override or det.get("question_cols") or []
        # if qcols is empty, fallback to all except emp_col
        if not qcols:
            qcols = [c for c in df.columns if c != (_find_col_ignore_case(df, emp_col) or emp_col)]
        long = wide_to_long_scored(df, emp_col, qcols)
        per_emp, per_q, merged = compute_scores_from_long_scored(long)
        # per_emp uses employee_name column
        return per_emp, per_q, merged, {"format":"wide","employee_col":emp_col,"question_cols":qcols}

    # Process long
    if fmt == "long":
        emp_col = emp_col_override or det.get("employee_col")
        q_col = question_col_override or det.get("question_col")
        ans_col = answer_col_override or det.get("answer_col")
        # check presence
        # first, try flexible mapping names to actual columns
        emp_actual = _find_col_ignore_case(df, emp_col) or emp_col
        q_actual = _find_col_ignore_case(df, q_col) or q_col
        ans_actual = _find_col_ignore_case(df, ans_col) or ans_col
        if emp_actual not in df.columns or q_actual not in df.columns or ans_actual not in df.columns:
            # unable to find required columns: raise
            raise ValueError("Long format requires employee, question and answer columns (use manual mapping).")
        # Use compute_scores_from_long_answers
        per_emp, per_q, merged = compute_scores_from_long_answers(df.rename(columns={emp_actual:"employee_id", q_actual:"question_id", ans_actual:"answer"}), key_df=key_df, use_modal=use_modal)
        # per_emp renamed to employee_name inside function
        return per_emp, per_q, merged, {"format":"long","employee_col":emp_actual,"question_col":q_actual,"answer_col":ans_actual}

    # Unknown
    raise ValueError("Unable to determine file format (wide/long). Use manual mapping fields to help the app.")

# -------------------------
# Run processing for uploaded files
# -------------------------
# process pre
pre_per_emp = pre_per_q = pre_merged = None
post_per_emp = post_per_q = post_merged = None
errors = []

if pre_df is None and post_df is None:
    st.info("Upload at least one pre-test or post-test file (wide or long).")
else:
    # process pre
    if pre_df is not None:
        try:
            pre_det_local = p_det or detect_format_and_columns(pre_df)
            pre_per_emp, pre_per_q, pre_merged, pre_det_local = process_uploaded_test(
                pre_df, pre_det_local,
                emp_col_override=pre_emp_col,
                question_col_override=pre_q_col,
                answer_col_override=pre_ans_col,
                wide_qcols_override=wide_pre_qcols
            )
        except Exception as e:
            errors.append(f"Pre-test processing error: {e}")
            pre_per_emp = pre_per_q = pre_merged = None

    if post_df is not None:
        try:
            post_det_local = s_det or detect_format_and_columns(post_df)
            post_per_emp, post_per_q, post_merged, post_det_local = process_uploaded_test(
                post_df, post_det_local,
                emp_col_override=post_emp_col,
                question_col_override=post_q_col,
                answer_col_override=post_ans_col,
                wide_qcols_override=wide_post_qcols
            )
        except Exception as e:
            errors.append(f"Post-test processing error: {e}")
            post_per_emp = post_per_q = post_merged = None

# show errors if any
if errors:
    for e in errors:
        st.error(e)

# -------------------------
# Build analysis outputs (lists, merges, visualizations)
# -------------------------
if (pre_per_emp is None and post_per_emp is None):
    st.info("No processed scoring data available to analyze (fix mapping or upload proper files).")
else:
    # get employee name lists
    pre_names = set(pre_per_emp["employee_name"].astype(str).unique()) if pre_per_emp is not None else set()
    post_names = set(post_per_emp["employee_name"].astype(str).unique()) if post_per_emp is not None else set()
    both = sorted(list(pre_names.intersection(post_names)))
    pre_only = sorted(list(pre_names - post_names))
    post_only = sorted(list(post_names - pre_names))

    c1,c2,c3 = st.columns(3)
    c1.metric("Pre only", len(pre_only))
    c2.metric("Post only", len(post_only))
    c3.metric("Both", len(both))

    with st.expander("See lists of employees"):
        st.subheader("Pre only")
        st.write(pre_only)
        st.subheader("Post only")
        st.write(post_only)
        st.subheader("Both")
        st.write(both)

    # Merge per-employee scores into unified table
    # normalize column names to employee_name, pre_score, post_score
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

    st.subheader("Employee Scores & Improvement")
    st.dataframe(merged_scores.sort_values("improvement", ascending=False).reset_index(drop=True))

    # KPIs
    avg_pre = round(float(merged_scores["pre_score"].mean()), 2) if not merged_scores.empty else 0.0
    avg_post = round(float(merged_scores["post_score"].mean()), 2) if not merged_scores.empty else 0.0
    avg_impr = round((avg_post - avg_pre), 2)
    k1,k2,k3 = st.columns(3)
    k1.metric("Average pre score (%)", avg_pre)
    k2.metric("Average post score (%)", avg_post)
    k3.metric("Average improvement (pp)", avg_impr)

    st.subheader("Pre vs Post mean (bar)")
    hist_df = merged_scores[["employee_name","pre_score","post_score"]].melt(id_vars="employee_name", var_name="test", value_name="score")
    st.bar_chart(hist_df.groupby("test")["score"].mean())

    # Per-question improvement: join per-question from pre and post if available
    if pre_per_q is not None and post_per_q is not None:
        pre_q = pre_per_q.rename(columns={pre_per_q.columns[0]:"question_id", pre_per_q.columns[1]:"pre_pct"}).set_index("question_id")
        post_q = post_per_q.rename(columns={post_per_q.columns[0]:"question_id", post_per_q.columns[1]:"post_pct"}).set_index("question_id")
        per_q = pre_q.join(post_q, how="outer").fillna(0)
        per_q["delta_pp"] = (per_q["post_pct"] - per_q["pre_pct"]).round(2)
        st.subheader("Per-question improvement")
        st.dataframe(per_q.reset_index().sort_values("delta_pp", ascending=False))
        per_q_export = per_q.reset_index()
    else:
        # if only one of them exists, show that
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

    # Prepare downloadables
    out_dfs = {
        "employee_scores": merged_scores,
        "pre_details": pre_merged if pre_merged is not None else pd.DataFrame(),
        "post_details": post_merged if post_merged is not None else pd.DataFrame(),
        "per_question": per_q_export
    }

    st.markdown("---")
    st.subheader("Download Report")
    excel_bytes = make_downloadable_excel(out_dfs)
    st.download_button("Download full report (Excel)", data=excel_bytes,
                       file_name=f"prepost_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.xlsx",
                       mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

    if REPORTLAB_AVAILABLE:
        try:
            pdf_bytes = make_downloadable_pdf(merged_scores, per_q_export, include_chart=pdf_include_chart, top_n=int(pdf_top_n))
            st.download_button("Download summary report (PDF)", data=pdf_bytes,
                               file_name=f"prepost_report_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.pdf",
                               mime="application/pdf")
        except Exception as e:
            st.error(f"PDF generation failed: {e}")
    else:
        st.info("PDF export disabled (reportlab/matplotlib not available).")

st.markdown("---")
st.caption("This app accepts both 'wide' scored files (rows = employee, cols = Q1..Qn with 0/1) and 'long' answer files. If you want automatic saving of reports to outputs/ or additional stats (paired t-test, Cohen's d), ask and I'll add them.")
