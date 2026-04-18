import json
from pathlib import Path

import joblib
import lightgbm as lgb
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st

# RAG explanation engine (rag_utils.py must be in the same folder)
try:
    from rag_utils import generate_rag_explanation, load_rag_resources
    RAG_AVAILABLE = True
except ImportError:
    RAG_AVAILABLE = False

# ============================================================
# PAGE CONFIG
# ============================================================

st.set_page_config(
    page_title="Macro-Aware Credit Risk Dashboard",
    page_icon="📉",
    layout="wide"
)

# ============================================================
# PATHS
# ============================================================

BASE_DIR = Path(__file__).resolve().parent
ARTIFACT_DIR = BASE_DIR / "artifacts"

MODEL_PATH = ARTIFACT_DIR / "model.txt"
PREPROCESSOR_PATH = ARTIFACT_DIR / "preprocessor.pkl"
SCHEMA_PATH = ARTIFACT_DIR / "feature_schema.json"
TRAIN_REF_PATH = ARTIFACT_DIR / "training_reference.csv"
GRADE_PATH = ARTIFACT_DIR / "grade_thresholds.json"

# ============================================================
# GLOBAL STYLING
# ============================================================

st.markdown(
    """
    <style>
    :root {
        --bg: #f6f8fb;
        --surface: #ffffff;
        --surface-soft: #f8fafc;
        --border: #dbe3ee;
        --text: #0f172a;
        --muted: #64748b;
        --accent: #24c7a8;
        --accent-dark: #18b193;
        --success: #10b981;
        --warning: #f59e0b;
        --orange: #f97316;
        --danger: #ef4444;
    }

    .stApp {
        background: var(--bg);
    }

    html, body, [class*="css"] {
        color: var(--text);
    }

    #MainMenu, footer, header {
        visibility: hidden;
    }

    .block-container {
        padding-top: 1.05rem;
        padding-bottom: 2rem;
        max-width: 1500px;
    }

    /* Sidebar */
    section[data-testid="stSidebar"] {
        background: #ffffff;
        border-right: 1px solid var(--border);
        width: 355px !important;
        min-width: 355px !important;
    }

    section[data-testid="stSidebar"] > div {
        padding-top: 0;
    }

    section[data-testid="stSidebar"] * {
        color: var(--text) !important;
    }

    .sidebar-header {
        padding: 22px 20px 16px 20px;
        border-bottom: 1px solid var(--border);
    }

    .sidebar-header-row {
        display: flex;
        align-items: center;
        gap: 14px;
    }

    .sidebar-icon {
        width: 46px;
        height: 46px;
        border-radius: 16px;
        background: rgba(36, 199, 168, 0.10);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 24px;
        color: var(--accent);
        font-weight: 700;
    }

    .sidebar-title {
        font-size: 18px;
        font-weight: 800;
        line-height: 1.2;
        color: var(--text);
    }

    .sidebar-subtitle {
        font-size: 13px;
        color: var(--muted);
        margin-top: 2px;
    }

    .sidebar-body {
        padding: 18px 18px 8px 18px;
    }

    .sidebar-footer {
        padding: 16px 18px 24px 18px;
        border-top: 1px solid var(--border);
        margin-top: 8px;
    }

    .divider-soft {
        height: 1px;
        background: var(--border);
        margin: 10px 0 18px 0;
    }

    /* Main headings */
    .main-title {
        font-size: 42px;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: var(--text);
        line-height: 1.05;
        margin-bottom: 6px;
    }

    .main-subtitle {
        font-size: 17px;
        color: var(--muted);
        margin-bottom: 20px;
    }

    /* KPI cards */
    .kpi-card {
        background: var(--surface);
        border: 1px solid var(--border);
        border-radius: 24px;
        padding: 18px 20px 16px 20px;
        min-height: 122px;
    }

    .kpi-card-highlight {
        background: #fff8ef;
        border: 1px solid #fdc98d;
    }

    .kpi-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--muted);
        line-height: 1.2;
        margin-bottom: 16px;
    }

    .kpi-value {
        font-size: 28px;
        font-weight: 900;
        letter-spacing: -0.02em;
        color: var(--text);
        line-height: 1.12;
        margin: 0;
    }

    .kpi-sub {
        font-size: 13px;
        color: var(--muted);
        margin-top: 8px;
    }

    .risk-low { color: #059669; }
    .risk-moderate { color: #d97706; }
    .risk-elevated { color: #c2410c; }
    .risk-high { color: #dc2626; }

    /* Section label */
    .section-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.11em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 14px;
    }

    .big-paragraph {
        font-size: 17px;
        line-height: 1.75;
        color: #334155;
    }

    /* True white Streamlit cards */
    div[data-testid="stVerticalBlockBorderWrapper"] {
        background: #ffffff !important;
        border: 1px solid var(--border) !important;
        border-radius: 24px !important;
        padding: 14px 18px 16px 18px !important;
        box-shadow: 0 1px 2px rgba(15, 23, 42, 0.03) !important;
    }

    div[data-testid="stVerticalBlockBorderWrapper"] > div {
        background: transparent !important;
    }

    /* Empty state */
    .empty-wrap {
        min-height: 68vh;
        display: flex;
        align-items: center;
        justify-content: center;
        text-align: center;
        padding: 32px 24px;
    }

    .empty-icon {
        width: 92px;
        height: 92px;
        border-radius: 28px;
        background: rgba(36, 199, 168, 0.10);
        display: flex;
        align-items: center;
        justify-content: center;
        font-size: 44px;
        color: var(--accent);
        margin: 0 auto 22px auto;
    }

    .empty-title {
        font-size: 28px;
        font-weight: 900;
        color: var(--text);
        margin-bottom: 8px;
    }

    .empty-text {
        font-size: 16px;
        line-height: 1.8;
        color: var(--muted);
        max-width: 650px;
        margin: 0 auto;
    }

    /* Explanation factor groups */
    .factor-box {
        padding-top: 16px;
        border-top: 1px solid #e9eef5;
        margin-top: 16px;
    }

    .factor-head {
        font-size: 12px;
        font-weight: 900;
        letter-spacing: 0.08em;
        text-transform: uppercase;
        margin-bottom: 10px;
    }

    .factor-red { color: #dc2626; }
    .factor-amber { color: #d97706; }
    .factor-green { color: #059669; }

    .factor-item {
        font-size: 15px;
        color: #334155;
        margin-bottom: 8px;
    }

    /* What-if */
    .sim-dti {
        font-size: 14px;
        color: var(--muted);
        margin-top: -2px;
        margin-bottom: 18px;
    }

    .sim-dti strong {
        color: var(--text);
    }

    .pd-box {
        background: #fbfcfe;
        border: 1px solid #e4eaf2;
        border-radius: 22px;
        padding: 20px 24px;
        min-height: 128px;
    }

    .pd-box-label {
        font-size: 12px;
        font-weight: 800;
        letter-spacing: 0.10em;
        text-transform: uppercase;
        color: var(--muted);
        margin-bottom: 12px;
    }

    .pd-box-value {
        font-size: 44px;
        font-weight: 900;
        letter-spacing: -0.03em;
        color: var(--text);
        line-height: 1;
    }

    .pd-delta-up {
        color: #dc2626;
        font-size: 14px;
        font-weight: 700;
        margin-top: 8px;
    }

    .pd-delta-down {
        color: #059669;
        font-size: 14px;
        font-weight: 700;
        margin-top: 8px;
    }

    .center-arrow {
        font-size: 48px;
        color: #64748b;
        text-align: center;
        padding-top: 28px;
    }

    .footnote {
        text-align: center;
        color: #94a3b8;
        font-size: 15px;
        margin-top: 20px;
    }

    .small-metric {
        font-size: 13px;
        color: var(--muted);
    }

    /* Widget labels */
    .stDateInput label,
    .stNumberInput label,
    .stSelectbox label,
    .stSlider label {
        font-size: 12px !important;
        font-weight: 800 !important;
        letter-spacing: 0.10em !important;
        text-transform: uppercase !important;
        color: var(--muted) !important;
    }

    /* Inputs */
    .stTextInput input,
    .stDateInput input,
    .stNumberInput input {
        background: #f8fafc !important;
        border: 1px solid #dbe3ee !important;
        border-radius: 18px !important;
        color: var(--text) !important;
        min-height: 52px !important;
        font-size: 16px !important;
        box-shadow: none !important;
    }

    [data-testid="stDateInput"] div[data-baseweb="input"],
    [data-testid="stNumberInput"] div[data-baseweb="input"] {
        background: #f8fafc !important;
        border: 1px solid #dbe3ee !important;
        border-radius: 18px !important;
        box-shadow: none !important;
    }

    [data-testid="stDateInput"] div[data-baseweb="input"] > div,
    [data-testid="stNumberInput"] div[data-baseweb="input"] > div {
        background: transparent !important;
    }

    .stSelectbox div[data-baseweb="select"] > div {
        background: #f8fafc !important;
        border: 1px solid #dbe3ee !important;
        border-radius: 18px !important;
        min-height: 52px !important;
        color: var(--text) !important;
        box-shadow: none !important;
    }

    /* Number buttons */
    [data-testid="stNumberInput"] button {
        background: #f8fafc !important;
        color: #f8fafc !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stNumberInput"] button:hover {
        background: #f8fafc !important;
        color: #f8fafc !important;
    }

    [data-testid="stNumberInput"] button svg {
        fill: #9aa7b8 !important;
        color: #9aa7b8 !important;
    }

    /* Date icon */
    [data-testid="stDateInput"] button {
        background: #f8fafc !important;
        border: none !important;
        box-shadow: none !important;
    }

    [data-testid="stDateInput"] button svg {
        fill: #64748b !important;
        color: #64748b !important;
    }

    /* Sliders */
    .stSlider [data-baseweb="slider"] {
        padding-top: 8px;
    }

    .stSlider div[role="slider"] {
        border: 2px solid #94a3b8 !important;
        box-shadow: 0 2px 6px rgba(15, 23, 42, 0.15) !important;
    }

    /* Sidebar buttons */
    section[data-testid="stSidebar"] button[kind="primary"] {
        background: var(--accent) !important;
        color: #ffffff !important;
        border-radius: 18px !important;
        min-height: 54px !important;
        font-size: 17px !important;
        font-weight: 800 !important;
        border: none !important;
        box-shadow: 0 6px 14px rgba(36, 199, 168, 0.22) !important;
    }

    section[data-testid="stSidebar"] button[kind="primary"]:hover {
        background: var(--accent-dark) !important;
        color: #ffffff !important;
    }

    section[data-testid="stSidebar"] button[kind="secondary"] {
        background: #f8fafc !important;
        color: #64748b !important;
        border-radius: 18px !important;
        min-height: 48px !important;
        font-size: 15px !important;
        font-weight: 700 !important;
        border: 1px solid #dbe3ee !important;
        box-shadow: none !important;
    }

    section[data-testid="stSidebar"] button[kind="secondary"]:hover {
        background: #eef4f8 !important;
        color: #475569 !important;
    }

    /* Dataframe */
    [data-testid="stDataFrame"] {
        border-radius: 20px;
        overflow: hidden;
        border: 1px solid var(--border);
    }

    [data-testid="stDataFrame"] div[role="table"] {
        background: white !important;
    }

    /* Plotly chart wrapper */
    [data-testid="stPlotlyChart"] {
        background: transparent !important;
    }

    /* Hide plotly toolbar */
    .modebar {
        display: none !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ============================================================
# LOAD ARTIFACTS
# ============================================================

@st.cache_resource
def load_artifacts():
    required_paths = [
        MODEL_PATH,
        PREPROCESSOR_PATH,
        SCHEMA_PATH,
        TRAIN_REF_PATH,
        GRADE_PATH
    ]

    missing = [p.name for p in required_paths if not p.exists()]
    if missing:
        raise FileNotFoundError(f"Missing artifact files: {missing}")

    model = lgb.Booster(model_file=str(MODEL_PATH))
    preprocessor = joblib.load(PREPROCESSOR_PATH)

    with open(SCHEMA_PATH, "r") as f:
        schema = json.load(f)

    train_ref = pd.read_csv(TRAIN_REF_PATH)

    with open(GRADE_PATH, "r") as f:
        grade_info = json.load(f)

    return model, preprocessor, schema, train_ref, grade_info

model, preprocessor, schema, train_ref, grade_info = load_artifacts()

grade_labels = grade_info["grade_labels"]
grade_thresholds = grade_info["grade_thresholds"]
cat_cols = schema["cat_cols"]
num_cols = schema["num_cols"]
feature_cols = schema["feature_cols"]
feature_names_transformed = preprocessor.get_feature_names_out()

# ============================================================
# FRED HELPERS
# ============================================================

@st.cache_data(ttl=3600)
def fetch_fred_series_csv(series_code: str, start_date: str, end_date: str) -> pd.DataFrame:
    url = f"https://fred.stlouisfed.org/graph/fredgraph.csv?id={series_code}"
    s = pd.read_csv(url)

    if s.shape[1] < 2:
        raise ValueError(f"FRED response for {series_code} does not have 2 columns. Columns found: {list(s.columns)}")

    s = s.rename(columns={s.columns[0]: "DATE", s.columns[1]: series_code})
    s["DATE"] = pd.to_datetime(s["DATE"], errors="coerce")
    s[series_code] = pd.to_numeric(s[series_code], errors="coerce")

    s = s[(s["DATE"] >= pd.Timestamp(start_date)) & (s["DATE"] <= pd.Timestamp(end_date))].copy()
    s = s.dropna(subset=["DATE"])
    s = s.set_index("DATE")
    s.index = s.index.to_period("M").to_timestamp()
    return s[[series_code]]

@st.cache_data(ttl=3600)
def get_macro_features_for_date(application_date) -> dict:
    app_month = pd.Timestamp(application_date).to_period("M").to_timestamp()

    start_date = (app_month - pd.DateOffset(months=18)).strftime("%Y-%m-%d")
    end_date = app_month.strftime("%Y-%m-%d")

    fred_codes = {
        "Inflation": "CPIAUCSL",
        "FedFunds": "FEDFUNDS",
        "HomePrices": "CSUSHPISA",
        "UNRATE": "UNRATE"
    }

    macro = pd.DataFrame()
    for name, code in fred_codes.items():
        s = fetch_fred_series_csv(code, start_date, end_date)
        macro[name] = s[code]

    macro = macro.sort_index().ffill()

    return {
        "Inflation_L6": float(macro["Inflation"].shift(6).dropna().iloc[-1]),
        "FedFunds_L3": float(macro["FedFunds"].shift(3).dropna().iloc[-1]),
        "HomePrices_L12": float(macro["HomePrices"].shift(12).dropna().iloc[-1]),
        "UNRATE_L6": float(macro["UNRATE"].shift(6).dropna().iloc[-1]),
    }

# ============================================================
# MODEL HELPERS
# ============================================================

def calculate_auto_dti(loan_amnt: float, annual_inc: float) -> float:
    loan_amnt = float(loan_amnt) if loan_amnt is not None else 0.0
    annual_inc = float(annual_inc) if annual_inc is not None else 0.0

    if annual_inc <= 0:
        return 100.0

    dti = (loan_amnt / annual_inc) * 100.0
    return float(np.clip(dti, 0.0, 100.0))

def build_single_applicant_row(applicant_dict: dict, application_date):
    macro_dict = get_macro_features_for_date(application_date)
    row = {**applicant_dict, **macro_dict}
    row_df = pd.DataFrame([row])

    for col in feature_cols:
        if col not in row_df.columns:
            row_df[col] = np.nan

    row_df = row_df[feature_cols]
    return row_df, macro_dict

def predict_single_applicant(applicant_dict: dict, application_date):
    row_df, macro_dict = build_single_applicant_row(applicant_dict, application_date)
    X_row = preprocessor.transform(row_df)
    pd_hat = float(model.predict(X_row)[0])
    return pd_hat, row_df, X_row, macro_dict

def get_credit_grade(pd_hat: float):
    bins = np.r_[-np.inf, grade_thresholds, np.inf]
    grade = pd.cut(
        [pd_hat],
        bins=bins,
        labels=grade_labels,
        include_lowest=True,
        right=True
    )[0]
    return str(grade)

def get_grade_class(grade: str):
    if grade == "A":
        return "risk-low"
    elif grade == "B":
        return "risk-low"
    elif grade == "C":
        return "risk-moderate"
    elif grade == "D":
        return "risk-elevated"
    elif grade == "E":
        return "risk-high"
    return "risk-high"

def get_recommendation_from_grade(grade: str):
    if grade in ["A", "B"]:
        return "Approve"
    elif grade in ["C", "D"]:
        return "Review"
    return "Manual Review"

# ============================================================
# EXPLANATION HELPERS
# ============================================================

pretty_raw_name_map = {
    "fico_range_low": "FICO score",
    "loan_amnt": "Loan amount",
    "dti": "Debt-to-income ratio",
    "revol_util": "Revolving utilization",
    "annual_inc": "Annual income",
    "open_acc": "Open accounts",
    "delinq_2yrs": "Recent delinquencies",
    "pub_rec": "Public records",
    "inq_last_6mths": "Recent credit inquiries",
    "mort_acc": "Mortgage accounts",
    "revol_bal": "Revolving balance",
    "Inflation_L6": "Inflation backdrop",
    "FedFunds_L3": "Fed funds backdrop",
    "HomePrices_L12": "Home price backdrop",
    "UNRATE_L6": "Unemployment backdrop",
    "term": "Loan term",
    "home_ownership": "Home ownership",
    "purpose": "Loan purpose",
    "emp_length": "Employment length"
}

def get_raw_feature_name(transformed_name: str, cat_cols: list, num_cols: list):
    if transformed_name.startswith("num__"):
        return transformed_name.replace("num__", "")

    if transformed_name.startswith("cat__"):
        cat_part = transformed_name.replace("cat__", "")
        for c in cat_cols:
            prefix = c + "_"
            if cat_part.startswith(prefix):
                return c
        return cat_part

    return transformed_name

def group_contributions(contrib_df: pd.DataFrame, cat_cols: list, num_cols: list):
    tmp = contrib_df.copy()
    tmp["raw_feature"] = tmp["feature"].apply(lambda x: get_raw_feature_name(x, cat_cols, num_cols))

    grouped = (
        tmp.groupby("raw_feature", as_index=False)
        .agg({"shap_value": "sum"})
    )

    grouped["abs_shap"] = grouped["shap_value"].abs()
    grouped["pretty_feature"] = grouped["raw_feature"].map(pretty_raw_name_map).fillna(grouped["raw_feature"])
    grouped = grouped.sort_values("abs_shap", ascending=False).reset_index(drop=True)
    return grouped

def classify_reason_groups(grouped_df: pd.DataFrame, top_positive=4, top_negative=3):
    pos_df = grouped_df[grouped_df["shap_value"] > 0].sort_values("shap_value", ascending=False).copy()
    neg_df = grouped_df[grouped_df["shap_value"] < 0].sort_values("shap_value", ascending=True).copy()

    high_risk = pos_df.head(2)["pretty_feature"].tolist()
    medium_risk = pos_df.iloc[2:top_positive]["pretty_feature"].tolist()

    reducing = (
        neg_df.assign(rank_value=neg_df["shap_value"].abs())
        .sort_values("rank_value", ascending=False)
        .head(top_negative)["pretty_feature"]
        .tolist()
    )

    return {
        "high_risk_factors": high_risk,
        "medium_risk_factors": medium_risk,
        "risk_reducing_factors": reducing
    }

def join_nicely(items: list):
    if not items:
        return ""
    if len(items) == 1:
        return items[0]
    if len(items) == 2:
        return f"{items[0]} and {items[1]}"
    return ", ".join(items[:-1]) + f", and {items[-1]}"

def make_explanation_paragraph(reason_groups: dict, predicted_pd=None, credit_grade=None):
    high_risk = reason_groups.get("high_risk_factors", [])
    medium_risk = reason_groups.get("medium_risk_factors", [])
    reducing = reason_groups.get("risk_reducing_factors", [])

    intro = ""
    if predicted_pd is not None and credit_grade is not None:
        intro = (
            f"This applicant is assigned credit grade <strong>{credit_grade}</strong> "
            f"with an estimated probability of default of <strong>{predicted_pd:.1%}</strong>. "
        )

    para = intro
    if high_risk:
        para += f"The strongest contributors to higher risk are {join_nicely(high_risk)}. "
    if medium_risk:
        para += f"Additional moderate risk pressure comes from {join_nicely(medium_risk)}. "
    if reducing:
        para += f"However, some factors help reduce the predicted risk, particularly {join_nicely(reducing)}."
    return para.strip()

def get_prediction_explanation(X_row):
    contrib = model.predict(X_row, pred_contrib=True)
    sv = np.array(contrib)[0]

    if len(sv) == len(feature_names_transformed) + 1:
        sv = sv[:-1]

    contrib_df = pd.DataFrame({
        "feature": feature_names_transformed,
        "shap_value": sv
    })

    contrib_df["abs_shap"] = contrib_df["shap_value"].abs()
    contrib_df = contrib_df.sort_values("abs_shap", ascending=False).reset_index(drop=True)

    grouped = group_contributions(contrib_df, cat_cols, num_cols)
    return contrib_df, grouped

# ============================================================
# CHART HELPERS
# ============================================================

def make_colored_gauge(pd_hat: float):
    value = pd_hat * 100
    t = [x * 100 for x in grade_thresholds]  # convert to %
    a, b, c, d, e = t

    fig = go.Figure(go.Indicator(
        mode="gauge",
        value=value,
        gauge={
            "shape": "angular",
            "axis": {
                "range": [0, 100],
                "tickmode": "array",
                "tickvals": [0, 100],
                "ticktext": ["0%", "100%"],
                "tickfont": {"size": 15, "color": "#94a3b8"}
            },
            "bar": {"color": "#0f172a", "thickness": 0.16},
            "bgcolor": "white",
            "borderwidth": 0,
            "steps": [
                {"range": [0, a], "color": "#86efac"},
                {"range": [a, b], "color": "#bbf7d0"},
                {"range": [b, c], "color": "#fde68a"},
                {"range": [c, d], "color": "#fdba74"},
                {"range": [d, e], "color": "#fca5a5"},
                {"range": [e, 100], "color": "#f87171"},
            ],
            "threshold": {
                "line": {"color": "#0f172a", "width": 5},
                "thickness": 0.75,
                "value": value
            }
        }
    ))

    fig.update_layout(
        height=360,
        margin=dict(l=24, r=24, t=24, b=0),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#0f172a")
    )
    return fig

def make_contribution_chart(grouped_df: pd.DataFrame):
    plot_df = grouped_df.copy().sort_values("shap_value", ascending=True)
    colors = ["#34d399" if v < 0 else "#f87171" for v in plot_df["shap_value"]]

    fig = go.Figure(
        go.Bar(
            x=plot_df["shap_value"],
            y=plot_df["pretty_feature"],
            orientation="h",
            marker_color=colors,
            hovertemplate="<b>%{y}</b><br>Contribution: %{x:.4f}<extra></extra>"
        )
    )

    dynamic_height = max(380, min(1000, 34 * len(plot_df)))

    fig.update_layout(
        height=dynamic_height,
        margin=dict(l=10, r=10, t=10, b=10),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        showlegend=False,
        font=dict(color="#334155", size=13),
        xaxis=dict(
            title="Contribution",
            title_font=dict(color="#94a3b8", size=14),
            tickfont=dict(color="#94a3b8", size=12),
            showgrid=True,
            gridcolor="rgba(148,163,184,0.18)",
            zeroline=True,
            zerolinecolor="rgba(148,163,184,0.35)"
        ),
        yaxis=dict(
            title="",
            tickfont=dict(color="#94a3b8", size=12),
            automargin=True
        )
    )
    return fig

# ============================================================
# RENDER HELPERS
# ============================================================

def render_sidebar_header():
    st.sidebar.markdown(
        """
        <div class="sidebar-header">
            <div class="sidebar-header-row">
                <div class="sidebar-icon">∿</div>
                <div>
                    <div class="sidebar-title">Borrower Profile</div>
                    <div class="sidebar-subtitle">Enter applicant details</div>
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

def render_kpi_card(label, value, sublabel=None, highlight=False, value_class=""):
    klass = "kpi-card kpi-card-highlight" if highlight else "kpi-card"
    sub_html = f'<div class="kpi-sub">{sublabel}</div>' if sublabel else ""
    st.markdown(
        f"""
        <div class="{klass}">
            <div class="kpi-label">{label}</div>
            <div class="kpi-value {value_class}">{value}</div>
            {sub_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_factor_column(title, items, css_class):
    items_html = "".join([f'<div class="factor-item">• {x}</div>' for x in items]) if items else '<div class="factor-item">—</div>'
    st.markdown(
        f"""
        <div class="factor-box">
            <div class="factor-head {css_class}">{title}</div>
            {items_html}
        </div>
        """,
        unsafe_allow_html=True
    )

def render_empty_state():
    st.markdown(
        """
        <div class="empty-wrap">
            <div>
                <div class="empty-icon">∿</div>
                <div class="empty-title">Ready to Predict</div>
                <div class="empty-text">
                    Fill in the borrower profile in the sidebar and click <strong>Run Prediction</strong>
                    to estimate the probability of default with macro-aware context.
                </div>
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )

# ============================================================
# SESSION STATE
# ============================================================

default_state = {
    "prediction_ready": False,
    "saved_applicant": None,
    "saved_application_date": None,
    "saved_pd_hat": None,
    "saved_row_df": None,
    "saved_macro_dict": None,
    "saved_grouped_contrib_df": None,
    "saved_credit_grade": None,
    "saved_grade_class": None, 
    "saved_recommendation": None,
    "saved_reason_groups": None,
    "saved_explanation_paragraph": None,
    "saved_rag_explanation": None,
}

for k, v in default_state.items():
    if k not in st.session_state:
        st.session_state[k] = v

# Pre-load RAG resources once at startup (silent — errors are caught later)
if RAG_AVAILABLE:
    try:
        load_rag_resources()
    except Exception:
        pass  # will surface gracefully when user clicks Run Prediction

# ============================================================
# SIDEBAR
# ============================================================

render_sidebar_header()

st.sidebar.markdown('<div class="sidebar-body">', unsafe_allow_html=True)

saved = st.session_state.saved_applicant

application_date = st.sidebar.date_input(
    "Application Date",
    value=st.session_state.saved_application_date if st.session_state.saved_application_date else pd.Timestamp("2018-01-01").date()
)

st.sidebar.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

term_options = ["36 months", "60 months"]
home_ownership_options = ["RENT", "MORTGAGE", "OWN", "OTHER"]
purpose_options = [
    "debt_consolidation", "credit_card", "home_improvement", "small_business",
    "major_purchase", "car", "medical", "vacation", "moving", "house", "wedding", "other"
]
emp_length_options = [
    "< 1 year", "1 year", "2 years", "3 years", "4 years",
    "5 years", "6 years", "7 years", "8 years", "9 years", "10+ years"
]

term = st.sidebar.selectbox(
    "Loan Term",
    term_options,
    index=0 if not saved else term_options.index(saved["term"])
)

home_ownership = st.sidebar.selectbox(
    "Home Ownership",
    home_ownership_options,
    index=0 if not saved else home_ownership_options.index(saved["home_ownership"])
)

purpose = st.sidebar.selectbox(
    "Loan Purpose",
    purpose_options,
    format_func=lambda x: x.replace("_", " ").title(),
    index=0 if not saved else purpose_options.index(saved["purpose"])
)

emp_length = st.sidebar.selectbox(
    "Employment Length",
    emp_length_options,
    index=3 if not saved else emp_length_options.index(saved["emp_length"])
)

st.sidebar.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

fico_range_low = st.sidebar.number_input(
    "FICO Score",
    min_value=300,
    max_value=850,
    value=680 if not saved else int(saved["fico_range_low"]),
    step=1
)

loan_amnt = st.sidebar.number_input(
    "Loan Amount ($)",
    min_value=500,
    max_value=100000,
    value=15000 if not saved else int(saved["loan_amnt"]),
    step=500
)

revol_util = st.sidebar.number_input(
    "Revolving Utilization (%)",
    min_value=0.0,
    max_value=150.0,
    value=45.0 if not saved else float(saved["revol_util"]),
    step=0.1
)

annual_inc = st.sidebar.number_input(
    "Annual Income ($)",
    min_value=0.0,
    max_value=1000000.0,
    value=70000.0 if not saved else float(saved["annual_inc"]),
    step=1000.0
)

dti = calculate_auto_dti(loan_amnt, annual_inc)
st.sidebar.markdown(
    f'<div class="small-metric">Derived DTI: <strong style="color:#0f172a;">{dti:.1f}%</strong></div>',
    unsafe_allow_html=True
)

st.sidebar.markdown('<div class="divider-soft"></div>', unsafe_allow_html=True)

open_acc = st.sidebar.number_input(
    "Open Accounts",
    min_value=0,
    max_value=100,
    value=8 if not saved else int(saved["open_acc"]),
    step=1
)

delinq_2yrs = st.sidebar.number_input(
    "Recent Delinquencies",
    min_value=0,
    max_value=20,
    value=0 if not saved else int(saved["delinq_2yrs"]),
    step=1
)

pub_rec = st.sidebar.number_input(
    "Public Records",
    min_value=0,
    max_value=20,
    value=0 if not saved else int(saved["pub_rec"]),
    step=1
)

inq_last_6mths = st.sidebar.number_input(
    "Recent Inquiries (6m)",
    min_value=0,
    max_value=50,
    value=1 if not saved else int(saved["inq_last_6mths"]),
    step=1
)

mort_acc = st.sidebar.number_input(
    "Mortgage Accounts",
    min_value=0,
    max_value=50,
    value=0 if not saved else int(saved["mort_acc"]),
    step=1
)

revol_bal = st.sidebar.number_input(
    "Revolving Balance ($)",
    min_value=0.0,
    max_value=500000.0,
    value=12000.0 if not saved else float(saved["revol_bal"]),
    step=500.0
)

st.sidebar.markdown('</div>', unsafe_allow_html=True)
st.sidebar.markdown('<div class="sidebar-footer">', unsafe_allow_html=True)

run_btn = st.sidebar.button("Run Prediction", use_container_width=True, type="primary")
clear_btn = st.sidebar.button("↺   Clear Results", use_container_width=True, type="secondary")

st.sidebar.markdown('</div>', unsafe_allow_html=True)

# ============================================================
# BUTTON ACTIONS
# ============================================================

if clear_btn:
    for k, v in default_state.items():
        st.session_state[k] = v
    st.rerun()

if run_btn:
    applicant = {
        "term": term,
        "home_ownership": home_ownership,
        "purpose": purpose,
        "emp_length": emp_length,
        "fico_range_low": fico_range_low,
        "loan_amnt": loan_amnt,
        "dti": dti,
        "revol_util": revol_util,
        "annual_inc": annual_inc,
        "open_acc": open_acc,
        "delinq_2yrs": delinq_2yrs,
        "pub_rec": pub_rec,
        "inq_last_6mths": inq_last_6mths,
        "mort_acc": mort_acc,
        "revol_bal": revol_bal,
    }

    try:
        pd_hat, row_df, X_row, macro_dict = predict_single_applicant(applicant, application_date)
        _, grouped_contrib_df = get_prediction_explanation(X_row)

        credit_grade = get_credit_grade(pd_hat)
        grade_class = get_grade_class(credit_grade)
        recommendation = get_recommendation_from_grade(credit_grade)
        reason_groups = classify_reason_groups(grouped_contrib_df)
        explanation_paragraph = make_explanation_paragraph(reason_groups, pd_hat, credit_grade)

        # ── RAG-powered explanation ───────────────────────────────────────────
        rag_explanation = None
        if RAG_AVAILABLE:
            with st.spinner("🧠 Generating macro-aware explanation via RAG…"):
                try:
                    rag_explanation = generate_rag_explanation(
                        grouped_contrib_df=grouped_contrib_df,
                        macro_dict=macro_dict,
                        application_date=application_date,
                        credit_grade=credit_grade,
                        pd_hat=pd_hat,
                    )
                except Exception as rag_err:
                    rag_explanation = (
                        f"RAG explanation unavailable: {rag_err}. "
                        "Make sure Ollama is running (`ollama run phi3:mini`)."
                    )
        # ─────────────────────────────────────────────────────────────────────

        st.session_state.prediction_ready = True
        st.session_state.saved_applicant = applicant
        st.session_state.saved_application_date = application_date
        st.session_state.saved_pd_hat = pd_hat
        st.session_state.saved_row_df = row_df
        st.session_state.saved_macro_dict = macro_dict
        st.session_state.saved_grouped_contrib_df = grouped_contrib_df
        st.session_state.saved_credit_grade = credit_grade
        st.session_state.saved_grade_class = grade_class
        st.session_state.saved_recommendation = recommendation
        st.session_state.saved_reason_groups = reason_groups
        st.session_state.saved_explanation_paragraph = explanation_paragraph
        st.session_state.saved_rag_explanation = rag_explanation

    except Exception as e:
        st.error(f"Prediction failed: {e}")

# ============================================================
# MAIN CONTENT
# ============================================================

if not st.session_state.prediction_ready:
    render_empty_state()
else:
    applicant = st.session_state.saved_applicant
    application_date = st.session_state.saved_application_date
    pd_hat = st.session_state.saved_pd_hat
    row_df = st.session_state.saved_row_df
    macro_dict = st.session_state.saved_macro_dict
    grouped_contrib_df = st.session_state.saved_grouped_contrib_df
    credit_grade = st.session_state.saved_credit_grade
    grade_class = st.session_state.saved_grade_class
    recommendation = st.session_state.saved_recommendation
    reason_groups = st.session_state.saved_reason_groups
    explanation_paragraph = st.session_state.saved_explanation_paragraph
    rag_explanation = st.session_state.saved_rag_explanation

    portfolio_default_rate = float(train_ref["default"].mean()) if "default" in train_ref.columns else np.nan
    base_rate_text = f"{portfolio_default_rate:.2%}" if pd.notna(portfolio_default_rate) else "N/A"

    st.markdown('<div class="main-title">Macro-Aware Credit Risk Scoring</div>', unsafe_allow_html=True)
    st.markdown(
        '<div class="main-subtitle">Prediction results with macro-economic context and model explanation</div>',
        unsafe_allow_html=True
    )

    c1, c2, c3, c4 = st.columns(4)
    with c1:
        render_kpi_card("Predicted PD", f"{pd_hat:.2%}")
    with c2:
        render_kpi_card("Credit Grade", credit_grade, highlight=True, value_class=grade_class)
    with c3:
        render_kpi_card("Recommendation", recommendation)
    with c4:
        render_kpi_card("Portfolio Base Rate", base_rate_text, sublabel="Training default rate")

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    top_left, top_right = st.columns([1.05, 1], gap="large")

    with top_left:
        with st.container(border=True):
            st.markdown('<div class="section-label">Default Probability</div>', unsafe_allow_html=True)
            st.plotly_chart(make_colored_gauge(pd_hat), use_container_width=True, config={"displayModeBar": False})

            st.markdown(
                f"""
                <div style="text-align:center; margin-top:-8px;">
                    <div style="font-size:46px; font-weight:900; color:#0f172a; line-height:1;">{pd_hat:.1%}</div>
                    <div style="font-size:15px; color:#64748b; margin-top:10px;">Predicted default probability</div>
                </div>
                """,
                unsafe_allow_html=True
            )

            legend_cols = st.columns(6)
            legend_data = [
                ("#86efac", "A"),
                ("#bbf7d0", "B"),
                ("#fde68a", "C"),
                ("#fdba74", "D"),
                ("#fca5a5", "E"),
                ("#f87171", "F"),
            ]
            for col, (color, label) in zip(legend_cols, legend_data):
                with col:
                    st.markdown(
                        f"""
                        <div style="display:flex; align-items:center; gap:8px; justify-content:center; margin-top:10px;">
                            <div style="width:14px; height:14px; border-radius:999px; background:{color};"></div>
                            <div style="font-size:13px; color:#64748b;">{label}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with top_right:
        with st.container(border=True):
            st.markdown('<div class="section-label">Why This Result?</div>', unsafe_allow_html=True)

            # ── RAG explanation (primary) ─────────────────────────────────────
            if rag_explanation and not rag_explanation.startswith("RAG explanation unavailable") and not rag_explanation.startswith("⚠️"):
                # Show RAG paragraph as the main explanation
                st.markdown(
                    f'<div class="big-paragraph">{rag_explanation}</div>',
                    unsafe_allow_html=True
                )
                # Collapsed fallback to show the template-based version
                with st.expander("Show rule-based explanation", expanded=False):
                    st.markdown(
                        f'<div style="font-size:15px; color:#475569; line-height:1.7;">'
                        f'{explanation_paragraph}</div>',
                        unsafe_allow_html=True
                    )
            else:
                # Ollama not available — fall back to template explanation
                st.markdown(
                    f'<div class="big-paragraph">{explanation_paragraph}</div>',
                    unsafe_allow_html=True
                )
                if rag_explanation:
                    st.markdown(
                        f'<div style="font-size:12px; color:#94a3b8; margin-top:8px;">'
                        f'ℹ️ {rag_explanation}</div>',
                        unsafe_allow_html=True
                    )
            # ─────────────────────────────────────────────────────────────────

            fr1, fr2, fr3 = st.columns(3)
            with fr1:
                render_factor_column("High Risk", reason_groups["high_risk_factors"], "factor-red")
            with fr2:
                render_factor_column("Moderate Risk", reason_groups["medium_risk_factors"], "factor-amber")
            with fr3:
                render_factor_column("Risk Reducing", reason_groups["risk_reducing_factors"], "factor-green")

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    m1, m2 = st.columns([1, 1.15], gap="large")

    with m1:
        with st.container(border=True):
            st.markdown('<div class="section-label">Macro Context</div>', unsafe_allow_html=True)

            macro_name_map = {
                "Inflation_L6": "Inflation (CPI)",
                "FedFunds_L3": "Federal Funds Rate",
                "HomePrices_L12": "Home Price Index",
                "UNRATE_L6": "Unemployment Rate"
            }

            macro_value_map = {
                "Inflation_L6": lambda x: f"{x:,.3f}",
                "FedFunds_L3": lambda x: f"{x:.2f}%",
                "HomePrices_L12": lambda x: f"{x:,.3f}",
                "UNRATE_L6": lambda x: f"{x:.1f}%"
            }

            entries = list(macro_dict.items())
            mc1, mc2 = st.columns(2)
            card_cols = [mc1, mc2]

            for i, (k, v) in enumerate(entries):
                with card_cols[i % 2]:
                    st.markdown(
                        f"""
                        <div style="background:#f8fafc; border:1px solid #e3eaf3; border-radius:18px; padding:16px 18px; margin-bottom:12px;">
                            <div style="font-size:12px; color:#64748b; margin-bottom:8px;">{macro_name_map.get(k, k)}</div>
                            <div style="font-size:24px; font-weight:800; color:#0f172a;">{macro_value_map.get(k, lambda z: z)(v)}</div>
                        </div>
                        """,
                        unsafe_allow_html=True
                    )

    with m2:
        with st.container(border=True):
            st.markdown('<div class="section-label">Feature Contributions</div>', unsafe_allow_html=True)

            lg1, lg2 = st.columns(2)
            with lg1:
                st.markdown(
                    """
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                        <div style="width:13px; height:13px; border-radius:999px; background:#f87171;"></div>
                        <div style="font-size:13px; color:#64748b;">Increases Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
            with lg2:
                st.markdown(
                    """
                    <div style="display:flex; align-items:center; gap:8px; margin-bottom:10px;">
                        <div style="width:13px; height:13px; border-radius:999px; background:#34d399;"></div>
                        <div style="font-size:13px; color:#64748b;">Reduces Risk</div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )

            st.plotly_chart(
                make_contribution_chart(grouped_contrib_df),
                use_container_width=True,
                config={"displayModeBar": False}
            )

    st.markdown("<div style='height:18px;'></div>", unsafe_allow_html=True)

    with st.container(border=True):
        st.markdown('<div class="section-label">What-If Simulation</div>', unsafe_allow_html=True)

        r1c1, r1c2, r1c3 = st.columns(3)
        with r1c1:
            sim_loan = st.slider("Loan Amount", 500, 100000, int(applicant["loan_amnt"]), step=500, format="$%d")
        with r1c2:
            sim_income = st.slider("Annual Income", 1000, 1000000, int(applicant["annual_inc"]), step=1000, format="$%d")
        with r1c3:
            sim_fico = st.slider("FICO Score", 300, 850, int(applicant["fico_range_low"]), step=1)

        sim_dti = calculate_auto_dti(sim_loan, sim_income)
        st.markdown(
            f'<div class="sim-dti">Simulated DTI: <strong>{sim_dti:.1f}%</strong> <span style="color:#94a3b8;">(auto-calculated from loan amount &amp; income)</span></div>',
            unsafe_allow_html=True
        )

        r2c1, r2c2, r2c3 = st.columns(3)
        with r2c1:
            sim_inq = st.slider("Recent Inquiries", 0, 50, int(applicant["inq_last_6mths"]), step=1)
        with r2c2:
            sim_mort = st.slider("Mortgage Accounts", 0, 50, int(applicant["mort_acc"]), step=1)
        with r2c3:
            sim_revol_bal = st.slider("Revolving Balance", 0, 500000, int(applicant["revol_bal"]), step=500, format="$%d")

        sim_applicant = applicant.copy()
        sim_applicant["loan_amnt"] = sim_loan
        sim_applicant["dti"] = sim_dti
        sim_applicant["annual_inc"] = sim_income
        sim_applicant["fico_range_low"] = sim_fico
        sim_applicant["inq_last_6mths"] = sim_inq
        sim_applicant["mort_acc"] = sim_mort
        sim_applicant["revol_bal"] = sim_revol_bal

        sim_pd, _, _, _ = predict_single_applicant(sim_applicant, application_date)
        delta = sim_pd - pd_hat
        sim_grade = get_credit_grade(sim_pd)

        pb1, pb2, pb3 = st.columns([1, 0.2, 1])
        with pb1:
            st.markdown(
                f"""
                <div class="pd-box">
                    <div class="pd-box-label">Current PD</div>
                    <div class="pd-box-value">{pd_hat:.2%}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

        with pb2:
            st.markdown('<div class="center-arrow">→</div>', unsafe_allow_html=True)

        with pb3:
            delta_class = "pd-delta-up" if delta > 0 else "pd-delta-down" if delta < 0 else ""
            delta_text = f"{delta:+.2%}" if abs(delta) > 0.0001 else "No change"

            st.markdown(
                f"""
                <div class="pd-box">
                    <div class="pd-box-label">Simulated PD</div>
                    <div class="pd-box-value">{sim_pd:.2%}</div>
                    <div style="font-size:14px; color:#64748b; margin-top:8px;">Grade: <strong>{sim_grade}</strong></div>
                    <div class="{delta_class}">{delta_text}</div>
                </div>
                """,
                unsafe_allow_html=True
            )

    with st.expander("Show applicant input and processed values"):
        st.dataframe(row_df, use_container_width=True)

    st.markdown(
        '<div class="footnote">This tool is for analytical and educational use and should support, not replace, human credit judgment.</div>',
        unsafe_allow_html=True
    )