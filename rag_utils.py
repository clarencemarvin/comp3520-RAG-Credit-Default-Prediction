import os
from pathlib import Path
from typing import Any

import faiss
import numpy as np
import pandas as pd
import requests
from sentence_transformers import SentenceTransformer

# ─────────────────────────────────────────────────────────────────────────────
# CONFIG  (paths are relative to this file's location, matching your notebook)
# ─────────────────────────────────────────────────────────────────────────────

BASE_DIR = Path(__file__).resolve().parent

EMBEDDINGS_DIR      = BASE_DIR / "Embeddings"
FAISS_INDEX_PATH    = EMBEDDINGS_DIR / "monetary_policy_faiss.index"
METADATA_CSV_PATH   = EMBEDDINGS_DIR / "monetary_policy_metadata.csv"

EMBEDDING_MODEL_NAME = "all-MiniLM-L6-v2"
OLLAMA_MODEL_NAME    = "llama3.2:1b"
OLLAMA_URL           = "http://localhost:11434/api/generate"
REQUEST_TIMEOUT      = 180   # seconds
TOP_K                = 3     # chunks to retrieve per year

# ─────────────────────────────────────────────────────────────────────────────
# LOAD RESOURCES
# ─────────────────────────────────────────────────────────────────────────────

_embedding_model: SentenceTransformer | None = None
_faiss_index: Any | None = None
_df_meta: pd.DataFrame | None = None


def load_rag_resources() -> tuple[SentenceTransformer, Any, pd.DataFrame]:
    """
    Load (and cache module-level) the embedding model, FAISS index,
    and chunk metadata CSV.  Call once at app startup.

    Returns
    -------
    (embedding_model, faiss_index, df_metadata)
    """
    global _embedding_model, _faiss_index, _df_meta

    if _embedding_model is not None:
        return _embedding_model, _faiss_index, _df_meta

    if not FAISS_INDEX_PATH.exists():
        raise FileNotFoundError(
            f"FAISS index not found: {FAISS_INDEX_PATH}\n"
            "Run the embedding pipeline in RAG_Agent_Consolidated.ipynb first."
        )
    if not METADATA_CSV_PATH.exists():
        raise FileNotFoundError(
            f"Metadata CSV not found: {METADATA_CSV_PATH}\n"
            "Run the embedding pipeline in RAG_Agent_Consolidated.ipynb first."
        )

    _embedding_model = SentenceTransformer(EMBEDDING_MODEL_NAME)
    _faiss_index     = faiss.read_index(str(FAISS_INDEX_PATH))
    _df_meta         = pd.read_csv(METADATA_CSV_PATH)

    return _embedding_model, _faiss_index, _df_meta


# ─────────────────────────────────────────────────────────────────────────────
# RETRIEVAL
# ─────────────────────────────────────────────────────────────────────────────

def retrieve_chunks_by_year(
    query: str,
    year: int,
    embedding_model: SentenceTransformer,
    index: Any,
    df_meta: pd.DataFrame,
    k: int = TOP_K,
) -> list[dict]:
    """
    Retrieve the top-k chunks from the Fed report corpus that are
    (a) most semantically similar to `query` AND
    (b) belong to reports from `year`.

    Returns a list of dicts with keys: section, subsection, text.
    Falls back to global search if no year-specific data exists.
    """
    df_filtered = df_meta[df_meta["year"] == year].reset_index(drop=True)

    # Fall back gracefully when the year isn't in the corpus
    if len(df_filtered) == 0:
        # try adjacent years ±1
        for delta in [1, -1, 2, -2]:
            alt_year = year + delta
            df_filtered = df_meta[df_meta["year"] == alt_year].reset_index(drop=True)
            if len(df_filtered) > 0:
                break

    if len(df_filtered) == 0:
        return []   # nothing we can do

    # Indices in the *full* df_meta that correspond to this year
    # (needed to map FAISS results back)
    year_col = "year" if "year" in df_meta.columns else None
    if year_col:
        full_year_indices = set(df_meta.index[df_meta["year"] == df_filtered["year"].iloc[0]])
    else:
        full_year_indices = set(df_meta.index)

    query_vec = embedding_model.encode(
        [query], convert_to_numpy=True
    ).astype("float32")

    # Over-fetch so we have enough after filtering by year
    distances, indices = index.search(query_vec, min(k * 10, index.ntotal))

    results = []
    for idx in indices[0]:
        if idx in full_year_indices:
            row = df_meta.iloc[idx]
            results.append({
                "section":    row.get("section", ""),
                "subsection": row.get("subsection", ""),
                "text":       row.get("text", ""),
            })
        if len(results) >= k:
            break

    return results


# ─────────────────────────────────────────────────────────────────────────────
# OLLAMA
# ─────────────────────────────────────────────────────────────────────────────

def call_ollama(prompt: str, model: str = OLLAMA_MODEL_NAME) -> str:
    """
    Send `prompt` to the locally running Ollama instance and return
    the raw text response.

    Raises RuntimeError if Ollama is unreachable or returns an error.
    """
    try:
        resp = requests.post(
            OLLAMA_URL,
            json={"model": model, "prompt": prompt, "stream": False},
            timeout=REQUEST_TIMEOUT,
        )
        resp.raise_for_status()
        data = resp.json()
        if "response" not in data:
            raise ValueError(f"Unexpected Ollama response format: {data}")
        return data["response"].strip()
    except requests.exceptions.ConnectionError:
        raise RuntimeError(
            "Cannot reach Ollama. Make sure it is running:\n"
            "    ollama run phi3:mini"
        )


# ─────────────────────────────────────────────────────────────────────────────
# PROMPT BUILDER
# ─────────────────────────────────────────────────────────────────────────────

def _build_shap_lines(grouped_contrib_df: pd.DataFrame, top_n: int = 7) -> str:
    """
    Convert grouped SHAP contributions into a compact, readable string.
    Positive SHAP → increases default risk.
    Negative SHAP → reduces default risk.
    """
    df = grouped_contrib_df.copy()
    df = df.sort_values("abs_shap", ascending=False).head(top_n)

    lines = []
    for _, row in df.iterrows():
        direction = "increases risk" if row["shap_value"] > 0 else "reduces risk"
        lines.append(
            f"  • {row['pretty_feature']}: {direction} "
            f"(SHAP contribution = {row['shap_value']:+.4f})"
        )
    return "\n".join(lines)


def _build_macro_lines(macro_dict: dict) -> str:
    """Format FRED macro numbers into readable bullet points."""
    label_map = {
        "Inflation_L6":     "Inflation / CPI (6-month lag)",
        "FedFunds_L3":      "Federal Funds Rate (3-month lag)",
        "HomePrices_L12":   "Home Price Index (12-month lag)",
        "UNRATE_L6":        "Unemployment Rate (6-month lag)",
    }
    fmt_map = {
        "Inflation_L6":   lambda v: f"{v:,.3f}",
        "FedFunds_L3":    lambda v: f"{v:.2f}%",
        "HomePrices_L12": lambda v: f"{v:,.3f}",
        "UNRATE_L6":      lambda v: f"{v:.1f}%",
    }
    lines = []
    for key, val in macro_dict.items():
        label = label_map.get(key, key)
        fmt   = fmt_map.get(key, lambda v: str(v))
        lines.append(f"  • {label}: {fmt(val)}")
    return "\n".join(lines)


def _build_rag_lines(chunks: list[dict]) -> str:
    """Format retrieved Fed report chunks into numbered evidence blocks."""
    if not chunks:
        return "  (No Federal Reserve report data available for this period.)"
    lines = []
    for i, chunk in enumerate(chunks, 1):
        section = chunk.get("section", "")
        subsect = chunk.get("subsection", "")
        text    = chunk.get("text", "").strip()[:300]   # cap length
        header  = " > ".join(filter(None, [section, subsect]))
        lines.append(f"  [{i}] {header}\n      {text}")
    return "\n\n".join(lines)


def build_explanation_prompt(
    grouped_contrib_df: pd.DataFrame,
    macro_dict: dict,
    chunks: list[dict],
    credit_grade: str,
    pd_hat: float,
    year: int,
) -> str:
    shap_block  = _build_shap_lines(grouped_contrib_df)
    macro_block = _build_macro_lines(macro_dict)
    rag_block   = _build_rag_lines(chunks)

    prompt = f"""You are a senior credit risk analyst writing a brief explanation
for a loan officer reviewing an application.

APPLICANT SUMMARY
-----------------
Credit Grade : {credit_grade}
Predicted Probability of Default: {pd_hat:.1%}
Application Year: {year}

SHAP RISK DRIVERS (model explanation)
--------------------------------------
{shap_block}

MACROECONOMIC INDICATORS AT TIME OF APPLICATION
------------------------------------------------
{macro_block}

FEDERAL RESERVE REPORT CONTEXT ({year})
----------------------------------------
{rag_block}

TASK
----
Write exactly ONE concise paragraph of 60-100 words maximum that explains to
the loan officer why this applicant received grade {credit_grade} with a
{pd_hat:.1%} default probability.

Rules you must follow:
1. Reference specific SHAP drivers by name and describe whether they raise
   or lower risk.
2. Connect the macroeconomic conditions listed above to the applicant's
   risk profile — explain *how* those conditions amplify or dampen the
   borrower-level factors.
3. Only use Federal Reserve context if it directly relates to household 
   credit risk or borrower behavior — ignore bond spreads, 
   capitalization rates, or institutional finance concepts.
4. Do NOT say "industry standard", "benchmark", or any threshold 
   you were not explicitly given. Only use numbers from the data above.
5. When referencing macro indicators, use the actual values provided 
   (e.g. "with unemployment at X% and inflation at Y...") — 
   do not speak in generalities.
6. Do NOT invent numbers or make up facts not present in the data above.
7. Write in plain, professional language suitable for a dashboard tooltip.
8. Output only the paragraph — no headings, bullet points, or JSON.

Paragraph:"""

    return prompt


# ─────────────────────────────────────────────────────────────────────────────
# MAIN ENTRY POINT  (called from app.py)
# ─────────────────────────────────────────────────────────────────────────────

def generate_rag_explanation(
    grouped_contrib_df: pd.DataFrame,
    macro_dict: dict,
    application_date,
    credit_grade: str,
    pd_hat: float,
) -> str:
    """
    Full pipeline: retrieve relevant Fed report chunks for the application year,
    build a rich prompt fusing SHAP + macro + RAG context, call Ollama,
    and return the explanation paragraph.

    Parameters
    ----------
    grouped_contrib_df : pd.DataFrame
        Output of group_contributions() in app.py — must have columns
        pretty_feature, shap_value, abs_shap.
    macro_dict : dict
        FRED macro values from get_macro_features_for_date() in app.py.
    application_date : date or str
        The loan application date (used to extract the year).
    credit_grade : str
        Grade letter, e.g. "F".
    pd_hat : float
        Predicted probability of default, e.g. 0.418.

    Returns
    -------
    str  — one paragraph of explanation, or an error message string.
    """
    # Extract year
    try:
        year = pd.Timestamp(application_date).year
    except Exception:
        year = 2018   # safe fallback

    # Load RAG resources (cached after first call)
    try:
        emb_model, faiss_idx, df_meta = load_rag_resources()
    except FileNotFoundError as exc:
        return (
            f"⚠️ RAG resources unavailable — {exc}  "
            "The explanation below is based on model features only."
        )

    # Build retrieval query from top SHAP features + macro context
    top_features = (
        grouped_contrib_df
        .sort_values("abs_shap", ascending=False)
        .head(3)["pretty_feature"]
        .tolist()
    )
    query = (
        f"In {year}, how did macroeconomic conditions such as inflation, "
        f"unemployment, and interest rates affect household credit risk "
        f"and loan default probability? Focus on borrower factors: "
        f"{', '.join(top_features)}."
    )

    chunks = retrieve_chunks_by_year(
        query=query,
        year=year,
        embedding_model=emb_model,
        index=faiss_idx,
        df_meta=df_meta,
        k=TOP_K,
    )

    prompt = build_explanation_prompt(
        grouped_contrib_df=grouped_contrib_df,
        macro_dict=macro_dict,
        chunks=chunks,
        credit_grade=credit_grade,
        pd_hat=pd_hat,
        year=year,
    )

    try:
        paragraph = call_ollama(prompt)
    except RuntimeError as exc:
        paragraph = str(exc)

    return paragraph