# Macro-Aware Credit Risk Dashboard
### An Agent-Based RAG Framework for Credit Default Risk Assessment Across Economic Cycles

> COMP3520 — The University of Hong Kong  
> Achal Agarwal · Tanujaya Clarence Marvin · Zheng Choi I Chloe

---

## Overview

This project implements a **macro-aware credit risk scoring system** that combines:

- A **LightGBM model** trained on 1M+ LendingClub loan records (2007–2019)
- **FRED macroeconomic indicators** aligned to each loan's origination date (CPI, Fed Funds Rate, Unemployment, Home Price Index)
- A **RAG agent (Agent B)** that retrieves relevant Federal Reserve Monetary Policy Reports and generates a dynamic, data-grounded explanation for each prediction using a local LLM

The dashboard takes a borrower profile as input and outputs a predicted probability of default, credit grade (A–F), SHAP-based risk factor breakdown, and a natural language explanation that fuses model outputs with macroeconomic context.

---

## Project Structure

```
RAG Agent - Copy/
├── app.py                          # Streamlit dashboard
├── rag_utils.py                    # RAG pipeline (retrieval + prompt + Ollama)
├── RAG_Agent_Consolidated.ipynb    # Pipeline: PDF → chunks → embeddings
├── main.ipynb                      # Model training (LightGBM + SHAP)
├── requirements.txt
├── artifacts/                      # Trained model and preprocessor (not tracked)
│   ├── model.txt                   # LightGBM booster
│   ├── preprocessor.pkl            # Sklearn preprocessor
│   ├── feature_schema.json
│   ├── grade_thresholds.json
│   └── training_reference.csv
├── Embeddings/                     # FAISS index (not tracked — generate locally)
│   ├── monetary_policy_faiss.index
│   └── monetary_policy_metadata.csv
├── RAW Reports/                    # Fed Monetary Policy PDFs (not tracked)
├── Cleaned Reports/                # Extracted text (not tracked)
├── Chunks/                         # Chunked text (not tracked)
└── Metadata/                       # Report metadata (not tracked)
```

---

## Prerequisites

- Python 3.12
- [Ollama](https://ollama.com/download/mac) installed and running
- Federal Reserve Monetary Policy Report PDFs placed in `RAW Reports/` — named as `mpr_yyyy_mm.pdf`
- Trained model artifacts in `artifacts/`

---

## Installation

### 1. Clone the repo

```bash
git clone https://github.com/clarencemarvin/comp3520-RAG-Credit-Default-Prediction.git
cd comp3520-RAG-Credit-Default-Prediction
```

### 2. Install Python dependencies

```bash
pip install -r requirements.txt
```

### 3. Install faiss via conda (required — pip version breaks on Mac)

```bash
conda install -c conda-forge faiss-cpu
```

### 4. Fix OpenMP conflict on Mac (if you get a libiomp5 error)

```bash
echo 'export KMP_DUPLICATE_LIB_OK=TRUE' >> ~/.zshrc
source ~/.zshrc
```

### 5. Install and start Ollama

Download from [ollama.com](https://ollama.com/download/mac), then:

```bash
ollama pull llama3.2:1b
ollama run llama3.2:1b
```

Keep this terminal open — Ollama must be running when you use the dashboard.

---

## Building the RAG Embeddings

Before running the dashboard for the first time, you need to build the FAISS index from your Fed report PDFs.

Open `RAG_Agent_Consolidated.ipynb` in Jupyter and run the cells **in order**, stopping after the **Embeddings cell**:

```
1. Install cell
2. Paths cell
3. Metadata cell       → generates Metadata/report_metadata.csv
4. Text Extraction     → generates Cleaned Reports/*.txt
5. Chunking cell       → generates Chunks/monetary_policy_chunks_v3.csv
6. Embeddings cell     → generates Embeddings/monetary_policy_faiss.index
                                     Embeddings/monetary_policy_metadata.csv
```

> ⚠️ Do **not** run the LLM section of the notebook — that is now handled by `rag_utils.py`.

**Important:** In the Embeddings cell, make sure the save lines read:

```python
faiss.write_index(index, faiss_path)
df.to_csv(metadata_path, index=False)
```

Not `FAISS_INDEX_PATH` / `METADATA_CSV_PATH` (those are defined later in the notebook and will cause a crash).

Verify the output:

```bash
ls Embeddings/
# monetary_policy_faiss.index
# monetary_policy_metadata.csv
```

---

## Running the Dashboard

```bash
streamlit run app.py
```

Open [http://localhost:8501](http://localhost:8501) in your browser.

---

## How It Works

### Prediction pipeline (`app.py`)
1. User fills in borrower profile in the sidebar
2. FRED macroeconomic indicators are fetched for the application date
3. LightGBM predicts probability of default
4. SHAP values explain which features drove the prediction
5. Credit grade (A–F) and recommendation are assigned

### RAG explanation pipeline (`rag_utils.py`)
1. Top SHAP features are extracted to form a retrieval query
2. FAISS retrieves the most relevant Fed report chunks for the application year
3. A prompt is built combining SHAP values + FRED macro numbers + retrieved Fed report context
4. The prompt is sent to the local Ollama LLM (`llama3.2:1b`)
5. The LLM generates a concise, grounded explanation paragraph
6. The explanation replaces the hardcoded template in the "Why This Result?" panel

---

## Configuration

Key settings in `rag_utils.py`:

| Variable | Default | Description |
|---|---|---|
| `OLLAMA_MODEL_NAME` | `llama3.2:1b` | Local LLM for explanation generation |
| `EMBEDDING_MODEL_NAME` | `all-MiniLM-L6-v2` | Sentence embedding model — **do not change** |
| `TOP_K` | `3` | Number of Fed report chunks to retrieve |

> ⚠️ Do **not** change `EMBEDDING_MODEL_NAME` — it must match the model used to build the FAISS index.

---

## Troubleshooting

| Error | Fix |
|---|---|
| `zsh: command not found: ollama` | Add to PATH: `echo 'export PATH="/Applications/Ollama.app/Contents/Resources:$PATH"' >> ~/.zshrc && source ~/.zshrc` |
| `OMP: Error #15: libiomp5.dylib` | Run `export KMP_DUPLICATE_LIB_OK=TRUE` |
| `zsh: segmentation fault` | Reinstall faiss via conda: `conda install -c conda-forge faiss-cpu` |
| `ModuleNotFoundError: faiss` | Use conda, not pip: `conda install -c conda-forge faiss-cpu` |
| RAG explanation unavailable | Make sure Ollama is running: `ollama run llama3.2:1b` |
| Streamlit connection error | Check terminal for Python errors — likely a missing dependency |

---

## Team

| Name | Student ID |
|---|---|
| Achal Agarwal | 3036030893 |
| Tanujaya Clarence Marvin | 3035993933 |
| Zheng Choi I Chloe | 3035987788 |

---

## License

For academic use only — COMP3520, The University of Hong Kong, 2026.
