# AutoRAG 360: Automotive Knowledge & Recall Intelligence Platform

AutoRAG 360 is a Streamlit prototype for automotive recall intelligence,
complaint analysis, governed document retrieval, predictive ML, and operational
monitoring. It was built in three phases to show how a lightweight RAG and ML
platform can be delivered without Docker, Kafka, Milvus, Kubernetes, or required
API keys.

The app is designed for Streamlit Cloud free tier and runs with local files,
scikit-learn, SQLite, and deterministic fallbacks.

## Architecture

```text
Recall CSVs, complaint CSVs, service/quality/supplier docs
        |
        v
Data loading and cleaning
        |
        v
Chunking + metadata tagging + TF-IDF embeddings
        |
        v
Role-based access control
        |
        v
Local cosine retrieval
        |
        v
Grounded generation: Groq -> Gemini -> extractive fallback
        |
        v
RAG Assistant, Access Demo, Evaluation, Monitoring

AI4I + Automobile + complaint text
        |
        v
Maintenance, vehicle risk, complaint severity models
        |
        v
Predictions + feature importance
        |
        v
Monitoring database + Executive Overview
```

## Full Feature List

- Executive Overview with recall count, complaint count, top recall components,
  maintenance model health, and prediction counter.
- Automotive RAG Assistant with role selection, grounded answers, source
  citations, access rule display, and visible answer mode.
- Access Control Demo comparing two roles side by side.
- Predictive Maintenance page using AI4I-style machine-failure classification.
- Vehicle Risk Scoring page using UCI Automobile-style normalized-loss risk
  categories and nearest-neighbor vehicle profiles.
- Complaint Severity page using rule-labeled complaint text and TF-IDF Logistic
  Regression.
- Golden-question RAG Evaluation page with precision@k, hit rate, MRR, latency,
  and no-answer rate.
- Monitoring page backed by SQLite audit logs for RAG queries and ML
  predictions.
- Architecture page with project rationale, diagram, JD mapping, dataset
  sources, and phase history.
- Optional LLM generation through Groq first, then Gemini, then extractive
  fallback.
- No required API keys. If no LLM key is present, answers are extractive and
  cited.

## JD Alignment

| JD capability | AutoRAG 360 implementation |
| --- | --- |
| Retrieval-augmented generation | TF-IDF RAG over recalls, complaints, and automotive documents |
| Responsible AI | Grounded answer modes, citations, and extractive fallback |
| Access control | Four-role filtering by `document_type` and `sensitivity_level` |
| Machine learning | Maintenance, severity, and vehicle risk models |
| Evaluation | Golden-question precision, hit rate, MRR, latency, no-answer metrics |
| Monitoring | SQLite audit logs for queries, predictions, fallbacks, latency, access denials |
| Executive analytics | KPI cards, model health, and prediction activity |
| Cloud-friendly delivery | Streamlit app with local artifacts and lightweight dependencies |

## Dataset Sources

| Dataset | Used for | Status |
| --- | --- | --- |
| NHTSA-style recalls and complaints | RAG corpus and complaint analytics | Generated fallback sample unless replaced with real NHTSA CSVs |
| AI4I 2020 predictive maintenance | Machine failure classification | Real file used when present; synthetic fallback generated if absent |
| UCI Automobile | Vehicle risk category modeling | UCI-style raw file normalized to `data/raw/automobile.csv` when present; synthetic fallback if absent |
| Sample automotive documents | Service manual, quality report, recall policy, supplier notice | Project-created Phase 1 sample documents |

## Data Notes

The bundled recall and complaint CSVs are synthetic NHTSA-style fallback data in
this workspace. They are realistic enough for a demo, but they are not official
NHTSA data.

The AI4I file in this workspace is a real AI4I-style CSV with the expected
columns. If it is removed, the training script creates a clearly commented
synthetic fallback.

The automobile file is normalized from a UCI Automobile-style raw file already
present in `data/raw/automobile.csv.txt`. If neither file is present, the
training script creates a clearly commented synthetic fallback.

Optional Groq and Gemini generation are disabled unless API keys are provided.
Without keys, the RAG assistant uses extractive answers from retrieved chunks
only.

## Setup Locally

```powershell
cd "C:\Users\admin\Desktop\AutoRAG 360\autorag-360"
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install --upgrade pip
python -m pip install -r requirements.txt
```

Train models and explainability artifacts:

```powershell
python -m src.ml.train_maintenance
python -m src.ml.train_severity
python -m src.ml.train_vehicle_risk
python -m src.ml.explainability
```

Run evaluation and tests:

```powershell
python -m src.rag.evaluation
python -m pytest -q
```

Run the app:

```powershell
streamlit run app.py
```

Open `http://localhost:8501`.

## Streamlit Cloud Setup

1. Push the `autorag-360` folder to a GitHub repository.
2. In Streamlit Cloud, set `app.py` as the app entry point.
3. Keep `requirements.txt` as-is for the lightweight free-tier install.
4. Optional secrets:
   - `GROQ_API_KEY`
   - `GEMINI_API_KEY`
5. If no secrets are configured, the app still runs with extractive RAG answers.

## Monitoring Checks

The app writes audit data to `artifacts/monitoring.db`.

```powershell
@'
import sqlite3
from pathlib import Path

db = Path("artifacts/monitoring.db")
with sqlite3.connect(db) as con:
    print("rag_queries", con.execute("select count(*) from rag_queries").fetchone()[0])
    print("ml_predictions", con.execute("select count(*) from ml_predictions").fetchone()[0])
'@ | python -
```

## Resume Bullets

- Built a Streamlit automotive intelligence platform combining RAG, access
  control, ML risk models, evaluation, and monitoring.
- Implemented local TF-IDF retrieval with role-based filtering and grounded
  answer fallback for no-key deployments.
- Trained scikit-learn models for maintenance failure, complaint severity, and
  vehicle risk classification with saved metrics and feature importances.
- Added golden-question RAG evaluation with precision@k, hit rate, MRR, latency,
  and no-answer metrics.
- Designed SQLite audit monitoring for query latency, fallback usage, access
  denials, and ML prediction activity.

## Built in 3 Phases

Phase 1 built the data ingestion, RAG pipeline, access control, Executive
Overview, RAG Assistant, and Access Control Demo.

Phase 2 added predictive maintenance, complaint severity, vehicle risk scoring,
model metrics, and explainability.

Phase 3 added optional LLM generation, golden-question evaluation, SQLite
monitoring, architecture documentation, and the final nine-page sidebar.
