# main.py — ConvoInsight BE (Flask, Cloud Run ready)
# Polars-first + PandasAI wrappers; GCS/Firestore persistence; provider keys via Firestore (encrypted)

import os, io, csv, json, time, uuid, re, html
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace
from collections import defaultdict

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# --- Polars + PandasAI (Polars-first)
import polars as pl
import pandasai as pai
from pandasai_litellm.litellm import LiteLLM
from litellm import completion, model_list as LITELLM_MODEL_LIST

# kept imports for compatibility (not used directly in the new path)
from pandasai import SmartDataframe, SmartDatalake   # noqa: F401
from pandasai.core.response.dataframe import DataFrameResponse  # noqa: F401
import pandas as pd  # only used as shim when PandasAI wrapper rejects Polars

# --- Google Cloud
from google.cloud import storage
from google.cloud import firestore

# Signed URL on Cloud Run (no private key)
import google.auth
from google.auth.transport.requests import Request as GoogleAuthRequest
try:
    from google.auth import iam
except Exception:  # pragma: no cover
    iam = None
from google.oauth2 import service_account

import requests
from cryptography.fernet import Fernet

# --- Optional .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# --- Optional PDF export
_REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False

# -------- App Config --------
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
GCP_PROJECT_ID = os.getenv("GCP_PROJECT_ID")

FERNET_KEY = os.getenv("FERNET_KEY")
fernet = Fernet(FERNET_KEY.encode()) if FERNET_KEY else None

CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://convoinsight.vercel.app"
).split(",")

DATASETS_ROOT = os.getenv("DATASETS_ROOT", os.path.abspath("./datasets"))
CHARTS_ROOT   = os.getenv("CHARTS_ROOT",   os.path.abspath("./charts"))
os.makedirs(DATASETS_ROOT, exist_ok=True)
os.makedirs(CHARTS_ROOT,   exist_ok=True)

# GCS / Firestore
GCS_BUCKET                  = os.getenv("GCS_BUCKET")
GCS_DATASETS_PREFIX         = os.getenv("GCS_DATASETS_PREFIX", "datasets")
GCS_DIAGRAMS_PREFIX         = os.getenv("GCS_DIAGRAMS_PREFIX", "diagrams")
GCS_SIGNED_URL_TTL_SECONDS  = int(os.getenv("GCS_SIGNED_URL_TTL_SECONDS", "604800"))  # 7 days
GOOGLE_SERVICE_ACCOUNT_EMAIL = os.getenv("GOOGLE_SERVICE_ACCOUNT_EMAIL")

# Firestore collections
FIRESTORE_COLLECTION_SESSIONS  = os.getenv("FIRESTORE_SESSIONS_COLLECTION", "convo_sessions")
FIRESTORE_COLLECTION_DATASETS  = os.getenv("FIRESTORE_DATASETS_COLLECTION", "datasets_meta")
FIRESTORE_COLLECTION_PROVIDERS = os.getenv("FIRESTORE_PROVIDERS_COLLECTION", "convo_providers")
FIRESTORE_COLLECTION_PG        = os.getenv("FIRESTORE_PG_COLLECTION", "pg_connections")  # new

# --- PostgreSQL (transaction pooler) ---
from sqlalchemy import create_engine, inspect
from sqlalchemy.pool import NullPool
PG_SSLMODE = "require"  # a0.0.8

# --- Flask init
app = Flask(__name__)
CORS(app, origins=[o.strip() for o in CORS_ORIGINS if o.strip()], supports_credentials=True)

# --- GCP clients
_storage_client = storage.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else storage.Client()
_firestore_client = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else firestore.Client()

# --- Global cancel flags
_CANCEL_FLAGS = set()  # holds session_id

# =========================
# Utilities
# =========================
def slug(s: str) -> str:
    return "-".join(s.strip().split()).lower()

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

def get_content(r):
    """Extract content from LiteLLM response (robust)."""
    try:
        msg = r.choices[0].message
        return msg["content"] if isinstance(msg, dict) else msg.content
    except Exception:
        pass
    if isinstance(r, dict):
        return r.get("choices", [{}])[0].get("message", {}).get("content", "")
    try:
        chunks = []
        for ev in r:
            delta = getattr(ev.choices[0], "delta", None)
            if delta and getattr(delta, "content", None):
                chunks.append(delta.content)
        return "".join(chunks)
    except Exception:
        return str(r)

def _safe_json_loads(s: str):
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        start = s.find("{")
        end = s.rfind("}")
        if start >= 0 and end >= 0 and end > start:
            return json.loads(s[start:end+1])
        raise

def _should_cancel(session_id: str) -> bool:
    return session_id in _CANCEL_FLAGS

def _cancel_if_needed(session_id: str):
    if _should_cancel(session_id):
        _CANCEL_FLAGS.discard(session_id)
        raise RuntimeError("CANCELLED_BY_USER")

# =========================
# Providers (keys, models)
# =========================
def get_provider_config(provider: str, api_key: str):
    if provider == "openai":
        return {"url": "https://api.openai.com/v1/models", "headers": {"Authorization": f"Bearer {api_key}"}}
    elif provider == "groq":
        return {"url": "https://api.groq.com/openai/v1/models", "headers": {"Authorization": f"Bearer {api_key}"}}
    elif provider == "anthropic":
        return {"url": "https://api.anthropic.com/v1/models", "headers": {"x-api-key": api_key}}
    elif provider == "google":
        return {"url": f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}", "headers": {}}
    else:
        raise ValueError("Provider not supported")

def save_provider_key(user_id: str, provider: str, encrypted_key: str, models: list):
    try:
        doc_id = f"{user_id}_{provider}"
        doc_ref = _firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS).document(doc_id)
        data = {
            "user_id": user_id,
            "provider": provider,
            "token": encrypted_key,
            "models": models,
            "is_active": True,
            "updated_at": firestore.SERVER_TIMESTAMP,
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        doc_ref.set(data, merge=True)
        return True
    except Exception as e:
        print("Firestore save error:", e)
        return False

def _compose_model_id(provider: str, model: Optional[str]) -> str:
    """Make a litellm id like 'gemini/gemini-2.5-pro' or 'openai/gpt-4o-mini'."""
    if model and "/" in model:
        return model
    prefix_map = {"google": "gemini", "gemini": "gemini", "openai": "openai", "anthropic": "anthropic", "groq": "groq"}
    prefix = prefix_map.get((provider or "").lower(), "gemini")
    default_model_map = {
        "gemini": "gemini-2.5-pro",
        "openai": "gpt-4o-mini",
        "anthropic": "claude-3-5-sonnet-20241022",
        "groq": "llama-3.1-70b-versatile",
    }
    chosen = model or default_model_map.get(prefix, "gemini-2.5-pro")
    return f"{prefix}/{chosen}"

def _get_user_provider_token(user_id: str, provider: str) -> Optional[str]:
    """Fetch encrypted token from Firestore, decrypt with Fernet."""
    try:
        doc_id = f"{user_id}_{provider}"
        doc = _firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS).document(doc_id).get()
        if not doc.exists:
            return None
        enc = (doc.to_dict() or {}).get("token")
        if not enc or not fernet:
            return None
        return fernet.decrypt(enc.encode()).decode()
    except Exception:
        return None

# =========================
# Conversation state (Firestore)
# =========================
def _fs_default_state():
    return {
        "history": [],
        "last_visual_gcs_path": "",
        "last_visual_signed_url": "",
        "last_visual_kind": "",
        "last_analyzer_text": "",
        "last_plan": None,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "created_at": firestore.SERVER_TIMESTAMP,
    }

def _fs_sess_ref(session_id: str):
    return _firestore_client.collection(FIRESTORE_COLLECTION_SESSIONS).document(session_id)

def _get_conv_state(session_id: str) -> dict:
    doc = _fs_sess_ref(session_id).get()
    if doc.exists:
        data = doc.to_dict() or {}
        for k, v in _fs_default_state().items():
            if k not in data:
                data[k] = v
        return data
    else:
        state = _fs_default_state()
        _fs_sess_ref(session_id).set(state, merge=True)
        return state

def _save_conv_state(session_id: str, state: dict):
    st = dict(state)
    st["updated_at"] = firestore.SERVER_TIMESTAMP
    _fs_sess_ref(session_id).set(st, merge=True)

def _append_history(state: dict, role: str, content: str, max_len=10_000, keep_last=100):
    content = str(content)
    if len(content) > max_len:
        content = content[:max_len] + " …"
    hist = state.get("history") or []
    hist.append({"role": role, "content": content, "ts": time.time()})
    state["history"] = hist[-keep_last:]

# =========================
# GCS helpers & Signed URLs
# =========================
def _gcs_bucket():
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not set")
    return _storage_client.bucket(GCS_BUCKET)

def _metadata_sa_email() -> Optional[str]:
    try:
        r = requests.get(
            "http://metadata.google.internal/computeMetadata/v1/instance/service-accounts/default/email",
            headers={"Metadata-Flavor": "Google"},
            timeout=1.5,
        )
        if r.status_code == 200:
            return r.text.strip()
    except Exception:
        pass
    return None

def _signed_url(blob, filename: str, content_type: str, ttl_seconds: int) -> str:
    """V4 signed URL that works on Cloud Run (IAM signer if possible)."""
    credentials, _ = google.auth.default(
        scopes=[
            "https://www.googleapis.com/auth/cloud-platform",
            "https://www.googleapis.com/auth/iam",
        ]
    )
    auth_req = GoogleAuthRequest()
    try:
        credentials.refresh(auth_req)
    except Exception:
        pass

    sa_email = getattr(credentials, "service_account_email", None)
    if not sa_email or sa_email.lower() == "default":
        sa_email = GOOGLE_SERVICE_ACCOUNT_EMAIL or _metadata_sa_email()

    if iam is not None and sa_email:
        try:
            signer = iam.Signer(auth_req, credentials, sa_email)
            signing_creds = service_account.Credentials(
                signer=signer,
                service_account_email=sa_email,
                token_uri="https://oauth2.googleapis.com/token",
            )
            return blob.generate_signed_url(
                version="v4",
                expiration=timedelta(seconds=ttl_seconds),
                method="GET",
                response_disposition=f'inline; filename="{filename}"',
                response_type=content_type,
                credentials=signing_creds,
            )
        except Exception:
            pass

    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=ttl_seconds),
        method="GET",
        response_disposition=f'inline; filename="{filename}"',
        response_type=content_type,
    )

# =========================
# Polars helpers (a0.0.8)
# =========================
def _normalize_columns_to_str(df: pl.DataFrame) -> pl.DataFrame:
    new_names = [str(c) for c in df.columns]
    if new_names != df.columns:
        df = df.set_column_names(new_names)
    return df

def _polars_info_string(df: pl.DataFrame) -> str:
    lines = [f"shape: {df.shape[0]} rows x {df.shape[1]} columns", "dtypes/nulls:"]
    try:
        nulls = df.null_count()
        nulls_map = {col: int(nulls[col][0]) for col in nulls.columns}
    except Exception:
        nulls_map = {c: None for c in df.columns}
    for name, dtype in df.schema.items():
        n = nulls_map.get(name, "n/a")
        lines.append(f"  - {name}: {dtype} (nulls={n})")
    return "\n".join(lines)

def _to_polars_dataframe(obj):
    if isinstance(obj, pl.DataFrame):
        return _normalize_columns_to_str(obj)
    try:
        df = pl.from_dataframe(obj)
        return _normalize_columns_to_str(df)
    except Exception:
        return None

def _as_pai_df(df):
    """Prefer Polars; only fall back to Pandas if wrapper complains."""
    if isinstance(df, pl.DataFrame):
        df = _normalize_columns_to_str(df)
    try:
        return pai.DataFrame(df)
    except TypeError:
        if isinstance(df, pl.DataFrame):
            pdf = df.to_pandas()
        else:
            pdf = df
        if hasattr(pdf, "columns"):
            pdf.columns = [str(c) for c in pdf.columns]
        return pai.DataFrame(pdf)

# ---- CSV & Excel universal readers (local & GCS)
def _sniff_csv_separator_text(sample: str, default: str = "|") -> str:
    candidates = [",", ";", "\t", "|", ":"]
    try:
        dialect = csv.Sniffer().sniff(sample, delimiters=candidates)
        return dialect.delimiter
    except Exception:
        for d in candidates:
            if d in sample.splitlines()[0]:
                return d
        return default

def _read_csv_bytes_to_polars(data: bytes, sep_candidates: List[str] = (",","|",";","\t")) -> pl.DataFrame:
    last_err = None
    for sep in sep_candidates:
        try:
            df = pl.read_csv(io.BytesIO(data), separator=sep)
            return _normalize_columns_to_str(df)
        except Exception as e:
            last_err = e
            continue
    try:
        df = pl.read_csv(io.BytesIO(data))
        return _normalize_columns_to_str(df)
    except Exception as e:
        raise last_err or e

def _read_local_csv_to_polars(path: str) -> pl.DataFrame:
    with open(path, "rb") as f:
        data = f.read()
    return _read_csv_bytes_to_polars(data)

def _is_excel_file(name: str) -> bool:
    return name.lower().endswith((".xlsx", ".xls", ".xlsm"))

def _read_any_local_tabular(path: str) -> Optional[pl.DataFrame]:
    if _is_excel_file(path):
        try:
            pdf = pd.read_excel(path)  # shim
            return _normalize_columns_to_str(pl.from_pandas(pdf))
        except Exception:
            return None
    else:
        try:
            return _read_local_csv_to_polars(path)
        except Exception:
            return None

def list_gcs_tabular(domain: str) -> List[storage.Blob]:
    safe = slug(domain)
    prefix = f"{GCS_DATASETS_PREFIX}/{safe}/"
    return list(_gcs_bucket().list_blobs(prefix=prefix))

def read_gcs_tabular_to_polars(blob: storage.Blob) -> Optional[pl.DataFrame]:
    data = blob.download_as_bytes()
    name = blob.name.lower()
    if name.endswith((".xlsx", ".xls", ".xlsm")):
        try:
            pdf = pd.read_excel(io.BytesIO(data))
            return _normalize_columns_to_str(pl.from_pandas(pdf))
        except Exception:
            return None
    else:
        return _read_csv_bytes_to_polars(data)

# ---- Upload helpers
def _upload_dataset_file_local(file_storage, *, domain: str) -> dict:
    safe_domain = slug(domain)
    folder = ensure_dir(os.path.join(DATASETS_ROOT, safe_domain))
    filename = file_storage.filename
    dest = os.path.join(folder, filename)
    file_storage.save(dest)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return {"filename": filename, "gs_uri": "", "signed_url": "", "size_bytes": size, "local_path": dest}

def upload_dataset_file(file_storage, *, domain: str) -> dict:
    if not GCS_BUCKET:
        return _upload_dataset_file_local(file_storage, domain=domain)
    try:
        safe_domain = slug(domain)
        filename = file_storage.filename
        blob_name = f"{GCS_DATASETS_PREFIX}/{safe_domain}/{filename}"
        bucket = _gcs_bucket()
        blob = bucket.blob(blob_name)
        content_type = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if _is_excel_file(filename) else "text/csv"
        blob.cache_control = "private, max-age=0"
        blob.content_type = content_type
        file_storage.stream.seek(0)
        blob.upload_from_file(file_storage.stream, rewind=True, size=None, content_type=content_type)
        size = blob.size or 0
        gs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        try:
            signed = _signed_url(blob, filename, content_type, GCS_SIGNED_URL_TTL_SECONDS)
        except Exception:
            signed = ""
        try:
            _save_dataset_meta(domain, filename, gs_uri, size)
        except Exception:
            pass
        return {"filename": filename, "gs_uri": gs_uri, "signed_url": signed, "size_bytes": size}
    except Exception:
        return _upload_dataset_file_local(file_storage, domain=domain)

def list_gcs_csvs(domain: str) -> List[storage.Blob]:
    return [b for b in list_gcs_tabular(domain) if b.name.lower().endswith(".csv")]

def read_gcs_csv_to_pl_df(gs_uri_or_blobname: str) -> pl.DataFrame:
    if gs_uri_or_blobname.startswith("gs://"):
        _, bucket_name, *rest = gs_uri_or_blobname.replace("gs://","").split("/")
        blob_name = "/".join(rest)
        bucket = _storage_client.bucket(bucket_name)
    else:
        bucket = _gcs_bucket()
        blob_name = gs_uri_or_blobname
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    return _read_csv_bytes_to_polars(data)

def delete_gcs_object(blob_name_or_gs_uri: str):
    if blob_name_or_gs_uri.startswith("gs://"):
        _, bucket_name, *rest = blob_name_or_gs_uri.replace("gs://","").split("/")
        blob_name = "/".join(rest)
        bucket = _storage_client.bucket(bucket_name)
    else:
        bucket = _gcs_bucket()
        blob_name = blob_name_or_gs_uri
    bucket.blob(blob_name).delete()

# ---- Diagrams (upload to GCS)
def _detect_diagram_kind(local_html_path: str, visual_hint: str) -> str:
    try:
        with open(local_html_path, "r", encoding="utf-8", errors="ignore") as f:
            head = f.read(20000).lower()
        if "plotly" in head or "plotly.newplot" in head:
            return "charts"
        if "<table" in head:
            return "tables"
    except Exception:
        pass
    return "tables" if str(visual_hint).lower().strip() == "table" else "charts"

def upload_diagram_to_gcs(local_path: str, *, domain: str, session_id: str, run_id: str, kind: str) -> dict:
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    safe_domain = slug(domain)
    filename = f"{session_id}_{run_id}.html"
    kind = "tables" if kind == "tables" else "charts"
    blob_name = f"{GCS_DIAGRAMS_PREFIX}/{kind}/{safe_domain}/{filename}"
    bucket = _gcs_bucket()
    blob = bucket.blob(blob_name)
    blob.cache_control = "public, max-age=86400"
    blob.content_type = "text/html; charset=utf-8"
    blob.upload_from_filename(local_path)
    return {
        "blob_name": blob_name,
        "gs_uri": f"gs://{GCS_BUCKET}/{blob_name}",
        "signed_url": _signed_url(blob, filename, "text/html", GCS_SIGNED_URL_TTL_SECONDS),
        "public_url": f"https://storage.googleapis.com/{GCS_BUCKET}/{blob_name}",
        "kind": kind,
    }

# =========================
# Router / Orchestrator / Agent system prompts (a0.0.8)
# =========================
router_system_configuration = """Make sure all of the information below is applied.
1. You are the Orchestration Router: decide which agents/LLMs to run for a business data prompt.
2. Output must be STRICT one-line JSON with keys: need_manipulator, need_visualizer, need_analyzer, need_compiler, compiler_model, visual_hint, reason.
3. Precedence & overrides: Direct user prompt > Router USER config > Router DOMAIN config > Router SYSTEM defaults.
4. Flexibility: treat system defaults as fallbacks (e.g., default colors, currency, timezone). If the user or domain requests a different value, obey that without changing core routing logic.
5. Use recent conversation context when deciding (short follow-ups may reuse prior data/visual).
6. Consider user phrasing to infer needs (e.g., “use bar chart” => visualizer needed).
7. Identify data manipulation needs (clean/aggregate/compute shares/rates) when raw data is not analysis/visual-ready.
8. Identify analysis needs for why/driver/trend/explain, or for optimization/allocation/gap-closure style tasks.
9. Agents vs compiler: Manipulator/Visualizer/Analyzer are data-capable agents; Compiler is an LLM-only formatter with no direct data access.
10. Data flow: Visualizer and Analyzer consume the manipulated dataframe produced by the Manipulator.
11. Rules of thumb: if prompt contains “chart/plot/visualize/graph/bar/line/table” then need_visualizer=true.
12. Rules of thumb: if prompt contains “why/driver/explain/root cause/trend/surprise” then need_analyzer=true.
13. Rules of thumb: if prompt mentions allocation, optimization, plan, gap-closure, “minimum number of additional takers”, set need_analyzer=true and set visual_hint="table".
14. If follow-up with no new data ops implied and a processed df exists, set need_manipulator=false to reuse the previous dataframe.
15. Compiler always runs; default compiler_model="gemini/gemini-2.5-pro" unless the domain/user requires otherwise.
16. visual_hint ∈ {"bar","line","table","auto"}; pick the closest fit and prefer "table" for plan/allocation outputs.
17. Keep the reason short (≤120 chars). No prose beyond the JSON.
18. In short: choose the most efficient set of agents/LLMs to answer the prompt well while respecting overrides.
19. By default, Manipulator and Analyzer should always be used in most scenario, because response compiler did not have access to the complete detailed data.
"""

orchestrator_system_configuration = """1. Honor precedence: direct user prompt > USER specific configuration > DOMAIN specific configuration > SYSTEM defaults.
2. Think step by step.
3. You orchestrate 3 LLM PandasAI Agents for business data analysis.
4. The 3 agents are: Data Manipulator, Data Visualizer, Data Analyser.
5. Emit a specific prompt for each of those 3 agents.
6. Each prompt is a numbered, step-by-step instruction set.
7. Prompts must be clear, detailed, and complete to avoid ambiguity.
8. The number of steps may differ per agent.
9. Example user tasks include: (a) revenue this week vs last; (b) why revenue dropped; (c) surprises this month; (d) notable trends; (e) correlation between revenue and bounces; (f) whether a conversion rate is normal for the season.
10. Reason strictly from the user-provided data.
11. Convert a short business question into three specialist prompts.
12. Use the Router Context Hint and Visualization hint when applicable.
13. Respect the user- and domain-level configurations injected below; overrides must not alter core process.
14. All specialists operate in Python using PandasAI Semantic DataFrames (pai.DataFrame) backed by Polars DataFrames.
15. Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
16. Each value must be a single-line string. No extra keys, no prose, no markdown/code fences.
"""

data_manipulator_system_configuration = """1. Honor precedence: direct user prompt > USER specific configuration > DOMAIN specific configuration > SYSTEM defaults.
2. Enforce data hygiene before analysis.
3. Parse dates to datetime; create explicit period columns (day/week/month).
4. Set consistent dtypes for numeric fields; strip/normalize categorical labels; standardize currency units if present.
5. Handle missing values: impute or drop only when necessary; keep legitimate zeros.
6. Mind each dataset’s name; avoid collisions in merges/aggregations.
7. Produce exactly the minimal, analysis-ready dataframe(s) needed for the user question, with stable, well-named columns.
8. Include the percentage version of appropriate raw value columns (share-of-total where relevant).
9. End by returning only:
    result = {"type":"dataframe","value": <THE_FINAL_DATAFRAME>}
10. Honor any user-level and domain-level instructions injected below.
"""

data_visualizer_system_configuration = """1. Honor precedence: direct user prompt > USER specific configuration > DOMAIN specific configuration > SYSTEM defaults.
2. Produce exactly ONE interactive visualization (a Plotly diagram or a table) per request.
3. Choose the best form based on the user's question: Plotly diagrams for trends/comparisons; Table for discrete, plan, or allocation outputs.
4. For explicit user preference: if prompt says “plotly table” use Plotly Table.
5. For Plotly diagrams: prevent overlaps (rotate axis ticks ≤45°), wrap long labels, ensure margins, place legend outside plot.
6. For Plotly diagrams: insight-first formatting (clear title/subtitle, axis units, thousands separators, rich hover).
7. Aggregate data to sensible granularity (day/week/month) and cap extreme outliers for readability (note in subtitle).
8. Use bar, grouped bar, or line chart; apply a truncated monochromatic colorscale by sampling from 0.25–1.0 of a standard scale (e.g., Blues).
9. Output Python code only (no prose/comments/markdown). Import os and datetime. Build an export dir and a run-scoped timestamped filename using globals()["_RUN_ID"].
10. Write the file exactly once using an atomic lock (.lock) to avoid duplicates across retries; write fig HTML or table HTML as appropriate.
11. Ensure file_path is a plain Python string; do not print/return anything else.
12. The last line of code MUST be exactly:
    result = {"type": "string", "value": file_path}
13. DO NOT rely on pandas-specific styling; prefer Plotly Table when a table is needed.
"""

data_analyzer_system_configuration = """1. Honor precedence: direct user prompt > USER configuration specific > DOMAIN specific configuration > SYSTEM defaults.
2. Write like you’re speaking to a person; be concise and insight-driven.
3. Quantify where possible (deltas, % contributions, time windows); reference exact columns/filters used.
4. Return only:
    result = {"type":"string","value":"<3–6 crisp bullets or 2 short paragraphs of insights>"}
"""

response_compiler_system_configuration = """1. Honor precedence: direct user prompt > USER specific configuration > DOMAIN specific configuration > SYSTEM defaults.
2. Brevity: ≤180 words; bullets preferred; no code blocks, no JSON, no screenshots.
3. Lead with the answer: 1–2 sentence “Bottom line” with main number, time window, and delta.
4. Quantified drivers: top 3 with magnitude, direction, and approx contribution (absolute and % where possible).
5. Next actions: 2–4 prioritized, concrete actions with expected impact/rationale.
6. Confidence & caveats: one short line on data quality/assumptions/gaps; include Confidence: High/Medium/Low.
7. Minimal tables: ≤1 table only if essential (≤5×3); otherwise avoid tables.
8. No repetition: do not restate agent text; synthesize it.
9. Do not try to show images; Do not mention the path of the generated file if there is one..
10. Always include units/currency and exact comparison window (e.g., “Aug 2025 vs Jul 2025”, “W34 vs W33”).
11. Show both absolute and % change where sensible (e.g., “+$120k (+8.4%)”).
12. Round smartly (money to nearest K unless < $10k; rates 1–2 decimals).
13. If any agent fails or data is incomplete, still produce the best insight; mark gaps in Caveats and adjust Confidence.
14. If the user asks “how much/which/why,” the first sentence must provide the number/entity/reason.
15. Exact compiler_instruction template the orchestrator should emit (single line; steps separated by ';'):
16. Read the user prompt, data_info, and all three agent responses;
17. Compute the direct answer including the main number and compare period;
18. Identify the top 3 quantified drivers with direction and contribution;
19. Draft 'Bottom line' in 1–2 sentences answering plainly;
20. List 2–4 prioritized Next actions with expected impact;
21. Add a one-line Caveats with Confidence and any gaps;
22. Keep ≤180 words, use bullets, avoid tables unless ≤5×3 and essential;
23. Include units, absolute and % deltas, and explicit dates;
24. Do not repeat agent text verbatim or include code/JSON.
25. Format hint (shape, not literal):
26. Bottom line — <answer with number + timeframe>.
27. Drivers — <A: +X (≈Y%); B: −X (≈Y%); C: ±X (≈Y%)>.
28. Next actions — 1) <action>; 2) <action>; 3) <action>.
29. Caveats — <one line>. Confidence: <High/Medium/Low>.
30. compiler_instruction must contain clear, step-by-step instructions to assemble the final response.
31. The final response must be decision-ready and insight-first, not raw data.
32. The compiler_instruction is used as the compiler LLM’s system content.
33. Compiler user content will be: f"User Prompt:{user_prompt}. \nData Info:{data_info}. \nData Describe:{data_describe}. \nData Manipulator Response:{data_manipulator_response}. \nData Visualizer Response:{data_visualizer_response}. \nData Analyzer Response:{data_analyzer_response}".
34. `data_info` is a string summary of dataframe types/shape.
35. `data_manipulator_response` is a PandasAI DataFrameResponse.
36. `data_visualizer_response` is a file path to an HTML/PNG inside {"type":"string","value": ...} with a plain Python string path.
37. `data_analyzer_response` is a PandasAI StringResponse.
38. Your goal in `compiler_instruction` is to force brevity, decisions, and insights.
39. Mention the dataset name involved of each statement.
40. SHOULD BE STRICTLY ONLY respond in HTML format.
"""

# --- User/Domain configs (kept)
user_specific_configuration = """1. (no user-specific instructions provided yet)."""

domain_specific_configuration = """1. Use period labels like m0 (current month) and m1 (prior month); apply consistently.
2. Use IDR as currency, for example: Rp93,000.00 atau Rp354,500.00.
3. Use blue themed chart and table colors.
4. target should be in mn (million).
5. %TUR is take up rate percentage.
6. % Taker, % Transaction, and % Revenue squad is the percentage of each product of all product Revenue all is in bn which is billion idr.
7. Revenue Squad is in mn wich is million idr.
8. rev/subs and rev/trx should be in thousands of idr.
9. MoM is month after month in percentage
10. Subs is taker.
"""

# =========================
# Data Loading (Polars-first) — CSV, Excel, Postgres
# =========================
def _read_sql_to_polars(conn, query: str) -> pl.DataFrame:
    """Try polars native; fallback to pandas.read_sql then convert."""
    try:
        return pl.read_database(query, conn)
    except Exception as e:
        print(f"[warn] pl.read_database failed, falling back to pandas.read_sql: {e}")
        pdf = pd.read_sql(query, conn)
        return pl.from_pandas(pdf)

def _pg_engine(pg_uri: str):
    return create_engine(
        pg_uri,
        poolclass=NullPool,                 # let PgBouncer handle pooling
        connect_args={"sslmode": PG_SSLMODE},
    )

def load_postgres_tables_to_dfs(
    engine_url: str,
    tables: Optional[List[str]] = None,
    schema: str = "public",
    limit: Optional[int] = None,
    key_prefix: str = "pg"
):
    """Load one/many tables into global dfs dict (polars)."""
    engine = _pg_engine(engine_url)
    with engine.connect() as conn:
        insp = inspect(conn)
        available = set(insp.get_table_names(schema=schema))
        target = tables or sorted(list(available))
        if not target:
            print(f"No tables found in schema '{schema}'.")
            return
        for t in target:
            if t not in available:
                print(f"[skip] Table '{t}' not found in schema '{schema}'. Available: {sorted(list(available))}")
                continue
            q = f'SELECT * FROM "{schema}"."{t}"'
            if limit is not None:
                q += f" LIMIT {int(limit)}"
            df = _read_sql_to_polars(conn, q)
            df = _normalize_columns_to_str(df)
            key = f"{key_prefix}:{schema}.{t}"
            _PG_CONTEXT["dfs"][key] = df
            _PG_CONTEXT["data_info"][key] = _polars_info_string(df)

def load_postgres_sql_to_dfs(engine_url: str, query: str, name: Optional[str] = None, key_prefix: str = "pg") -> str:
    """Run arbitrary SQL and store in dfs dict."""
    engine = _pg_engine(engine_url)
    with engine.connect() as conn:
        df = _read_sql_to_polars(conn, query)
        df = _normalize_columns_to_str(df)
    key = name if name else f"{key_prefix}:query:{abs(hash(query))}"
    _PG_CONTEXT["dfs"][key] = df
    _PG_CONTEXT["data_info"][key] = _polars_info_string(df)
    return key

# In-memory PG context merged into request-run dfs
_PG_CONTEXT = {"dfs": {}, "data_info": {}}

def _load_domain_dataframes(domain: str, dataset_filters: Optional[set]) -> Tuple[Dict[str, pl.DataFrame], Dict[str, str], Dict[str, str]]:
    """Load CSV+Excel (GCS/local)."""
    dfs: Dict[str, pl.DataFrame] = {}
    data_info: Dict[str, str] = {}
    data_describe: Dict[str, str] = {}

    # GCS first
    try:
        if GCS_BUCKET:
            for b in list_gcs_tabular(domain):
                name = os.path.basename(b.name)
                if dataset_filters and name not in dataset_filters:
                    continue
                if not (name.lower().endswith(".csv") or _is_excel_file(name)):
                    continue
                df = read_gcs_tabular_to_polars(b)
                if df is None:
                    continue
                dfs[name] = df
                info_str = _polars_info_string(df)
                data_info[name] = info_str
                try:
                    data_describe[name] = df.describe().to_pandas().to_json()
                except Exception:
                    data_describe[name] = ""
    except Exception:
        pass

    # Local fallback
    domain_dir = ensure_dir(os.path.join(DATASETS_ROOT, slug(domain)))
    for name in sorted(os.listdir(domain_dir)):
        if not (name.lower().endswith(".csv") or _is_excel_file(name)):
            continue
        if dataset_filters and name not in dataset_filters:
            continue
        if name in dfs:
            continue
        path = os.path.join(domain_dir, name)
        try:
            df = _read_any_local_tabular(path)
            if df is None:
                continue
            dfs[name] = df
            info_str = _polars_info_string(df)
            data_info[name] = info_str
            try:
                data_describe[name] = df.describe().to_pandas().to_json()
            except Exception:
                data_describe[name] = ""
        except Exception:
            pass

    # Merge any Postgres-loaded frames for this run (if any were requested before)
    for k, v in _PG_CONTEXT["dfs"].items():
        dfs[k] = v
        data_info[k] = _PG_CONTEXT["data_info"].get(k, _polars_info_string(v))
        try:
            data_describe[k] = v.describe().to_pandas().to_json()
        except Exception:
            data_describe[k] = ""

    return dfs, data_info, data_describe

# =========================
# Router & Orchestrator calls (a0.0.8)
# =========================
def _run_router(user_prompt: str, data_info, data_describe, state: dict, *, llm_model: str, llm_api_key: Optional[str]) -> dict:
    router_start = time.time()
    recent_context = json.dumps(state.get("history", [])[-6:], ensure_ascii=False)
    router_response = completion(
        model=llm_model,
        messages=[
            {"role": "system", "content": router_system_configuration.strip()},
            {"role": "user", "content":
                f"""Make sure all of the information below is applied.
                User Prompt: {user_prompt}
                Recent Context: {recent_context}
                Data Info (summary): {data_info}
                Data Describe (summary): {data_describe}"""
            },
        ],
        seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high",
        api_key=llm_api_key
    )
    router_content = get_content(router_response)
    try:
        plan = _safe_json_loads(router_content)
    except Exception:
        p = user_prompt.lower()
        need_visual = bool(re.search(r"\b(chart|plot|graph|visual|bar|line|table)\b", p))
        need_analyzer = bool(re.search(r"\b(why|driver|explain|root cause|trend|surprise|allocate|allocation|optimi[sz]e|min(?:imum)? number|gap closure|takers?)\b", p))
        plan = {
            "need_manipulator": True,
            "need_visualizer": need_visual,
            "need_analyzer": need_analyzer,
            "need_compiler": True,
            "compiler_model": llm_model,
            "visual_hint": "table" if need_analyzer else "auto",
            "reason": "Fallback heuristic",
        }
    router_end = time.time()
    plan["_elapsed"] = float(router_end - router_start)
    return plan

def _run_orchestrator(user_prompt: str, domain: str, data_info, data_describe, visual_hint: str, context_hint: dict, *, llm_model: str, llm_api_key: Optional[str]):
    resp = completion(
        model=llm_model,
        messages=[
            {"role": "system", "content": f"""
            You are the Orchestrator.
            Make sure all of the information below is applied.

            orchestrator_system_configuration:
            {orchestrator_system_configuration}

            data_manipulator_system_configuration:
            {data_manipulator_system_configuration}

            data_visualizer_system_configuration:
            {data_visualizer_system_configuration}

            data_analyzer_system_configuration:
            {data_analyzer_system_configuration}

            response_compiler_system_configuration:
            {response_compiler_system_configuration}

            user_specific_configuration:
            {user_specific_configuration}

            domain_specific_configuration:
            {domain_specific_configuration}"""},
            {"role": "user", "content":
                f"""Make sure all of the information below is applied.
                User Prompt: {user_prompt}
                Datasets Domain name: {domain}.
                df.info of each dfs key(file name)-value pair:
                {data_info}.
                df.describe of each dfs key(file name)-value pair:
                {data_describe}.
                Router Context Hint: {json.dumps(context_hint)}
                Visualization hint (from router): {visual_hint}"""
            }
        ],
        seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high",
        api_key=llm_api_key
    )
    content = get_content(resp)
    try:
        spec = _safe_json_loads(content)
    except Exception:
        spec = {"manipulator_prompt":"", "visualizer_prompt":"", "analyzer_prompt":"", "compiler_instruction":""}
    return spec

# =========================
# Health & Static
# =========================
@app.get("/health")
def health():
    return jsonify({"status": "healthy", "ts": datetime.utcnow().isoformat()})

@app.get("/")
def root():
    return jsonify({"ok": True, "service": "ConvoInsight BE", "health": "/health"})

@app.route("/charts/<path:relpath>")
def serve_chart(relpath):
    full = os.path.join(CHARTS_ROOT, relpath)
    base = os.path.dirname(full)
    filename = os.path.basename(full)
    return send_from_directory(base, filename)

# =========================
# Provider key management
# =========================
def _require_fernet():
    if not fernet:
        raise RuntimeError("FERNET_KEY is not configured on server")

@app.route("/validate-key", methods=["POST"])
def validate_key():
    """Validate provider API key and store encrypted copy + model list."""
    try:
        data = request.get_json() or {}
        provider = data.get("provider")
        api_key = data.get("apiKey")
        user_id = data.get("userId")
        if not provider or not api_key or not user_id:
            return jsonify({"valid": False, "error": "Missing provider, apiKey, or userId"}), 400

        _require_fernet()
        cfg = get_provider_config(provider, api_key)
        res = requests.get(cfg["url"], headers=cfg["headers"], timeout=6)
        if res.status_code == 200:
            try:
                models_json = res.json()
                if isinstance(models_json, dict) and "data" in models_json:
                    models = [m.get("id","") for m in models_json.get("data", [])]
                elif isinstance(models_json, dict) and "models" in models_json:
                    models = [m.get("name","") for m in models_json.get("models", [])]
                else:
                    models = []
            except Exception:
                models = []

            encrypted = fernet.encrypt(api_key.encode()).decode()
            ok = save_provider_key(user_id, provider, encrypted, models)
            return jsonify({"valid": True, "saved": ok, "models": models})
        else:
            return jsonify({"valid": False, "status": res.status_code, "detail": res.text[:300]})
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 500

# =========================
# Datasets
# =========================
def _save_dataset_meta(domain: str, filename: str, gs_uri: str, size_bytes: int):
    try:
        doc_id = f"{slug(domain)}::{filename}"
        _firestore_client.collection(FIRESTORE_COLLECTION_DATASETS).document(doc_id).set({
            "domain": slug(domain),
            "filename": filename,
            "gs_uri": gs_uri,
            "size_bytes": int(size_bytes),
            "updated_at": firestore.SERVER_TIMESTAMP,
            "created_at": firestore.SERVER_TIMESTAMP,
        }, merge=True)
    except Exception as e:
        print("save dataset meta error:", e)

def _list_dataset_meta(*, domain: str) -> List[dict]:
    out = []
    try:
        q = _firestore_client.collection(FIRESTORE_COLLECTION_DATASETS).where("domain","==",slug(domain)).stream()
        for d in q:
            if not d.exists: continue
            out.append(d.to_dict() or {})
    except Exception:
        pass
    return out

@app.post("/domains/<domain>/datasets/upload")
def upload_datasets(domain: str):
    try:
        items = []
        for _, file in request.files.items():
            meta = upload_dataset_file(file, domain=domain)
            items.append(meta)
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.get("/domains/<domain>/datasets")
def list_domain_datasets(domain: str):
    try:
        items: List[dict] = []

        # Firestore metas (try add signed_url)
        try:
            fs_items = _list_dataset_meta(domain=domain)
            for it in fs_items:
                gs_uri = it.get("gs_uri","")
                if gs_uri:
                    try:
                        _, bucket_name, *rest = gs_uri.replace("gs://","").split("/")
                        blob_name = "/".join(rest)
                        blob = _storage_client.bucket(bucket_name).blob(blob_name)
                        fn = it.get("filename","")
                        ctype = "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet" if _is_excel_file(fn) else "text/csv"
                        it["signed_url"] = _signed_url(blob, fn, ctype, GCS_SIGNED_URL_TTL_SECONDS)
                    except Exception:
                        it.setdefault("signed_url","")
            items.extend(fs_items)
        except Exception:
            pass

        # Local directory listing (CSV+Excel)
        try:
            domain_dir = os.path.join(DATASETS_ROOT, slug(domain))
            if os.path.isdir(domain_dir):
                known = { i.get("filename") for i in items }
                for name in sorted(os.listdir(domain_dir)):
                    if (name.lower().endswith(".csv") or _is_excel_file(name)) and name not in known:
                        path = os.path.join(domain_dir, name)
                        size = os.path.getsize(path) if os.path.exists(path) else 0
                        items.append({"domain": slug(domain), "filename": name, "gs_uri": "", "size_bytes": size, "signed_url": ""})
        except Exception:
            pass

        return jsonify({"items": items, "datasets": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# Backward-compat duplicate (existing)
@app.get("/domains/<domain>/datasets/")
def list_domain_datasets_trailing(domain: str):
    return list_domain_datasets(domain)

# ---------- Compat route aliases (to avoid 404 from frontend) ----------
# 1) GET /datasets?domain=Campaign:1
@app.get("/datasets")
def list_datasets_qs():
    domain = request.args.get("domain")
    if not domain:
        return jsonify({"detail": "Missing query param 'domain'"}), 400
    return list_domain_datasets(domain)

# 2) GET /datasets/<domain>
@app.get("/datasets/<path:domain>")
def list_datasets_path(domain):
    return list_domain_datasets(domain)

# 3) GET /domains/datasets?domain=Campaign:1
@app.get("/domains/datasets")
def list_datasets_domains_qs():
    domain = request.args.get("domain")
    if not domain:
        return jsonify({"detail": "Missing query param 'domain'"}), 400
    return list_domain_datasets(domain)

# 4) POST /datasets/upload?domain=Campaign:1
@app.post("/datasets/upload")
def upload_datasets_qs():
    domain = request.args.get("domain")
    if not domain:
        return jsonify({"detail": "Missing query param 'domain'"}), 400
    return upload_datasets(domain)

# 5) POST /datasets/<domain>/upload
@app.post("/datasets/<path:domain>/upload")
def upload_datasets_path(domain):
    return upload_datasets(domain)

# 6) POST /domains/datasets/upload?domain=Campaign:1
@app.post("/domains/datasets/upload")
def upload_datasets_domains_qs():
    domain = request.args.get("domain")
    if not domain:
        return jsonify({"detail": "Missing query param 'domain'"}), 400
    return upload_datasets(domain)
# ----------------------------------------------------------------------

# =========================
# Sessions / History / Export
# =========================
@app.get("/sessions")
def sessions_list():
    try:
        limit = int(request.args.get("limit","20"))
        col = _firestore_client.collection(FIRESTORE_COLLECTION_SESSIONS)
        docs = col.order_by("updated_at", direction=firestore.Query.DESCENDING).limit(limit).stream()
        items = []
        for d in docs:
            if not d.exists: continue
            data = d.to_dict() or {}
            items.append({
                "session_id": d.id,
                "updated_at": str(data.get("updated_at","")),
                "created_at": str(data.get("created_at","")),
                "last_visual_signed_url": data.get("last_visual_signed_url","") or "",
                "last_visual_kind": data.get("last_visual_kind",""),
                "last_plan": data.get("last_plan"),
            })
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.get("/sessions/<session_id>/history")
def sessions_history(session_id):
    try:
        st = _get_conv_state(session_id)
        return jsonify({
            "session_id": session_id,
            "history": st.get("history", []),
            "last_visual_signed_url": st.get("last_visual_signed_url",""),
            "last_visual_kind": st.get("last_visual_kind",""),
            "last_plan": st.get("last_plan"),
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# =========================
# PostgreSQL connection storage (optional)
# =========================
@app.post("/pg/save")
def pg_save():
    """Save/replace user's PG connection (host,user,pass,db,port) in Firestore."""
    try:
        body = request.get_json() or {}
        user_id = body.get("userId")
        cfg = body.get("config") or {}
        if not user_id or not cfg:
            return jsonify({"ok": False, "detail": "Missing userId or config"}), 400
        _firestore_client.collection(FIRESTORE_COLLECTION_PG).document(user_id).set({
            "config": cfg, "updated_at": firestore.SERVER_TIMESTAMP
        }, merge=True)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)}), 500

def _pg_uri_from_cfg(cfg: dict) -> str:
    user = cfg.get("user") or cfg.get("username")
    pwd  = cfg.get("password")
    host = cfg.get("host")
    port = cfg.get("port", 5432)
    db   = cfg.get("database") or cfg.get("db")
    return f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}?sslmode={PG_SSLMODE}"

def _load_pg_for_request(user_id: Optional[str], pg: Optional[dict]) -> Optional[str]:
    """Return engine URI if available (from the request or Firestore)."""
    cfg = None
    if isinstance(pg, dict) and pg.get("host"):
        cfg = pg
    elif user_id:
        try:
            d = _firestore_client.collection(FIRESTORE_COLLECTION_PG).document(user_id).get()
            if d.exists:
                cfg = (d.to_dict() or {}).get("config")
        except Exception:
            cfg = None
    if cfg:
        return _pg_uri_from_cfg(cfg)
    return None

# =========================
# /think — response explainer (plan)
# =========================
def _force_strict_html(s: str) -> str:
    s = (s or "").strip()
    if s.startswith("<"):
        return s
    return f"<div><p>{html.escape(s)}</p></div>"

@app.post("/think")
def think():
    """Return a 1-paragraph plan explainer for the given prompt + data context."""
    try:
        body = request.get_json() or {}
        session_id = body.get("sessionId") or str(uuid.uuid4())
        user_id = body.get("userId") or "anon"
        provider = (body.get("provider") or "google").lower()
        model = body.get("model")
        chosen_model_id = _compose_model_id(provider, model)
        chosen_api_key = _get_user_provider_token(user_id, provider)

        domain = body.get("domain") or "default"
        datasets = body.get("datasets") or []  # optional filter
        dataset_filters = set(datasets) if datasets else None

        # Optional PG load for context
        _PG_CONTEXT["dfs"].clear(); _PG_CONTEXT["data_info"].clear()
        pg_cfg = body.get("pg")  # {"tables": [...]} or {"sql": "..."} + connection (or saved)
        engine_uri = _load_pg_for_request(user_id, pg_cfg)
        if engine_uri:
            if isinstance(pg_cfg, dict) and pg_cfg.get("tables"):
                load_postgres_tables_to_dfs(engine_uri, tables=list(pg_cfg["tables"]), schema=pg_cfg.get("schema","public"), limit=pg_cfg.get("limit"))
            if isinstance(pg_cfg, dict) and pg_cfg.get("sql"):
                load_postgres_sql_to_dfs(engine_uri, pg_cfg["sql"], name=pg_cfg.get("name"))

        # Load tabulars
        dfs, data_info, data_describe = _load_domain_dataframes(domain, dataset_filters)
        prompt = body.get("prompt") or ""
        state = _get_conv_state(session_id)

        # Router → Orchestrator (to get plan)
        agent_plan = _run_router(prompt, data_info, data_describe, state, llm_model=chosen_model_id, llm_api_key=chosen_api_key)
        visual_hint = agent_plan.get("visual_hint", "auto")
        context_hint = {"router_plan": agent_plan, "dataset_filter": (sorted(datasets) if datasets else "ALL")}
        spec = _run_orchestrator(prompt, domain, data_info, data_describe, visual_hint, context_hint, llm_model=chosen_model_id, llm_api_key=chosen_api_key)

        # Plan explainer text
        plan_explainer_response = completion(
            model=chosen_model_id,
            messages=[
                {"role":"system","content":(
                    "Make sure all of the information below is applied.\n"
                    "1) The prompt you get is the exact system plan for answering the user.\n"
                    "2) Summarize it into one paragraph like a thought-process of what the system will do.\n"
                    "3) Keep it to a single paragraph and include reasons for crucial steps."
                )},
                {"role":"user","content":(
                    f"User Prompt: {prompt}\n"
                    f"Domain: {domain}\n"
                    f"df.info: {data_info}\n"
                    f"df.describe: {data_describe}\n"
                    f"Router plan: {json.dumps(agent_plan)}\n"
                    f"Orchestrator spec: {json.dumps(spec)}"
                )}
            ],
            seed=1, stream=False, verbosity="medium", drop_params=True, reasoning_effort="high",
            api_key=chosen_api_key
        )
        thinking = get_content(plan_explainer_response) or ""
        thinking = re.sub(r"\s+", " ", thinking).strip()
        state["last_plan"] = {"router": agent_plan, "orchestrator": spec, "thinking": thinking}
        _save_conv_state(session_id, state)
        return jsonify({"sessionId": session_id, "thinking": thinking, "plan": agent_plan, "orchestrator": spec})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# =========================
# /query — main pipeline (a0.0.8)
# =========================
@app.post("/query")
def query():
    try:
        body = request.get_json() or {}
        session_id = body.get("sessionId") or str(uuid.uuid4())
        user_id = body.get("userId") or "anon"
        provider = (body.get("provider") or "google").lower()
        model = body.get("model")
        chosen_model_id = _compose_model_id(provider, model)
        chosen_api_key = _get_user_provider_token(user_id, provider)

        domain = body.get("domain") or "default"
        datasets = body.get("datasets") or []  # optional filter by file names
        dataset_filters = set(datasets) if datasets else None
        prompt = body.get("prompt") or ""
        include_insight = body.get("includeInsight", True)
        want_thinking = bool(body.get("thinking", False))  # optional: return plan explainer

        # Optional PG load for this request
        _PG_CONTEXT["dfs"].clear(); _PG_CONTEXT["data_info"].clear()
        pg_cfg = body.get("pg")  # may contain connection or rely on saved one
        engine_uri = _load_pg_for_request(user_id, pg_cfg)
        if engine_uri:
            if isinstance(pg_cfg, dict) and pg_cfg.get("tables"):
                load_postgres_tables_to_dfs(engine_uri, tables=list(pg_cfg["tables"]), schema=pg_cfg.get("schema","public"), limit=pg_cfg.get("limit"))
            if isinstance(pg_cfg, dict) and pg_cfg.get("sql"):
                load_postgres_sql_to_dfs(engine_uri, pg_cfg["sql"], name=pg_cfg.get("name"))

        # Load dataframes
        dfs, data_info, data_describe = _load_domain_dataframes(domain, dataset_filters)
        if not dfs:
            available = []
            if GCS_BUCKET:
                try:
                    available.extend(sorted({os.path.basename(b.name) for b in list_gcs_tabular(domain)}))
                except Exception:
                    pass
            try:
                domain_dir = os.path.join(DATASETS_ROOT, slug(domain))
                if os.path.isdir(domain_dir):
                    available.extend(sorted([n for n in os.listdir(domain_dir) if (n.lower().endswith(".csv") or _is_excel_file(n))]))
            except Exception:
                pass
            return jsonify({"code":"NEED_UPLOAD", "detail": f"No datasets found in domain '{domain}'", "domain": domain, "available": available}), 409

        if dataset_filters:
            missing = dataset_filters.difference(set(dfs.keys()))
            if missing:
                available = sorted(list(dfs.keys()))
                return jsonify({"code":"DATASET_NOT_FOUND", "detail": f"Requested datasets {sorted(list(missing))} not found in domain '{domain}'.", "domain": domain, "available": available}), 404

        state = _get_conv_state(session_id)

        agent_plan = _run_router(prompt, data_info, data_describe, state, llm_model=chosen_model_id, llm_api_key=chosen_api_key)
        need_manip = bool(agent_plan.get("need_manipulator", True))
        need_visual = bool(agent_plan.get("need_visualizer", True))
        need_analyze = include_insight and bool(agent_plan.get("need_analyzer", True))
        need_compile = bool(agent_plan.get("need_compiler", True))
        compiler_model = agent_plan.get("compiler_model") or chosen_model_id
        visual_hint = agent_plan.get("visual_hint", "auto")

        context_hint = {
            "router_plan": agent_plan,
            "last_visual_path": state.get("last_visual_gcs_path",""),
            "has_prev_df_processed": False,
            "last_analyzer_excerpt": (state.get("last_analyzer_text") or "")[:400],
            "dataset_filter": (sorted(datasets) if datasets else "ALL"),
        }

        _cancel_if_needed(session_id)
        spec = _run_orchestrator(prompt, domain, data_info, data_describe, visual_hint, context_hint, llm_model=chosen_model_id, llm_api_key=chosen_api_key)
        manipulator_prompt = spec.get("manipulator_prompt", "")
        visualizer_prompt  = spec.get("visualizer_prompt", "")
        analyzer_prompt    = spec.get("analyzer_prompt", "")
        compiler_instruction = spec.get("compiler_instruction", "")

        thinking = ""
        if want_thinking:
            pe = completion(
                model=chosen_model_id,
                messages=[
                    {"role":"system","content":(
                        "Summarize the following plan into a single paragraph that explains what will be done and why key steps matter."
                    )},
                    {"role":"user","content":(
                        f"User Prompt: {prompt}\n"
                        f"Plan: {json.dumps(agent_plan)}\n"
                        f"Spec: {json.dumps(spec)}"
                    )}
                ],
                seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high",
                api_key=chosen_api_key
            )
            thinking = re.sub(r"\s+"," ", get_content(pe) or "").strip()

        llm = LiteLLM(model=chosen_model_id, api_key=chosen_api_key)
        pai.config.set({"llm": llm, "seed": 1, "stream": False, "verbosity": "low", "drop_params": True, "save_charts": False, "open_charts": False, "conversational": False, "enforce_privacy": True, "reasoning_effort": "high", "save_charts_path": "./charts"})

        _cancel_if_needed(session_id)
        df_processed = None
        dm_resp = None
        if need_manip or (need_visual or need_analyze):
            semantic_dfs = []
            for key, d in dfs.items():
                try:
                    semantic_dfs.append(pai.DataFrame(d))
                except TypeError:
                    pdf = d.to_pandas()
                    pdf.columns = [str(c) for c in pdf.columns]
                    semantic_dfs.append(pai.DataFrame(pdf))
            dm_resp = pai.chat(manipulator_prompt, *semantic_dfs)
            val = getattr(dm_resp, "value", dm_resp)
            df_processed = _to_polars_dataframe(val)

        _cancel_if_needed(session_id)
        dv_resp = SimpleNamespace(value="")
        diagram_signed_url = None
        diagram_kind = ""
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        globals()["_RUN_ID"] = run_id

        if need_visual:
            assert df_processed is not None, "Visualization requested but no processed dataframe available."
            pai.config.set({"llm": llm, "conversational": False})
            semantic_df = _as_pai_df(df_processed)
            dv_resp = semantic_df.chat(visualizer_prompt)

            local_path = ""
            try:
                export_dir = "./charts"; os.makedirs(export_dir, exist_ok=True)
                if hasattr(dv_resp, "save"):
                    img_path = os.path.join(export_dir, f"viz_{int(time.time())}.png")
                    dv_resp.save(img_path)
                    local_path = img_path
                else:
                    v = str(getattr(dv_resp, "value", "") or "")
                    local_path = v.split(" ")[-1]
            except Exception:
                pass

            if local_path and GCS_BUCKET:
                kind = _detect_diagram_kind(local_path, visual_hint)
                up = upload_diagram_to_gcs(local_path, domain=domain, session_id=session_id, run_id=run_id, kind=kind)
                diagram_signed_url = up["signed_url"]
                diagram_kind = up["kind"]
                state["last_visual_gcs_path"] = up["gs_uri"]
                state["last_visual_kind"] = up["kind"]

        _cancel_if_needed(session_id)
        pai.config.set({"llm": llm, "conversational": True, "enforce_privacy": False})
        da_resp_text = ""
        if need_analyze:
            assert df_processed is not None, "Analyzer requested but no processed dataframe available."
            semantic_df = _as_pai_df(df_processed)
            da_resp = semantic_df.chat(analyzer_prompt)
            da_resp_text = str(da_resp)
            state["last_analyzer_text"] = da_resp_text or ""

        final_content = ""
        if need_compile:
            if isinstance(df_processed, pl.DataFrame):
                data_info_runtime = _polars_info_string(df_processed)
            else:
                data_info_runtime = "\n".join(data_info.values())

            final_response = completion(
                model=compiler_model or chosen_model_id,
                messages=[
                    {"role": "system", "content": compiler_instruction},
                    {"role": "user", "content": (
                        f"User Prompt:{prompt}. \n"
                        f"Datasets Domain name: {domain}. \n"
                        f"df.info of each dfs key(file name)-value pair:\n{data_info_runtime}. \n"
                        f"df.describe of each dfs key(file name)-value pair:\n{data_describe}. \n"
                        f"Data Manipulator Response:{dm_resp}. \n"
                        f"Data Visualizer Response:{getattr(dv_resp, 'value', '')}. \n"
                        f"Data Analyzer Response:{da_resp_text}."
                    )},
                ],
                seed=1, stream=False, verbosity="medium", drop_params=True, reasoning_effort="high",
                api_key=chosen_api_key
            )
            final_content = get_content(final_response)
            final_content = _force_strict_html(final_content)
        else:
            final_content = _force_strict_html("<div><p>Compiler skipped by router decision.</p></div>")

        _append_history(state, "user", prompt)
        _append_history(state, "assistant", final_content[:600])
        state["last_plan"] = agent_plan
        _save_conv_state(session_id, state)

        return jsonify({
            "sessionId": session_id,
            "compiledHtml": final_content,
            "diagramSignedUrl": diagram_signed_url or state.get("last_visual_signed_url",""),
            "diagramKind": diagram_kind or state.get("last_visual_kind",""),
            "thinking": thinking,
            "plan": agent_plan
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# =========================
# Cancel in-flight run
# =========================
@app.post("/cancel")
def cancel():
    try:
        body = request.get_json() or {}
        session_id = body.get("sessionId")
        if not session_id:
            return jsonify({"ok": False, "detail": "Missing sessionId"}), 400
        _CANCEL_FLAGS.add(session_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"ok": False, "detail": str(e)}), 500

# =========================
# Run
# =========================
if __name__ == "__main__":
    app.run(host="0.0.0.0", port=int(os.getenv("PORT", "8080")))
