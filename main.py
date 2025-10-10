# main.py — ConvoInsight BE (Flask, Cloud Run ready)
# Upgraded: a0.0.8 system configs + Firestore-centric dataset registry (GCS & Supabase PG)
# Additions are annotated with:  # === NEW a0.0.8 === or  # --- PG (Supabase) ---
# NEW (2025-10-09): Firestore-stored PG creds + litellm.model_list endpoint + encrypted user provider keys flow

import os, io, json, time, uuid, re, html
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple
from types import SimpleNamespace

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

# --- Polars + PandasAI (Polars-first)
import polars as pl
import pandasai as pai
from litellm import completion
# === NEW: use litellm.model_list programmatically
try:
    import litellm
    from litellm import model_list as litellm_model_list  # type: ignore
except Exception:
    litellm = None
    litellm_model_list = None

from pandasai import SmartDataframe, SmartDatalake  # kept import for compatibility, not used in new path
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse

# --- (pandas kept import for minimal surface compatibility, not used in pipeline)
import pandas as pd  # not used for pipeline; retained to avoid non-pipeline breakages elsewhere

# --- GCP clients ---
from google.cloud import storage
from google.cloud import firestore

# --- PG (Supabase) ---
from sqlalchemy import create_engine, inspect, text
from sqlalchemy.pool import NullPool

import requests
from cryptography.fernet import Fernet
from contextlib import contextmanager  # === NEW

# -------- optional: load .env --------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

# -------- Optional PDF deps (ReportLab) --------
_REPORTLAB_AVAILABLE = False
try:
    from reportlab.lib.pagesizes import A4
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.enums import TA_LEFT
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer
    _REPORTLAB_AVAILABLE = True
except Exception:
    _REPORTLAB_AVAILABLE = False

# -------- Config --------
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

FIRESTORE_COLLECTION_SESSIONS  = os.getenv("FIRESTORE_SESSIONS_COLLECTION", "convo_sessions")
FIRESTORE_COLLECTION_DATASETS  = os.getenv("FIRESTORE_DATASETS_COLLECTION", "datasets_meta")
FIRESTORE_COLLECTION_PROVIDERS = os.getenv("FIRESTORE_PROVIDERS_COLLECTION", "convo_providers")

# === NEW: Firestore collection for PG creds
FIRESTORE_COLLECTION_PGCREDS = os.getenv("FIRESTORE_PG_COLLECTION", "convo_pg_credentials")
FIRESTORE_PG_DOC_ID          = os.getenv("FIRESTORE_PG_DOC_ID", "default")

# --- PG (Supabase) env (transaction pooler) ---
# kept as fallback if Firestore creds not set
SUPABASE_PG_HOST = os.getenv("SUPABASE_PG_HOST", "aws-1-ap-southeast-1.pooler.supabase.com")
SUPABASE_PG_PORT = int(os.getenv("SUPABASE_PG_PORT", "6543"))
SUPABASE_PG_DB   = os.getenv("SUPABASE_PG_DB", "postgres")
SUPABASE_PG_USER = os.getenv("SUPABASE_PG_USER", "postgres.sbkynxkkvuirnzvakhfi")  # use your actual user
SUPABASE_PG_PASSWORD = os.getenv("SUPABASE_PG_PASSWORD")  # DO NOT hardcode
SUPABASE_PG_SSLMODE  = os.getenv("SUPABASE_PG_SSLMODE", "require")

# --- Init Flask ---
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# --- Init GCP clients ---
_storage_client = storage.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else storage.Client()
_firestore_client = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else firestore.Client()

# --- Cancel flags ---
_CANCEL_FLAGS = set()  # holds session_id


# =========================
# Utilities & Helpers
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

def _parse_gs_uri(gs_uri: str) -> Tuple[str, str]:
    if not gs_uri.startswith("gs://"):
        raise ValueError("Not a gs:// URI")
    rest = gs_uri[5:]
    if "/" not in rest:
        return rest, ""
    bucket, blob = rest.split("/", 1)
    return bucket, blob

# === NEW: provider env mapping + temp injection ===
_PROVIDER_ENV = {
    "openai": "OPENAI_API_KEY",
    "anthropic": "ANTHROPIC_API_KEY",
    "google": "GEMINI_API_KEY",
    "groq": "GROQ_API_KEY",
}

def get_provider_config(provider: str, api_key: str):
    if provider == "openai":
        return {"url": "https://api.openai.com/v1/models", "headers": {"Authorization": f"Bearer {api_key}"}}
    elif provider == "groq":
        return {"url": "https://api.groq.com/openai/v1/models", "headers": {"Authorization": f"Bearer {api_key}"}}
    elif provider == "anthropic":
        return {"url": "https://api.anthropic.com/v1/models", "headers": {"x-api-key": api_key, "anthropic-version": "2023-06-01"}}
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
            "updated_at": datetime.utcnow(),
            "created_at": firestore.SERVER_TIMESTAMP,
        }
        doc_ref.set(data, merge=True)
        return True
    except Exception as e:
        print("Firestore save error:", e)
        return False

# === NEW: helpers for per-user provider keys (decrypt)
def _get_user_provider_keymap(user_id: str) -> Dict[str, str]:
    """
    Returns {provider: plain_api_key} for given user_id by reading Firestore and decrypting.
    """
    if not user_id:
        return {}
    try:
        docs = (_firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS)
                .where("user_id", "==", user_id).stream())
        out = {}
        for d in docs:
            obj = d.to_dict() or {}
            tok = obj.get("token")
            if tok and fernet:
                try:
                    out[str(obj.get("provider"))] = fernet.decrypt(tok.encode()).decode()
                except Exception:
                    pass
            elif tok:
                out[str(obj.get("provider"))] = tok
        return out
    except Exception as e:
        print("read provider keys error:", e)
        return {}

def _effective_gemini_key(user_id: Optional[str]) -> Optional[str]:
    """
    Prefer user's google key from Firestore; fallback to env GEMINI_API_KEY.
    """
    if user_id:
        km = _get_user_provider_keymap(user_id)
        if km.get("google"):
            return km["google"]
    return GEMINI_API_KEY

@contextmanager
def _temp_provider_env(keymap: Dict[str, str]):
    """
    Temporarily set provider envs (e.g., GEMINI_API_KEY) for the duration of a call.
    """
    backup = {}
    try:
        for prov, key in keymap.items():
            env_name = _PROVIDER_ENV.get(prov)
            if env_name and key:
                backup[env_name] = os.environ.get(env_name)
                os.environ[env_name] = key
        yield
    finally:
        for env_name, old in backup.items():
            if old is None:
                os.environ.pop(env_name, None)
            else:
                os.environ[env_name] = old

# --- Firestore-backed conversation state -----------
def _fs_default_state():
    return {
        "history": [],
        "last_visual_gcs_path": "",
        "last_visual_signed_url": "",
        "last_visual_kind": "",
        "last_analyzer_text": "",
        "last_plan": None,
        "plan_explainer": "",                 # === NEW a0.0.8 ===
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
                data[k] = v if k not in ("created_at","updated_at") else firestore.SERVER_TIMESTAMP
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


# --- GCS helpers -----------------------------------
def _gcs_bucket():
    if not GCS_BUCKET:
        raise RuntimeError("GCS_BUCKET is not set")
    return _storage_client.bucket(GCS_BUCKET)

def _signed_url(blob, filename: str, content_type: str, ttl_seconds: int) -> str:
    return blob.generate_signed_url(
        version="v4",
        expiration=timedelta(seconds=ttl_seconds),
        method="GET",
        response_disposition=f'inline; filename="{filename}"',
        response_type=content_type,
    )


# ---- Local helpers / robust dev mode --------------
def _upload_dataset_file_local(file_storage, *, domain: str) -> dict:
    safe_domain = slug(domain)
    folder = ensure_dir(os.path.join(DATASETS_ROOT, safe_domain))
    filename = file_storage.filename
    dest = os.path.join(folder, filename)
    file_storage.save(dest)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return {"filename": filename, "gs_uri": "", "signed_url": "", "size_bytes": size, "local_path": dest}

def _save_bytes_local(domain: str, filename: str, data: bytes) -> dict:
    safe_domain = slug(domain)
    folder = ensure_dir(os.path.join(DATASETS_ROOT, safe_domain))
    dest = os.path.join(folder, filename)
    with open(dest, "wb") as f:
        f.write(data)
    size = os.path.getsize(dest)
    return {"filename": filename, "gs_uri": "", "signed_url": "", "size_bytes": size, "local_path": dest}

# =========================
# Polars DataFrame helpers
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
    if isinstance(df, pl.DataFrame):
        df = _normalize_columns_to_str(df)
    try:
        return pai.DataFrame(df)
    except TypeError:
        pdf = df.to_pandas() if isinstance(df, pl.DataFrame) else df
        if hasattr(pdf, "columns"):
            pdf.columns = [str(c) for c in pdf.columns]
        return pai.DataFrame(pdf)

def _read_csv_bytes_to_polars(data: bytes, sep_candidates: List[str] = (",", "|", ";", "\t")) -> pl.DataFrame:
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

def _read_local_csv_to_polars(path: str, sep_candidates: List[str] = (",", "|", ";", "\t")) -> pl.DataFrame:
    with open(path, "rb") as f:
        data = f.read()
    return _read_csv_bytes_to_polars(data, sep_candidates=sep_candidates)


# ---- Upload (GCS when possible, safe local fallback otherwise) ----------------
def upload_dataset_file(file_storage, *, domain: str) -> dict:
    """
    Upload a Werkz eug FileStorage to GCS or local.
    Also registers Firestore metadata with source='gcs'.  # === NEW a0.0.8 ===
    """
    if not GCS_BUCKET:
        meta = _upload_dataset_file_local(file_storage, domain=domain)
        try:
            _save_dataset_meta(domain, meta["filename"], "", meta["size_bytes"], source="gcs")
        except Exception:
            pass
        return meta

    try:
        safe_domain = slug(domain)
        filename = file_storage.filename
        blob_name = f"{GCS_DATASETS_PREFIX}/{safe_domain}/{filename}"
        bucket = _gcs_bucket()
        blob = bucket.blob(blob_name)
        blob.cache_control = "private, max-age=0"
        blob.content_type = "text/csv"
        file_storage.stream.seek(0)
        blob.upload_from_file(file_storage.stream, rewind=True, size=None, content_type="text/csv")
        size = blob.size or 0
        gs_uri = f"gs://{GCS_BUCKET}/{blob_name}"
        try:
            _save_dataset_meta(domain, filename, gs_uri, size, source="gcs")
        except Exception:
            pass
        return {
            "filename": filename,
            "gs_uri": gs_uri,
            "signed_url": _signed_url(blob, filename, "text/csv", GCS_SIGNED_URL_TTL_SECONDS),
            "size_bytes": size,
        }
    except Exception:
        meta = _upload_dataset_file_local(file_storage, domain=domain)
        try:
            _save_dataset_meta(domain, meta["filename"], "", meta["size_bytes"], source="gcs")
        except Exception:
            pass
        return meta


def list_gcs_csvs(domain: str) -> List[storage.Blob]:
    safe_domain = slug(domain)
    prefix = f"{GCS_DATASETS_PREFIX}/{safe_domain}/"
    return list(_gcs_bucket().list_blobs(prefix=prefix))

def read_gcs_csv_to_pl_df(gs_uri_or_blobname: str, *, sep_candidates: List[str] = (",","|",";","\t")) -> pl.DataFrame:
    if gs_uri_or_blobname.startswith("gs://"):
        bucket_name, blob_name = _parse_gs_uri(gs_uri_or_blobname)
        bucket = _storage_client.bucket(bucket_name)
    else:
        bucket = _gcs_bucket()
        blob_name = gs_uri_or_blobname
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    return _read_csv_bytes_to_polars(data, sep_candidates=sep_candidates)

def delete_gcs_object(blob_name_or_gs_uri: str):
    if blob_name_or_gs_uri.startswith("gs://"):
        bucket_name, blob_name = _parse_gs_uri(blob_name_or_gs_uri)
        bucket = _storage_client.bucket(bucket_name)
    else:
        bucket = _gcs_bucket()
        blob_name = blob_name_or_gs_uri
    bucket.blob(blob_name).delete()


# ---- Diagrams (charts|tables) helper ------------
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
# a0.0.8 System Configs  (synced with ml_bi_pipeline_a0_0_8.py)
# =========================
# (UNCHANGED CONFIG BLOCKS BELOW)
router_system_configuration = f"""Make sure all of the information below is applied.
1. You are the Orchestration Router: decide which agents/LLMs to run for a business data prompt.
2. Output must be STRICT one-line JSON with keys: need_manipulator, need_visualizer, need_analyzer, need_plan_explainer, need_compiler, compiler_model, plan_explainer_model, visual_hint, reason.
3. Precedence & overrides: Direct user prompt > Router USER config > Router DOMAIN config > Router SYSTEM defaults.
4. Flexibility: treat system defaults as fallbacks (e.g., default colors, currency, timezone). If the user or domain requires a different value, obey that without changing core routing logic.
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
15. Compiler always runs by default; default compiler_model="gemini/gemini-2.5-pro" unless the domain/user requires otherwise.
16. visual_hint ∈ {"{"}bar","line","table","auto"{"}"}; pick the closest fit and prefer "table" for plan/allocation outputs.
17. Keep the reason short (≤120 chars). No prose beyond the JSON.
18. In short: choose the most efficient set of agents/LLMs to answer the prompt well while respecting overrides.
19. By default, Manipulator and Analyzer should always be used in most scenario, because response compiler did not have access to the complete detailed data.
20. Plan explainer is optional; enable it for complex, multi-agent plans or first runs; default plan_explainer_model="gemini/gemini-2.5-pro"."""  # noqa: E501

orchestrator_system_configuration = f"""1. Honor precedence: direct user prompt > DOMAIN specific configuration > USER specific configuration > SYSTEM defaults.
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
14. All specialists operate in Python using PandasAI Semantic DataFrames (`pai.DataFrame`) backed by Polars DataFrames.
15. Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
16. Each value must be a single-line string. No extra keys, no prose, no markdown/code fences."""  # noqa: E501

data_manipulator_system_configuration = f"""1. Honor precedence: direct user prompt > DOMAIN specific configuration > USER specific configuration > SYSTEM defaults.
2. Enforce data hygiene before analysis.
3. Parse dates to Polars datetime; create explicit period columns (day/week/month).
4. Set consistent dtypes for numeric fields; strip/normalize categorical labels; standardize currency units if present.
5. Handle missing values: impute or drop only when necessary; keep legitimate zeros.
6. Mind each dataset’s name; avoid collisions in merges/aggregations.
7. Produce exactly the minimal, analysis-ready dataframe(s) needed for the user question, with stable, well-named columns.
8. Include the percentage version of appropriate raw value columns (share-of-total where relevant).
9. End by returning only:
   result = {{"type":"dataframe","value": <THE_FINAL_DATAFRAME>}}
10. Honor any user-level and domain-level instructions injected below."""  # noqa: E501

data_visualizer_system_configuration = f"""1. Honor precedence: direct user prompt > DOMAIN specific configuration > USER specific configuration > SYSTEM defaults.
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
    result = {{"type": "string", "value": file_path}}
13. DO NOT rely on pandas-specific styling; prefer Plotly Table when a table is needed."""  # noqa: E501

data_analyzer_system_configuration = f"""1. Honor precedence: direct user prompt > DOMAIN specific configuration > USER specific configuration > SYSTEM defaults.
2. Write like you’re speaking to a person; be concise and insight-driven.
3. Quantify where possible (deltas, % contributions, time windows); reference exact columns/filters used.
4. Return only:
   result = {{"type":"string","value":"<3–6 crisp bullets or 2 short paragraphs of insights>"}}"""  # noqa: E501

response_compiler_system_configuration = f"""1. Honor precedence: direct user prompt > DOMAIN specific configuration > USER specific configuration > SYSTEM defaults.
2. Output MUST be valid, self-contained HTML only. Do NOT use markdown (no **, *, -, numbered lines, code fences).
3. Brevity: ≤180 words; bullets preferred via <ul><li>…</li></ul>; no code blocks, no JSON, no screenshots.
4. Lead with the answer: 1–2 sentence “Bottom line” with main number, time window, and delta.
5. Quantified drivers: top 3 with magnitude, direction, and approx contribution (absolute and % where possible).
6. Next actions: 2–4 prioritized, concrete actions with expected impact/rationale.
7. Confidence & caveats: one short line on data quality/assumptions/gaps; include Confidence: High/Medium/Low.
8. Minimal tables: ≤1 table only if essential (≤5×3); otherwise avoid tables.
9. No repetition: do not restate agent text; synthesize it.
10. Do not try to show images; Do not mention the path of the generated file if there is one..
11. Always include units/currency and exact comparison window (e.g., “Aug 2025 vs Jul 2025”, “W34 vs W33”).
12. Show both absolute and % change where sensible (e.g., “+Rp120m (+8.4%)”).
13. Round smartly (money to nearest K unless < Rp10m; rates 1–2 decimals).
14. If any agent fails or data is incomplete, still produce the best insight; mark gaps in Caveats and adjust Confidence.
15. If the user asks “how much/which/why,” the first sentence must provide the number/entity/reason.
16. Exact compiler_instruction template the orchestrator should emit (single line; steps separated by ';'):
17. Read the user prompt, data_info, and all three agent responses;
18. Compute the direct answer including the main number and compare period;
19. Identify the top 3 quantified drivers with direction and contribution;
20. Draft 'Bottom line' in 1–2 sentences answering plainly;
21. List 2–4 prioritized Next actions with expected impact;
22. Add a one-line Caveats with Confidence and any gaps;
23. Keep ≤180 words; use only HTML tags: <div>, <p>, <strong>, <em>, <ul>, <ol>, <li>, <table>, <thead>, <tbody>, <tr>, <th>, <td>;
24. Include units, absolute and % deltas, and explicit dates;
25. Do not repeat agent text verbatim or include code/JSON.
26. Format hint (shape, not literal):
    <div>
      <p><strong>Bottom line —</strong> …</p>
      <p><strong>Drivers —</strong></p>
      <ul><li>…</li><li>…</li><li>…</li></ul>
      <p><strong>Next actions —</strong></p>
      <ol><li>…</li><li>…</li></ol>
      <p><strong>Caveats —</strong> … Confidence: …</p>
    </div>
27. The final response MUST be strictly HTML; never emit markdown markers like **, *, -, or code fences.
28. The compiler_instruction is used as the compiler LLM’s system content.
29. Compiler user content will be: f"User Prompt:{{user_prompt}}. \nData Info:{{data_info}}. \nData Describe:{{data_describe}}. \nData Manipulator Response:{{data_manipulator_response}}. \nData Visualizer Response:{{data_visualizer_response}}. \nData Analyzer Response:{{data_analyzer_response}}".
30. `data_info` is a string summary of dataframe types/shape.
31. `data_manipulator_response` is a PandasAI DataFrameResponse.
32. `data_visualizer_response` is a file path to an HTML/PNG inside {{"type":"string","value": ...}} with a plain Python string path.
33. `data_analyzer_response` is a PandasAI StringResponse.
34. Your goal in `compiler_instruction` is to force brevity, decisions, and insights.
35. Mention the data source of each statement.
36. STRICTLY respond in HTML only (no markdown)."""  # noqa: E501

# --- User/Domain configs (kept)
user_specific_configuration = """1. (no user-specific instructions provided yet)."""
domain_specific_configuration = """1. Use period labels like m0 (current month) and m1 (prior month); apply consistently.
2. Use IDR as currency, for example: Rp93,000.00 or Rp354,500.00.
3. Use blue themed chart and table colors.
4. target should be in mn (million).
5. %TUR is take up rate percentage.
6. % Taker, % Transaction, and % Revenue squad is the percentage of each product of all product Revenue all is in bn which is billion idr.
7. Revenue Squad is in mn wich is million idr.
8. rev/subs and rev/trx should be in thousands of idr.
9. MoM is month after month in percentage
10. Subs is taker."""

# =========================
# PG (Supabase) helpers — central registry via Firestore  (UPDATED)
# =========================

# === NEW: read/write PG creds on Firestore (encrypted password)
def _pg_doc_ref(doc_id: Optional[str] = None):
    return _firestore_client.collection(FIRESTORE_COLLECTION_PGCREDS).document(doc_id or FIRESTORE_PG_DOC_ID)

def _save_pg_credentials_to_fs(payload: dict):
    """
    payload keys: host, port, db, user, password, sslmode
    password will be encrypted with Fernet if available.
    """
    data = {
        "host": payload.get("host") or "",
        "port": int(payload.get("port") or 0),
        "db": payload.get("db") or "postgres",
        "user": payload.get("user") or "",
        "sslmode": payload.get("sslmode") or "require",
        "updated_at": datetime.utcnow(),
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    pwd = payload.get("password")
    if pwd is not None:
        data["password"] = fernet.encrypt(pwd.encode()).decode() if fernet else pwd
    _pg_doc_ref().set(data, merge=True)

def _load_pg_credentials_from_fs() -> Optional[dict]:
    try:
        doc = _pg_doc_ref().get()
        if not doc.exists:
            return None
        d = doc.to_dict() or {}
        password_enc = d.get("password")
        password_plain = None
        if password_enc:
            if fernet:
                try:
                    password_plain = fernet.decrypt(password_enc.encode()).decode()
                except Exception:
                    password_plain = None
            else:
                password_plain = password_enc
        return {
            "host": d.get("host") or "",
            "port": int(d.get("port") or 0),
            "db": d.get("db") or "postgres",
            "user": d.get("user") or "",
            "password": password_plain,
            "sslmode": d.get("sslmode") or "require",
        }
    except Exception as e:
        print("load PG creds error:", e)
        return None

def _pg_engine_uri_from_env() -> Optional[str]:
    """
    UPDATED: Prefer Firestore-stored credentials (encrypted), fallback to env vars.
    """
    # 1) Try Firestore
    fs = _load_pg_credentials_from_fs()
    if fs and fs.get("host") and fs.get("user") and fs.get("password"):
        host = fs["host"]
        port = int(fs.get("port") or 6543)
        db   = fs.get("db") or "postgres"
        user = fs["user"]
        pwd  = fs["password"]
        ssl  = fs.get("sslmode") or "require"
        return (
            f"postgresql+psycopg2://{user}:{pwd}@{host}:{port}/{db}"
            f"?sslmode={ssl}"
        )
    # 2) Fallback ENV (legacy)
    if not SUPABASE_PG_PASSWORD:
        return None
    return (
        f"postgresql+psycopg2://{SUPABASE_PG_USER}:{SUPABASE_PG_PASSWORD}"
        f"@{SUPABASE_PG_HOST}:{SUPABASE_PG_PORT}/{SUPABASE_PG_DB}"
        f"?sslmode={SUPABASE_PG_SSLMODE}"
    )

def _pg_read_to_polars(query: str) -> pl.DataFrame:
    uri = _pg_engine_uri_from_env()
    if not uri:
        raise RuntimeError("Supabase PG credentials not configured. Set via /pg/credentials or env SUPABASE_*.")
    engine = create_engine(uri, poolclass=NullPool, connect_args={"sslmode": ( _load_pg_credentials_from_fs() or {} ).get("sslmode", SUPABASE_PG_SSLMODE)})
    with engine.connect() as conn:
        try:
            return pl.read_database(query, conn)
        except Exception:
            pdf = pd.read_sql(query, conn)
            return pl.from_pandas(pdf)

def _pg_table_df(schema: str, table: str, limit: Optional[int] = None) -> pl.DataFrame:
    q = f'SELECT * FROM "{schema}"."{table}"'
    if limit is not None:
        q += f" LIMIT {int(limit)}"
    return _pg_read_to_polars(q)

def _pg_query_df(sql: str) -> pl.DataFrame:
    return _pg_read_to_polars(sql)

# =========================
# Shared Data Loading (Firestore-centric, GCS + PG)
# =========================
def _load_domain_dataframes(domain: str, dataset_filters: Optional[set]) -> Tuple[Dict[str, pl.DataFrame], Dict[str, str], Dict[str, str]]:
    dfs: Dict[str, pl.DataFrame] = {}
    data_info: Dict[str, str] = {}
    data_describe: Dict[str, str] = {}

    # --- 1) Firestore metadata (single source of truth)
    try:
        metas = _list_dataset_meta(domain=domain, limit=500)
        for m in metas:
            name = m.get("filename") or m.get("name")
            if not name:
                continue
            if dataset_filters and name not in dataset_filters:
                continue
            source = (m.get("source") or "gcs").lower()

            if source == "gcs":
                # Prefer GCS; fall back to local if not present
                gs_uri = m.get("gs_uri", "")
                try:
                    if gs_uri:
                        df = read_gcs_csv_to_pl_df(gs_uri)
                    else:
                        # maybe stored by blob name
                        blob_name = f"{GCS_DATASETS_PREFIX}/{slug(domain)}/{name}"
                        df = read_gcs_csv_to_pl_df(blob_name)
                except Exception:
                    # local fallback
                    local_path = os.path.join(DATASETS_ROOT, slug(domain), name)
                    if not os.path.exists(local_path):
                        continue
                    df = _read_local_csv_to_polars(local_path)
            elif source in ("pg", "postgres", "supabase"):
                # Either table or stored SQL
                pg_schema = m.get("pg_schema") or "public"
                pg_table  = m.get("pg_table") or ""
                pg_query  = m.get("pg_query") or ""
                pg_limit  = int(m.get("pg_limit") or 0) or None

                if pg_query:
                    df = _pg_query_df(pg_query)
                elif pg_table:
                    df = _pg_table_df(pg_schema, pg_table, limit=pg_limit)
                else:
                    # malformed meta
                    continue
            else:
                continue

            df = _normalize_columns_to_str(df)
            dfs[name] = df
            info_str = _polars_info_string(df)
            data_info[name] = info_str
            try:
                desc_df = df.describe()
                data_describe[name] = desc_df.to_pandas().to_json()
            except Exception:
                data_describe[name] = ""
    except Exception as e:
        print("load meta error:", e)

    # --- 2) Local directory (compatibility / dev) — keep but don't duplicate names
    domain_dir = ensure_dir(os.path.join(DATASETS_ROOT, slug(domain)))
    for name in sorted(os.listdir(domain_dir)):
        if not name.lower().endswith(".csv"):
            continue
        if dataset_filters and name not in dataset_filters:
            continue
        if name in dfs:
            continue
        path = os.path.join(domain_dir, name)
        try:
            df = _read_local_csv_to_polars(path)
            dfs[name] = df
            info_str = _polars_info_string(df)
            data_info[name] = info_str
            try:
                desc_df = df.describe()
                data_describe[name] = desc_df.to_pandas().to_json()
            except Exception:
                data_describe[name] = ""
        except Exception:
            pass

    return dfs, data_info, data_describe


# =========================
# Router (a0.0.8)
# =========================
def _run_router(user_prompt: str, data_info, data_describe, state: dict) -> dict:
    router_start = time.time()
    recent_context = json.dumps(state.get("history", [])[-6:], ensure_ascii=False)

    router_response = completion(
        model="gemini/gemini-2.5-pro",
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
    )
    router_content = get_content(router_response)
    try:
        plan = _safe_json_loads(router_content)
    except Exception:
        p = user_prompt.lower()
        need_visual = bool(re.search(r"\b(chart|plot|graph|visual|bar|line|table)\b", p))
        optimize_terms = bool(re.search(r"\b(allocate|allocation|optimal|optimi[sz]e|plan|planning|min(?:imum)? number|minimum number|close (?:the )?gap|gap closure|takers?)\b", p))
        need_analyze = bool(re.search(r"\b(why|driver|explain|root cause|trend|surprise|reason)\b", p)) or optimize_terms
        follow_up = bool(re.search(r"\b(what about|and|how about|ok but|also)\b", p)) or len(p.split()) <= 8
        need_manip = not follow_up
        visual_hint = "bar" if "bar" in p else ("line" if "line" in p else ("table" if ("table" in p or optimize_terms) else "auto"))
        plan = {
            "need_manipulator": bool(need_manip),
            "need_visualizer": bool(need_visual or optimize_terms),
            "need_analyzer": bool(need_analyze or not need_visual),
            "need_plan_explainer": True,
            "need_compiler": True,
            "compiler_model": "gemini/gemini-2.5-pro",
            "plan_explainer_model": "gemini/gemini-2.5-pro",
            "visual_hint": visual_hint,
            "reason": "heuristic fallback",
        }

    p_low = user_prompt.lower()
    if re.search(r"\b(min(?:imum)? number|minimum number of additional takers|additional takers|close (?:the )?gap|gap closure|optimal allocation|allocate|allocation|optimi[sz]e)\b", p_low):
        plan["need_analyzer"] = True
        plan["need_visualizer"] = True if not plan.get("need_visualizer") else plan["need_visualizer"]
        if plan.get("visual_hint", "auto") == "auto":
            plan["visual_hint"] = "table"
        plan["reason"] = (plan.get("reason") or "") + " + analyzer-for-gap/allocation tasks"

    router_end = time.time()
    plan["_elapsed"] = float(router_end - router_start)
    return plan


# =========================
# Orchestrate (a0.0.8)
# =========================
def _run_orchestrator(user_prompt: str, domain: str, data_info, data_describe, visual_hint: str, context_hint: dict):
    resp = completion(
        model="gemini/gemini-2.5-pro",
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

@app.route("/charts/<path:relpath>")
def serve_chart(relpath):
    full = os.path.join(CHARTS_ROOT, relpath)
    base = os.path.dirname(full)
    filename = os.path.basename(full)
    return send_from_directory(base, filename)


# =========================
# Provider key management (kept; used by configuser FE)
# =========================
@app.route("/validate-key", methods=["POST"])
def validate_key():
    try:
        data = request.get_json()
        provider = data.get("provider")
        api_key = data.get("apiKey")
        user_id = data.get("userId")
        if not provider or not api_key or not user_id:
            return jsonify({"valid": False, "error": "Missing provider, apiKey, or userId"}), 400

        cfg = get_provider_config(provider, api_key)
        res = requests.get(cfg["url"], headers=cfg["headers"], timeout=6)
        if res.status_code == 200:
            j = res.json()
            models = []
            if "data" in j:
                models = [m.get("id") for m in j["data"] if "id" in m]
            elif "models" in j:
                models = [m.get("name") or m.get("id") for m in j["models"]]
            encrypted_key = fernet.encrypt(api_key.encode()).decode() if fernet else api_key
            save_provider_key(user_id, provider, encrypted_key, models)
            return jsonify({"valid": True, "provider": provider, "models": models, "token": encrypted_key})
        else:
            return jsonify({"valid": False, "provider": provider, "status": res.status_code, "detail": res.text}), 400
    except Exception as e:
        return jsonify({"valid": False, "error": str(e)}), 500

@app.route("/get-provider-keys", methods=["GET"])
def get_provider_keys():
    try:
        user_id = request.args.get("userId")
        if not user_id:
            return jsonify({"error": "Missing userId"}), 400

        docs = (_firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS)
                .where("user_id", "==", user_id).stream())

        items = []
        for doc in docs:
            d = doc.to_dict()
            items.append({
                "id": doc.id,
                "provider": d.get("provider"),
                "models": d.get("models", []),
                "is_active": d.get("is_active", True),
                "updated_at": d.get("updated_at").isoformat() if d.get("updated_at") else None,
            })
        return jsonify({"items": items, "count": len(items)})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.route("/update-provider-key", methods=["PUT"])
def update_provider_key():
    try:
        data = request.get_json()
        user_id = data.get("userId")
        provider = data.get("provider")
        api_key = data.get("apiKey")
        if not user_id or not provider or not api_key:
            return jsonify({"updated": False, "error": "Missing fields"}), 400

        cfg = get_provider_config(provider, api_key)
        res = requests.get(cfg["url"], headers=cfg["headers"], timeout=6)
        if res.status_code != 200:
            return jsonify({"updated": False, "error": "Invalid API key"}), 400

        j = res.json()
        models = []
        if "data" in j:
            models = [m.get("id") for m in j["data"] if "id" in m]
        elif "models" in j:
            models = [m.get("name") or m.get("id") for m in j["models"]]
        encrypted_key = fernet.encrypt(api_key.encode()).decode() if fernet else api_key

        doc_ref = _firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS).document(f"{user_id}_{provider}")
        doc_ref.set({"user_id": user_id, "provider": provider, "token": encrypted_key, "models": models,
                     "updated_at": datetime.utcnow().isoformat()}, merge=True)

        return jsonify({"updated": True, "models": models})
    except Exception as e:
        return jsonify({"updated": False, "error": str(e)}), 500

@app.route("/delete-provider-key", methods=["DELETE"])
def delete_provider_key():
    try:
        data = request.get_json()
        user_id = data.get("userId")
        provider = data.get("provider")
        if not user_id or not provider:
            return jsonify({"error": "Missing userId or provider"}), 400

        doc_id = f"{user_id}_{provider}"
        doc_ref = _firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS).document(doc_id)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"error": "Key not found"}), 404
        doc_ref.delete()
        return jsonify({"deleted": True})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# === NEW: LLM provider & model listing via litellm.model_list =================
@app.get("/litellm/model-list")
def litellm_model_listing():
    """
    Lists all models accessible for the user across supported providers using litellm.model_list().
    - Query param: userId (required to read provider keys from Firestore)
    - It temporarily sets provider env vars so litellm can resolve providers.
    """
    try:
        if litellm_model_list is None:
            return jsonify({"error": "litellm not available in runtime"}), 500

        user_id = request.args.get("userId")
        if not user_id:
            return jsonify({"error": "Missing userId"}), 400

        keymap = _get_user_provider_keymap(user_id)  # {'openai': 'sk-...', 'google': '...'}
        if not keymap:
            return jsonify({"models": [], "providers": [], "note": "No provider keys stored for this user"}), 200

        # Map provider -> ENV var expected by litellm
        provider_env = {
            "openai": "OPENAI_API_KEY",
            "anthropic": "ANTHROPIC_API_KEY",
            "google": "GEMINI_API_KEY",      # litellm supports 'gemini' provider; uses GEMINI_API_KEY
            "groq": "GROQ_API_KEY",
        }

        # backup and set envs temporarily
        backup_env = {}
        for prov, key in keymap.items():
            env_name = provider_env.get(prov)
            if env_name and key:
                backup_env[env_name] = os.environ.get(env_name)
                os.environ[env_name] = key

        try:
            ml = litellm_model_list() or []
        finally:
            # restore env
            for k, v in backup_env.items():
                if v is None:
                    os.environ.pop(k, None)
                else:
                    os.environ[k] = v

        # Normalize result
        models = []
        providers_seen = set()
        for m in ml:
            mid = m.get("id") or m.get("name") or ""
            prov = m.get("provider") or m.get("owned_by") or ""
            if prov:
                providers_seen.add(prov)
            if mid:
                models.append({"id": mid, "provider": prov})

        return jsonify({"models": models, "providers": sorted(list(providers_seen))})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# =========================
# Datasets CRUD + Domain listing (Firestore as registry)
# =========================
@app.get("/domains")
def list_domains():
    """
    Aggregate domains that have at least one dataset (from FS meta or local CSVs).
    """
    result = {}
    try:
        # From local folder (dev)
        for d in sorted(os.listdir(DATASETS_ROOT)):
            p = os.path.join(DATASETS_ROOT, d)
            if os.path.isdir(p):
                csvs = [f for f in sorted(os.listdir(p)) if f.lower().endswith(".csv")]
                if csvs:
                    result[d] = csvs
        # From Firestore meta
        try:
            metas = _list_dataset_meta(limit=1000)
            for m in metas:
                d = m.get("domain","")
                f = m.get("filename") or m.get("name")
                if not d or not f: 
                    continue
                result.setdefault(d, [])
                if f not in result[d]:
                    result[d].append(f)
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

def _ds_ref(domain: str, filename_or_name: str):
    key = f"{slug(domain)}::{filename_or_name}"
    return _firestore_client.collection(FIRESTORE_COLLECTION_DATASETS).document(key)

def _save_dataset_meta(domain: str, filename: str, gs_uri: str, size: int, *, source: str="gcs", **extra):
    """
    Upsert dataset metadata (single source of truth).
    `source`: "gcs" | "pg"
    For PG, pass extra like pg_schema, pg_table, pg_query, pg_limit, name, etc.
    """
    meta = {
        "domain": slug(domain),
        "source": source,
        "filename": filename,     # for gcs; for pg we still keep a 'name' below
        "name": extra.get("name", filename),  # FE can use this as display key
        "gs_uri": gs_uri,
        "size_bytes": size,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    # include PG fields if any
    for k in ("pg_schema","pg_table","pg_query","pg_limit"):
        if k in extra and extra[k] is not None:
            meta[k] = extra[k]
    _ds_ref(domain, meta["name"]).set(meta, merge=True)

def _delete_dataset_meta(domain: str, filename_or_name: str):
    _ds_ref(domain, filename_or_name).delete()

def _list_dataset_meta(domain: Optional[str]=None, limit: int=200) -> List[dict]:
    col = _firestore_client.collection(FIRESTORE_COLLECTION_DATASETS)
    q = col.order_by("updated_at", direction=firestore.Query.DESCENDING)
    if domain:
        q = q.where("domain", "==", slug(domain))
    docs = q.limit(limit).stream()
    return [d.to_dict() for d in docs if d.exists]

# --- Register a PG dataset (no upload) ---------------------------------
@app.post("/datasets/register/pg")
def register_pg_dataset():
    """
    Body JSON:
    {
      "domain": "campaign",
      "name": "pg:public.sales",   // unique key shown on FE (used as filename param)
      "pg_schema": "public",
      "pg_table": "sales",         // OR supply "pg_query" (one of table OR query is required)
      "pg_query": null,
      "pg_limit": 0                // optional preview cap
    }
    """
    try:
        body = request.get_json(force=True)
        domain = body.get("domain")
        name   = body.get("name")
        pg_schema = body.get("pg_schema") or "public"
        pg_table  = body.get("pg_table") or ""
        pg_query  = body.get("pg_query") or ""
        pg_limit  = int(body.get("pg_limit") or 0)

        if not domain or not name:
            return jsonify({"detail":"Missing 'domain' or 'name'"}), 400
        if not (pg_table or pg_query):
            return jsonify({"detail":"Provide either 'pg_table' or 'pg_query'"}), 400

        # validate connection quickly if password provided
        try:
            if pg_query:
                _ = _pg_query_df(pg_query).head(1)
            else:
                _ = _pg_table_df(pg_schema, pg_table, limit=1)
        except Exception as e:
            return jsonify({"detail": f"PG validation failed: {e}"}), 400

        _save_dataset_meta(domain, filename=name, gs_uri="", size=0, source="pg",
                           name=name, pg_schema=pg_schema, pg_table=pg_table, pg_query=pg_query, pg_limit=pg_limit)
        return jsonify({"registered": True, "domain": slug(domain), "name": name, "source":"pg"}), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.put("/datasets/register/pg/<domain>/<path:name>")
def update_pg_dataset(domain, name):
    """
    Update PG metadata (edit on Datasets page).
    Body JSON can include: pg_schema, pg_table, pg_query, pg_limit, new_name
    """
    try:
        body = request.get_json(force=True) if request.data else {}
        fields = {}
        for k in ("pg_schema","pg_table","pg_query","pg_limit"):
            if k in body:
                fields[k] = body[k]
        new_name = body.get("new_name")

        doc_ref = _ds_ref(domain, name)
        doc = doc_ref.get()
        if not doc.exists:
            return jsonify({"detail":"Metadata not found"}), 404

        if new_name and new_name != name:
            # create new doc with same payload + new key; delete old
            data = doc.to_dict()
            data["name"] = new_name
            data["filename"] = new_name
            data.update(fields)
            _ds_ref(domain, new_name).set(data, merge=True)
            doc_ref.delete()
            return jsonify({"updated": True, "renamed_to": new_name})

        if fields:
            doc_ref.set(fields, merge=True)
        return jsonify({"updated": True})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.delete("/datasets/register/pg/<domain>/<path:name>")
def delete_pg_dataset(domain, name):
    try:
        _delete_dataset_meta(domain, name)
        return jsonify({"deleted": True, "domain": slug(domain), "name": name})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- Upload CSVs (GCS/local)  — unchanged, now also tags source='gcs'
@app.post("/datasets/upload")
def datasets_upload():
    try:
        domain = request.form.get("domain")
        file = request.files.get("file")
        if not domain or not file:
            return jsonify({"detail":"Missing 'domain' or 'file'"}), 400
        uploaded = upload_dataset_file(file, domain=domain)
        return jsonify(uploaded), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- List datasets for a domain (merged)
@app.get("/datasets")
def datasets_list():
    try:
        domain = request.args.get("domain")
        add_signed = request.args.get("signed","false").lower() in ("1","true","yes")
        items = []
        try:
            items = _list_dataset_meta(domain=domain, limit=1000)
            if add_signed:
                for it in items:
                    gs_uri = it.get("gs_uri","")
                    if not gs_uri: 
                        continue
                    try:
                        bucket_name, blob_name = _parse_gs_uri(gs_uri)
                        blob = _storage_client.bucket(bucket_name).blob(blob_name)
                        it["signed_url"] = _signed_url(blob, it.get("filename") or it.get("name"), "text/csv", GCS_SIGNED_URL_TTL_SECONDS)
                    except Exception:
                        it["signed_url"] = ""
        except Exception:
            items = []

        # local fallback merge (dev only)
        if domain:
            domain_dir = os.path.join(DATASETS_ROOT, slug(domain))
            if os.path.isdir(domain_dir):
                known = {(i.get("domain"), (i.get("filename") or i.get("name"))) for i in items}
                for name in sorted(os.listdir(domain_dir)):
                    if name.lower().endswith(".csv") and (slug(domain), name) not in known:
                        path = os.path.join(domain_dir, name)
                        size = os.path.getsize(path)
                        items.append({"domain": slug(domain), "source": "gcs", "filename": name, "name": name, "gs_uri":"", "size_bytes": size, "signed_url": ""})
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- View sample rows (works for GCS or PG) ---
@app.get("/datasets/<domain>/<path:name>")
def datasets_read(domain, name):
    try:
        n = int(request.args.get("n","50"))
        as_fmt = request.args.get("as","json")

        # Look up meta first
        meta = None
        try:
            doc = _ds_ref(domain, name).get()
            if doc.exists:
                meta = doc.to_dict()
        except Exception:
            meta = None

        if meta and (meta.get("source") or "").lower() in ("pg","postgres","supabase"):
            # PG dataset
            pg_schema = meta.get("pg_schema") or "public"
            pg_table  = meta.get("pg_table") or ""
            pg_query  = meta.get("pg_query") or ""
            limit = n if n > 0 else None
            if pg_query:
                df = _pg_query_df(pg_query)
                if limit:
                    df = df.head(limit)
            else:
                df = _pg_table_df(pg_schema, pg_table, limit=limit)
        else:
            # GCS/local (compat)
            if GCS_BUCKET:
                blob_name = f"{GCS_DATASETS_PREFIX}/{slug(domain)}/{name}"
                df = read_gcs_csv_to_pl_df(blob_name)
            else:
                local_path = os.path.join(DATASETS_ROOT, slug(domain), name)
                if not os.path.exists(local_path):
                    return jsonify({"detail": "File not found"}), 404
                df = _read_local_csv_to_polars(local_path)
            if n > 0:
                df = df.head(n)

        if as_fmt == "csv":
            out = io.StringIO()
            df.write_csv(out)
            return out.getvalue(), 200, {"Content-Type":"text/csv; charset=utf-8"}
        return jsonify({"records": df.to_dicts()})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- Delete (GCS/Local or just PG meta)
@app.delete("/datasets/<domain>/<path:name>")
def datasets_delete(domain, name):
    try:
        # read meta to know source
        doc = _ds_ref(domain, name).get()
        if doc.exists:
            meta = doc.to_dict()
            src = (meta.get("source") or "gcs").lower()
            if src == "gcs":
                if GCS_BUCKET and meta.get("gs_uri"):
                    delete_gcs_object(meta["gs_uri"])
                # local fallback
                local_path = os.path.join(DATASETS_ROOT, slug(domain), name)
                try:
                    if os.path.exists(local_path):
                        os.remove(local_path)
                except Exception:
                    pass
            # remove meta for both
            _delete_dataset_meta(domain, name)
            return jsonify({"deleted": True, "domain": slug(domain), "filename": name})
        # if no meta, fall back to local delete (legacy)
        if GCS_BUCKET:
            blob_name = f"{GCS_DATASETS_PREFIX}/{slug(domain)}/{name}"
            try:
                delete_gcs_object(blob_name)
            except Exception:
                pass
        local_path = os.path.join(DATASETS_ROOT, slug(domain), name)
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass
        return jsonify({"deleted": True, "domain": slug(domain), "filename": name})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# FE compatibility
@app.post("/upload_datasets/<domain>")
def compat_upload_datasets(domain: str):
    try:
        files: List = []
        single = request.files.get("file")
        if single: files.append(single)
        files.extend(request.files.getlist("files"))
        files.extend(request.files.getlist("files[]"))

        uploads = []
        for f in files:
            uploads.append(upload_dataset_file(f, domain=domain))

        if not uploads and request.data:
            fname = request.args.get("filename") or request.headers.get("X-Filename") or f"upload_{int(time.time())}.csv"
            uploads.append(_save_bytes_local(domain, fname, request.data))
            try:
                _save_dataset_meta(domain, fname, "", os.path.getsize(os.path.join(DATASETS_ROOT, slug(domain), fname)), source="gcs")
            except Exception:
                pass

        if not uploads:
            return jsonify({"detail": "No file provided"}), 400

        return jsonify({"items": uploads}), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.get("/domains/<domain>/datasets>")
def compat_list_domain_datasets_trailing(domain: str):
    return compat_list_domain_datasets(domain)

@app.get("/domains/<domain>/datasets")
def compat_list_domain_datasets(domain: str):
    try:
        items: List[dict] = []
        try:
            fs_items = _list_dataset_meta(domain=domain, limit=1000)
            # inject signed URLs for GCS items if requested elsewhere
            for it in fs_items:
                gs_uri = it.get("gs_uri","")
                if gs_uri:
                    try:
                        bucket_name, blob_name = _parse_gs_uri(gs_uri)
                        blob = _storage_client.bucket(bucket_name).blob(blob_name)
                        it["signed_url"] = _signed_url(blob, it.get("filename") or it.get("name"), "text/csv", GCS_SIGNED_URL_TTL_SECONDS)
                    except Exception:
                        it.setdefault("signed_url","")
            items.extend(fs_items)
        except Exception:
            pass

        try:
            domain_dir = os.path.join(DATASETS_ROOT, slug(domain))
            if os.path.isdir(domain_dir):
                known_names = { (i.get("filename") or i.get("name")) for i in items }
                for name in sorted(os.listdir(domain_dir)):
                    if name.lower().endswith(".csv") and name not in known_names:
                        path = os.path.join(domain_dir, name)
                        size = os.path.getsize(path)
                        items.append({"domain": slug(domain), "source":"gcs", "filename": name, "name": name, "gs_uri": "", "size_bytes": size, "signed_url": ""})
        except Exception:
            pass

        return jsonify({"items": items, "datasets": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


# =========================
# Sessions / PDF Export / Cancel
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
                "plan_explainer": data.get("plan_explainer",""),
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
            "plan_explainer": st.get("plan_explainer",""),
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.get("/sessions/<session_id>/export/pdf")
def sessions_export_pdf(session_id: str):
    if not _REPORTLAB_AVAILABLE:
        return jsonify({"detail": "PDF export requires 'reportlab'. Install first: uv pip install reportlab"}), 501
    try:
        state = _get_conv_state(session_id)
        history: List[dict] = state.get("history", [])
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        title = styles["Heading1"]
        meta  = styles["Normal"]
        body  = ParagraphStyle("Body", parent=styles["BodyText"], fontSize=10, leading=14, alignment=TA_LEFT)
        role_style = styles["Heading3"]

        story: List = []
        story.append(Paragraph(f"Chat History — Session {html.escape(session_id)}", title))
        story.append(Spacer(1, 6))
        story.append(Paragraph(f"Generated at: {datetime.utcnow().isoformat()}Z", meta))
        story.append(Spacer(1, 12))
        if not history:
            story.append(Paragraph("No messages yet.", body))
        else:
            for i, item in enumerate(history, 1):
                role = str(item.get("role", "unknown")).capitalize()
                ts = item.get("ts")
                ts_str = ""
                if isinstance(ts, (int, float)):
                    ts_str = datetime.utcfromtimestamp(ts).isoformat() + "Z"
                content = item.get("content", "")
                if not isinstance(content, str):
                    content = json.dumps(content, ensure_ascii=False, indent=2)
                safe = html.escape(content).replace("\n", "<br/>")

                story.append(Paragraph(f"{i}. <b>{role}</b> <font size=9 color='#666666'>({ts_str})</font>", role_style))
                story.append(Paragraph(safe, body))
                story.append(Spacer(1, 8))

        doc.build(story)
        buf.seek(0)
        filename = f"chat_session_{session_id}.pdf"
        return send_file(buf, mimetype="application/pdf", as_attachment=True, download_name=filename)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.post("/query/cancel")
def query_cancel():
    try:
        body = request.get_json(force=True) if request.data else {}
        session_id = body.get("session_id")
        if not session_id:
            return jsonify({"detail":"Missing 'session_id'"}), 400
        _CANCEL_FLAGS.add(session_id)
        return jsonify({"ok": True})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


# =========================
# NEW: Suggestion Endpoint (per-user key default)
# =========================
@app.post("/suggest")
def suggest():
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        domain_in  = body.get("domain")
        dataset_field = body.get("dataset")
        user_id = body.get("userId") or body.get("user_id")

        if not domain_in:
            return jsonify({"detail":"Missing 'domain'"}), 400

        # pick effective key (prefer user)
        effective_key = _effective_gemini_key(user_id)
        if not effective_key:
            return jsonify({"detail": "No API key available. Please save a Google (Gemini) key in Config User."}), 401

        domain = slug(domain_in)
        if isinstance(dataset_field, list):
            datasets = [s.strip() for s in dataset_field if isinstance(dataset_field, list) and isinstance(s, str) and s.strip()]
            dataset_filters = set(datasets) if datasets else None
        elif isinstance(dataset_field, str) and dataset_field.strip():
            dataset_filters = {dataset_field.strip()}
        else:
            dataset_filters = None

        dfs, data_info, data_describe = _load_domain_dataframes(domain, dataset_filters)

        # inject envs for this request (use all user providers if available)
        keymap = _get_user_provider_keymap(user_id) if user_id else {"google": effective_key}
        if not keymap.get("google"):
            keymap["google"] = effective_key

        with _temp_provider_env(keymap):
            r = completion(
                model="gemini/gemini-2.5-pro",
                messages=[
                    {"role":"system","content":"""
                    Make sure all of the information below is applied.
                    1. Based on the provided dataset(s), Suggest exactly 4 realistic user prompt in a format of a STRICT one-line JSON with keys: suggestion1, suggestion2, suggestion3, suggestion4.
                    2. Each suggestion should be less than 100 characters. No prose beyond the JSON.
                    3. Each value must be a single-line string. No extra keys, no prose, no markdown/code fences.
                    """},
                    {"role":"user","content":
                        f"""Make sure all of the information below is applied.
                        Datasets Domain name: {domain}.
                        df.info of each dfs key(file name)-value pair:
                        {data_info}.
                        df.describe of each dfs key(file name)-value pair:
                        {data_describe}."""
                    }
                ],
                seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high",
            )
        content = get_content(r)
        try:
            m = re.search(r'\{.*\}', content, re.DOTALL)
            js = json.loads(m.group(0)) if m else {}
        except Exception:
            js = {}
        suggestions = [js.get("suggestion1",""), js.get("suggestion2",""), js.get("suggestion3",""), js.get("suggestion4","")]
        elapsed = time.time() - t0
        return jsonify({"suggestions": suggestions, "elapsed": elapsed, "data_info": data_info, "data_describe": data_describe})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


# =========================
# Query/Inferencing Endpoint (per-user key default)
# =========================
@app.post("/query")
def query():
    """
    Body (JSON):
      - domain (str, required)
      - prompt (str, required)
      - session_id (str, optional)
      - dataset (str | [str], optional)
      - userId | user_id (str, recommended)  -> choose per-user API key
    Returns:
      - session_id, response (HTML), diagram_* fields, timing & flags
    """
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        domain_in  = body.get("domain")
        prompt     = body.get("prompt")
        session_id = body.get("session_id") or str(uuid.uuid4())
        user_id    = body.get("userId") or body.get("user_id")

        # dataset selection (single or multi)
        dataset_field = body.get("dataset")
        if isinstance(dataset_field, list):
            datasets = [s.strip() for s in dataset_field if isinstance(s, str) and s.strip()]
            dataset_filters = set(datasets) if datasets else None
        elif isinstance(dataset_field, str) and dataset_field.strip():
            datasets = [dataset_field.strip()]
            dataset_filters = {datasets[0]}
        else:
            datasets = []
            dataset_filters = None

        if not domain_in or not prompt:
            return jsonify({"detail": "Missing 'domain' or 'prompt'"}), 400

        effective_key = _effective_gemini_key(user_id)
        if not effective_key:
            return jsonify({"detail": "No API key available. Please save a Google (Gemini) key in Config User."}), 401

        domain = slug(domain_in)

        # state & history
        state = _get_conv_state(session_id)
        _append_history(state, "user", prompt)
        _save_conv_state(session_id, state)

        _cancel_if_needed(session_id)

        # load data (Polars) — now via Firestore registry (GCS + PG)
        dfs, data_info, data_describe = _load_domain_dataframes(domain, dataset_filters)
        if not dfs:
            if dataset_filters:
                available = []
                domain_dir = os.path.join(DATASETS_ROOT, domain)
                if os.path.isdir(domain_dir):
                    available.extend(sorted([f for f in os.listdir(domain_dir) if f.lower().endswith(".csv")]))
                try:
                    metas = _list_dataset_meta(domain=domain, limit=500)
                    available.extend(sorted({(m.get("name") or m.get("filename")) for m in metas}))
                except Exception:
                    pass
                return jsonify({
                    "code": "DATASET_NOT_FOUND",
                    "detail": f"Requested datasets {sorted(list(dataset_filters))} not found in domain '{domain}'.",
                    "domain": domain,
                    "available": sorted(list(set(available))),
                }), 404
            return jsonify({"code":"NEED_UPLOAD", "detail": f"No datasets found in domain '{domain}'", "domain": domain}), 409

        # Inject per-user provider envs for the duration of this request
        keymap = _get_user_provider_keymap(user_id) if user_id else {"google": effective_key}
        if not keymap.get("google"):
            keymap["google"] = effective_key

        with _temp_provider_env(keymap):
            # Router
            agent_plan = _run_router(prompt, data_info, data_describe, state)
            need_manip = bool(agent_plan.get("need_manipulator", True))
            need_visual = bool(agent_plan.get("need_visualizer", True))
            need_analyze = bool(agent_plan.get("need_analyzer", True))
            need_plan_explainer = bool(agent_plan.get("need_plan_explainer", False))
            compiler_model = agent_plan.get("compiler_model") or "gemini/gemini-2.5-pro"
            plan_explainer_model = agent_plan.get("plan_explainer_model") or "gemini/gemini-2.5-pro"
            visual_hint = agent_plan.get("visual_hint", "auto")

            context_hint = {
                "router_plan": agent_plan,
                "last_visual_path": "",
                "has_prev_df_processed": False,   # process fresh each call (stateless df)
                "last_analyzer_excerpt": (state.get("last_analyzer_text") or "")[:400],
                "dataset_filter": (sorted(datasets) if datasets else "ALL"),
            }

            # Orchestrator
            _cancel_if_needed(session_id)
            spec = _run_orchestrator(prompt, domain, data_info, data_describe, visual_hint, context_hint)
            manipulator_prompt = spec.get("manipulator_prompt", "")
            visualizer_prompt  = spec.get("visualizer_prompt", "")
            analyzer_prompt    = spec.get("analyzer_prompt", "")
            compiler_instruction = spec.get("compiler_instruction", "")

            # Optional: Plan explainer text
            if need_plan_explainer:
                try:
                    pe = completion(
                        model=plan_explainer_model,
                        messages=[
                            {"role":"system","content":"""
                            Make sure all of the information below is applied.
                            1. The prompt that will be given to you is the details of what the systems is going to do to respond to the user prompt.
                            2. Your objective is to summarize the prompt that will be given to you into an easy to understand thought-process-like-explanation of what you (the system) are going to do for the user to read while they wait.
                            3. Your respond must be in a form of a single paragraph.
                            4. Include reason behind each crucial steps taken."""},
                            {"role":"user","content":
                             f"User Prompt: {prompt}. Datasets Domain name: {domain}. "
                             f"df.info: {data_info}. df.describe: {data_describe}. "
                             f"Plan: {json.dumps(agent_plan)}. Detailed instructions: {json.dumps(spec)}."
                            }
                        ],
                        seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high"
                    )
                    state["plan_explainer"] = get_content(pe) or ""
                except Exception:
                    state["plan_explainer"] = ""
                _save_conv_state(session_id, state)

            # Shared LLM
            llm = LiteLLM(model="gemini/gemini-2.5-pro", api_key=effective_key)
            pai.config.set({"llm": llm})

            # Manipulator
            _cancel_if_needed(session_id)
            df_processed = None
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

            # Visualizer
            _cancel_if_needed(session_id)
            dv_resp = SimpleNamespace(value="")
            chart_url = None
            diagram_signed_url = None
            diagram_gs_uri = None
            diagram_kind = ""
            run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
            globals()["_RUN_ID"] = run_id

            if need_visual:
                if df_processed is None:
                    return jsonify({"detail": "Visualization requested but no processed dataframe available."}), 500
                data_visualizer = _as_pai_df(df_processed)
                dv_resp = data_visualizer.chat(visualizer_prompt)

                # Move produced HTML to CHARTS_ROOT (local dev) + upload to GCS
                chart_path = getattr(dv_resp, "value", None)
                if isinstance(chart_path, str) and os.path.exists(chart_path):
                    out_dir = ensure_dir(os.path.join(CHARTS_ROOT, domain))
                    filename = f"{session_id}_{run_id}.html"
                    dest = os.path.join(out_dir, filename)
                    try:
                        os.replace(chart_path, dest)
                    except Exception:
                        import shutil; shutil.copyfile(chart_path, dest)
                    chart_url = f"/charts/{domain}/{filename}"

                    diagram_kind = _detect_diagram_kind(dest, visual_hint)
                    if GCS_BUCKET:
                        uploaded = upload_diagram_to_gcs(dest, domain=domain, session_id=session_id, run_id=run_id, kind=diagram_kind)
                        diagram_signed_url = uploaded["signed_url"]
                        diagram_gs_uri     = uploaded["gs_uri"]

                        state["last_visual_gcs_path"]   = diagram_gs_uri
                        state["last_visual_signed_url"] = diagram_signed_url
                        state["last_visual_kind"]       = diagram_kind

            # Analyzer
            _cancel_if_needed(session_id)
            da_resp = ""
            if need_analyze:
                if df_processed is None:
                    return jsonify({"detail": "Analyzer requested but no processed dataframe available."}), 500
                data_analyzer = _as_pai_df(df_processed)
                da_obj = data_analyzer.chat(analyzer_prompt)
                da_resp = get_content(da_obj)
                state["last_analyzer_text"] = da_resp or ""

            # Compiler
            _cancel_if_needed(session_id)
            data_info_runtime = _polars_info_string(df_processed) if isinstance(df_processed, pl.DataFrame) else data_info
            final_response = completion(
                model=compiler_model or "gemini/gemini-2.5-pro",
                messages=[
                    {"role": "system", "content": compiler_instruction},
                    {"role": "user", "content":
                        f"User Prompt:{prompt}. \n"
                        f"Datasets Domain name: {domain}. \n"
                        f"df.info of each dfs key(file name)-value pair:\n{data_info_runtime}. \n"
                        f"df.describe of each dfs key(file name)-value pair:\n{data_describe}. \n"
                        f"Data Visualizer Response:{getattr(dv_resp, 'value', '')}. \n"
                        f"Data Analyzer Response:{da_resp}."
                    },
                ],
                seed=1, stream=False, verbosity="medium", drop_params=True, reasoning_effort="high",
            )
            final_content = get_content(final_response)

        # Persist summary
        _append_history(state, "assistant", {
            "plan": agent_plan,
            "visual_path": "",
            "visual_signed_url": state.get("last_visual_signed_url",""),
            "visual_gs_uri": state.get("last_visual_gcs_path",""),
            "visual_kind": state.get("last_visual_kind",""),
            "analyzer_excerpt": (state.get("last_analyzer_text") or "")[:400],
            "final_preview": final_content[:600]
        })
        _save_conv_state(session_id, state)

        exec_time = time.time() - t0
        return jsonify({
            "session_id": session_id,
            "response": final_content,        # HTML
            "chart_url": chart_url,           # dev preview
            "diagram_kind": state.get("last_visual_kind",""),
            "diagram_gs_uri": state.get("last_visual_gcs_path",""),
            "diagram_signed_url": state.get("last_visual_signed_url",""),
            "execution_time": exec_time,
            "need_visualizer": need_visual,
            "need_analyzer": need_analyze,
            "need_manipulator": need_manip,
            "plan_explainer": state.get("plan_explainer",""),
        })
    except RuntimeError as rexc:
        if "CANCELLED_BY_USER" in str(rexc):
            return jsonify({"code":"CANCELLED","detail":"Processing cancelled by user."}), 409
        return jsonify({"detail": str(rexc)}), 500
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


# === NEW: PG credentials endpoints ============================================
@app.get("/pg/credentials")
def pg_get_credentials():
    """
    Returns PG credentials metadata from Firestore (no password in cleartext).
    """
    try:
        d = _load_pg_credentials_from_fs()
        if not d:
            return jsonify({"exists": False})
        safe = dict(d)
        safe.pop("password", None)
        return jsonify({"exists": True, "credentials": safe})
    except Exception as e:
        return jsonify({"error": str(e)}), 500

@app.post("/pg/credentials")
@app.put("/pg/credentials")
def pg_set_credentials():
    """
    Accepts JSON: {host, port, db, user, password, sslmode}
    Stores to Firestore with encrypted password (if FERNET_KEY configured).
    """
    try:
        body = request.get_json(force=True)
        if not body:
            return jsonify({"error": "Missing body"}), 400
        _save_pg_credentials_to_fs(body)
        # quick connection test (optional but helpful)
        try:
            _ = _pg_read_to_polars("SELECT 1 as ok").head(1)
            ok = True
        except Exception as e:
            ok = False
        return jsonify({"saved": True, "connection_ok": ok})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


# --- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=True)