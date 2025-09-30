# main.py — Merged Flask API: GCS datasets CRUD, diagrams(charts|tables) to GCS,
# Firestore history (GPT-like), provider key mgmt, cancel "thinking", export to PDF
import os, io, json, time, uuid, re, html
from datetime import datetime, timedelta
from typing import Dict, List, Optional
from types import SimpleNamespace

from flask import Flask, request, jsonify, send_from_directory, send_file
from flask_cors import CORS

import pandas as pd
from litellm import completion
import pandasai as pai
from pandasai import SmartDataframe, SmartDatalake
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse

# --- GCP clients ---
from google.cloud import storage
from google.cloud import firestore

<<<<<<< HEAD
import requests
=======
import requests, os
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
from cryptography.fernet import Fernet

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


DATASETS_ROOT = os.getenv("DATASETS_ROOT", os.path.abspath("./datasets"))  # still supported for local/dev
CHARTS_ROOT   = os.getenv("CHARTS_ROOT",   os.path.abspath("./charts"))    # still supported for local/dev
os.makedirs(DATASETS_ROOT, exist_ok=True)
os.makedirs(CHARTS_ROOT,   exist_ok=True)

# GCS / Firestore
GCS_BUCKET                  = os.getenv("GCS_BUCKET")                        # REQUIRED on Cloud Run
GCS_DATASETS_PREFIX         = os.getenv("GCS_DATASETS_PREFIX", "datasets")   # datasets/<domain>/<filename>
GCS_DIAGRAMS_PREFIX         = os.getenv("GCS_DIAGRAMS_PREFIX", "diagrams")   # diagrams/(charts|tables)/<domain>/<session>_<run>.html
GCS_SIGNED_URL_TTL_SECONDS  = int(os.getenv("GCS_SIGNED_URL_TTL_SECONDS", "604800"))  # 7 days

<<<<<<< HEAD
FIRESTORE_COLLECTION_SESSIONS  = os.getenv("FIRESTORE_COLLECTION", "convo_sessions")
FIRESTORE_COLLECTION_DATASETS  = os.getenv("FIRESTORE_DATASETS_COLLECTION", "datasets_meta")
=======
FIRESTORE_COLLECTION_SESSIONS = os.getenv("FIRESTORE_COLLECTION", "convo_sessions")
FIRESTORE_COLLECTION_DATASETS = os.getenv("FIRESTORE_DATASETS_COLLECTION", "datasets_meta")
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
FIRESTORE_COLLECTION_PROVIDERS = os.getenv("FIRESTORE_COLLECTION", "convo_providers")

# --- Init Flask ---
app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# --- Init GCP clients (ADC creds on Cloud Run) ---
_storage_client = storage.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else storage.Client()
_firestore_client = firestore.Client(project=GCP_PROJECT_ID) if GCP_PROJECT_ID else firestore.Client()

# --- Cancel flags (cooperative cancel between stages) ---
_CANCEL_FLAGS = set()  # holds session_id

# -------- Helpers --------
def slug(s: str) -> str:
    # Normalisasi domain agar konsisten FE/BE (case/space safe)
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
    
def get_provider_config(provider: str, api_key: str):
    """Return endpoint and headers based on provider"""
    if provider == "openai":
        return {
            "url": "https://api.openai.com/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"}
        }

    elif provider == "groq":
        return {
            "url": "https://api.groq.com/openai/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"}
        }

    elif provider == "anthropic":
        return {
            "url": "https://api.anthropic.com/v1/models",
            "headers": {"x-api-key": api_key}
        }

    elif provider == "google":
        # Google pakai query param ?key=
        return {
            "url": f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            "headers": {}
        }

    else:
        raise ValueError("Provider not supported")

def save_provider_key(user_id: str, provider: str, encrypted_key: str, models: list):
    """
    Save the encrypted API key to the Firestore collection
    """
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
        print(f"Saved {provider} for user {user_id}")
        return True
    except Exception as e:
        print("Firestore save error:", e)
        return False

def get_provider_config(provider: str, api_key: str):
    """Return endpoint and headers based on provider"""
    if provider == "openai":
        return {
            "url": "https://api.openai.com/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"}
        }
    elif provider == "groq":
        return {
            "url": "https://api.groq.com/openai/v1/models",
            "headers": {"Authorization": f"Bearer {api_key}"}
        }
    elif provider == "anthropic":
        return {
            "url": "https://api.anthropic.com/v1/models",
            "headers": {"x-api-key": api_key}
        }
    elif provider == "google":
        # Google pakai query param ?key=
        return {
            "url": f"https://generativelanguage.googleapis.com/v1beta/models?key={api_key}",
            "headers": {}
        }
    else:
        raise ValueError("Provider not supported")

def save_provider_key(user_id: str, provider: str, encrypted_key: str, models: list):
    """
    Save the encrypted API key to the Firestore collection
    """
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
        print(f"Saved {provider} for user {user_id}")
        return True
    except Exception as e:
        print("Firestore save error:", e)
        return False

# --- Firestore-backed conversation state --------------------------------------
def _fs_default_state():
    return {
        "history": [],             # list of dicts: {role, content, ts}
        "last_visual_gcs_path": "",
        "last_visual_signed_url": "",
        "last_visual_kind": "",    # "charts" | "tables"
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
        # ensure keys exist
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

# --- Firestore datasets meta ---------------------------------------------------
def _ds_ref(domain: str, filename: str):
    key = f"{slug(domain)}::{filename}"
    return _firestore_client.collection(FIRESTORE_COLLECTION_DATASETS).document(key)

def _save_dataset_meta(domain: str, filename: str, gs_uri: str, size: int):
    meta = {
        "domain": slug(domain),
        "filename": filename,
        "gs_uri": gs_uri,
        "size_bytes": size,
        "updated_at": firestore.SERVER_TIMESTAMP,
        "created_at": firestore.SERVER_TIMESTAMP,
    }
    _ds_ref(domain, filename).set(meta, merge=True)

def _delete_dataset_meta(domain: str, filename: str):
    _ds_ref(domain, filename).delete()

def _list_dataset_meta(domain: Optional[str]=None, limit: int=200) -> List[dict]:
    col = _firestore_client.collection(FIRESTORE_COLLECTION_DATASETS)
    q = col.order_by("updated_at", direction=firestore.Query.DESCENDING)
    if domain:
        q = q.where("domain", "==", slug(domain))
    docs = q.limit(limit).stream()
    return [d.to_dict() for d in docs if d.exists]

# --- GCS helpers ---------------------------------------------------------------
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

# ---- Local helpers / robust dev mode -----------------------------------------
def _upload_dataset_file_local(file_storage, *, domain: str) -> dict:
    safe_domain = slug(domain)
    folder = ensure_dir(os.path.join(DATASETS_ROOT, safe_domain))
    filename = file_storage.filename
    dest = os.path.join(folder, filename)
    file_storage.save(dest)
    size = os.path.getsize(dest) if os.path.exists(dest) else 0
    return {
        "filename": filename,
        "gs_uri": "",
        "signed_url": "",
        "size_bytes": size,
        "local_path": dest,
    }

def _save_bytes_local(domain: str, filename: str, data: bytes) -> dict:
    safe_domain = slug(domain)
    folder = ensure_dir(os.path.join(DATASETS_ROOT, safe_domain))
    dest = os.path.join(folder, filename)
    with open(dest, "wb") as f:
        f.write(data)
    size = os.path.getsize(dest)
    return {
        "filename": filename,
        "gs_uri": "",
        "signed_url": "",
        "size_bytes": size,
        "local_path": dest,
    }

def _read_local_csv_to_df(path: str, sep_candidates: List[str] = (",", "|", ";", "\\t")) -> pd.DataFrame:
    with open(path, "rb") as f:
        data = f.read()
    for sep in sep_candidates:
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep if sep != "\\t" else "\t")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data))

# ---- Upload (GCS when possible, safe local fallback otherwise) ----------------
def upload_dataset_file(file_storage, *, domain: str) -> dict:
    """
    Upload a Werkzeug FileStorage.
    - If GCS_BUCKET is set → try GCS (and write Firestore meta).
    - If GCS upload/meta fails (common in local dev) → automatically fall back to local save.
    - If GCS_BUCKET is not set → save locally.
    """
    if not GCS_BUCKET:
        return _upload_dataset_file_local(file_storage, domain=domain)

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
            _save_dataset_meta(domain, filename, gs_uri, size)
        except Exception:
            # ignore meta failure in local/dev
            pass
        return {
            "filename": filename,
            "gs_uri": gs_uri,
            "signed_url": _signed_url(blob, filename, "text/csv", GCS_SIGNED_URL_TTL_SECONDS),
            "size_bytes": size,
        }
    except Exception:
        # Any GCS error → use local fallback
        return _upload_dataset_file_local(file_storage, domain=domain)

def list_gcs_csvs(domain: str) -> List[storage.Blob]:
    safe_domain = slug(domain)
    prefix = f"{GCS_DATASETS_PREFIX}/{safe_domain}/"
    return list(_gcs_bucket().list_blobs(prefix=prefix))

def read_gcs_csv_to_df(gs_uri_or_blobname: str, *, sep_candidates: List[str] = (",","|",";","\\t")) -> pd.DataFrame:
    """
    Download a CSV from GCS and parse to DataFrame with simple separator heuristics.
    Accepts 'gs://bucket/path' or 'path/inside/bucket'.
    """
    if gs_uri_or_blobname.startswith("gs://"):
        _, bucket_name, *rest = gs_uri_or_blobname.replace("gs://","").split("/")
        blob_name = "/".join(rest)
        bucket = _storage_client.bucket(bucket_name)
    else:
        bucket = _gcs_bucket()
        blob_name = gs_uri_or_blobname
    blob = bucket.blob(blob_name)
    data = blob.download_as_bytes()
    for sep in sep_candidates:
        try:
            return pd.read_csv(io.BytesIO(data), sep=sep if sep!="\\t" else "\t")
        except Exception:
            continue
    return pd.read_csv(io.BytesIO(data))

def delete_gcs_object(blob_name_or_gs_uri: str):
    if blob_name_or_gs_uri.startswith("gs://"):
        _, bucket_name, *rest = blob_name_or_gs_uri.replace("gs://","").split("/")
        blob_name = "/".join(rest)
        bucket = _storage_client.bucket(bucket_name)
    else:
        bucket = _gcs_bucket()
        blob_name = blob_name_or_gs_uri
    bucket.blob(blob_name).delete()

# ---- Diagrams (charts|tables) helper -----------------------------------------
def _detect_diagram_kind(local_html_path: str, visual_hint: str) -> str:
    """
    Heuristics:
      - if contains 'plotly' or 'Plotly.newPlot' -> charts
      - elif contains '<table' -> tables
      - else fallback to visual_hint (table -> tables) otherwise charts
    """
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
    """
    Upload local HTML diagram to GCS under diagrams/<kind>/<domain>/<session>_<run>.html
    kind in {"charts","tables"}
    """
    if not os.path.exists(local_path):
        raise FileNotFoundError(local_path)
    safe_domain = slug(domain)
    filename = f"{session_id}_{run_id}.html"
    kind = "tables" if kind == "tables" else "charts"  # sanitize
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

# --- Router: decides which agents/LLM to run based on prompt + context --------
def plan_agents(user_prompt: str, data_info: dict, data_describe: dict, state: dict):
    """
    Returns a dict:
      {
        "need_manipulator": bool,
        "need_visualizer": bool,
        "need_analyzer": bool,
        "need_compiler": bool,
        "compiler_model": "gemini/gemini-2.5-pro" (or other),
        "visual_hint": "bar|line|table|auto",
        "reason": "<brief>"
      }
    """
    router_start_time = time.time()
    recent_context = json.dumps(state.get("history", [])[-6:], ensure_ascii=False)

    router_response = completion(
        model="gemini/gemini-2.5-pro",
        messages=[
            {
                "role": "system",
                "content": """
                You are the Orchestration Router.
                Decide which agents to run for a business data prompt, considering:
                - User’s explicit phrasing (e.g., “use bar chart” => visualizer needed).
                - Whether this is a follow-up (short references like “why did it drop?” may not need a visual).
                - Data manipulation needs (clean/aggregate/compare/compute % share).
                - Analysis needs (explain/why/driver/trend/surprises).
                - Use recent conversation context if present.

                Return STRICT JSON with keys (booleans are true/false; strings are single-line):
                need_manipulator, need_visualizer, need_analyzer, need_compiler, compiler_model, visual_hint, reason.

                Rules of thumb:
                - If “chart/plot/visualize/graph/bar/line/table” in prompt => need_visualizer=true.
                - If “why/driver/explain/root cause/trend/surprise” => need_analyzer=true.
                - If follow-up AND no new data ops implied => need_manipulator=false (reuse previous df if available).
                - Compiler always runs; choose a capable model (default gemini/gemini-2.5-pro).
                - visual_hint in {"bar","line","table","auto"}.
                Keep it one line JSON, no prose.
                """.strip(),
            },
            {
                "role": "user",
                "content": (
                    f"User Prompt: {user_prompt}\n"
                    f"Recent Context: {recent_context}\n"
                    f"Data Info (summary): {str(data_info)[:3000]}\n"
                    f"Data Describe (summary): {str(data_describe)[:3000]}"
                ),
            },
        ],
        seed=1,
        stream=False,
        verbosity="low",
        drop_params=True,
        reasoning_effort="high",
    )
    router_content = get_content(router_response)
    try:
        plan = _safe_json_loads(router_content)
    except Exception:
        # Heuristic fallback if LLM JSON fails
        p = user_prompt.lower()
        need_visual = bool(re.search(r"\b(chart|plot|graph|visual|bar|line|table)\b", p))
        need_analyze = bool(re.search(r"\b(why|driver|explain|root cause|trend|surprise|reason)\b", p))
        follow_up = bool(re.search(r"\b(what about|and|how about|ok but|also)\b", p)) or len(p.split()) <= 8
        need_manip = not follow_up or state.get("last_df_processed") is None
        visual_hint = "bar" if "bar" in p else ("line" if "line" in p else ("table" if "table" in p else "auto"))
        plan = {
            "need_manipulator": bool(need_manip),
            "need_visualizer": bool(need_visual),
            "need_analyzer": bool(need_analyze or not need_visual),
            "need_compiler": True,
            "compiler_model": "gemini/gemini-2.5-pro",
            "visual_hint": visual_hint,
            "reason": "heuristic fallback",
        }
    router_end_time = time.time()
    print(f"Router elapsed: {router_end_time - router_start_time:.2f}s — Plan: {plan}")
    state["last_plan"] = plan
    return plan

# --- Static file serving for charts (dev preview) ------------------------------
@app.route("/charts/<path:relpath>")
def serve_chart(relpath):
    full = os.path.join(CHARTS_ROOT, relpath)
    base = os.path.dirname(full)
    filename = os.path.basename(full)
    return send_from_directory(base, filename)

# --- Health & domain listing ---------------------------------------------------
@app.get("/health")
def health():
    return jsonify({"status": "healthy", "ts": datetime.utcnow().isoformat()})

<<<<<<< HEAD
# --- Provider key management endpoints (kept from previous version) -----------
=======
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
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
        print(f"[{provider}] status:", res.status_code)

        if res.status_code == 200:
            j = res.json()
            models = []

            # ambil daftar model dari API
            if "data" in j:
                models = [m.get("id") for m in j["data"] if "id" in m]
            elif "models" in j:
                models = [m.get("name") or m.get("id") for m in j["models"]]

            # enkripsi API Key
            encrypted_key = fernet.encrypt(api_key.encode()).decode() if fernet else None

<<<<<<< HEAD
            # simpan ke Firestore
=======
            # 🔐 simpan ke Firestore
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
            save_provider_key(user_id, provider, encrypted_key, models)

            return jsonify({
                "valid": True,
                "provider": provider,
                "models": models,
                "token": encrypted_key
            })
        else:
            return jsonify({
                "valid": False,
                "provider": provider,
                "status": res.status_code,
                "detail": res.text
            }), 400

    except Exception as e:
        print("Validation error:", e)
        return jsonify({"valid": False, "error": str(e)}), 500
<<<<<<< HEAD

=======
    
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
@app.route("/get-provider-keys", methods=["GET"])
def get_provider_keys():
    try:
        user_id = request.args.get("userId")
        if not user_id:
            return jsonify({"error": "Missing userId"}), 400

        docs = (
            _firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS)
            .where("user_id", "==", user_id)
            .stream()
        )

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
        print("Error get-provider-keys:", e)
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

<<<<<<< HEAD
        # 1) Validasi dulu API key baru
=======
        # 1️⃣ Validasi dulu API key baru
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
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

<<<<<<< HEAD
        # 2) Enkripsi ulang API key
=======
        # 2️⃣ Enkripsi ulang API key
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
        encrypted_key = (
            fernet.encrypt(api_key.encode()).decode() if fernet else None
        )

<<<<<<< HEAD
        # 3) Update Firestore document
=======
        # 3️⃣ Update Firestore document
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
        doc_ref = _firestore_client.collection(FIRESTORE_COLLECTION_PROVIDERS).document(
            f"{user_id}_{provider}"
        )
        doc_ref.set(
            {
                "user_id": user_id,
                "provider": provider,
                "token": encrypted_key,
                "models": models,
                "updated_at": datetime.utcnow().isoformat(),
            },
<<<<<<< HEAD
            merge=True,
=======
            merge=True,  
>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
        )

        return jsonify({"updated": True, "models": models})

    except Exception as e:
        print("Update provider error:", e)
        return jsonify({"updated": False, "error": str(e)}), 500

<<<<<<< HEAD
=======

>>>>>>> 5dd0c49beeb1444b572aefe66c091049fee5bc21
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

        # bisa pilih: delete atau soft delete
        doc_ref.delete()
        print(f"Deleted provider={provider} for user_id={user_id}")
        return jsonify({"deleted": True})
    except Exception as e:
        print("Error delete-provider-key:", e)
        return jsonify({"error": str(e)}), 500

@app.get("/domains")
def list_domains():
    """List domain folders & CSV available in this instance (ephemeral-friendly) + GCS meta."""
    result = {}
    try:
        # local
        for d in sorted(os.listdir(DATASETS_ROOT)):
            p = os.path.join(DATASETS_ROOT, d)
            if os.path.isdir(p):
                csvs = [f for f in sorted(os.listdir(p)) if f.lower().endswith(".csv")]
                if csvs:
                    result[d] = csvs
        # gcs (merge by name)
        try:
            metas = _list_dataset_meta()
            for m in metas:
                d = m.get("domain","")
                f = m.get("filename","")
                if not d or not f: continue
                result.setdefault(d, [])
                if f not in result[d]:
                    result[d].append(f)
        except Exception:
            pass
        return jsonify(result)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- DATASETS: upload/list/read/delete (GCS) ----------------------------------
@app.post("/datasets/upload")
def datasets_upload():
    """
    Multipart upload:
      - domain (str, required)
      - file (file, required)
    """
    try:
        domain = request.form.get("domain")
        file = request.files.get("file")
        if not domain or not file:
            return jsonify({"detail":"Missing 'domain' or 'file'"}), 400
        uploaded = upload_dataset_file(file, domain=domain)
        return jsonify(uploaded), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.get("/datasets")
def datasets_list():
    """
    Query:
      - domain (optional)
      - signed=true|false (include signed_url)
    """
    try:
        domain = request.args.get("domain")
        add_signed = request.args.get("signed","false").lower() in ("1","true","yes")
        items = []
        try:
            items = _list_dataset_meta(domain=domain)
            if add_signed:
                for it in items:
                    gs_uri = it.get("gs_uri","")
                    if not gs_uri: continue
                    _, bucket_name, *rest = gs_uri.replace("gs://","").split("/")
                    blob_name = "/".join(rest)
                    blob = _storage_client.bucket(bucket_name).blob(blob_name)
                    it["signed_url"] = _signed_url(blob, it["filename"], "text/csv", GCS_SIGNED_URL_TTL_SECONDS)
        except Exception:
            items = []

        # local fallback merge
        if domain:
            domain_dir = os.path.join(DATASETS_ROOT, slug(domain))
            if os.path.isdir(domain_dir):
                known = { (i.get("domain"), i.get("filename")) for i in items }
                for name in sorted(os.listdir(domain_dir)):
                    if name.lower().endswith(".csv") and (slug(domain), name) not in known:
                        path = os.path.join(domain_dir, name)
                        size = os.path.getsize(path)
                        items.append({"domain": slug(domain), "filename": name, "gs_uri":"", "size_bytes": size, "signed_url": ""})
        return jsonify({"items": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.get("/datasets/<domain>/<path:filename>")
def datasets_read(domain, filename):
    """
    Read dataset content (preview).
    Query:
      - n (int) default 50
      - as=json|csv (default json)
    """
    try:
        n = int(request.args.get("n","50"))
        as_fmt = request.args.get("as","json")
        # prefer GCS if configured, else local
        if GCS_BUCKET:
            blob_name = f"{GCS_DATASETS_PREFIX}/{slug(domain)}/{filename}"
            df = read_gcs_csv_to_df(blob_name)
        else:
            local_path = os.path.join(DATASETS_ROOT, slug(domain), filename)
            if not os.path.exists(local_path):
                return jsonify({"detail": "File not found"}), 404
            df = _read_local_csv_to_df(local_path)

        if n > 0:
            df = df.head(n)
        if as_fmt == "csv":
            out = io.StringIO(); df.to_csv(out, index=False)
            return out.getvalue(), 200, {"Content-Type":"text/csv; charset=utf-8"}
        return jsonify({"records": json.loads(df.to_json(orient="records"))})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

@app.delete("/datasets/<domain>/<path:filename>")
def datasets_delete(domain, filename):
    try:
        if GCS_BUCKET:
            blob_name = f"{GCS_DATASETS_PREFIX}/{slug(domain)}/{filename}"
            delete_gcs_object(blob_name)
            try:
                _delete_dataset_meta(domain, filename)
            except Exception:
                pass
        # best-effort local cleanup
        local_path = os.path.join(DATASETS_ROOT, slug(domain), filename)
        try:
            if os.path.exists(local_path):
                os.remove(local_path)
        except Exception:
            pass
        return jsonify({"deleted": True, "domain": slug(domain), "filename": filename})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# ----------------- FE COMPAT ALIASES (fix 404/500 seen in console) -----------------
# POST /upload_datasets/<domain>  (UploadDropzone.tsx)
@app.post("/upload_datasets/<domain>")
def compat_upload_datasets(domain: str):
    """
    FE compatibility endpoint.
    Accepts:
      - single file via 'file'
      - multiple files via 'files' or 'files[]'
      - raw body (bytes) with ?filename=... or header X-Filename (fallback to timestamped CSV)
    Returns:
      { "items": [ {filename, gs_uri, signed_url, size_bytes, local_path?}, ... ] }
    """
    try:
        files: List = []
        # support various possible field names
        single = request.files.get("file")
        if single:
            files.append(single)
        files.extend(request.files.getlist("files"))
        files.extend(request.files.getlist("files[]"))

        uploads = []
        for f in files:
            uploads.append(upload_dataset_file(f, domain=domain))

        # raw body fallback (some uploaders stream bytes instead of multipart)
        if not uploads and request.data:
            fname = request.args.get("filename") or request.headers.get("X-Filename") or f"upload_{int(time.time())}.csv"
            uploads.append(_save_bytes_local(domain, fname, request.data))

        if not uploads:
            return jsonify({"detail": "No file provided (expected 'file', 'files', or 'files[]', or raw body)."}), 400

        return jsonify({"items": uploads}), 201
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# GET /domains/<domain>/datasets  (DatasetsPage.tsx)
@app.get("/domains/<domain>/datasets>")
def compat_list_domain_datasets_trailing(domain: str):
    # Guard for accidental trailing '>' routes from some routers — redirect to proper JSON.
    return compat_list_domain_datasets(domain)

@app.get("/domains/<domain>/datasets")
def compat_list_domain_datasets(domain: str):
    """
    FE compatibility endpoint.
    Always includes signed URLs when available (GCS mode).
    Returns both 'items' and 'datasets' keys for broader FE compatibility.
    """
    try:
        items: List[dict] = []

        # Firestore-backed meta (GCS mode)
        try:
            fs_items = _list_dataset_meta(domain=domain)
            for it in fs_items:
                gs_uri = it.get("gs_uri","")
                if gs_uri:
                    try:
                        _, bucket_name, *rest = gs_uri.replace("gs://","").split("/")
                        blob_name = "/".join(rest)
                        blob = _storage_client.bucket(bucket_name).blob(blob_name)
                        it["signed_url"] = _signed_url(blob, it["filename"], "text/csv", GCS_SIGNED_URL_TTL_SECONDS)
                    except Exception:
                        it.setdefault("signed_url","")
            items.extend(fs_items)
        except Exception:
            # ignore when Firestore not configured in local dev
            pass

        # Local files (dev) — merged & deduped by filename
        try:
            domain_dir = os.path.join(DATASETS_ROOT, slug(domain))
            if os.path.isdir(domain_dir):
                known_names = { i.get("filename") for i in items }
                for name in sorted(os.listdir(domain_dir)):
                    if name.lower().endswith(".csv") and name not in known_names:
                        path = os.path.join(domain_dir, name)
                        size = os.path.getsize(path)
                        items.append({
                            "domain": slug(domain),
                            "filename": name,
                            "gs_uri": "",
                            "size_bytes": size,
                            "signed_url": "",
                        })
        except Exception:
            pass

        return jsonify({"items": items, "datasets": items})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500
# -------------------------------------------------------------------------------

# --- Sessions / History (GPT-like persistence) --------------------------------
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

# --- NEW: Export chat history to PDF ------------------------------------------
@app.get("/sessions/<session_id>/export/pdf")
def sessions_export_pdf(session_id: str):
    """
    Export riwayat chat untuk sebuah session menjadi PDF dan kirim sebagai download.
    Jika ReportLab belum terpasang, kembalikan 501 dengan pesan instalasi.
    """
    if not _REPORTLAB_AVAILABLE:
        return jsonify({
            "detail": "PDF export requires 'reportlab'. Install first: uv pip install reportlab"
        }), 501

    try:
        state = _get_conv_state(session_id)
        history: List[dict] = state.get("history", [])

        # siapkan dokumen PDF (in-memory)
        buf = io.BytesIO()
        doc = SimpleDocTemplate(buf, pagesize=A4)
        styles = getSampleStyleSheet()
        title = styles["Heading1"]
        meta  = styles["Normal"]
        body  = ParagraphStyle(
            "Body",
            parent=styles["BodyText"],
            fontSize=10,
            leading=14,
            alignment=TA_LEFT
        )
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
                    # if content is dict (e.g., plan/preview), serialize compactly
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

# --- Cancel “thinking” (cooperative stop) -------------------------------------
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

# --- Query endpoint (GCS + local fallback) ------------------------------------
@app.post("/query")
def query():
    """Run the orchestrated 3-agent pipeline on datasets (GCS-first) with routing and persistent session context."""
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        domain_in  = body.get("domain")
        prompt     = body.get("prompt")
        session_id = body.get("session_id") or str(uuid.uuid4())
        # ---------- NEW: optional dataset filter ----------
        dataset_filter = (body.get("dataset") or "").strip() or None  # e.g., "product.csv"

        dataset_field = body.get("dataset")
        if isinstance(dataset_field, list):
            datasets = dataset_field
        elif isinstance(dataset_field, str) and dataset_field.strip():
            datasets = [dataset_field.strip()]
        else:
            datasets = []

        if not domain_in or not prompt:
            return jsonify({"detail": "Missing 'domain' or 'prompt'"}), 400
        if not GEMINI_API_KEY:
            return jsonify({"detail": "No API key configured"}), 500

        print("Datasets selected:", datasets)

        # Normalize domain
        domain = slug(domain_in)

        # Persistent state (Firestore)
        state = _get_conv_state(session_id)
        _append_history(state, "user", prompt)
        _save_conv_state(session_id, state)

        _cancel_if_needed(session_id)

        # Load datasets from GCS first; fallback to local
        dfs: Dict[str, pd.DataFrame] = {}
        data_info: Dict[str, str] = {}
        data_describe: Dict[str, str] = {}

        # GCS
        try:
            if GCS_BUCKET:
                for b in list_gcs_csvs(domain):
                    if not b.name.lower().endswith(".csv"):
                        continue
                    key = os.path.basename(b.name)
                    if dataset_filter and key != dataset_filter:
                        continue
                    df = read_gcs_csv_to_df(b.name)
                    dfs[key] = df
                    buf = io.StringIO(); df.info(buf=buf)
                    data_info[key] = buf.getvalue()
                    try:
                        data_describe[key] = df.describe(include="all").to_json()
                    except Exception:
                        data_describe[key] = "{}"
        except Exception:
            pass

        # Local fallback (dev)
        domain_dir = ensure_dir(os.path.join(DATASETS_ROOT, domain))
        for name in sorted(os.listdir(domain_dir)):
            if not name.lower().endswith(".csv"):
                continue
            if dataset_filter and name != dataset_filter:
                continue
            if name in dfs:
                continue
            path = os.path.join(domain_dir, name)
            try:
                df = _read_local_csv_to_df(path)
                dfs[name] = df
                buf = io.StringIO(); df.info(buf=buf)
                data_info[name] = buf.getvalue()
                try:
                    data_describe[name] = df.describe(include="all").to_json()
                except Exception:
                    data_describe[name] = "{}"
            except Exception:
                pass

        if not dfs:
            # No CSV uploaded for requested scope
            if dataset_filter:
                # dataset was explicitly requested but not found
                available = []
                # list local files
                if os.path.isdir(domain_dir):
                    available.extend(sorted([f for f in os.listdir(domain_dir) if f.lower().endswith(".csv")]))
                # add GCS names (best-effort)
                try:
                    if GCS_BUCKET:
                        available.extend(sorted({os.path.basename(b.name) for b in list_gcs_csvs(domain) if b.name.lower().endswith(".csv")}))
                except Exception:
                    pass
                return jsonify({
                    "code": "DATASET_NOT_FOUND",
                    "detail": f"Requested dataset '{dataset_filter}' not found in domain '{domain}'.",
                    "domain": domain,
                    "available": sorted(list(set(available))),
                }), 404
            return jsonify({
                "code": "NEED_UPLOAD",
                "detail": f"No CSV files found in domain '{domain}'",
                "domain": domain
            }), 409

        # ------------------- Router (decide which agents to run) -------------------
        agent_plan = plan_agents(
            user_prompt=prompt,
            data_info=data_info,
            data_describe=data_describe,
            state=state,
        )
        need_manip = bool(agent_plan.get("need_manipulator", True))
        need_visual = bool(agent_plan.get("need_visualizer", True))
        need_analyze = bool(agent_plan.get("need_analyzer", True))
        compiler_model = agent_plan.get("compiler_model") or "gemini/gemini-2.5-pro"
        visual_hint = agent_plan.get("visual_hint", "auto")

        # ------------------- Orchestrator (aware of context + router) -------------
        context_hint = {
            "router_plan": agent_plan,
            "last_visual_path": "",  # local path deprecated; use signed url below
            "has_prev_df_processed": False,
            "last_analyzer_excerpt": (state.get("last_analyzer_text") or "")[:400],
            "dataset_filter": dataset_filter or "ALL",  # expose selected dataset to LLM context
        }

        _cancel_if_needed(session_id)
        orchestrator_response = completion(
            model="gemini/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": """
                You are the Orchestrator.

                15 instructions you need to follow as the orchestrator:
                1. Think step by step.
                2. You orchestrate 3 LLM PandasAI Agents for business data analysis.
                3. The 3 agents are: Data Manipulator, Data Visualizer, Data Analyser.
                4. You will emit a specific prompt for each of those 3 agents.
                5. Each prompt is a numbered, step-by-step instruction set.
                6. Prompts must be clear, detailed, and complete to avoid ambiguity.
                7. The number of steps may differ per agent.
                8. Example user tasks include:
                8a. What is my revenue this week vs last week?
                8b. Why did my revenue drop this week?
                8c. Any surprises in revenue this month?
                8d. Notable trends this month?
                8e. Correlation between revenue and bounces?
                8f. Is this conversion rate normal for this time of year?
                9. Reason strictly from the user-provided data.
                10. Convert a short business question into three specialist prompts.
                11. If a currency is not explicitly stated assume its in Indonesian Rupiah.
                13. All specialists operate in Python using PandasAI SmartDataframe as `sdf` (backed by pandas `df`).
                14. Return STRICT JSON with keys: manipulator_prompt, visualizer_prompt, analyzer_prompt, compiler_instruction.
                15. Each value must be a **single line** string. No extra keys, no prose, no markdown/code fences.

                6 instructions for data manipulator prompt creation:
                1. Enforce data hygiene before analysis.
                1a. Parse dates to pandas datetime, create explicit period columns (day/week/month).
                1b. Set consistent dtypes for numeric fields; strip/normalize categorical labels; standardize currency units if present.
                1c. Handle missing values: impute or drop **only when necessary**; keep legitimate zeros.
                2. Mind the term like m0 and m1 and any other similar terms used to state if something is the former or later, in this case the m0 is the current month and the m1 is the 1 month earlier.
                3. Mind each of the datasets name.
                4. Produce exactly the minimal, analysis-ready dataframe(s) needed for the user question, with stable, well-named columns.
                5. Include the percentage version of the raw value on the column that you think is appropriate to include.
                6. End by returning only:
                   result = {"type":"dataframe","value": <THE_FINAL_DATAFRAME>}

                15 instructions for data visualizer prompt creation:
                1. Produce exactly ONE interactive visualization (a Plotly diagram or an HTML table) per request.
                2. Choose the best form of visualization based on the user's question. Use a Plotly diagram for trends and comparisons; use a styled HTML table for ranked lists or data with percentages.
                3. For Plotly diagrams: Prevent overlaps by rotating axis ticks ≤45°, wrapping long labels, ensuring adequate margins, and placing the legend outside the plot area.
                4. For Plotly diagrams: Apply insight-first formatting: include a clear title and subtitle, label axes with units, use thousands separators, and configure a rich hover-over.
                5. Aggregate data to a sensible granularity (e.g., day, week, or month) and cap extreme outliers for readability (noting this in the subtitle).
                6. For Plotly diagrams: To ensure high contrast, instruct the agent to use a truncated monochromatic colorscale by skipping the lightest 25% of a standard scale (e.g., Only shades of Blues, or only shades of reds).
                7. The prompt must specify how to truncate the scale, for example: "Create a custom colorscale by sampling 'Blues' from 0.25 to 1.0." The gradient must map to data values (lighter for low, darker for high).
                8. For Plotly diagrams: Use a bar chart, grouped bar chart, or line chart.
                9. If a table visualization is chosen, instruct the agent to use the Pandas Styler object to generate the final HTML, not Plotly. Enforce a monochromatic blue palette only (no red or green): use tints/shades of a single blue (e.g., very light → very dark) for accents and data bars. Apply zebra striping to table rows with alternating white and light grey backgrounds (e.g., #FFFFFF and #F5F5F5). Hide the index in the rendered table.
                10. The prompt must specify using the Styler.bar() method only on columns that represent share-of-total percentages and only when the column total ≈ 100% (allow small rounding tolerance). These are the columns analogous to “% of all” fields (e.g., “% Taker”, “% Transaction”, “% Revenue Squad” when present). Bars must be left-aligned, constrained with vmin=0, and use one blue color (single hue).
                10a. Example instruction: “Identify share-of-total percentage columns whose values sum to ~100%. Apply in-cell data bars with df.style.bar(subset=share_cols, align='left', color='#5DADE2', vmin=0). Ensure non-share percentage columns (e.g., rates like TUR) do not receive bars.”
                11. Output Python code only (no prose/comments/markdown). Import os and datetime. Build a directory and a run-scoped timestamped filename using a run ID stored in globals:
                    dir_path = "/content/exports/<domain>/<chart or table>"
                    os.makedirs(dir_path, exist_ok=True)
                    rid = globals().get("_RUN_ID")
                    if not isinstance(rid, str) or not rid:
                        import datetime as _dt
                        rid = _dt.datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
                        globals()["_RUN_ID"] = rid
                    file_path = os.path.join(dir_path, f"some-slug-{rid}.html")
                12. Write the file exactly once using an atomic lock to avoid duplicates across retries:
                    lock_path = file_path + ".lock"
                    try:
                        fd = os.open(lock_path, os.O_CREAT | os.O_EXCL | os.O_WRONLY)
                        try:
                            if "fig" in globals():
                                fig.write_html(file_path, include_plotlyjs="cdn", full_html=True)
                            else:
                                html_str = styled_html if "styled_html" in globals() else sdf._obj.to_html(index=False)
                                with open(file_path, "w", encoding="utf-8") as f:
                                    f.write(html_str)
                        finally:
                            os.close(fd)
                    except FileExistsError:
                        pass
                13. Ensure file_path is a plain Python string and do not print/return anything else:
                    file_path = str(file_path)
                14. The last line of code MUST be exactly:
                    result = {"type": "string", "value": file_path}
                15. DO NOT return the raw HTML string in the value field.

                3 instructions for data analyzer prompt creation:
                1. Write like you’re speaking to a person; be concise and insight-driven.
                2. Quantify where possible (deltas, % contributions, time windows); reference the exact columns/filters used.
                3. Return only:
                    result = {"type":"string","value":"<3–6 crisp bullets or 2 short paragraphs of insights>"}

                34 instructions for response compiler system content creation:
                1. Brevity: ≤180 words; bullets preferred; no code blocks, no JSON, no screenshots.
                2. Lead with the answer: 1–2 sentence “Bottom line” with main number, time window, and delta.
                3. Quantified drivers: top 3 with magnitude, direction, and approx contribution (absolute and % where possible).
                4. Next actions: 2–4 prioritized, concrete actions with expected impact/rationale.
                5. Confidence & caveats: one short line on data quality/assumptions/gaps; include Confidence: High/Medium/Low.
                6. Minimal tables: ≤1 table only if essential (≤5×3); otherwise avoid tables.
                7. No repetition: do not restate agent text; synthesize it.
                8. Do not try to show images; if a chart exists, mention the HTML path only.
                9. Always include units/currency and exact comparison window (e.g., “Aug 2025 vs Jul 2025”, “W34 vs W33”).
                10. Show both absolute and % change where sensible (e.g., “+$120k (+8.4%)”).
                11. Round smartly (money to nearest K unless < $10k; rates 1–2 decimals).
                12. If any agent fails or data is incomplete, still produce the best insight; mark gaps in Caveats and adjust Confidence.
                13. If the user asks “how much/which/why,” the first sentence must provide the number/entity/reason.
                14. Exact compiler_instruction template the orchestrator should emit (single line; steps separated by ';'):
                15. Read the user prompt, data_info, and all three agent responses;
                16. Compute the direct answer including the main number and compare period;
                17. Identify the top 3 quantified drivers with direction and contribution;
                18. Draft 'Bottom line' in 1–2 sentences answering plainly;
                19. List 2–4 prioritized Next actions with expected impact;
                20. Add a one-line Caveats with Confidence and any gaps;
                21. Keep ≤180 words, use bullets, avoid tables unless ≤5×3 and essential;
                22. Include units, absolute and % deltas, and explicit dates;
                23. Do not repeat agent text verbatim or include code/JSON.
                24. Format hint (shape, not literal):
                24a. Bottom line — <answer with number + timeframe>.
                24b. Drivers — <A: +X (≈Y%); B: −X (≈Y%); C: ±X (≈Y%)>.
                24c. Next actions — 1) <action>; 2) <action>; 3) <action>.
                24d. Caveats — <one line>. Confidence: <High/Medium/Low>.
                25. compiler_instruction must contain clear, step-by-step instructions to assemble the final response.
                26. The final response must be decision-ready and insight-first, not raw data.
                27. The compiler_instruction is used as the compiler LLM’s system content.
                28. Compiler user content will be: f"User Prompt:{user_prompt}. \nData Info:{data_info}. \nData Describe:{data_describe}. \nData Manipulator Response:{data_manipulator_response}. \nData Visualizer Response:{data_visualizer_response}. \nData Analyzer Response:{data_analyzer_response}".
                29. `data_info` is a string from `df.info()`.
                30. `data_manipulator_response` is a PandasAI DataFrameResponse.
                31. `data_visualizer_response` is a **file path to an HTML** inside `{"type":"string","value": ...}`. The `value` MUST be a plain Python string containing the path.
                32. `data_analyzer_response` is a PandasAI StringResponse.
                33. Your goal in `compiler_instruction` is to force brevity, decisions, and insights.
                34. The compiler must NOT echo raw dataframes, code, or long tables; it opens with the business answer, quantifies drivers, and closes with next actions.
                """},
                {
                    "role": "user",
                    "content": (
                        f"User Prompt: {prompt} \n"
                        f"Datasets Domain name: {domain}. \n"
                        f"df.info of each dfs key(file name)-value pair:\n{data_info}. \n"
                        f"df.describe of each dfs key(file name)-value pair:\n{data_describe}. \n"
                        f"Router Context Hint: {json.dumps(context_hint)} \n"
                        f"Visualization hint (from router): {visual_hint}"
                    )
                }
            ],
            seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high",
        )
        orchestrator_content = get_content(orchestrator_response)
        try:
            spec = _safe_json_loads(orchestrator_content)
        except Exception:
            spec = {"manipulator_prompt":"", "visualizer_prompt":"", "analyzer_prompt":"", "compiler_instruction":""}

        manipulator_prompt = spec.get("manipulator_prompt", "")
        visualizer_prompt  = spec.get("visualizer_prompt", "")
        analyzer_prompt    = spec.get("analyzer_prompt", "")
        compiler_instruction = spec.get("compiler_instruction", "")

        # ------------------- Shared LLM config ------------------------------------
        llm = LiteLLM(model="gemini/gemini-2.5-pro", api_key=GEMINI_API_KEY)
        pai.config.set({"llm": llm})

        # ---------- Data Manipulator (conditional) ----------
        _cancel_if_needed(session_id)
        df_processed = None
        if need_manip or (need_visual or need_analyze):
            data_manipulator = SmartDatalake(
                list(dfs.values()),
                config={
                    "llm": llm,
                    "seed": 1,
                    "stream": False,
                    "verbosity": "low",
                    "drop_params": True,
                    "save_charts": False,
                    "open_charts": False,
                    "conversational": False,
                    "enforce_privacy": True,
                    "reasoning_effort": "high",
                    "save_charts_path": "./charts"
                }
            )
            dm_resp = data_manipulator.chat(manipulator_prompt)
            if isinstance(dm_resp, DataFrameResponse):
                df_processed = dm_resp.value
            else:
                df_processed = dm_resp

        # ---------- Data Visualizer (conditional) ----------
        _cancel_if_needed(session_id)
        dv_resp = SimpleNamespace(value="")
        chart_url = None
        diagram_signed_url = None
        diagram_gs_uri = None
        diagram_kind = ""  # "charts" | "tables"
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        globals()["_RUN_ID"] = run_id

        if need_visual:
            if df_processed is None:
                return jsonify({"detail": "Visualization requested but no processed dataframe available."}), 500
            data_visualizer = SmartDataframe(
                df_processed,
                config={
                    "llm": llm,
                    "seed": 1,
                    "stream": False,
                    "verbosity": "low",
                    "drop_params": True,
                    "save_charts": False,
                    "open_charts": False,
                    "conversational": False,
                    "enforce_privacy": True,
                    "reasoning_effort": "high",
                    "save_charts_path": "./charts"
                }
            )
            dv_resp = data_visualizer.chat(visualizer_prompt)

            # Move produced HTML to CHARTS_ROOT for serving (local dev) + upload to GCS/diagrams
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

                # Determine diagram kind and upload
                diagram_kind = _detect_diagram_kind(dest, visual_hint)
                if GCS_BUCKET:
                    uploaded = upload_diagram_to_gcs(dest, domain=domain, session_id=session_id, run_id=run_id, kind=diagram_kind)
                    diagram_signed_url = uploaded["signed_url"]
                    diagram_gs_uri     = uploaded["gs_uri"]

                    # persist for GPT-like reload
                    state["last_visual_gcs_path"]   = diagram_gs_uri
                    state["last_visual_signed_url"] = diagram_signed_url
                    state["last_visual_kind"]       = diagram_kind

        # ---------- Data Analyzer (conditional) ----------
        _cancel_if_needed(session_id)
        da_resp = ""
        if need_analyze:
            if df_processed is None:
                return jsonify({"detail": "Analyzer requested but no processed dataframe available."}), 500
            data_analyzer = SmartDataframe(
                df_processed,
                config={
                    "llm": llm,
                    "seed": 1,
                    "stream": False,
                    "verbosity": "low",
                    "drop_params": True,
                    "save_charts": False,
                    "open_charts": False,
                    "conversational": True,
                    "enforce_privacy": False,
                    "reasoning_effort": "high",
                    "save_charts_path": "./charts"
                }
            )
            da_obj = data_analyzer.chat(analyzer_prompt)
            da_resp = get_content(da_obj)
            state["last_analyzer_text"] = da_resp or ""

        # ---------- Response Compiler (router-selected model) ----------
        _cancel_if_needed(session_id)
        data_info_runtime = data_info  # keep per-file infos

        final_response = completion(
            model=compiler_model or "gemini/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": compiler_instruction},
                {
                    "role": "user",
                    "content": (
                        f"User Prompt:{prompt}. \n"
                        f"Datasets Domain name: {domain}. \n"
                        f"df.info of each dfs key(file name)-value pair:\n{data_info_runtime}. \n"
                        f"df.describe of each dfs key(file name)-value pair:\n{data_describe}. \n"
                        f"Data Visualizer Response:{getattr(dv_resp, 'value', '')}. \n"
                        f"Data Analyzer Response:{da_resp}."
                    ),
                },
            ],
            seed=1,
            stream=False,
            verbosity="medium",
            drop_params=True,
            reasoning_effort="high",
        )
        final_content = get_content(final_response)

        # --- Persist summary to conversation history ---
        _append_history(state, "assistant", {
            "plan": agent_plan,
            "visual_path": "",  # local path deprecated
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
            "response": final_content,
            # local (dev) preview:
            "chart_url": chart_url,

            # preferred fields (diagrams on GCS):
            "diagram_kind": diagram_kind,               # "charts" | "tables"
            "diagram_gs_uri": diagram_gs_uri,           # gs://...
            "diagram_signed_url": diagram_signed_url,   # FE should use this

            # backward-compat (deprecated):
            "chart_gs_uri": diagram_gs_uri,
            "chart_signed_url": diagram_signed_url,

            "execution_time": exec_time,
            "need_visualizer": need_visual,
            "need_analyzer": need_analyze,
            "need_manipulator": need_manip,
        })
    except RuntimeError as rexc:
        if "CANCELLED_BY_USER" in str(rexc):
            return jsonify({"code":"CANCELLED","detail":"Processing cancelled by user."}), 409
        return jsonify({"detail": str(rexc)}), 500
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=True)