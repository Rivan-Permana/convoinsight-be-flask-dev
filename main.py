# main.py
from flask import Flask, request, jsonify
from flask_cors import CORS
import os, json, tempfile, re, io, base64
import requests
import duckdb
import pandas as pd
from dotenv import load_dotenv

# --- Matplotlib headless for server ---
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt

# ================== Setup ==================
load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "").strip().strip('"')
if not OPENAI_API_KEY:
    raise RuntimeError("OPENAI_API_KEY not found")

app = Flask(__name__)
CORS(app)

OPENAI_CHAT_URL = "https://api.openai.com/v1/chat/completions"


# ================== OpenAI Helper ==================
def openai_chat(messages, model: str = "gpt-4o", temperature: float = 0.0) -> str:
    headers = {
        "Authorization": f"Bearer {OPENAI_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {"model": model, "messages": messages, "temperature": temperature}
    resp = requests.post(OPENAI_CHAT_URL, headers=headers, json=payload, timeout=90)
    if resp.status_code != 200:
        raise RuntimeError(f"OpenAI error {resp.status_code}: {resp.text}")
    data = resp.json()
    try:
        return data["choices"][0]["message"]["content"]
    except Exception as exc:
        raise RuntimeError(f"Unexpected OpenAI response shape: {data}") from exc


# ================== SQL Utilities ==================
def sanitize_sql(raw: str) -> str:
    """
    Ambil SQL dari balasan LLM (buang code fence, komentar, dan ; di akhir)
    """
    m = re.search(r"```[a-zA-Z]*\s*([\s\S]*?)```", raw)
    s = m.group(1) if m else raw

    # hapus komentar
    s = re.sub(r"--.*?$", "", s, flags=re.MULTILINE)
    s = re.sub(r"/\*.*?\*/", "", s, flags=re.DOTALL)

    s = s.strip()
    if s.endswith(";"):
        s = s[:-1].strip()
    return s


def sanitize_limit(sql: str) -> str:
    """
    Hilangkan LIMIT ganda; pertahankan LIMIT pertama yang muncul.
    """
    tokens = sql.strip().split()
    result = []
    seen_limit = False
    skip_next_number = False
    for i, t in enumerate(tokens):
        low = t.lower()
        if low == "limit":
            if seen_limit:
                # skip "limit" kedua+
                skip_next_number = True
                continue
            seen_limit = True
            result.append(t)
            continue
        if skip_next_number:
            # lewati angka setelah limit yang kedua
            if re.fullmatch(r"\d+", t):
                skip_next_number = False
                continue
            # kalau bukan angka, matikan flag dan teruskan
            skip_next_number = False
        result.append(t)
    return " ".join(result)


# ================== Data Cleaning ==================
def to_snake(name: str) -> str:
    s = name.strip().lower()
    s = re.sub(r"[^a-z0-9]+", "_", s)
    s = re.sub(r"_+", "_", s).strip("_")
    return s or "col"


def build_data_clean(con: duckdb.DuckDBPyConnection) -> None:
    """
    Buat view data_clean:
    - semua kolom dari `data` diganti snake_case
    - tambahkan event_ts (TIMESTAMP) jika terdeteksi kolom tanggal/waktu
    """
    cols_df = con.execute("PRAGMA table_info('data');").fetchdf()
    orig_cols = [str(r["name"]) for _, r in cols_df.iterrows()]

    safe_cols = [to_snake(c) for c in orig_cols]
    select_parts = [f'"{o}" AS {s}' for o, s in zip(orig_cols, safe_cols)]

    # cari kandidat datetime tunggal
    candidates_single = [
        "event_ts", "timestamp", "datetime", "date_time",
        "pickup_datetime", "dropoff_datetime", "booking_datetime",
        "created_at", "updated_at", "created_time", "time_stamp",
        "pickup_time", "dropoff_time", "booking_time",
        "event_time"
    ]
    # cari kandidat date & time terpisah
    candidates_date = ["event_date", "date", "pickup_date", "dropoff_date", "booking_date"]
    candidates_time = ["event_time", "time", "pickup_time", "dropoff_time", "booking_time"]

    safe_set = set(safe_cols)
    dt_single = next((c for c in candidates_single if c in safe_set), None)
    dt_date = next((c for c in candidates_date if c in safe_set), None)
    dt_time = next((c for c in candidates_time if c in safe_set), None)

    event_ts_expr = None
    if dt_single:
        # satu kolom yang “mirip datetime”
        event_ts_expr = (
            f"COALESCE("
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%Y-%m-%d %H:%M:%S'),"
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%Y-%m-%d %H:%M'),"
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%Y-%m-%d'),"
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%d-%m-%Y %H:%M:%S'),"
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%d/%m/%Y %H:%M:%S'),"
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%d-%m-%Y'),"
            f"  try_strptime(CAST({dt_single} AS VARCHAR), '%d/%m/%Y')"
            f") AS event_ts"
        )
    elif dt_date and dt_time:
        # gabungan date + time
        event_ts_expr = (
            f"COALESCE("
            f"  try_strptime(CAST({dt_date} AS VARCHAR) || ' ' || CAST({dt_time} AS VARCHAR), '%Y-%m-%d %H:%M:%S'),"
            f"  try_strptime(CAST({dt_date} AS VARCHAR) || ' ' || CAST({dt_time} AS VARCHAR), '%Y-%m-%d %H:%M'),"
            f"  try_strptime(CAST({dt_date} AS VARCHAR) || ' ' || CAST({dt_time} AS VARCHAR), '%d-%m-%Y %H:%M:%S'),"
            f"  try_strptime(CAST({dt_date} AS VARCHAR) || ' ' || CAST({dt_time} AS VARCHAR), '%d/%m/%Y %H:%M:%S')"
            f") AS event_ts"
        )
    elif dt_date:
        # hanya date
        event_ts_expr = (
            f"COALESCE("
            f"  try_strptime(CAST({dt_date} AS VARCHAR), '%Y-%m-%d'),"
            f"  try_strptime(CAST({dt_date} AS VARCHAR), '%d-%m-%Y'),"
            f"  try_strptime(CAST({dt_date} AS VARCHAR), '%d/%m/%Y')"
            f") AS event_ts"
        )

    if event_ts_expr:
        select_parts.append(event_ts_expr)

    select_sql = ", ".join(select_parts)
    con.execute("DROP VIEW IF EXISTS data_clean;")
    con.execute(f"CREATE VIEW data_clean AS SELECT {select_sql} FROM data;")


# ================== Chart Helpers ==================
def fig_to_data_uri():
    buf = io.BytesIO()
    plt.tight_layout()
    plt.savefig(buf, format="png", bbox_inches="tight")
    plt.close()
    buf.seek(0)
    b64 = base64.b64encode(buf.read()).decode("ascii")
    return f"data:image/png;base64,{b64}"


def chart_timeseries_per_day(con):
    try:
        df = con.execute("""
            SELECT date_trunc('day', event_ts) AS d, COUNT(*) AS rides
            FROM data_clean
            WHERE event_ts IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """).fetchdf()
        if df.empty:
            return None
        plt.figure(figsize=(8, 3.5))
        plt.plot(df["d"], df["rides"])
        plt.title("Rides per Day")
        plt.xlabel("Date")
        plt.ylabel("Rides")
        return {"title": "Rides per Day", "data_uri": fig_to_data_uri()}
    except Exception:
        return None


def chart_hour_of_day(con):
    try:
        df = con.execute("""
            SELECT EXTRACT(hour FROM event_ts)::INT AS h, COUNT(*) AS rides
            FROM data_clean
            WHERE event_ts IS NOT NULL
            GROUP BY 1
            ORDER BY 1
        """).fetchdf()
        if df.empty:
            return None
        plt.figure(figsize=(8, 3.5))
        plt.bar(df["h"], df["rides"])
        plt.title("Rides by Hour")
        plt.xlabel("Hour (0-23)")
        plt.ylabel("Rides")
        return {"title": "Rides by Hour", "data_uri": fig_to_data_uri()}
    except Exception:
        return None


def chart_top_category(con, colname: str):
    try:
        df = con.execute(f"""
            SELECT {colname} AS key, COUNT(*) AS cnt
            FROM data_clean
            GROUP BY 1
            ORDER BY 2 DESC
            LIMIT 10
        """).fetchdf()
        if df.empty:
            return None
        plt.figure(figsize=(8, max(3.0, 0.35 * len(df))))
        plt.barh(df["key"].astype(str), df["cnt"])
        plt.gca().invert_yaxis()
        plt.title(f"Top 10 {colname}")
        plt.xlabel("Count")
        return {"title": f"Top 10 {colname}", "data_uri": fig_to_data_uri()}
    except Exception:
        return None


def chart_hist_numeric(con, colname: str):
    try:
        df = con.execute(f"""
            SELECT try_cast({colname} AS DOUBLE) AS v
            FROM data_clean
        """).fetchdf()
        v = df["v"].dropna()
        if v.empty:
            return None
        plt.figure(figsize=(8, 3.5))
        plt.hist(v, bins=40)
        plt.title(f"Histogram of {colname}")
        plt.xlabel(colname)
        plt.ylabel("Frequency")
        return {"title": f"Histogram of {colname}", "data_uri": fig_to_data_uri()}
    except Exception:
        return None


# ================== Profil Dataset ==================
def basic_profile(con):
    """
    Profil ringkas: jumlah baris/kolom, null per kolom, contoh kolom numeric/categorical, rentang tanggal
    """
    meta = {}
    info = con.execute("PRAGMA table_info('data_clean');").fetchdf()
    meta["columns"] = [{"name": r["name"], "type": r["type"]} for _, r in info.iterrows()]
    meta["ncols"] = len(meta["columns"])
    nrows = con.execute("SELECT COUNT(*) AS n FROM data_clean").fetchdf().iloc[0]["n"]
    meta["nrows"] = int(nrows)

    # null per kolom
    cols = [c["name"] for c in meta["columns"]]
    if cols:
        exprs = ", ".join([f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}_nulls" for c in cols])
        nulls_row = con.execute(f"SELECT {exprs} FROM data_clean").fetchdf().iloc[0].to_dict()
        meta["nulls_per_col"] = {c: int(nulls_row.get(f"{c}_nulls", 0)) for c in cols}
    else:
        meta["nulls_per_col"] = {}

    # kolom numeric & categorical (berdasarkan type DuckDB)
    numeric_cols = [c["name"] for c in meta["columns"] if any(t in c["type"].lower() for t in ["int", "decimal", "double", "float", "hugeint"])]
    categorical_cols = [c["name"] for c in meta["columns"] if "varchar" in c["type"].lower() or "string" in c["type"].lower()]

    meta["numeric_cols"] = numeric_cols
    meta["categorical_cols"] = categorical_cols

    # rentang tanggal jika ada event_ts
    try:
        rng = con.execute("SELECT MIN(event_ts) AS start, MAX(event_ts) AS end FROM data_clean").fetchdf().iloc[0]
        meta["date_range"] = {
            "start": str(rng["start"]) if pd.notna(rng["start"]) else None,
            "end": str(rng["end"]) if pd.notna(rng["end"]) else None
        }
    except Exception:
        meta["date_range"] = {"start": None, "end": None}

    # pilih sebuah kolom kategori (cardinality 2..50) untuk bar chart top-10
    best_cat = None
    best_cnt = 0
    for c in categorical_cols:
        cnt = con.execute(f"SELECT COUNT(DISTINCT {c}) AS n FROM data_clean").fetchdf().iloc[0]["n"]
        if 2 <= cnt <= 50 and cnt > best_cnt:
            best_cnt, best_cat = cnt, c
    meta["best_categorical"] = best_cat

    # contoh kolom numeric untuk histogram
    best_num = numeric_cols[0] if numeric_cols else None
    meta["best_numeric"] = best_num

    return meta


# ================== Endpoints ==================
@app.post("/api/chat")
def chat():
    try:
        data = request.get_json(silent=True) or {}
        messages = data.get("messages", [])
        if not isinstance(messages, list) or not messages:
            return jsonify({"error": "messages harus array"}), 400
        reply = openai_chat(messages=messages, model="gpt-4o", temperature=0.3)
        return jsonify({"reply": reply})
    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.post("/api/ask-dataset")
def ask_dataset():
    try:
        if "file" not in request.files:
            return jsonify({"error": "file wajib ada (multipart/form-data)"}), 400

        question = (request.form.get("question") or "").strip()
        kind = (request.form.get("kind") or "").lower()
        f = request.files["file"]
        if not question:
            return jsonify({"error": "question kosong"}), 400

        suffix = ".parquet" if (kind == "parquet" or f.filename.endswith(".parquet")) else ".csv"
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            f.save(tmp.name)
            tmp_path = tmp.name

        ql = question.lower().strip()

        # ===== Shortcut: sapaan sederhana =====
        greetings = {"halo", "hello", "hi", "hay", "hallo", "assalamualaikum"}
        if ql in greetings:
            return jsonify({
                "sql": "",
                "columns": [],
                "rows": [],
                "answer": "Halo! Saya ConvoInsight, asisten data kamu. Ada yang bisa saya bantu?",
                "analysis": "",
                "charts": [],
            })

        con = duckdb.connect()
        try:
            # 1) Load data ke tabel sementara
            if suffix == ".parquet":
                con.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM read_parquet(?);", [tmp_path])
            else:
                con.execute("CREATE OR REPLACE TABLE data AS SELECT * FROM read_csv_auto(?, sample_size=200000);", [tmp_path])

            # 2) Buat view 'data_clean'
            build_data_clean(con)

            # ===== Shortcut query tanpa LLM =====
            if "5 baris" in ql or "lima baris" in ql or "first 5" in ql or "preview" in ql:
                df = con.execute("SELECT * FROM data_clean LIMIT 5").fetchdf()
                rows = json.loads(df.to_json(orient="records", date_format="iso"))
                return jsonify({
                    "sql": "SELECT * FROM data_clean LIMIT 5",
                    "columns": list(df.columns),
                    "rows": rows,
                    "answer": "Berikut 5 baris contoh dari dataset.",
                    "analysis": "",
                    "charts": [],
                })

            if ("null" in ql and ("per kolom" in ql or "per column" in ql)) or ("missing" in ql and "per" in ql):
                cols = con.execute("PRAGMA table_info('data_clean');").fetchdf()["name"].tolist()
                exprs = ", ".join([f"SUM(CASE WHEN {c} IS NULL THEN 1 ELSE 0 END) AS {c}_nulls" for c in cols])
                df = con.execute(f"SELECT {exprs} FROM data_clean").fetchdf()
                out = pd.DataFrame({"column": cols, "nulls": [int(df.iloc[0][f"{c}_nulls"]) for c in cols]})
                rows = json.loads(out.to_json(orient="records"))
                return jsonify({
                    "sql": f"SELECT {exprs} FROM data_clean",
                    "columns": list(out.columns),
                    "rows": rows,
                    "answer": "Jumlah nilai NULL per kolom.",
                    "analysis": "",
                    "charts": [],
                })

            if "kolom apa saja" in ql or "columns" in ql or "schema" in ql:
                info = con.execute("PRAGMA table_info('data_clean');").fetchdf()[["name", "type"]]
                rows = json.loads(info.to_json(orient="records"))
                return jsonify({
                    "sql": "PRAGMA table_info('data_clean')",
                    "columns": ["name", "type"],
                    "rows": rows,
                    "answer": "Daftar kolom dan tipe data pada tabel.",
                    "analysis": "",
                    "charts": [],
                })

            if "jumlah baris" in ql or "row count" in ql or "count rows" in ql:
                df = con.execute("SELECT COUNT(*) AS row_count FROM data_clean").fetchdf()
                rows = json.loads(df.to_json(orient="records"))
                return jsonify({
                    "sql": "SELECT COUNT(*) AS row_count FROM data_clean",
                    "columns": list(df.columns),
                    "rows": rows,
                    "answer": f"Total baris pada dataset: {int(df.iloc[0]['row_count'])}.",
                    "analysis": "",
                    "charts": [],
                })

            # 3) Siapkan schema & prompt untuk OpenAI (generate SQL)
            cols_clean = con.execute("PRAGMA table_info('data_clean');").fetchdf()
            schema_str = "\n".join(f"- {row['name']} {row['type']}" for _, row in cols_clean.iterrows())

            guardrails = (
                "ONLY write a single SELECT SQL query over table `data_clean`.\n"
                "Return ONLY the SQL text (no code fences) and do NOT end with a semicolon.\n"
                "Do NOT reference table `data`; always use `data_clean`.\n"
                "If a numeric field may be stored as TEXT/VARCHAR (e.g., amounts/prices), wrap with try_cast(... AS DOUBLE).\n"
                "For time-based summaries, use `event_ts`.\n"
                "Never use INSERT/UPDATE/DELETE/CREATE/DROP/ATTACH/COPY.\n"
                "Always include LIMIT 50.\n"
            )

            prompt_sql = f"""
You are a SQL generator for DuckDB. Given a question and a schema, output ONLY the SQL (no explanation).

Schema (table `data_clean`):
{schema_str}

{guardrails}

Question: {question}
SQL:
""".strip()

            sql_raw = openai_chat(
                [
                    {"role": "system", "content": "You convert questions into a single valid DuckDB SQL SELECT with LIMIT 50 over table `data_clean`."},
                    {"role": "user", "content": prompt_sql},
                ]
            )

            sql = sanitize_sql(sql_raw)
            lowered = sql.lower()

            if not lowered.startswith("select"):
                return jsonify({"error": f"Refused non-SELECT SQL.\nModel returned:\n{sql_raw}"}), 400
            for bad in ["insert", "update", "delete", "create", "drop", "attach", "copy"]:
                if re.search(rf"\b{bad}\b", lowered):
                    return jsonify({"error": f"Refused unsafe SQL.\nModel returned:\n{sql_raw}"}), 400

            sql = sanitize_limit(sql)
            if not re.search(r"\blimit\s+\d+", sql, flags=re.IGNORECASE):
                sql = f"{sql} LIMIT 50"

            try:
                df = con.execute(sql).fetchdf()
            except Exception as ex:
                return jsonify({"error": f"DuckDB error: {ex}\nSQL:\n{sql}"}), 400

            # batas payload
            max_cells = 10000
            if df.size > max_cells:
                max_rows = max(1, max_cells // max(1, len(df.columns)))
                df = df.head(max_rows)

            rows = json.loads(df.to_json(orient="records", date_format="iso"))
            sample_text = df.head(10).to_json(orient="records", date_format="iso")

            # 4) Jawaban singkat spesifik pertanyaan
            answer_prompt = f"""
Pertanyaan: {question}

SQL yang dijalankan:
{sql}

Contoh hasil (JSON, maksimal 10 baris):
{sample_text}

Tulislah jawaban singkat (1–3 paragraf) menggunakan bahasa yang sama dengan pertanyaan.
Jika ada agregasi, jelaskan secara ringkas. Sebutkan jika hasil terpotong (truncated).
""".strip()

            answer = openai_chat(
                [
                    {"role": "system", "content": "Kamu adalah analis data. Jawablah ringkas, jelas, dan gunakan bahasa yang sama dengan pertanyaan user (Indonesia atau Inggris)."},
                    {"role": "user", "content": answer_prompt},
                ],
                temperature=0.2,
            )

            # 5) Profil dataset & Charts otomatis
            profile = basic_profile(con)

            charts = []
            ts_chart = chart_timeseries_per_day(con)
            if ts_chart: charts.append(ts_chart)

            hod_chart = chart_hour_of_day(con)
            if hod_chart: charts.append(hod_chart)

            if profile.get("best_categorical"):
                cat_chart = chart_top_category(con, profile["best_categorical"])
                if cat_chart: charts.append(cat_chart)

            if profile.get("best_numeric"):
                hist_chart = chart_hist_numeric(con, profile["best_numeric"])
                if hist_chart: charts.append(hist_chart)

            # 6) Analisis deskriptif (narasi 1–3 paragraf)
            nulls_preview = sorted(profile["nulls_per_col"].items(), key=lambda x: -x[1])[:5]
            nulls_text = ", ".join(f"{k}: {v}" for k, v in nulls_preview) if nulls_preview else "tidak signifikan"

            date_range = profile.get("date_range") or {}
            date_rng_text = (
                f"{date_range.get('start')} s/d {date_range.get('end')}"
                if (date_range.get("start") and date_range.get("end"))
                else "tak terdeteksi"
            )

            analysis_prompt = f"""
Kamu adalah analis data. Berikan analisis deskriptif singkat (maks 3 paragraf), bahasa mengikuti pertanyaan user.

Konteks dataset:
- Jumlah baris: {profile['nrows']}
- Jumlah kolom: {profile['ncols']}
- Rentang tanggal (event_ts): {date_rng_text}
- Kolom numeric contoh: {profile.get('best_numeric')}
- Kolom kategori contoh: {profile.get('best_categorical')}
- Top 5 nulls per kolom: {nulls_text}

Pertanyaan user:
{question}

SQL yang dijalankan:
{sql}

Contoh hasil (JSON, maksimal 10 baris):
{sample_text}

Fokuskan pada:
- Tren waktu (jika ada event_ts), variasi jam, atau musiman.
- Distribusi metrik penting (mis. fare/distance/duration jika ada).
- Perbandingan kategori (jika ada).
- Catat jika hasil terpotong (LIMIT 50) atau ada missing value signifikan.
""".strip()

            analysis_text = openai_chat(
                [
                    {"role": "system", "content": "Tulislah analisis deskriptif yang ringkas, jelas, dan actionable. Hindari klaim spekulatif."},
                    {"role": "user", "content": analysis_prompt},
                ],
                temperature=0.2,
            )

            return jsonify({
                "sql": sql,
                "columns": list(df.columns),
                "rows": rows,
                "answer": answer,            # jawaban spesifik
                "analysis": analysis_text,   # narasi EDA
                "charts": charts             # [{title, data_uri}]
            })
        finally:
            try:
                con.close()
            except:
                pass
            try:
                os.remove(tmp_path)
            except:
                pass

    except Exception as e:
        return jsonify({"error": str(e)}), 500


@app.get("/health")
def health():
    return jsonify({"ok": True})


# ================== Run ==================
if __name__ == "__main__":
    # Gunakan port 5000 secara default
    app.run(host="0.0.0.0", port=5000, debug=True)
