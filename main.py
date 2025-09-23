# main.py — Merged Flask API: latest ML pipeline + newer backend bits (slug, /domains, NEED_UPLOAD, CORS origins)
import os, io, json, time, uuid, re
from datetime import datetime
from typing import Dict
from types import SimpleNamespace

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd
from litellm import completion
import pandasai as pai
from pandasai import SmartDataframe, SmartDatalake
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse

# -------- optional: load .env --------
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173,https://convoinsight.vercel.app"
).split(",")

DATASETS_ROOT = os.getenv("DATASETS_ROOT", os.path.abspath("./datasets"))
CHARTS_ROOT   = os.getenv("CHARTS_ROOT",   os.path.abspath("./charts"))
os.makedirs(DATASETS_ROOT, exist_ok=True)
os.makedirs(CHARTS_ROOT,   exist_ok=True)

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

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

# --- Session-scoped conversation state ----------------------------------------
_SESSIONS: Dict[str, dict] = {}

def _get_conv_state(session_id: str) -> dict:
    st = _SESSIONS.get(session_id)
    if not isinstance(st, dict):
        st = {
            "history": [],             # recent (role, content) pairs
            "last_df_processed": None, # pandas DataFrame (post-manipulation)
            "last_visual_path": "",    # last HTML chart/table path
            "last_analyzer_text": "",  # last analyzer bullets/string
            "last_plan": None,         # last routing decision
        }
        _SESSIONS[session_id] = st
    return st

def _append_history(state: dict, role: str, content: str, max_len=10_000, keep_last=12):
    content = str(content)
    if len(content) > max_len:
        content = content[:max_len] + " …"
    state["history"].append({"role": role, "content": content, "ts": time.time()})
    state["history"] = state["history"][-keep_last:]

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
        reasoning_effort="low",
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

# --- Static file serving for charts -------------------------------------------
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

@app.get("/domains")
def list_domains():
    """List domain folders & CSV available in this instance (ephemeral-friendly)."""
    result = {}
    try:
        for d in sorted(os.listdir(DATASETS_ROOT)):
            p = os.path.join(DATASETS_ROOT, d)
            if os.path.isdir(p):
                csvs = [f for f in sorted(os.listdir(p)) if f.lower().endswith(".csv")]
                result[d] = csvs
        return jsonify(result)
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- Upload datasets -----------------------------------------------------------
@app.post("/upload_datasets/<domain>")
def upload_datasets(domain: str):
    """Accept CSV(s); save under ./datasets/<slug(domain)>/ for LOCAL/Cloud Run (ephemeral)."""
    try:
        domain = slug(domain)
        files = request.files.getlist("files")
        if not files:
            return jsonify({"detail": "No files uploaded. Field name should be 'files'."}), 400

        domain_dir = ensure_dir(os.path.join(DATASETS_ROOT, domain))
        saved = []
        for f in files:
            if not f.filename.lower().endswith(".csv"):
                return jsonify({"detail": f"Only CSV supported in local mode: {f.filename}"}), 400
            dest = os.path.join(domain_dir, f.filename)
            f.save(dest)
            saved.append({"filename": f.filename, "local_path": dest})
        return jsonify({"message": f"Uploaded {len(saved)} files for domain {domain}", "files": saved})
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- Query endpoint (orchestrated 3-agent pipeline with router + session) -----
@app.post("/query")
def query():
    """Run the orchestrated 3-agent pipeline on LOCAL datasets with routing and session context."""
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        domain_in  = body.get("domain")
        prompt     = body.get("prompt")
        session_id = body.get("session_id") or str(uuid.uuid4())

        if not domain_in or not prompt:
            return jsonify({"detail": "Missing 'domain' or 'prompt'"}), 400
        if not GEMINI_API_KEY:
            return jsonify({"detail": "No API key configured"}), 500

        # Normalize domain & ensure folder exists (ephemeral-friendly)
        domain = slug(domain_in)
        domain_dir = ensure_dir(os.path.join(DATASETS_ROOT, domain))

        # Session-scoped state
        state = _get_conv_state(session_id)
        globals()["_CONV_STATE"] = state  # used by helper functions
        _append_history(state, "user", prompt)

        # Load datasets (CSV)
        dfs: Dict[str, pd.DataFrame] = {}
        data_info: Dict[str, str] = {}
        data_describe: Dict[str, str] = {}

        for name in sorted(os.listdir(domain_dir)):
            if name.lower().endswith(".csv"):
                path = os.path.join(domain_dir, name)
                try:
                    df = pd.read_csv(path, sep="|")
                except Exception:
                    df = pd.read_csv(path)
                dfs[name] = df
                buf = io.StringIO()
                df.info(buf=buf)
                data_info[name] = buf.getvalue()
                try:
                    data_describe[name] = df.describe(include="all").to_json()
                except Exception:
                    data_describe[name] = "{}"

        if not dfs:
            # No CSV uploaded yet for this (possibly new) domain
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
            "last_visual_path": state.get("last_visual_path", ""),
            "has_prev_df_processed": state.get("last_df_processed") is not None,
            "last_analyzer_excerpt": (state.get("last_analyzer_text") or "")[:400],
        }

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
        df_processed = None
        if need_manip or (state.get("last_df_processed") is None and (need_visual or need_analyze)):
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
            state["last_df_processed"] = df_processed
        else:
            df_processed = state.get("last_df_processed")

        # ---------- Data Visualizer (conditional) ----------
        dv_resp = SimpleNamespace(value="")
        chart_url = None
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

            # Move produced HTML to CHARTS_ROOT for serving
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
                state["last_visual_path"] = dest

        # ---------- Data Analyzer (conditional) ----------
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
            "visual_path": state.get("last_visual_path", ""),
            "analyzer_excerpt": (state.get("last_analyzer_text") or "")[:400],
            "final_preview": final_content[:600]
        })
        _SESSIONS[session_id] = state  # save back

        exec_time = time.time() - t0
        return jsonify({
            "session_id": session_id,
            "response": final_content,
            "chart_url": chart_url,
            "execution_time": exec_time,
            "need_visualizer": need_visual,
            "need_analyzer": need_analyze,
            "need_manipulator": need_manip,
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 500

# --- Entry point ---------------------------------------------------------------
if __name__ == "__main__":
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=True)
