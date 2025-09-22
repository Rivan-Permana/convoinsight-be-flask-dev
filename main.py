# app.py — Flask API for "ML BI Pipeline" (LOCAL DEV)
import os, io, json, time, uuid
from datetime import datetime
from typing import Dict

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS

import pandas as pd

# LLM + PandasAI
from litellm import completion
import pandasai as pai
from pandasai import SmartDataframe, SmartDatalake
from pandasai_litellm.litellm import LiteLLM
from pandasai.core.response.dataframe import DataFrameResponse

# -------- Configuration --------
# optional: load .env
try:
    from dotenv import load_dotenv
    load_dotenv()
except Exception:
    pass

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY")  # REQUIRED
CORS_ORIGINS = os.getenv(
    "CORS_ORIGINS",
    "http://localhost:5173,http://127.0.0.1:5173"
).split(",")

DATASETS_ROOT = os.getenv("DATASETS_ROOT", os.path.abspath("./datasets"))
CHARTS_ROOT   = os.getenv("CHARTS_ROOT",   os.path.abspath("./charts"))
os.makedirs(DATASETS_ROOT, exist_ok=True)
os.makedirs(CHARTS_ROOT,   exist_ok=True)

app = Flask(__name__)
CORS(app, origins=CORS_ORIGINS, supports_credentials=True)

# -------- Helpers --------
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

def ensure_dir(p: str):
    os.makedirs(p, exist_ok=True)
    return p

# serve generated charts locally
@app.route("/charts/<path:relpath>")
def serve_chart(relpath):
    full = os.path.join(CHARTS_ROOT, relpath)
    base = os.path.dirname(full)
    filename = os.path.basename(full)
    return send_from_directory(base, filename)

# -------- Endpoints --------
@app.get("/health")
def health():
    return jsonify({"status": "healthy", "ts": datetime.utcnow().isoformat()})

@app.post("/upload_datasets/<domain>")
def upload_datasets(domain: str):
    """Accept CSV(s); save under ./datasets/<domain>/ for LOCAL dev."""
    try:
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

@app.post("/query")
def query():
    """Run the orchestrated 3-agent pipeline on LOCAL datasets."""
    t0 = time.time()
    try:
        body = request.get_json(force=True)
        domain     = body.get("domain")
        prompt     = body.get("prompt")
        session_id = body.get("session_id") or str(uuid.uuid4())

        if not domain or not prompt:
            return jsonify({"detail": "Missing 'domain' or 'prompt'"}), 400
        if not GEMINI_API_KEY:
            return jsonify({"detail": "No API key configured"}), 500

        # load datasets from ./datasets/<domain>
        domain_dir = os.path.join(DATASETS_ROOT, domain)
        if not os.path.isdir(domain_dir):
            return jsonify({"detail": f"No datasets folder for domain '{domain}'"}), 404

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
            return jsonify({"detail": f"No CSV files found in domain '{domain}'"}), 404

        # ===== Orchestrator (DO NOT CHANGE THIS SYSTEM PROMPT) =====
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
                2. Mind the term like m0 and m1 which means month 0 and 1 and any other similar terms used to decide if something is the former or later, in this case the m0 is the previous month and the m1 is the current or the next month.
                3. Mind each of the datasets name.
                4. Produce exactly the minimal, analysis-ready dataframe(s) needed for the user question, with stable, well-named columns.
                5. Include the percentage version of the raw value on the column that you think is appropriate to include.
                6. End by returning only: result = {"type":"dataframe","value": <THE_FINAL_DATAFRAME>}

                15 instructions for data visualizer prompt creation:
                1. Produce exactly ONE interactive visualization (a Plotly diagram or an HTML table) per request.
                2. Choose the best form of visualization based on the user's question. Use a Plotly diagram for trends and comparisons; use a styled HTML table for ranked lists or data with percentages.
                3. For Plotly diagrams: Prevent overlaps by rotating axis ticks ≤45°, wrapping long labels, ensuring adequate margins, and placing the legend outside the plot area.
                4. For Plotly diagrams: Apply insight-first formatting: include a clear title and subtitle, label axes with units, use thousands separators, and configure a rich hover-over.
                5. Aggregate data to a sensible granularity (e.g., day, week, or month) and cap extreme outliers for readability (noting this in the subtitle).
                6. For Plotly diagrams: To ensure high contrast, instruct the agent to use a truncated monochromatic colorscale by skipping the lightest 25% of a standard scale (e.g., 'Blues only').
                7. The prompt must specify how to truncate the scale, for example: "Create a custom colorscale by sampling 'Blues' from 0.25 to 1.0." The gradient must map to data values (lighter for low, darker for high).
                8. For Plotly diagrams: Use a bar chart, grouped bar chart, or line chart.
                9. If a table visualization is chosen, instruct the agent to use the Pandas Styler object to generate the final HTML, not Plotly.
                10. The prompt must specify using the Styler.bar() method only on columns that represent share-of-total percentages and only when the column total ≈ 100%.
                11. Output Python code only (no prose/comments/markdown). Import os and datetime. Build a directory and a run-scoped timestamped filename using a run ID stored in globals.
                12. Write the file exactly once using an atomic lock to avoid duplicates across retries.
                13. Ensure file_path is a plain Python string and do not print/return anything else.
                14. The last line of code MUST be exactly: result = {"type": "string", "value": file_path}
                15. DO NOT return the raw HTML string in the value field.

                3 instructions for data analyzer prompt creation:
                1. Write like you're speaking to a person; be concise and insight-driven.
                2. Quantify where possible (deltas, % contributions, time windows); reference the exact columns/filters used.
                3. Return only: result = {"type":"string","value":"<3–6 crisp bullets or 2 short paragraphs of insights>"}

                34 instructions for response compiler system content creation:
                1. Brevity: ≤180 words; bullets preferred; no code blocks, no JSON, no screenshots.
                2. Lead with the answer: 1–2 sentence "Bottom line" with main number, time window, and delta.
                3. Quantified drivers: top 3 with magnitude, direction, and approx contribution (absolute and % where possible).
                4. Next actions: 2–4 prioritized, concrete actions with expected impact/rationale.
                5. Confidence & caveats: one short line on data quality/assumptions/gaps; include Confidence: High/Medium/Low.
                6. Minimal tables: ≤1 table only if essential (≤5×3); otherwise avoid tables.
                7. No repetition: do not restate agent text; synthesize it.
                8. Do not try to show images; if a chart exists, mention the HTML path only.
                9. Always include units/currency and exact comparison window (e.g., "Aug 2025 vs Jul 2025", "W34 vs W33").
                10. Show both absolute and % change where sensible (e.g., "+$120k (+8.4%)").
                11. Round smartly (money to nearest K unless < $10k; rates 1–2 decimals).
                12. If any agent fails or data is incomplete, still produce the best insight; mark gaps in Caveats and adjust Confidence.
                13. If the user asks "how much/which/why," the first sentence must provide the number/entity/reason.
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
                24. Format hint (shape, not literal): 24a. Bottom line – <answer with number + timeframe>. 24b. Drivers – <A: +X (≈Y%); B: −X (≈Y%); C: ±X (≈Y%)>. 24c. Next actions – 1) <action>; 2) <action>; 3) <action>. 24d. Caveats – <one line>. Confidence: <High/Medium/Low>.
                25. compiler_instruction must contain clear, step-by-step instructions to assemble the final response.
                26. The final response must be decision-ready and insight-first, not raw data.
                27. The compiler_instruction is used as the compiler LLM's system content.
                28. Compiler user content will be: f"User Prompt:{user_prompt}. \nData Info:{data_info}. \nData Describe:{data_describe}. \nData Manipulator Response:{data_manipulator_response}. \nData Visualizer Response:{data_visualizer_response}. \nData Analyzer Response:{data_analyzer_response}".
                29. `data_info` is a string from `df.info()`.
                30. `data_manipulator_response` is a PandasAI DataFrameResponse.
                31. `data_visualizer_response` is a **file path to an HTML** inside {"type":"string","value": ...}. The `value` MUST be a plain Python string containing the path.
                32. `data_analyzer_response` is a PandasAI StringResponse.
                33. Your goal in `compiler_instruction` is to force brevity, decisions, and insights.
                34. The compiler must NOT echo raw dataframes, code, or long tables; it opens with the business answer, quantifies drivers, and closes with next actions.
                """},
                {"role": "user", "content": f"User Prompt: {prompt} \nDatasets Domain name: {domain}. \ndf.info of each dfs key(file name)-value pair:\n{data_info}. \df.describe of each dfs key(file name)-value pair:\n{data_describe}."}
            ],
            seed=1, stream=False, verbosity="low", drop_params=True, reasoning_effort="high",
        )
        orchestrator_content = get_content(orchestrator_response)
        try:
            spec = json.loads(orchestrator_content)
        except json.JSONDecodeError:
            start = orchestrator_content.find("{")
            end = orchestrator_content.rfind("}")
            spec = json.loads(orchestrator_content[start:end+1])

        manipulator_prompt = spec["manipulator_prompt"]
        visualizer_prompt  = spec["visualizer_prompt"]
        analyzer_prompt    = spec["analyzer_prompt"]
        compiler_instruction = spec["compiler_instruction"]

        # LLM & PandasAI
        llm = LiteLLM(model="gemini/gemini-2.5-pro", api_key=GEMINI_API_KEY)
        pai.config.set({"llm": llm})

        # Data Manipulator
        data_manipulator = SmartDatalake(
            list(dfs.values()),
            config={
                "llm": llm, "seed": 1, "stream": False, "verbosity": "low",
                "drop_params": True, "save_charts": False, "open_charts": False,
                "conversational": False, "enforce_privacy": True, "reasoning_effort": "high",
            }
        )
        dm_resp = data_manipulator.chat(manipulator_prompt)
        df_processed = dm_resp.value if isinstance(dm_resp, DataFrameResponse) else dm_resp

        # Data Visualizer
        data_visualizer = SmartDataframe(
            df_processed,
            config={
                "llm": llm, "seed": 1, "stream": False, "verbosity": "low",
                "drop_params": True, "save_charts": False, "open_charts": False,
                "conversational": False, "enforce_privacy": True, "reasoning_effort": "high",
            }
        )
        run_id = datetime.utcnow().strftime("%Y%m%d%H%M%S%f")
        globals()["_RUN_ID"] = run_id
        dv_resp = data_visualizer.chat(visualizer_prompt)

        # Data Analyzer
        data_analyzer = SmartDataframe(
            df_processed,
            config={
                "llm": llm, "seed": 1, "stream": False, "verbosity": "low",
                "drop_params": True, "save_charts": False, "open_charts": False,
                "conversational": True, "enforce_privacy": False, "reasoning_effort": "high",
            }
        )
        da_resp = data_analyzer.chat(analyzer_prompt)

        # Response compiler
        final_response = completion(
            model="gemini/gemini-2.5-pro",
            messages=[
                {"role": "system", "content": compiler_instruction},
                {"role": "user", "content":
                    f"User Prompt:{prompt}. \nDatasets Domain name: {domain}. "
                    f"\ndf.info of each dfs key(file name)-value pair:\n{data_info}. "
                    f"\df.describe of each dfs key(file name)-value pair:\n{data_describe}. "
                    f"\nData Visualizer Response:{getattr(dv_resp, 'value', None)}. "
                    f"\nData Analyzer Response:{da_resp}."
                },
            ],
            seed=1, stream=False, verbosity="medium", drop_params=True, reasoning_effort="high",
        )
        final_content = get_content(final_response)

        # publish chart (local path -> /charts URL)
        chart_url = None
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

        exec_time = time.time() - t0
        return jsonify({
            "session_id": session_id,
            "response": final_content,
            "chart_url": chart_url,
            "execution_time": exec_time,
        })
    except Exception as e:
        return jsonify({"detail": str(e)}), 500


if __name__ == "__main__":
    # Run: python app.py
    host = os.getenv("HOST", "127.0.0.1")
    port = int(os.getenv("PORT", "8000"))
    app.run(host=host, port=port, debug=True)