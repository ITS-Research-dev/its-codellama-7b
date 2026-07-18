import os
import re
import json
import datetime
from pathlib import Path
from flask import Flask, request, jsonify, send_file
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)

# ── Model config ──────────────────────────────────────────────────────────────
def _ensure_v1(host: str) -> str:
    host = host.rstrip("/")
    if not host.endswith("/v1"):
        host += "/v1"
    return host

SCORING_HOST  = _ensure_v1(os.getenv("SCORING_MODEL_HOST", "http://localhost:1234"))
SCORING_MODEL = os.getenv("SCORING_MODEL_NAME", "codellama-7b-instruct")

FEEDBACK_HOST  = _ensure_v1(os.getenv("FEEDBACK_MODEL_HOST", "http://localhost:11434"))
FEEDBACK_MODEL = os.getenv("FEEDBACK_MODEL_NAME", "codegemma:7b-instruct")

# ── QLoRA output ──────────────────────────────────────────────────────────────
FINETUNE_PATH = Path(
    os.getenv("FINETUNE_OUTPUT_PATH", "finetune_samples.jsonl")
)
if not FINETUNE_PATH.is_absolute():
    FINETUNE_PATH = Path(__file__).parent / FINETUNE_PATH


# ═══════════════════════════════════════════════════════════════════════════════
#  RUBRIC  (used in the scoring prompt so CodeLlama knows the scale)
# ═══════════════════════════════════════════════════════════════════════════════
RUBRIC = """SCORING RUBRIC
(use integer multiples of 5 ONLY: 0,5,10,...,100)

FUNCTIONALITY : 100=perfect output; 80=correct main case; 60=core runs incomplete; 40=wrong but basic understanding; 0=crashes.
CODE_STYLE    : 100=perfect PEP8; 80=minor style issues; 60=readable but errors; 40=messy; 0=unreadable.
DOCUMENTATION : 100=full docstrings+comments; 80=most code documented; 60=basic comments; 40=1-2 comments; 0=none.
LOGIC         : 100=optimal and elegant; 80=correct but improvable; 60=mostly correct with flaws; 40=partially correct; 0=no logical structure.
SYNTAX        : 100=perfect idiomatic Python; 80=correct not fully idiomatic; 60=small non-breaking errors; 40=1-2 errors; 0=cannot parse.
CONCEPT       : 100=best concept applied; 80=correct but improvable; 60=runs but not ideal; 40=misused; 0=none.

PROFICIENCY (for feedback model only — do NOT output this in scoring):
  Expert          : avg >= 90, nothing below 85
  Competent       : avg >= 75, nothing below 65
  Advance         : avg >= 60, nothing below 50
  Advance Beginner: avg >= 45, nothing below 35
  Beginner        : avg >= 30, nothing below 20
  Novice          : avg <  30 or fundamental errors
"""


# ═══════════════════════════════════════════════════════════════════════════════
#  PROMPT BUILDERS
# ═══════════════════════════════════════════════════════════════════════════════
def build_scoring_prompt(soal: str, code: str, expected_out: str,
                          simulated_in: str) -> str:
    """
    Focused scoring prompt for CodeLlama.
    Output: ONLY a JSON object with 6 scores + overall_score.
    """
    return f"""You are a strict Python code assessor. Score the student submission below using the rubric.

### PROBLEM:
{soal}

### EXPECTED OUTPUT:
{expected_out if expected_out else "(not specified)"}

### SIMULATED INPUT:
{simulated_in if simulated_in else "(none)"}

### STUDENT CODE:
```python
{code}
```

{RUBRIC}

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{
  "scores": {{
    "functionality": <int, multiple of 5, 0-100>,
    "code_style": <int, multiple of 5, 0-100>,
    "documentation": <int, multiple of 5, 0-100>,
    "logic": <int, multiple of 5, 0-100>,
    "syntax": <int, multiple of 5, 0-100>,
    "concept": <int, multiple of 5, 0-100>,
    "overall_score": <average of above 6, rounded to nearest int>
  }}
}}"""


def build_feedback_prompt(soal: str, code: str, expected_out: str,
                           simulated_in: str, scores: dict) -> str:
    """
    Feedback prompt for CodeGemma.
    Receives the scores already computed by CodeLlama, adds narrative.
    Output: feedback, proficiency, reasoning, per-rubric critique.
    """
    scores_summary = "\n".join(
        f"  {k}: {v}" for k, v in scores.items()
    )
    return f"""You are a friendly, encouraging Python programming tutor providing detailed feedback.

### PROBLEM:
{soal}

### EXPECTED OUTPUT:
{expected_out if expected_out else "(not specified)"}

### SIMULATED INPUT:
{simulated_in if simulated_in else "(none)"}

### STUDENT CODE:
```python
{code}
```

### SCORES ALREADY COMPUTED (do NOT change these):
{scores_summary}

{RUBRIC}

Based on the scores above, provide:
1. An encouraging overall_feedback (max 70 words, friendly English or Bahasa Indonesia).
2. The inferred_proficiency level (one of: Novice / Beginner / Advance Beginner / Advance / Competent / Expert).
3. A short reasoning for that proficiency (max 30 words).
4. Per-rubric critique: for each of the 6 aspects, if score < 80 quote the exact issue and how to fix it; if score >= 80 write "All good!".

Respond with ONLY valid JSON, no markdown fences, no extra text:
{{
  "feedback": "overall encouraging summary here",
  "inferred_proficiency": "level here",
  "reasoning": "why this level",
  "critique": {{
    "functionality": "critique or All good!",
    "code_style": "critique or All good!",
    "documentation": "critique or All good!",
    "logic": "critique or All good!",
    "syntax": "critique or All good!",
    "concept": "critique or All good!"
  }}
}}"""


# ═══════════════════════════════════════════════════════════════════════════════
#  LLM CALL HELPER
# ═══════════════════════════════════════════════════════════════════════════════
def _call_model(host: str, model: str, prompt: str,
                temperature: float = 0.0,
                max_tokens: int = 1024,
                top_p: float = 1.0) -> str:
    """Call any OpenAI-compatible endpoint and return raw text.
    
    Automatically selects the correct api_key:
      - LM Studio (port 1234) → 'lm-studio'
      - Ollama   (port 11434) → 'ollama'
    """
    api_key = "ollama" if "11434" in host else "lm-studio"
    client = OpenAI(base_url=host, api_key=api_key)
    response = client.chat.completions.create(
        model=model,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    return response.choices[0].message.content.strip()


# ═══════════════════════════════════════════════════════════════════════════════
#  JSON PARSING
# ═══════════════════════════════════════════════════════════════════════════════
def _parse_json_output(raw: str) -> dict | None:
    """Strip markdown fences, <think> blocks, then parse JSON."""
    raw = re.sub(r"<think>.*?</think>", "", raw, flags=re.DOTALL)
    raw = re.sub(r"```(?:json)?", "", raw).replace("```", "").strip()
    try:
        return json.loads(raw)
    except json.JSONDecodeError:
        pass
    # Try to extract first {...} block
    start = raw.find("{")
    if start == -1:
        return None
    depth = 0
    for i, ch in enumerate(raw[start:], start):
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                try:
                    return json.loads(raw[start:i + 1])
                except json.JSONDecodeError:
                    return None
    return None


def _normalize_scores(scores: dict) -> dict:
    """Clamp each score to 0-100 and ensure overall_score is recomputed."""
    rubric_keys = ["functionality", "code_style", "documentation",
                   "logic", "syntax", "concept"]
    for k in rubric_keys:
        v = scores.get(k, 0)
        # If model returned 0-10 scale, multiply by 10
        if isinstance(v, (int, float)) and v <= 10:
            v = v * 10
        scores[k] = max(0, min(100, int(v)))
    vals = [scores[k] for k in rubric_keys]
    scores["overall_score"] = round(sum(vals) / len(vals)) if vals else 0
    return scores


# ═══════════════════════════════════════════════════════════════════════════════
#  CORE EVALUATION (dual-model)
# ═══════════════════════════════════════════════════════════════════════════════
def run_dual_evaluation(
    soal: str, code: str, expected_out: str, simulated_in: str,
    # per-request overrides
    scoring_host: str = None, scoring_model: str = None,
    feedback_host: str = None, feedback_model: str = None,
    temperature: float = 0.0, max_tokens: int = 1024, top_p: float = 1.0,
) -> dict:
    s_host  = _ensure_v1(scoring_host  or SCORING_HOST)
    s_model = scoring_model  or SCORING_MODEL
    f_host  = _ensure_v1(feedback_host  or FEEDBACK_HOST)
    f_model = feedback_model or FEEDBACK_MODEL

    result = {
        "scores": {},
        "feedback": "",
        "inferred_proficiency": "N/A",
        "reasoning": "",
        "critique": {},
        "_meta": {
            "scoring_model_host":  s_host,
            "scoring_model_name":  s_model,
            "feedback_model_host": f_host,
            "feedback_model_name": f_model,
            "temperature": temperature,
            "max_tokens":  max_tokens,
            "top_p":       top_p,
        },
    }

    # ── CALL 1: CodeLlama → scores ────────────────────────────────────────────
    scoring_prompt = build_scoring_prompt(soal, code, expected_out, simulated_in)
    try:
        raw_scores = _call_model(s_host, s_model, scoring_prompt,
                                  temperature, max_tokens, top_p)
        result["_raw_scoring"] = raw_scores
        parsed_scores = _parse_json_output(raw_scores)
        if parsed_scores and "scores" in parsed_scores:
            result["scores"] = _normalize_scores(parsed_scores["scores"])
        elif parsed_scores:
            result["scores"] = _normalize_scores(parsed_scores)
        else:
            result["scores"] = {k: 0 for k in
                ["functionality", "code_style", "documentation",
                 "logic", "syntax", "concept", "overall_score"]}
            result["_scoring_parse_error"] = "Failed to parse scoring JSON"
    except Exception as exc:
        result["scores"] = {k: 0 for k in
            ["functionality", "code_style", "documentation",
             "logic", "syntax", "concept", "overall_score"]}
        result["_scoring_error"] = str(exc)

    # ── CALL 2: CodeGemma → feedback ─────────────────────────────────────────
    feedback_prompt = build_feedback_prompt(
        soal, code, expected_out, simulated_in, result["scores"]
    )
    try:
        raw_feedback = _call_model(f_host, f_model, feedback_prompt,
                                    temperature, max_tokens, top_p)
        result["_raw_feedback"] = raw_feedback
        parsed_feedback = _parse_json_output(raw_feedback)
        if parsed_feedback:
            result["feedback"]             = parsed_feedback.get("feedback", "")
            result["inferred_proficiency"] = parsed_feedback.get("inferred_proficiency", "N/A")
            result["reasoning"]            = parsed_feedback.get("reasoning", "")
            result["critique"]             = parsed_feedback.get("critique", {})
        else:
            result["feedback"] = "Feedback model returned an unparseable response."
            result["_feedback_parse_error"] = "Failed to parse feedback JSON"
    except Exception as exc:
        result["feedback"] = f"Feedback model error: {exc}"
        result["_feedback_error"] = str(exc)

    # Store prompts so the frontend can embed them in fine-tune samples
    result["_prompts"] = {
        "scoring_prompt":  scoring_prompt,
        "feedback_prompt": feedback_prompt,
    }
    return result


# ═══════════════════════════════════════════════════════════════════════════════
#  FLASK ROUTES
# ═══════════════════════════════════════════════════════════════════════════════

@app.route("/health", methods=["GET"])
def health():
    return jsonify({
        "status": "ok",
        "scoring_model":  SCORING_MODEL,
        "scoring_host":   SCORING_HOST,
        "feedback_model": FEEDBACK_MODEL,
        "feedback_host":  FEEDBACK_HOST,
    })


@app.route("/defaults", methods=["GET"])
def defaults():
    return jsonify({
        "scoring_model_host":  SCORING_HOST.replace("/v1", ""),
        "scoring_model_name":  SCORING_MODEL,
        "feedback_model_host": FEEDBACK_HOST.replace("/v1", ""),
        "feedback_model_name": FEEDBACK_MODEL,
        "temperature": 0.0,
        "max_tokens":  1024,
        "top_p":       1.0,
    })


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True, silent=True) or {}

    soal         = data.get("soal", "").strip()
    code         = data.get("code", "").strip()
    expected_out = data.get("expected_out", "").strip()
    simulated_in = data.get("simulated_in", "").strip()

    if not soal or not code:
        return jsonify({"error": "Fields 'soal' and 'code' are required."}), 400

    try:
        result = run_dual_evaluation(
            soal=soal,
            code=code,
            expected_out=expected_out,
            simulated_in=simulated_in,
            scoring_host=data.get("scoring_model_host"),
            scoring_model=data.get("scoring_model_name"),
            feedback_host=data.get("feedback_model_host"),
            feedback_model=data.get("feedback_model_name"),
            temperature=float(data.get("temperature", 0.0)),
            max_tokens=int(data.get("max_tokens", 1024)),
            top_p=float(data.get("top_p", 1.0)),
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


# ── QLoRA dataset routes ───────────────────────────────────────────────────────

@app.route("/save-sample", methods=["POST"])
def save_sample():
    """
    Append one Alpaca-format QLoRA training sample to the JSONL file.

    Expected body:
    {
      "source":      "scoring" | "feedback",
      "instruction": "<prompt sent to the model>",
      "output":      "<ideal model response>",
      "metadata":    { ... optional ... }
    }
    """
    data = request.get_json(force=True, silent=True) or {}
    instruction = data.get("instruction", "").strip()
    output_text = data.get("output", "").strip()
    source      = data.get("source", "scoring")
    metadata    = data.get("metadata", {})

    if not instruction or not output_text:
        return jsonify({"error": "'instruction' and 'output' are required."}), 400

    sample = {
        "instruction": instruction,
        "input": "",
        "output": output_text,
        "source": source,
        "metadata": {
            **metadata,
            "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
        },
    }

    try:
        with open(FINETUNE_PATH, "a", encoding="utf-8") as f:
            f.write(json.dumps(sample, ensure_ascii=False) + "\n")
        count = _count_samples()
        return jsonify({"saved": True, "total_samples": count,
                        "file": str(FINETUNE_PATH)})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/export-samples", methods=["GET"])
def export_samples():
    """Download the full finetune_samples.jsonl file."""
    if not FINETUNE_PATH.exists() or FINETUNE_PATH.stat().st_size == 0:
        return jsonify({"error": "No samples collected yet."}), 404
    return send_file(
        str(FINETUNE_PATH),
        mimetype="application/x-ndjson",
        as_attachment=True,
        download_name="finetune_samples.jsonl",
    )


@app.route("/clear-samples", methods=["DELETE"])
def clear_samples():
    """Wipe the JSONL file."""
    try:
        if FINETUNE_PATH.exists():
            FINETUNE_PATH.unlink()
        return jsonify({"cleared": True})
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/samples-count", methods=["GET"])
def samples_count():
    """Return number of collected samples."""
    return jsonify({"count": _count_samples(),
                    "file": str(FINETUNE_PATH)})


def _count_samples() -> int:
    if not FINETUNE_PATH.exists():
        return 0
    with open(FINETUNE_PATH, encoding="utf-8") as f:
        return sum(1 for line in f if line.strip())


# ─────────────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 5050))
    print(f"[api-simple] Scoring  model : {SCORING_MODEL} @ {SCORING_HOST}")
    print(f"[api-simple] Feedback model : {FEEDBACK_MODEL} @ {FEEDBACK_HOST}")
    print(f"[api-simple] Fine-tune file : {FINETUNE_PATH}")
    print(f"[api-simple] Listening on   : http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
