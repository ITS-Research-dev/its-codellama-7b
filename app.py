import json
import re
import os
import random
import requests
from openai import OpenAI
from dotenv import load_dotenv

load_dotenv()

# ─── SOAL UJIAN DATASET ───────────────────────────────────────────────────────
_DATASET_PATH = os.path.join(os.path.dirname(__file__), "datasets", "backend", "soal_ujian.json")

def _load_soal_ujian() -> list[dict]:
    """Load the soal_ujian dataset from disk (cached after first call)."""
    if not hasattr(_load_soal_ujian, "_cache"):
        try:
            with open(_DATASET_PATH, encoding="utf-8") as f:
                _load_soal_ujian._cache = json.load(f)
            print(f"[soal-ujian] Loaded {len(_load_soal_ujian._cache)} questions from dataset.")
        except FileNotFoundError:
            print(f"[soal-ujian] WARNING: Dataset not found at {_DATASET_PATH}")
            _load_soal_ujian._cache = []
        except json.JSONDecodeError as e:
            print(f"[soal-ujian] WARNING: Failed to parse dataset — {e}")
            _load_soal_ujian._cache = []
    return _load_soal_ujian._cache


def get_soal_ujian(soal_id: int | None = None, level: str | None = None) -> dict | list:
    """
    Retrieve exam questions from the soal_ujian dataset.

    Args:
        soal_id: If provided, return the single question with this id.
        level:   If provided, filter by difficulty ('easy', 'medium', 'hard').

    Returns:
        A single question dict (when soal_id given), a filtered list,
        or the full list when no arguments are supplied.
    """
    data = _load_soal_ujian()
    if soal_id is not None:
        matches = [q for q in data if q.get("id") == soal_id]
        return matches[0] if matches else {}
    if level:
        data = [q for q in data if q.get("level", "").lower() == level.lower()]
    return data


def get_random_soal(level: str | None = None) -> dict:
    """Return a random exam question, optionally filtered by level."""
    questions = get_soal_ujian(level=level)
    return random.choice(questions) if questions else {}

# ─── RUBRIC ──────────────────────────────────────────────────────────────────
RUBRIC = """
SCORING RUBRIC
(use integer multiples of 5 ONLY: 0,5,10,15,20,25,30,35,40,45,50,55,60,65,70,75,80,85,90,95,100)

FUNCTIONALITY:
  100=output perfect; 95=tiny edge case; 90=correct with minor weakness;
  85=mostly correct; 80=correct for main case; 75=mostly works but inconsistent;
  70=mostly correct with visible issues; 65=partially correct; 60=core runs incomplete;
  55=significant issues; 50=half requirements met; 45=some logic runs;
  40=wrong but shows basic understanding; 35=little progress; 30=runs with major errors;
  25=very limited correctness; 20=runs but mostly wrong; 15=barely functional;
  10=very serious errors; 5=almost unusable; 0=crashes or no valid output.

CODE_STYLE:
  100=perfect PEP8; 95=nearly perfect; 90=very clean; 85=good naming and format;
  80=minor style issues; 75=generally readable; 70=some issues; 65=readability suffers;
  60=understandable but errors; 55=many inconsistencies; 50=average; 45=hard to read;
  40=messy; 35=very inconsistent; 30=little readability; 25=very messy; 20=chaotic;
  15=nearly unreadable; 10=major issues; 5=almost impossible; 0=unreadable.

DOCUMENTATION:
  100=full docstrings+comments; 95=nearly complete; 90=very good; 85=mostly documented;
  80=most code documented; 75=decent but gaps; 70=present but uneven;
  65=inconsistent quality; 60=basic comments; 55=very minimal; 50=limited;
  45=very few; 40=1-2 comments; 35=very weak; 30=almost none; 25=little benefit;
  20=unclear; 15=almost none; 10=very poor; 5=fragments; 0=none.

LOGIC:
  100=optimal and elegant; 95=very good; 90=very strong; 85=correct minor gap;
  80=correct but improvable; 75=mostly correct; 70=works with weaknesses;
  65=inconsistencies; 60=mostly correct with flaws; 55=significant weakness;
  50=partial understanding; 45=many errors; 40=partially correct;
  35=weak structure; 30=minimal flow; 25=major misunderstanding;
  20=fundamental error; 15=barely makes sense; 10=almost nothing correct;
  5=random; 0=no logical structure.

SYNTAX:
  100=perfect idiomatic Python; 95=nearly flawless; 90=very good; 85=correct minor issues;
  80=correct not fully idiomatic; 75=slight inconsistencies; 70=mostly valid;
  65=some errors program runs; 60=small non-breaking errors; 55=noticeable errors;
  50=frequent weaknesses; 45=occasional runtime issues; 40=1-2 errors;
  35=many errors; 30=needs many fixes; 25=frequently fails; 20=many errors;
  15=barely runnable; 10=very serious; 5=almost impossible; 0=cannot parse.

CONCEPT:
  100=best concept applied; 95=very good choice; 90=very appropriate;
  85=correct with minor gap; 80=correct but improvable; 75=adequately applied;
  70=mostly appropriate; 65=partially fitting; 60=runs but not ideal;
  55=weak application; 50=moderate understanding; 45=confused in places;
  40=misused; 35=very weak; 30=minimal; 25=major misunderstanding;
  20=wrong concept; 15=nearly irrelevant; 10=very serious errors; 5=almost none; 0=none.

PROFICIENCY LEVEL (choose exactly one):
  Expert          : average >= 90, nothing below 85 — elegant, idiomatic, production-ready
  Competent       : average >= 75, nothing below 65 — solid and reliable, minor gaps only
  Advance         : average >= 60, nothing below 50 — generally correct with clear room to grow
  Advance Beginner: average >= 45, nothing below 35 — basic concepts present, inconsistencies remain
  Beginner        : average >= 30, nothing below 20 — limited grasp, frequent errors
  Novice          : average <  30 or fundamental errors present — very early stage
"""

# ─── CONFIG ───────────────────────────────────────────────────────────────────
DEBUG = os.getenv("DEBUG_ASSESSOR", "1") == "1"   # set DEBUG_ASSESSOR=0 in .env to silence
def _normalize_host(host: str) -> str:
    host = host.rstrip("/")
    if not host.endswith("/v1"):
        host += "/v1"
    return host

MODEL_HOST = _normalize_host(os.getenv("LMSTUDIO_HOST", os.getenv("MODEL_HOST", "http://localhost:1234/v1")))
MODEL_NAME = os.getenv("MODEL_NAME", "codellama-7b-instruct")

lm_client = OpenAI(base_url=MODEL_HOST, api_key="lm-studio")


def _dbg(label, text):
    if DEBUG:
        print(f"\n----- DEBUG [{label}] -----")
        print(text)
        print(f"----- END [{label}] -----\n")


# ─── LM STUDIO HELPERS ────────────────────────────────────────────────────────
def _call_lmstudio(model_name, system_msg, user_msg):
    """Call LM Studio and return the response text."""
    try:
        url = MODEL_HOST + "/chat/completions"
        payload = {
            "model": model_name or MODEL_NAME,   # FIX: actually use the passed-in model_name
            "messages": [
                {"role": "system", "content": system_msg},
                {"role": "user", "content": user_msg},
            ],
            "temperature": 0.1
        }
        res = requests.post(url, json=payload, headers={"Content-Type": "application/json"})
        res.raise_for_status()
        data = res.json()

        if "choices" in data and data["choices"]:
            content = data["choices"][0]["message"]["content"]
            _dbg("LLM RAW RESPONSE", content)
            return content
        else:
            print("Error: LM Studio did not return 'choices' in the JSON!")
            _dbg("LLM RAW RESPONSE (no choices)", json.dumps(data))
            return ""
    except Exception as e:
        print(f"Error calling LLM via HTTP request: {e}")
        return ""


def _extract_json(text):
    """Extract the first valid JSON object using bracket counting."""
    start = text.find("{")
    if start == -1:
        return None
    depth, end, in_str, esc = 0, -1, False, False
    for i, ch in enumerate(text[start:], start):
        if esc:
            esc = False; continue
        if ch == "\\" and in_str:
            esc = True; continue
        if ch == '"':
            in_str = not in_str; continue
        if in_str:
            continue
        if ch == "{":
            depth += 1
        elif ch == "}":
            depth -= 1
            if depth == 0:
                end = i + 1; break
    return text[start:end] if end != -1 else None


def _sanitize_json(text):
    if not text or not text.strip():
        return None

    try:
        return json.loads(text)
    except json.JSONDecodeError:
        pass

    extracted = _extract_json(text)
    if extracted:
        try:
            return json.loads(extracted)
        except json.JSONDecodeError:
            pass

    if extracted:
        chars = []
        in_str = esc = False
        for ch in extracted:
            if esc:
                chars.append(ch); esc = False; continue
            if ch == "\\" and in_str:
                chars.append(ch); esc = True; continue
            if ch == '"':
                in_str = not in_str; chars.append(ch); continue
            if in_str and ch in ('\n', '\r', '\t'):
                chars.append(' '); continue
            chars.append(ch)
        try:
            return json.loads(''.join(chars))
        except json.JSONDecodeError:
            pass

    src = extracted or text

    def rfield(pattern):
        m = re.search(pattern, src, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else ""

    def rscore(key):
        m = re.search(rf'"{key}"\s*:\s*(\d+)', src, re.IGNORECASE)
        return int(m.group(1)) if m else 0

    def rcrit(key):
        m = re.search(rf'"{key}"\s*:\s*"([^"]*?)"', src, re.DOTALL | re.IGNORECASE)
        return m.group(1).strip() if m else "See analysis above."

    skeys = ["functionality", "code_style", "documentation", "logic", "syntax", "concept"]
    scores = {k: rscore(k) for k in skeys}
    scores["overall_score"] = round(sum(scores[k] for k in skeys) / len(skeys)) if skeys else 0

    return {
        "feedback":             rfield(r'"feedback"\s*:\s*"([^"]*?)"') or "See analysis.",
        "inferred_proficiency": rfield(r'"inferred_proficiency"\s*:\s*"([^"]*?)"') or "N/A",
        "reasoning":            rfield(r'"reasoning"\s*:\s*"([^"]*?)"') or "",
        "scores":               scores,
        "critique":             {k: rcrit(k) for k in skeys},
        "_fallback_strategy":   "regex",
    }


def normalize_scores(scores: dict) -> dict:
    """If all scores <= 10, auto-scale to 100."""
    skeys = ['functionality', 'code_style', 'documentation', 'logic', 'syntax', 'concept']
    values = [scores.get(k, 0) for k in skeys]
    if values and max(values) <= 10:
        for k in skeys:
            scores[k] = scores.get(k, 0) * 10
    scores['overall_score'] = round(sum(scores.get(k, 0) for k in skeys) / len(skeys))
    return scores


# ─── SEMANTIC GATE ───────────────────────────────────────────────────────────
# Regex that matches any real algorithmic construct.
# Two or more hits in a submission = definitely real logic, not a hardcoded answer.
_LOGIC_RE = re.compile(
    r'\b(?:def |for |while |if |elif |else:|class |import |from |lambda |try:|except\b|return )'
)


def _heuristic_code_check(code: str) -> tuple:
    """
    Fast, deterministic pre-screen for hardcoded submissions.
    Returns (result, reason) where result is:
      True  → clearly real code, auto-PASS (skip LLM)
      None  → ambiguous, defer to LLM
    """
    logic_hits = _LOGIC_RE.findall(code)
    n_logic = len(logic_hits)

    if n_logic >= 2:
        # Has multiple algorithmic constructs — definitely real code
        return True, f"Heuristic auto-pass: {n_logic} logic constructs detected (real algorithmic code)."

    # Count non-blank, non-comment lines
    real_lines = [l for l in code.strip().splitlines()
                  if l.strip() and not l.strip().startswith('#')]

    if len(real_lines) <= 3 and n_logic == 0:
        # Suspiciously short with zero logic — let LLM decide
        return None, "Short code with no logic constructs — deferring to LLM check."

    # Longer code with at least one logic keyword
    return True, f"Heuristic auto-pass: code has {len(real_lines)} lines with algorithmic structure."


def _check_semantic(model_name, system_msg, soal, expected_output, code):
    """
    Returns (passed: bool, reason: str, raw_text: str).
    Runs a fast heuristic first; only calls the LLM for suspiciously short
    submissions where hardcoding is actually plausible.
    Defaults to PASS on any ambiguity so a flaky/small model
    doesn't zero out legitimate student work.
    """
    # ── Step 1: heuristic pre-screen (no LLM cost) ───────────────────────────
    heuristic_result, heuristic_reason = _heuristic_code_check(code)
    if heuristic_result is True:
        _dbg("SEMANTIC HEURISTIC", f"Auto-passed: {heuristic_reason}")
        return True, heuristic_reason, ""
    # heuristic_result is None → fall through to LLM check for short/ambiguous code
    prompt_semantic = f"""You are a Python code reviewer. Your only job is to detect whether a student CHEATED by hardcoding the answer instead of implementing real logic.

Problem the student must solve:
{soal}

Expected output (for reference only — used to check for literal hardcoding):
{expected_output}

Student code:
{code}

Answer the following THREE questions with YES, NO, or N-A. Then give one sentence explanation.

1. SOLVES_PROBLEM:
   - Answer YES if the code implements the actual algorithm or logic that the problem requires, even if it uses predefined/hardcoded test data (a list, variable, etc.) instead of input().
   - Answer NO ONLY if the code literally prints or returns the answer with NO computation at all (e.g. print("15") for a sum problem, or return [1,2,3,4] for a sort problem).
   - IMPORTANT: having a predefined list or variable is NOT cheating. Cheating means the output is hardwired with zero logic.

2. USES_INPUT:
   - Answer YES if the code reads user input with input() and the problem EXPLICITLY asks for user input.
   - Answer N-A if the problem does NOT explicitly require user input, or if demonstrating with predefined data is a valid approach (e.g. algorithm demos, function definitions).
   - Answer NO only if the problem clearly says "menerima masukan pengguna" / "gunakan input()" but the code never calls input().
   - NEVER let a N-A answer here cause a FAIL verdict on its own.

3. CORRECT_APPROACH: Is the general algorithm or approach appropriate for solving this problem?

Rules for SEMANTIC_VERDICT:
   - Output PASS if SOLVES_PROBLEM is YES and CORRECT_APPROACH is YES.
   - Output FAIL ONLY if SOLVES_PROBLEM is NO (literal hardcoded answer detected).
   - USES_INPUT: N-A alone is NOT a reason to FAIL.

Respond in this exact format:
SOLVES_PROBLEM: YES/NO — explanation
USES_INPUT: YES/NO/N-A — explanation
CORRECT_APPROACH: YES/NO — explanation
SEMANTIC_VERDICT: PASS or FAIL
SEMANTIC_REASON: one sentence summary of why it passes or fails
"""
    semantic_text = _call_lmstudio(model_name, system_msg, prompt_semantic)

    if not semantic_text or not semantic_text.strip():
        # No response at all -> don't punish the student for an LLM/connection hiccup
        return True, "Semantic check produced no output; auto-passed.", semantic_text

    verdict_match = re.search(r'SEMANTIC_VERDICT\s*:\s*(PASS|FAIL)', semantic_text, re.IGNORECASE)
    reason_match = re.search(r'SEMANTIC_REASON\s*:\s*(.+)', semantic_text, re.IGNORECASE)

    if verdict_match:
        passed = verdict_match.group(1).upper() == "PASS"
    else:
        # No explicit verdict line found. Instead of guessing from stray words,
        # look specifically at the SOLVES_PROBLEM answer, which is the actual
        # hardcode-detector. Only fail if it explicitly says NO.
        solves_match = re.search(r'SOLVES_PROBLEM\s*:\s*(YES|NO)', semantic_text, re.IGNORECASE)
        if solves_match:
            passed = solves_match.group(1).upper() == "YES"
        else:
            # Truly unparseable -> auto-pass, let the rubric scoring do the real work
            passed = True

    reason = reason_match.group(1).strip() if reason_match else (
        semantic_text.strip().splitlines()[0] if semantic_text.strip() else "N/A"
    )
    return passed, reason, semantic_text


# ─── MAIN EVALUATION ─────────────────────────────────────────────────────────
def evaluate_and_classify(code, soal, expected_output, simulated_input, execution_result=None):
    model_name = MODEL_NAME
    system_msg = "You are a Python programming assessor. Be strict and objective."

    error_fallback = {
        "feedback": "A technical error occurred. Please try again.",
        "inferred_proficiency": "N/A",
        "reasoning": "System error.",
        "scores": {k: 0 for k in ["functionality", "code_style", "documentation",
                                   "logic", "syntax", "concept", "overall_score"]},
        "critique": {k: "N/A" for k in ["functionality", "code_style", "documentation",
                                         "logic", "syntax", "concept"]},
    }

    raw_text = ""
    try:
        # ── CALL 0: semantic validation ──────────────────────────────────────
        semantic_pass, semantic_reason, semantic_text = _check_semantic(
            model_name, system_msg, soal, expected_output, code
        )

        if not semantic_pass:
            zero_scores = {k: 0 for k in ["functionality", "code_style", "documentation",
                                           "logic", "syntax", "concept", "overall_score"]}
            return {
                "feedback": (
                    f"Your output matched, but your code does not actually solve the problem. "
                    f"{semantic_reason} "
                    f"Avoid hardcoding answers — implement the real logic."
                ),
                "inferred_proficiency": "Novice",
                "reasoning": "Code does not genuinely solve the problem.",
                "scores": zero_scores,
                "critique": {
                    "functionality": f"Output matched but code doesn't solve the problem: {semantic_reason}",
                    "code_style": "Not evaluated — code must solve the problem first.",
                    "documentation": "Not evaluated — code must solve the problem first.",
                    "logic": f"Logic does not address the problem requirements: {semantic_reason}",
                    "syntax": "Not evaluated — code must solve the problem first.",
                    "concept": "Not evaluated — code must solve the problem first.",
                },
                "_semantic_failed": True,
                "_semantic_raw": semantic_text,
            }

        # ── CALL 1: narrative analysis ────────────────────────────────────────
        if execution_result:
            actual_out = execution_result["actual_output"]
            actual_err = execution_result["actual_error"]
            passed = execution_result.get("passed", False)
            exec_ctx = f"""
CODE EXECUTION RESULT (determined by actually running the code — treat this as ground truth):
  Actual Output : {repr(actual_out)}
  Expected Output: {repr(expected_output)}
  Runtime Error  : {actual_err if actual_err else "None"}
  Output Match   : {"YES — output matches expected" if passed else "NO — output does NOT match expected"}

IMPORTANT RULES based on execution result:
- If Output Match is NO, functionality score MUST be below 60. Do NOT give high functionality scores when the output is wrong.
- If Runtime Error is not None, functionality score MUST be 0.
- If Output Match is YES, you may give functionality >= 80 depending on code quality.
"""
        else:
            exec_ctx = "No execution result available. Evaluate functionality based on code logic only."

        prompt_analysis = f"""Analyze this Python student code for the problem below.
Problem : {soal}
Expected: {expected_output}
Input   : {simulated_input or "None"}

{exec_ctx}

{RUBRIC}

For each of the 6 aspects (functionality, code_style, documentation, logic, syntax, concept):
- Give a score (multiple of 5, from 0 to 100) based strictly on the rubric.
- If score < 80, quote the exact line or issue and explain clearly how to fix it.
- If score >= 80, write "All good!"

Also provide:
- overall_feedback: short encouraging summary in friendly English (max 60 words)
- proficiency: one of Novice / Beginner / Advance Beginner / Advance / Competent / Expert
- reasoning: why this proficiency level, in friendly English (max 30 words)

Student code:
{code}
"""
        analysis_text = _call_lmstudio(model_name, system_msg, prompt_analysis)

        if not analysis_text or not analysis_text.strip():
            error_fallback["feedback"] = "AI produced no analysis text (empty response from LM Studio)."
            return error_fallback

        # ── CALL 2: convert to JSON ───────────────────────────────────────────
        prompt_json = f"""Convert the following analysis into a single JSON object.
RULES:
- Output ONLY the raw JSON. No markdown, no explanation, no text before or after.
- Every string value must be on ONE line only. No newlines inside string values.
- Use double quotes for all strings. Escape any double quote inside a value with backslash.
- Do NOT add trailing commas.

Analysis:
{analysis_text}

Required JSON format (fill actual values, keep all keys exactly as shown):
{{
  "feedback": "overall_feedback here",
  "inferred_proficiency": "proficiency level here",
  "reasoning": "reasoning here",
  "scores": {{
    "functionality": 0,
    "code_style": 0,
    "documentation": 0,
    "logic": 0,
    "syntax": 0,
    "concept": 0,
    "overall_score": 0
  }},
  "critique": {{
    "functionality": "critique or All good!",
    "code_style": "critique or All good!",
    "documentation": "critique or All good!",
    "logic": "critique or All good!",
    "syntax": "critique or All good!",
    "concept": "critique or All good!"
  }}
}}
overall_score = average of the 6 aspect scores, rounded to nearest integer.
"""
        raw_text = _call_lmstudio(model_name, system_msg, prompt_json)

        raw_text = re.sub(r"<think>.*?</think>", "", raw_text, flags=re.DOTALL)
        raw_text = re.sub(r"```(?:json)?", "", raw_text).replace("```", "").strip()

        if not raw_text:
            error_fallback["feedback"] = "AI returned no output. Please submit again."
            error_fallback["_raw_response"] = "(empty)"
            return error_fallback

        result = _sanitize_json(raw_text)
        if result is None:
            error_fallback["feedback"] = "Could not parse AI response after all attempts."
            error_fallback["_raw_response"] = raw_text
            return error_fallback

        result["scores"] = normalize_scores(result.get("scores", {}))
        return result

    except json.JSONDecodeError as e:
        result = _sanitize_json(raw_text)
        if result:
            result["scores"] = normalize_scores(result.get("scores", {}))
            return result
        error_fallback["feedback"] = f"Invalid JSON format: {str(e)}"
        error_fallback["_raw_response"] = raw_text
        return error_fallback

    except Exception as e:
        error_fallback["feedback"] = f"Technical error: {str(e)}"
        error_fallback["_raw_response"] = raw_text
        return error_fallback


if __name__ == "__main__":
    # ── Test question: Binary Search (soal id=22, medium difficulty) ──────────
    soal_dummy = (
        "Buatlah program Python untuk mencari nama seorang murid pada daftar nama "
        "menggunakan Linear Search dan Binary Search. Bandingkan efisiensi kedua "
        "algoritma dan jelaskan kapan masing-masing lebih tepat digunakan."
    )
    code_dummy = """\
def linear_search(names, target):
    comparisons = 0
    for i, name in enumerate(names):
        comparisons += 1
        if name == target:
            return i, comparisons
    return -1, comparisons

def binary_search(names, target):
    names_sorted = sorted(names)
    low, high = 0, len(names_sorted) - 1
    comparisons = 0
    while low <= high:
        mid = (low + high) // 2
        comparisons += 1
        if names_sorted[mid] == target:
            return mid, comparisons
        elif names_sorted[mid] < target:
            low = mid + 1
        else:
            high = mid - 1
    return -1, comparisons

names = ['Andi', 'Budi', 'Citra', 'Dewi', 'Eka']
target = 'Citra'

ls_idx, ls_cmp = linear_search(names, target)
bs_idx, bs_cmp = binary_search(sorted(names), target)

print(f"Linear Search: ditemukan di indeks {ls_idx} ({ls_cmp} kali perbandingan)")
print(f"Binary Search: ditemukan di indeks {bs_idx} ({bs_cmp} kali perbandingan)")
print("Linear Search cocok untuk data kecil/tidak terurut.")
print("Binary Search lebih efisien untuk data besar yang sudah terurut.")
"""
    # expected_out matches what the code above actually prints
    expected_out_dummy = (
        "Linear Search: ditemukan di indeks 2 (3 kali perbandingan)\n"
        "Binary Search: ditemukan di indeks 2 (2 kali perbandingan)\n"
        "Linear Search cocok untuk data kecil/tidak terurut.\n"
        "Binary Search lebih efisien untuk data besar yang sudah terurut."
    )
    simulated_in_dummy = ""

    print("Testing codellama evaluation — Binary Search topic...")
    res = evaluate_and_classify(code_dummy, soal_dummy, expected_out_dummy, simulated_in_dummy)
    print(json.dumps(res, indent=2))