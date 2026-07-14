import os
import json
from flask import Flask, request, jsonify
from flask_cors import CORS
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

app = Flask(__name__)
CORS(app)  # allow the standalone index.html to call this API

# ── Default LM Studio connection (can be overridden per-request) ─────────────
DEFAULT_HOST  = os.getenv("LMSTUDIO_HOST", "http://localhost:1234/v1")
DEFAULT_MODEL = os.getenv("MODEL_NAME",    "codellama-7b-instruct")


def build_evaluation_prompt(soal, code, expected_out, simulated_in,
                             custom_instructions: str = "") -> str:
    base = f"""Kamu adalah asisten evaluasi kode Python untuk sistem pembelajaran pemrograman.

Tugas kamu: evaluasi apakah kode Python berikut menjawab soal dengan benar, lalu klasifikasikan hasilnya.

### SOAL:
{soal}

### KODE JAWABAN SISWA:
```python
{code}
```

### INPUT SIMULASI (jika ada):
{simulated_in if simulated_in else "(tidak ada input)"}

### OUTPUT YANG DIHARAPKAN:
{expected_out}

### INSTRUKSI:
1. Jalankan secara logis kode di atas (mental execution), bandingkan hasil keluarannya dengan OUTPUT YANG DIHARAPKAN.
2. Nilai apakah kode tersebut benar secara fungsional, efisien, dan sesuai dengan konsep yang diminta soal.
3. Berikan penjelasan singkat mengenai kebenaran logika, kompleksitas waktu (Big-O) jika relevan, dan gaya penulisan kode.
4. Klasifikasikan hasil evaluasi ke dalam salah satu kategori: "Benar", "Salah", atau "Sebagian Benar".
"""
    if custom_instructions and custom_instructions.strip():
        base += f"\n### INSTRUKSI TAMBAHAN:\n{custom_instructions.strip()}\n"

    base += """
Jawab HANYA dalam format JSON berikut, tanpa teks tambahan:
{
  "is_correct": true/false,
  "classification": "Benar" | "Salah" | "Sebagian Benar",
  "explanation": "penjelasan singkat dalam Bahasa Indonesia",
  "complexity_analysis": "penjelasan singkat kompleksitas algoritma jika relevan"
}
"""
    return base


def evaluate_and_classify(code, soal, expected_out, simulated_in,
                           model_host: str  = DEFAULT_HOST,
                           model_name: str  = DEFAULT_MODEL,
                           temperature: float = 0.0,
                           max_tokens: int  = 1024,
                           top_p: float     = 1.0,
                           custom_instructions: str = "") -> dict:
    if not model_host.rstrip("/").endswith("/v1"):
        model_host = model_host.rstrip("/") + "/v1"

    client = OpenAI(base_url=model_host, api_key="lm-studio")
    prompt = build_evaluation_prompt(soal, code, expected_out, simulated_in,
                                     custom_instructions)

    response = client.chat.completions.create(
        model=model_name,
        messages=[{"role": "user", "content": prompt}],
        temperature=temperature,
        max_tokens=max_tokens,
        top_p=top_p,
    )
    raw_output = response.choices[0].message.content.strip()

    if raw_output.startswith("```"):
        raw_output = raw_output.strip("`")
        if raw_output.startswith("json"):
            raw_output = raw_output[4:].strip()

    try:
        result = json.loads(raw_output)
    except json.JSONDecodeError:
        result = {
            "is_correct": None,
            "classification": "Unknown",
            "explanation": "Gagal mem-parsing output model sebagai JSON.",
            "complexity_analysis": None,
            "raw_output": raw_output,
        }

    result["_meta"] = {
        "model_host": model_host,
        "model_name": model_name,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
    }
    return result


@app.route("/health", methods=["GET"])
def health():
    return jsonify({"status": "ok", "default_model": DEFAULT_MODEL})


@app.route("/evaluate", methods=["POST"])
def evaluate():
    data = request.get_json(force=True, silent=True) or {}

    soal         = data.get("soal",         "").strip()
    code         = data.get("code",         "").strip()
    expected_out = data.get("expected_out", "").strip()
    simulated_in = data.get("simulated_in", "").strip()

    if not soal or not code:
        return jsonify({"error": "Fields 'soal' and 'code' are required."}), 400

    model_host   = data.get("model_host",   DEFAULT_HOST)
    model_name   = data.get("model_name",   DEFAULT_MODEL)
    temperature  = float(data.get("temperature",  0.0))
    max_tokens   = int(data.get("max_tokens",     1024))
    top_p        = float(data.get("top_p",        1.0))
    custom_instructions = data.get("custom_instructions", "")

    try:
        result = evaluate_and_classify(
            code=code,
            soal=soal,
            expected_out=expected_out,
            simulated_in=simulated_in,
            model_host=model_host,
            model_name=model_name,
            temperature=temperature,
            max_tokens=max_tokens,
            top_p=top_p,
            custom_instructions=custom_instructions,
        )
        return jsonify(result)
    except Exception as exc:
        return jsonify({"error": str(exc)}), 500


@app.route("/defaults", methods=["GET"])
def defaults():
    return jsonify({
        "model_host":  DEFAULT_HOST,
        "model_name":  DEFAULT_MODEL,
        "temperature": 0.0,
        "max_tokens":  1024,
        "top_p":       1.0,
    })


if __name__ == "__main__":
    port = int(os.getenv("API_PORT", 5050))
    print(f"[api-simple] Listening on http://localhost:{port}")
    app.run(host="0.0.0.0", port=port, debug=True)
