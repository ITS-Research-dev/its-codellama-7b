import os
import json
from dotenv import load_dotenv
from openai import OpenAI

load_dotenv()

client = OpenAI(
    base_url=os.getenv("LMSTUDIO_HOST", "http://localhost:1234/v1"),
    api_key="lm-studio"
)

def build_evaluation_prompt(soal, code, expected_out, simulated_in):
    prompt = f"""Kamu adalah asisten evaluasi kode Python untuk sistem pembelajaran pemrograman.

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

Jawab HANYA dalam format JSON berikut, tanpa teks tambahan:
{{
  "is_correct": true/false,
  "classification": "Benar" | "Salah" | "Sebagian Benar",
  "explanation": "penjelasan singkat dalam Bahasa Indonesia",
  "complexity_analysis": "penjelasan singkat kompleksitas algoritma jika relevan"
}}
"""
    return prompt


def evaluate_and_classify(code, soal, expected_out, simulated_in):
    prompt = build_evaluation_prompt(soal, code, expected_out, simulated_in)

    response = client.chat.completions.create(
        model=os.getenv("MODEL_NAME", "codellama-7b-instruct"),
        messages=[
            {"role": "user", "content": prompt}
        ],
        temperature=0
    )

    raw_output = response.choices[0].message.content.strip()

    # CodeLlama-7B sometimes wraps JSON in ```json ... ``` fences — strip them
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
            "raw_output": raw_output
        }

    return result


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