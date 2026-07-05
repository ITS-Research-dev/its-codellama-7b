"""
Scraper sederhana untuk CodingBat Python (https://codingbat.com/python)

Yang diambil per soal:
- section     : nama section, misal "Warmup-1"
- name        : nama fungsi/soal, misal "sleep_in"
- url         : link ke halaman soal
- description : teks soal
- examples    : list contoh input -> output, misal ["sleep_in(False, False) -> True", ...]
- solution    : kode solusi (kalau ada di halaman, biasanya dari div#results > pre)

Cara pakai:
    python codingbat_scraper.py

Hasil disimpan ke file "codingbat_python.json"

Struktur ini gampang dimodif:
- Mau tambah/kurangi section -> edit list SECTIONS di bawah.
- Mau ambil field lain (misal difficulty) -> tambahin di fungsi scrape_problem().
"""

import re
import json
import time
import requests
from urllib.parse import unquote
from bs4 import BeautifulSoup

BASE_URL = "https://codingbat.com"

# Kalau mau scrape semua section otomatis, biarkan None.
# Kalau mau scrape section tertentu saja, isi manual, contoh:
# SECTIONS = ["Warmup-1", "String-1"]
SECTIONS = None

HEADERS = {
    "User-Agent": "Mozilla/5.0 (compatible; simple-scraper/1.0)"
}

MAX_RETRIES = 3
RETRY_DELAY = 3  # detik, jeda sebelum coba lagi


def get_soup(url: str) -> BeautifulSoup:
    """Ambil HTML dari url dan parse jadi BeautifulSoup object.
    Kalau gagal konek (timeout/putus), coba ulang beberapa kali dulu."""
    last_error = None
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, headers=HEADERS, timeout=20)
            resp.raise_for_status()
            return BeautifulSoup(resp.text, "html.parser")
        except requests.exceptions.RequestException as e:
            last_error = e
            print(f"    [retry {attempt}/{MAX_RETRIES}] gagal ambil {url}: {e}")
            time.sleep(RETRY_DELAY)
    # kalau semua percobaan gagal, lempar error terakhir ke pemanggil
    raise last_error


def get_sections() -> list[str]:
    """Ambil daftar nama section dari halaman utama, contoh: Warmup-1, String-1, dst."""
    soup = get_soup(f"{BASE_URL}/python")
    sections = []
    for a in soup.find_all("a", href=True):
        m = re.match(r"^/python/([A-Za-z0-9\-]+)$", a["href"])
        if m:
            sections.append(m.group(1))
    # buang duplikat, jaga urutan
    return list(dict.fromkeys(sections))


def get_problem_links(section: str) -> list[dict]:
    """Ambil semua link soal (nama + url) di dalam satu section."""
    soup = get_soup(f"{BASE_URL}/python/{section}")
    problems = []
    for a in soup.find_all("a", href=True):
        m = re.match(r"^/prob/(p\d+)$", a["href"])
        if m:
            problems.append({
                "name": a.get_text(strip=True),
                "url": f"{BASE_URL}/prob/{m.group(1)}",
            })
    return problems


def scrape_problem(url: str) -> dict:
    """Ambil deskripsi soal + contoh + solusi (kalau ada) dari satu halaman soal."""
    soup = get_soup(url)

    # Cari <td> yang isinya mengandung tanda panah "->" atau "→"
    # itu adalah cell yang berisi soal + contoh.
    target_td = None
    for td in soup.find_all("td"):
        text = td.get_text()
        if "→" in text or "->" in text:
            target_td = td
            break

    description = ""
    examples = []

    if target_td:
        # ambil teks baris per baris (setiap <br> jadi baris baru)
        lines = target_td.get_text(separator="\n").split("\n")
        lines = [l.strip() for l in lines if l.strip()]

        desc_lines = []
        for line in lines:
            line = line.replace("→", "->")
            # pola contoh: nama_fungsi(...) -> hasil
            if re.match(r"^[a-zA-Z_]\w*\(.*\)\s*->\s*.+$", line):
                examples.append(line)
            elif not examples:
                # selama belum ketemu contoh pertama, anggap ini bagian deskripsi
                desc_lines.append(line)

        description = " ".join(desc_lines)

    # Cari solusi. Solusinya sebenarnya sudah ada di HTML mentah, tersimpan
    # di dalam atribut onclick tombol "Show Solution", dalam bentuk:
    #   unescape("def sleep_in(...):%0a  if ...%0a  ...")
    # %0a dkk itu url-encoding, jadi tinggal di-decode pakai unquote().
    solution = ""
    show_solution_btn = soup.find(
        "button", onclick=re.compile(r"unescape\(")
    )
    if show_solution_btn:
        onclick = show_solution_btn.get("onclick", "")
        m = re.search(r'unescape\("(.*?)"\s*\+\s*"', onclick)
        if m:
            solution = unquote(m.group(1))

    return {
        "description": description,
        "examples": examples,
        "solution": solution,
    }


def main():
    sections = SECTIONS if SECTIONS else get_sections()
    print(f"Section yang akan discrape: {sections}")

    all_data = []

    for section in sections:
        print(f"\n=== Scraping section: {section} ===")
        problems = get_problem_links(section)

        for p in problems:
            print(f"  - {p['name']} ({p['url']})")
            try:
                detail = scrape_problem(p["url"])
            except Exception as e:
                print(f"    [!] dilewati, tetap gagal setelah beberapa kali coba: {e}")
                continue

            all_data.append({
                "section": section,
                "name": p["name"],
                "url": p["url"],
                "description": detail["description"],
                "examples": detail["examples"],
                "solution": detail["solution"],
            })

            time.sleep(0.5)  # jaga-jaga biar tidak membebani server

            # simpan progres tiap kali dapat 1 soal, biar kalau tiba-tiba
            # berhenti di tengah jalan, data yang sudah didapat tidak hilang
            with open("codingbat_python.json", "w", encoding="utf-8") as f:
                json.dump(all_data, f, indent=2, ensure_ascii=False)

    print(f"\nSelesai. Total {len(all_data)} soal disimpan ke codingbat_python.json")


if __name__ == "__main__":
    main()