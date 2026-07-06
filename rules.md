--------------------------------------------------------------------
CONTOH (FEW-SHOT EXAMPLE) — gunakan format ini sebagai acuan ketat
--------------------------------------------------------------------

### INPUT CONTOH ###

RUBRIK:
{
  "bobot": {"sintaks_logika": 0.35, "keterbacaan": 0.20, "efisiensi": 0.20, "capaian_pembelajaran": 0.25},
  "kriteria": {
    "sintaks_logika": "Kode berjalan tanpa error dan mengembalikan hasil faktorial yang benar untuk semua n >= 0",
    "keterbacaan": "Penamaan fungsi/variabel jelas, mengikuti PEP-8, docstring ada bila perlu",
    "efisiensi": "Tidak ada komputasi berulang yang tidak perlu, kompleksitas sesuai kebutuhan tugas",
    "capaian_pembelajaran": "Menunjukkan pemahaman konsep rekursi (base case dan recursive case) sesuai CPL"
  }
}

CONTOH_SOLUSI:
def faktorial(n):
    """Menghitung faktorial n secara rekursif."""
    if n <= 1:
        return 1
    return n * faktorial(n - 1)

CAPAIAN_PEMBELAJARAN:
"Peserta didik mampu menerapkan konsep rekursi (base case dan recursive case) untuk menyelesaikan masalah komputasi sederhana."

RIWAYAT_KESALAHAN_SISWA:
"Siswa sebelumnya sering lupa menambahkan base case pada fungsi rekursif."

SOAL:
"Buatlah fungsi Python bernama faktorial(n) yang menghitung nilai faktorial dari sebuah bilangan bulat non-negatif n menggunakan rekursi."

KODE_SISWA:
def faktorial(n):
    hasil = 1
    for i in range(1, n+1):
        hasil = hasil * i
    return hasil

### OUTPUT CONTOH (JSON YANG BENAR) ###

{
  "skor": {
    "sintaks_logika": 95,
    "keterbacaan": 80,
    "efisiensi": 90,
    "capaian_pembelajaran": 20
  },
  "skor_tertimbang": 68.75,
  "umpan_balik": {
    "aspek_positif": "Kode berjalan dengan benar dan menghasilkan nilai faktorial yang tepat untuk semua kasus uji, termasuk n=0.",
    "perlu_diperbaiki": "Tugas ini secara eksplisit meminta penyelesaian menggunakan rekursi, namun kamu menggunakan perulangan (for loop), bukan rekursi.",
    "saran_spesifik": "Coba ubah fungsi menjadi rekursif: definisikan base case 'if n <= 1: return 1', lalu recursive case 'return n * faktorial(n-1)'."
  },
  "kesalahan_terdeteksi": [
    {
      "jenis": "concept_mismatch",
      "lokasi": "seluruh fungsi faktorial",
      "deskripsi": "Solusi menggunakan pendekatan iteratif, sedangkan CPL menuntut pendekatan rekursif dengan base case dan recursive case."
    }
  ],
  "sumber_konteks_rag": ["RUBRIK.capaian_pembelajaran", "CONTOH_SOLUSI"],
  "flag_review_guru": false,
  "alasan_flag": ""
}

--------------------------------------------------------------------
CATATAN UNTUK MODEL
--------------------------------------------------------------------
Perhatikan pada contoh di atas: meskipun sintaks_logika dan efisiensi tinggi
(kode benar dan berjalan cepat), skor capaian_pembelajaran tetap RENDAH (20)
karena siswa tidak menggunakan konsep rekursi yang diminta. Ini adalah pola
wajib: KEBENARAN OUTPUT TIDAK OTOMATIS BERARTI CAPAIAN_PEMBELAJARAN TINGGI.
Selalu periksa apakah konsep yang diminta CPL benar-benar digunakan dalam kode,
bukan hanya apakah outputnya benar.