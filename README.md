# Duolingo Reviews Sentiment Analysis

Analisis sentimen terhadap ulasan pengguna aplikasi Duolingo di Google Play Store, menggunakan pendekatan machine learning (SVM, Logistic Regression) dan deep learning (LSTM).

## 📋 Deskripsi

Project ini terdiri dari dua tahap utama:
1. **Scraping** ulasan aplikasi Duolingo dari Google Play Store
2. **Training** model klasifikasi sentimen (positif/negatif) dari ulasan tersebut

## 📁 Struktur File

```
.
├── scraping_duolingo.ipynb     # Notebook untuk scraping data ulasan dari Google Play
├── sentiment_training.ipynb    # Notebook untuk preprocessing & training model sentimen
├── duolingo_reviews.csv        # Dataset hasil scraping (30.000 ulasan)
└── requirements.txt            # Daftar dependency Python
```

## 📊 Dataset

Dataset `duolingo_reviews.csv` berisi **30.000 ulasan** aplikasi Duolingo (`com.duolingo`) yang diambil dari Google Play Store (bahasa: `en`, negara: `us`), diurutkan berdasarkan yang terbaru.

Kolom yang tersedia:
| Kolom | Deskripsi |
|---|---|
| `userName` | Nama pengguna yang memberi ulasan |
| `score` | Rating bintang (1–5) |
| `content` | Isi teks ulasan |
| `at` | Tanggal ulasan diberikan |

Distribusi rating pada dataset:
| Rating | Jumlah |
|---|---|
| 5 | 21.883 |
| 4 | 3.840 |
| 1 | 2.097 |
| 3 | 1.449 |
| 2 | 731 |

## ⚙️ Instalasi

```bash
pip install -r requirements.txt
```

Dependency yang digunakan:
- `pandas`, `numpy` — manipulasi data
- `scikit-learn` — model ML (SVM, Logistic Regression) & evaluasi
- `tensorflow` — model deep learning (LSTM)
- `google-play-scraper` — scraping ulasan dari Google Play Store

## 🚀 Cara Menjalankan

### 1. Scraping Data (`scraping_duolingo.ipynb`)
Notebook ini mengambil ulasan Duolingo langsung dari Google Play Store menggunakan library `google-play-scraper`, lalu menyimpannya sebagai `duolingo_reviews.csv`. Jika hanya ingin melakukan training, langkah ini bisa dilewati karena dataset sudah tersedia di repo ini.

### 2. Training Model (`sentiment_training.ipynb`)
Alur pemrosesan pada notebook ini:
1. **Load dataset** dari `duolingo_reviews.csv`
2. **Buang rating 3** dan ubah label menjadi biner:
   - Rating 1–2 → **negatif (0)**
   - Rating 4–5 → **positif (1)**
3. **Balancing dataset** (undersampling kelas mayoritas agar seimbang)
4. **Text cleaning**: lowercase, hapus URL, hapus angka, normalisasi spasi
5. **Split data** train/test (80/20, stratified)
6. **Training 3 model**:
   - **SVM (LinearSVC)** dengan fitur TF-IDF (unigram + bigram)
   - **Logistic Regression** dengan fitur TF-IDF yang sama
   - **LSTM** (Embedding → SpatialDropout1D → LSTM → Dense sigmoid) menggunakan tokenizer & padding sequence Keras
7. **Evaluasi** menggunakan accuracy score & classification report
8. Fungsi `predict_with_score()` untuk menguji model LSTM terhadap kalimat baru

## 🧠 Model

| Model | Fitur | Keterangan |
|---|---|---|
| SVM (LinearSVC) | TF-IDF (1–2 gram, max 20.000 fitur) | Baseline ML |
| Logistic Regression | TF-IDF (sama seperti di atas) | Pembanding baseline |
| LSTM | Word embedding (128 dim) + Tokenizer | Deep learning, max_len=100 |

## 📝 Catatan

- Label sentimen dibuat biner (positif/negatif); ulasan dengan rating 3 (netral) dikeluarkan dari data training.
- Dataset di-balance dengan undersampling agar jumlah kelas positif dan negatif sama, sehingga model tidak bias ke kelas mayoritas.
