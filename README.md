# 🎓 Prediksi Kinerja & Risiko Dropout Mahasiswa

> **Sistem Peringatan Dini Berbasis Machine Learning untuk Risiko Dropout Mahasiswa**

Sebuah aplikasi web interaktif yang dibangun menggunakan **Streamlit** untuk memprediksi apakah seorang mahasiswa berisiko mengalami *dropout* (putus kuliah) berdasarkan indikator akademik dan sosial ekonomi. Proyek ini memanfaatkan *machine learning* untuk membantu institusi pendidikan mengidentifikasi mahasiswa berisiko sejak dini dan menerapkan intervensi secara tepat waktu.

---

## 📋 Daftar Isi

- [Ikhtisar](#-ikhtisar)
- [Fitur Utama](#-fitur-utama)
- [Teknologi (Tech Stack)](#-teknologi-tech-stack)
- [Struktur Proyek](#-struktur-proyek)
- [Instalasi](#-instalasi)
- [Penggunaan](#-penggunaan)
- [Performa Model](#-performa-model)
- [Dataset](#-dataset)
- [Penggunaan AI Tools](#-penggunaan-ai-tools)
- [Penulis](#-penulis)

---

## 🔍 Ikhtisar

*Dropout* mahasiswa adalah masalah kritis di pendidikan tinggi yang berdampak pada institusi maupun mahasiswa itu sendiri. Proyek ini menggunakan model **Logistic Regression** (dengan *StandardScaler pipeline*) yang dilatih pada dataset [Higher Education Predictors of Student Retention](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention) yang bersumber dari Kaggle.

### Insight Utama dari EDA
- Sekitar **32%** mahasiswa pada dataset mengalami *dropout*.
- Status pembayaran biaya kuliah memiliki korelasi yang sangat kuat dengan risiko *dropout*.
- Pemegang beasiswa memiliki tingkat kelulusan yang jauh lebih tinggi.
- Nilai dan jumlah mata kuliah yang lulus pada semester 1 adalah indikator awal yang menjadi kunci prediksi.
- Status debitur berhubungan dengan peningkatan risiko *dropout*.

---

## ✨ Fitur Utama

| Fitur | Deskripsi |
|---------|-------------|
| 🏠 **Beranda** | Gambaran umum proyek, metrik utama, dan wawasan (insight) kunci dari dataset. |
| 📊 **Dashboard EDA** | Visualisasi interaktif menggunakan Plotly dengan filter kategorikal (Semua, Akademik, Finansial, Beasiswa, Demografi). |
| 🔍 **Prediksi Risiko** | Sistem inferensi *machine learning* dengan antarmuka yang dikelompokkan secara rapi. Memberikan label risiko (Rendah/Sedang/Tinggi), distribusi probabilitas, faktor dominan penyebab, dan rekomendasi tindakan konkrit. |
| ℹ️ **Tentang Proyek** | Perbandingan performa model ML dan alasan spesifik mengapa Logistic Regression dipilih sebagai algoritma utama. |

---

## 🛠️ Teknologi (Tech Stack)

*   **Pemrograman Dasar:** Python 3.x
*   **Web Framework:** Streamlit
*   **Machine Learning:** Scikit-Learn
*   **Visualisasi Data:** Plotly Express & Plotly Graph Objects
*   **Manipulasi Data:** Pandas & NumPy
*   **Serialization:** Joblib

---

## 📁 Struktur Proyek

```text
student-dropout-prediction/
│
├── data/
│   └── dataset.csv                # Dataset asli (tidak disertakan di git jika terlalu besar)
│
├── models/
│   ├── best_model.joblib          # Model Logistic Regression (Pipeline + Scaler)
│   └── random_forest.joblib       # Model RF (Digunakan khusus untuk Feature Importance)
│
├── src/
│   └── app.py                     # Kode aplikasi utama (Streamlit UI & Logic)
│
├── requirements.txt               # Daftar dependensi library Python
└── README.md                      # Dokumentasi proyek
```

---

## 🚀 Instalasi

Pastikan Anda telah menginstal **Python 3.8+**.

1. **Clone repository ini:**
   ```bash
   git clone https://github.com/username/student-dropout-prediction.git
   cd student-dropout-prediction
   ```

2. **Buat dan aktifkan Virtual Environment (Disarankan):**
   ```bash
   # Windows
   python -m venv venv
   venv\Scripts\activate
   
   # macOS/Linux
   python3 -m venv venv
   source venv/bin/activate
   ```

3. **Install semua requirement library:**
   ```bash
   pip install -r requirements.txt
   ```

---

## 💻 Penggunaan

Untuk menjalankan aplikasi Streamlit di komputer lokal, jalankan perintah berikut di terminal:

```bash
streamlit run src/app.py
```

Aplikasi akan otomatis terbuka pada *browser* bawaan di alamat `http://localhost:8501`.

---

## 📊 Performa Model

Sistem membandingkan beberapa model yang berbeda. **Logistic Regression** dipilih karena memiliki rasio F1-Score dan *explainability* (keterbukaan interpretasi) terbaik untuk kasus pendidikan.

- **Akurasi (Accuracy):** 86%
- **Presisi (Precision):** 78%
- **Recall:** 81%
- **F1-Score:** 79%

*Catatan: Pipeline telah menggunakan `class_weight='balanced'` untuk menangani proporsi dataset yang imbalanced.*

---

## 💾 Dataset

Dataset yang digunakan dalam proyek ini bersumber dari repositori Kaggle. Dataset ini mencakup fitur-fitur demografi, sosial ekonomi, makroekonomi, hingga performa akademik pada pendaftaran pertama.

- **Sumber:** [Higher Education Predictors of Student Retention (Kaggle)](https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention)
- **Ukuran:** 4.424 Baris, 35 Kolom (Atribut)

---

## 🤖 Penggunaan AI Tools

Proyek ini dikembangkan dengan bantuan berbagai alat Kecerdasan Buatan (AI Tools) untuk mendukung efisiensi pengembangan.

AI Tools yang digunakan dalam siklus pengembangan:
*   **ChatGPT / Claude / Gemini / Antigravity:** Digunakan untuk *brainstorming*, penulisan *docstring*, pendeteksian *bug* (*debugging*), *UI/UX polishing*, penyelarasan translasi bahasa, dan penulisan dokumentasi teknis.

*Seluruh proses analisis fundamental data, pemilihan dan evaluasi arsitektur model, serta integrasi komponen sistem tetap dilakukan dan diverifikasi secara mandiri dan kritis oleh pengembang.*

---

## 👤 Penulis

**Raihan Azka Hidayat**

*Proyek ini dibangun sebagai portofolio akhir akademik dalam ranah Data Science dan penerapan Machine Learning praktis.*
