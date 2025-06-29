# 🩺 Aplikasi Prediksi Risiko Diabetes Berbasis Streamlit Menggunakan Random Forest

📌[APK-PrediksiDiabetes](https://predikdiabetes.streamlit.app/)

## 📌 Deskripsi Proyek
Aplikasi ini merupakan sistem prediksi risiko diabetes yang dikembangkan menggunakan algoritma **Random Forest** dan antarmuka berbasis **Streamlit**. Dengan input data kesehatan seperti usia, BMI, kadar gula darah, dan riwayat medis, sistem ini memprediksi risiko seseorang terkena diabetes dan mengkategorikannya menjadi **Rendah**, **Sedang**, atau **Tinggi**.

## 👨‍💻 Anggota Kelompok
| Nama                    | NIM        |
|-------------------------|------------|
| Muhammad Rizqi Maulana | 312210360  |
| Dhefi Nurkholik         | 312210414  |
| Sandy Ramadhan          | 312210633  |

## 🎯 Tujuan
Membangun aplikasi berbasis web yang mampu membantu pengguna dalam mengetahui tingkat risiko diabetes secara cepat dan interaktif menggunakan data kesehatan dan machine learning.

## 🧠 Teknologi & Tools
- **Python**
- **Streamlit**
- **Pandas, NumPy**
- **Matplotlib, Seaborn**
- **Scikit-learn**
- **Imbalanced-learn (SMOTE)**

## 📂 Dataset
Dataset yang digunakan berasal dari [Kaggle](https://www.kaggle.com/datasets/iammustafatz/diabetes-prediction-dataset), berisi informasi kesehatan seperti:
- Umur, Gender
- BMI
- HbA1c Level
- Glukosa Darah
- Riwayat Merokok
- Riwayat Hipertensi dan Penyakit Jantung
- Label `diabetes` (Yes/No)

## 🖼️ Fitur Aplikasi
1. **Tampilan Dataset** – Melihat data awal yang digunakan.
2. **Evaluasi Model** – Menampilkan akurasi dan classification report.
3. **Visualisasi** – ROC Curve, Feature Importance, Learning Curve.
4. **Prediksi** – Input data manual untuk mengecek risiko diabetes.

## 🚀 Cara Menjalankan Aplikasi
1. Clone repositori ini:
   ```bash
   git clone https://github.com/rizqimaulana04/APP_Prediksi_Diabetes.git
   cd APP_Prediksi_Diabetes
2. Buat environment dan aktifkan (Windows):
   ```bash
   python -m venv venv
   .\venv\Scripts\activate
3. Install dependensi:
   ```bash
   pip install -r requirements.txt
4. Jalankan aplikasi:
   ```bash
   streamlit run app.py
