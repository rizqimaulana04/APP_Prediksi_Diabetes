import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import numpy as np

# Konfigurasi halaman
st.set_page_config(page_title="Prediksi Risiko Diabetes", layout="centered")
st.title("📊 Aplikasi Prediksi Risiko Diabetes")
st.markdown("""
Aplikasi ini menggunakan algoritma **Random Forest** yang dilatih dari data kesehatan untuk memprediksi tingkat risiko diabetes seseorang.  
Masukkan data pribadi seperti usia, BMI, kadar gula darah, dan lainnya lalu sistem akan mengklasifikasikan risiko Anda ke dalam kategori **Rendah, Sedang, atau Tinggi**.  
""")

# Load dataset
try:
    df = pd.read_csv("diabetes_prediction_dataset.csv")
    st.success("✅ Dataset berhasil dimuat!")
except FileNotFoundError:
    st.error("❌ File tidak ditemukan: pastikan `diabetes_prediction_dataset.csv` ada di folder.")
    st.stop()

# Tampilkan 20 baris pertama
st.subheader("📂 Dataset (20 Baris Awal)")
st.dataframe(df.head(20))

# Fitur dan target
fitur = [
    "gender", "age", "bmi", "HbA1c_level",
    "blood_glucose_level", "hypertension", "heart_disease", "smoking_history"
]

if "diabetes" not in df.columns:
    st.error("❌ Kolom target 'diabetes' tidak ditemukan.")
    st.stop()

# Preprocessing
X = pd.get_dummies(df[fitur], drop_first=True)
y = df["diabetes"]

# Pastikan target y bertipe numerik (0/1)
if y.dtype == object:
    y = y.map({"Yes": 1, "No": 0})

# Split dan oversampling
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
sm = SMOTE(random_state=42)
X_train_res, y_train_res = sm.fit_resample(X_train, y_train)

# Model
clf = RandomForestClassifier(n_estimators=100, random_state=42)
clf.fit(X_train_res, y_train_res)
y_pred = clf.predict(X_test)

# Evaluasi
st.subheader("📈 Evaluasi Model")
st.write(f"Akurasi Model: **{accuracy_score(y_test, y_pred):.2f}**")

# Classification Report
st.subheader("📑 Classification Report")
st.text(classification_report(y_test, y_pred))

# Feature Importance
st.subheader("📌 Feature Importance")
importances = clf.feature_importances_
feat_names = X.columns
feat_df = pd.DataFrame({"Fitur": feat_names, "Importance": importances})
feat_df = feat_df.sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x="Importance", y="Fitur", data=feat_df)
plt.title("Fitur yang Paling Berpengaruh")
st.pyplot(plt.gcf())
plt.clf()

# ROC Curve
st.subheader("📈 ROC Curve")
y_proba = clf.predict_proba(X_test)[:, 1]
fpr, tpr, _ = roc_curve(y_test, y_proba)
roc_auc = auc(fpr, tpr)

plt.figure()
plt.plot(fpr, tpr, label=f"AUC = {roc_auc:.2f}")
plt.plot([0, 1], [0, 1], 'k--')
plt.xlabel("False Positive Rate")
plt.ylabel("True Positive Rate")
plt.title("ROC Curve")
plt.legend(loc="lower right")
st.pyplot(plt.gcf())
plt.clf()

# Learning Curve
st.subheader("📚 Learning Curve")
train_sizes, train_scores, test_scores = learning_curve(
    clf, X_train_res, y_train_res, cv=5, scoring='accuracy', n_jobs=-1,
    train_sizes=np.linspace(0.1, 1.0, 5)
)

train_mean = np.mean(train_scores, axis=1)
test_mean = np.mean(test_scores, axis=1)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")
plt.xlabel("Jumlah Data Latih")
plt.ylabel("Akurasi")
plt.title("Learning Curve Random Forest")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()

# Form Input
st.subheader("🧮 Prediksi Risiko Diabetes")
with st.form("form_input"):
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Umur (tahun):", 18, 100, 40)
    bmi = st.slider("BMI (Body Mass Index):", 10.0, 60.0, 25.0)
    hba1c = st.number_input("HbA1c Level (%):", 4.0, 14.0, 6.0)
    glu = st.number_input("Kadar Glukosa Darah (mg/dL):", 70, 300, 120)
    hypertension = st.selectbox("Riwayat Hipertensi:", ["Tidak", "Ya"])
    heart = st.selectbox("Riwayat Penyakit Jantung:", ["Tidak", "Ya"])
    smoking = st.selectbox("Riwayat Merokok:", df["smoking_history"].dropna().unique().tolist())
    ok = st.form_submit_button("Prediksi Risiko")

if ok:
    inp = pd.DataFrame([{
        "gender_Male": int(gender == "Male"),
        "age": age,
        "bmi": bmi,
        "HbA1c_level": hba1c,
        "blood_glucose_level": glu,
        "hypertension": int(hypertension == "Ya"),
        "heart_disease": int(heart == "Ya"),
        f"smoking_history_{smoking}": 1
    }])
    for col in X.columns:
        if col not in inp.columns:
            inp[col] = 0
    inp = inp[X.columns]

    prob = clf.predict_proba(inp)[0][1]
    if prob < 0.25:
        status = "🔵 Risiko Rendah"
    elif prob < 0.60:
        status = "🟡 Risiko Sedang"
    else:
        status = "🔴 Risiko Tinggi"

    st.markdown(f"### 🎯 Hasil Prediksi Terkena Diabetes: **{status}** (Peluang = {prob:.2f})")
