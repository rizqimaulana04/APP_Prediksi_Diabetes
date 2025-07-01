import streamlit as st
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split, learning_curve
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix,
    roc_curve, auc
)
from imblearn.over_sampling import SMOTE
import numpy as np

# -------------------- Konfigurasi Halaman --------------------
st.set_page_config(page_title="Prediksi Risiko Diabetes", layout="centered")
st.title("📊 Aplikasi Prediksi Risiko Diabetes")
st.markdown("""
Aplikasi ini menggunakan algoritma **Decision Tree** untuk memprediksi tingkat risiko diabetes berdasarkan data kesehatan.
Masukkan data seperti usia, BMI, kadar glukosa darah, dan riwayat medis Anda. 
Sistem akan mengklasifikasikan risiko ke dalam kategori **Rendah, Sedang, atau Tinggi**.
""")

# -------------------- Load Dataset --------------------
@st.cache_data
def load_data():
    return pd.read_csv("diabetes_prediction_dataset.csv")

try:
    df = load_data()
    st.success("✅ Dataset berhasil dimuat!")
except FileNotFoundError:
    st.error("❌ File tidak ditemukan: pastikan `diabetes_prediction_dataset.csv` ada di folder.")
    st.stop()

# -------------------- Preprocessing --------------------
fitur = [
    "gender", "age", "bmi", "HbA1c_level",
    "blood_glucose_level", "hypertension", "heart_disease", "smoking_history"
]

if "diabetes" not in df.columns:
    st.error("❌ Kolom target 'diabetes' tidak ditemukan.")
    st.stop()

# Encode input
X = pd.get_dummies(df[fitur], drop_first=True)
y = df["diabetes"].map({"Yes": 1, "No": 0}) if df["diabetes"].dtype == object else df["diabetes"]

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# -------------------- Train Model --------------------
@st.cache_resource
def train_model(X_train, y_train):
    sm = SMOTE(random_state=42)
    X_res, y_res = sm.fit_resample(X_train, y_train)
    clf = DecisionTreeClassifier(max_depth=4, random_state=42)
    clf.fit(X_res, y_res)
    return clf, X_res, y_res

clf, X_train_res, y_train_res = train_model(X_train, y_train)

# -------------------- Evaluasi Model --------------------
y_pred = clf.predict(X_test)
st.subheader("📈 Evaluasi Model")
st.write(f"Akurasi Model: **{accuracy_score(y_test, y_pred):.2f}**")
st.text(classification_report(y_test, y_pred))

# -------------------- Feature Importance --------------------
st.subheader("📌 Feature Importance")
feat_df = pd.DataFrame({
    "Fitur": X.columns,
    "Importance": clf.feature_importances_
}).sort_values(by="Importance", ascending=False)

plt.figure(figsize=(8, 4))
sns.barplot(x="Importance", y="Fitur", data=feat_df)
plt.title("Fitur yang Paling Berpengaruh")
st.pyplot(plt.gcf())
plt.clf()

# -------------------- ROC Curve --------------------
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

# -------------------- Learning Curve --------------------
def get_learning_curve(_model, X, y):
    train_sizes, train_scores, test_scores = learning_curve(
        _model, X, y, cv=5, scoring='accuracy', n_jobs=-1,
        train_sizes=np.linspace(0.1, 1.0, 5)
    )
    return train_sizes, np.mean(train_scores, axis=1), np.mean(test_scores, axis=1)

st.subheader("📚 Learning Curve")
train_sizes, train_mean, test_mean = get_learning_curve(clf, X_train_res, y_train_res)

plt.figure()
plt.plot(train_sizes, train_mean, label="Training Accuracy")
plt.plot(train_sizes, test_mean, label="Validation Accuracy")
plt.xlabel("Jumlah Data Latih")
plt.ylabel("Akurasi")
plt.title("Learning Curve Decision Tree")
plt.legend()
st.pyplot(plt.gcf())
plt.clf()

# -------------------- Form Input --------------------
st.subheader("🧮 Prediksi Risiko Diabetes")
with st.form("form_input"):
    gender = st.selectbox("Gender:", ["Male", "Female"])
    age = st.number_input("Umur (tahun):", 18, 100, 40)
    bmi = st.slider("BMI (Body Mass Index):", 10.0, 60.0, 25.0)
    hba1c = st.number_input("HbA1c Level (%):", 4.0, 14.0, 6.0)
    glu = st.number_input("Kadar Glukosa Darah (mg/dL):", 70, 300, 120)
    hypertension = st.selectbox("Riwayat Hipertensi:", ["Tidak", "Ya"])
    heart = st.selectbox("Riwayat Penyakit Jantung:", ["Tidak", "Ya"])
    
    smoking_options = df["smoking_history"].dropna().unique().tolist()
    smoking = st.selectbox("Riwayat Merokok:", smoking_options)
    
    ok = st.form_submit_button("Prediksi Risiko")

if ok:
    with st.spinner("⏳ Menghitung prediksi..."):
        input_data = {
            "age": age,
            "bmi": bmi,
            "HbA1c_level": hba1c,
            "blood_glucose_level": glu,
            "hypertension": int(hypertension == "Ya"),
            "heart_disease": int(heart == "Ya"),
        }

        # Encode gender
        if "gender_Male" in X.columns:
            input_data["gender_Male"] = int(gender == "Male")
        # Encode smoking history
        for s in [col for col in X.columns if col.startswith("smoking_history_")]:
            input_data[s] = 0
        smoking_col = f"smoking_history_{smoking}"
        if smoking_col in X.columns:
            input_data[smoking_col] = 1

        # Pastikan semua kolom ada
        for col in X.columns:
            if col not in input_data:
                input_data[col] = 0

        inp_df = pd.DataFrame([input_data])[X.columns]

        # Debug input
        st.write("🔎 Data yang Dimasukkan ke Model:")
        st.dataframe(inp_df)

        prob = clf.predict_proba(inp_df)[0][1]
        if prob < 0.25:
            status = "🔵 Risiko Rendah"
        elif prob < 0.60:
            status = "🟡 Risiko Sedang"
        else:
            status = "🔴 Risiko Tinggi"

        st.markdown(f"### 🎯 Hasil Prediksi Terkena Diabetes: **{status}** (Peluang = {prob:.2f})")