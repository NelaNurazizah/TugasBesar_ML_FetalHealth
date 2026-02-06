import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.neighbors import KNeighborsClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score, f1_score, roc_curve, auc
from imblearn.over_sampling import SMOTE
from sklearn.preprocessing import label_binarize

# --- KONFIGURASI HALAMAN ---
st.set_page_config(
    page_title="Deteksi Kesehatan Janin",
    page_icon="👶",
    layout="wide"
)

# --- JUDUL & DESKRIPSI ---
st.title("👶 Aplikasi Deteksi Dini Patologi Janin")
st.markdown("""
Aplikasi ini menggunakan algoritma **K-Nearest Neighbor (KNN)** dan **Random Forest** dengan teknik **SMOTE** untuk mengklasifikasikan kondisi kesehatan janin.
""")

# --- 1. LOAD DATA ---
@st.cache_data
def load_data():
    df = pd.read_csv('fetal_health.csv')
    return df

try:
    df = load_data()
except FileNotFoundError:
    st.error("File 'fetal_health.csv' tidak ditemukan. Pastikan file ada di folder yang sama dengan app.py")
    st.stop()

# --- SIDEBAR MENU ---
menu = st.sidebar.radio("Navigasi", ["Beranda & Data", "Exploratory Data Analysis (EDA)", "Evaluasi Model", "Simulasi Prediksi"])

# --- FUNGSI TRAINING MODEL (CACHED) ---
@st.cache_resource
def train_models(df):
    # Split Data
    X = df.drop('fetal_health', axis=1)
    y = df['fetal_health']
    
    # Train Test Split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # SMOTE (Balancing Data)
    smote = SMOTE(random_state=42)
    X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train_resampled)
    X_test_scaled = scaler.transform(X_test)
    
    # --- PROSES MENCARI K TERBAIK (ELBOW METHOD) ---
    k_range = range(1, 21)
    k_scores = []
    for k in k_range:
        knn_temp = KNeighborsClassifier(n_neighbors=k)
        knn_temp.fit(X_train_scaled, y_train_resampled)
        y_pred_temp = knn_temp.predict(X_test_scaled)
        k_scores.append(accuracy_score(y_test, y_pred_temp))
    
    best_k = k_range[np.argmax(k_scores)] # Ambil K dengan akurasi tertinggi
    
    # 1. Train Final KNN dengan Best K
    knn = KNeighborsClassifier(n_neighbors=best_k)
    knn.fit(X_train_scaled, y_train_resampled)
    y_pred_knn = knn.predict(X_test_scaled)
    
    # 2. Train Random Forest
    rf = RandomForestClassifier(n_estimators=100, random_state=42)
    rf.fit(X_train_scaled, y_train_resampled)
    y_pred_rf = rf.predict(X_test_scaled)
    
    return {
        "X_train_resampled": X_train_resampled,
        "y_train_resampled": y_train_resampled,
        "X_test": X_test,
        "y_test": y_test,
        "X_test_scaled": X_test_scaled,
        "scaler": scaler,
        "knn_model": knn,
        "rf_model": rf,
        "y_pred_knn": y_pred_knn,
        "y_pred_rf": y_pred_rf,
        "k_range": list(k_range),   # Data untuk Plot Elbow
        "k_scores": k_scores,       # Data untuk Plot Elbow
        "best_k": best_k
    }

# Load & Train Models (Otomatis saat pertama run)
models = train_models(df)

# --- HALAMAN 1: BERANDA & DATA ---
if menu == "Beranda & Data":
    st.header("Dataset Kesehatan Janin")
    st.write(f"Jumlah Baris: {df.shape[0]} | Jumlah Kolom: {df.shape[1]}")
    st.dataframe(df.head(10))
    
    st.subheader("Keterangan Kelas Target")
    st.info("1.0 = Normal | 2.0 = Suspect (Mencurigakan) | 3.0 = Pathological (Patologi)")

# --- HALAMAN 2: EDA ---
elif menu == "Exploratory Data Analysis (EDA)":
    st.header("Exploratory Data Analysis")
    
    # Tabulasi EDA
    tab_dist, tab_corr, tab_scatter = st.tabs(["Distribusi Kelas", "Korelasi Fitur", "Scatter Plot"])
    
    with tab_dist:
        col1, col2 = st.columns(2)
        with col1:
            st.subheader("Distribusi Awal (Imbalanced)")
            fig, ax = plt.subplots()
            sns.countplot(x='fetal_health', data=df, palette='viridis', ax=ax)
            st.pyplot(fig)
            
        with col2:
            st.subheader("Distribusi Setelah SMOTE (Balanced)")
            fig, ax = plt.subplots()
            y_resampled = models['y_train_resampled']
            sns.countplot(x=y_resampled, palette='viridis', ax=ax)
            st.pyplot(fig)
            st.caption(f"Total Data Training setelah SMOTE: {len(y_resampled)}")

    with tab_corr:
        st.subheader("Heatmap Korelasi")
        fig, ax = plt.subplots(figsize=(12, 10))
        sns.heatmap(df.corr(), cmap='coolwarm', annot=False, ax=ax)
        st.pyplot(fig)
        
    with tab_scatter:
        st.subheader("Scatter Plot Interaktif")
        st.write("Lihat hubungan antara dua fitur:")
        
        col_x, col_y = st.columns(2)
        with col_x:
            feat_x = st.selectbox("Pilih Sumbu X", df.columns[:-1], index=7) # Default: abnormal_short_term...
        with col_y:
            feat_y = st.selectbox("Pilih Sumbu Y", df.columns[:-1], index=0) # Default: baseline value
            
        fig, ax = plt.subplots(figsize=(8, 6))
        sns.scatterplot(data=df, x=feat_x, y=feat_y, hue='fetal_health', palette='deep', ax=ax)
        plt.title(f"{feat_x} vs {feat_y}")
        st.pyplot(fig)

# --- HALAMAN 3: EVALUASI MODEL ---
elif menu == "Evaluasi Model":
    st.header("Perbandingan Performa Model")
    
    # Ambil data
    y_test = models['y_test']
    y_pred_knn = models['y_pred_knn']
    y_pred_rf = models['y_pred_rf']
    
    # Metrics Score Cards
    acc_knn = accuracy_score(y_test, y_pred_knn)
    f1_knn = f1_score(y_test, y_pred_knn, average='weighted')
    
    acc_rf = accuracy_score(y_test, y_pred_rf)
    f1_rf = f1_score(y_test, y_pred_rf, average='weighted')
    
    col1, col2 = st.columns(2)
    with col1:
        st.info(f"**KNN (K={models['best_k']}) Accuracy:** {acc_knn:.2%}")
        st.write(f"KNN F1-Score: {f1_knn:.2%}")
    with col2:
        st.success(f"**Random Forest Accuracy:** {acc_rf:.2%}")
        st.write(f"Random Forest F1-Score: {f1_rf:.2%}")
    
    st.divider()
    
    # Tab Evaluasi Lengkap
    tab1, tab2, tab3, tab4 = st.tabs([
        "Elbow Method (KNN)", 
        "Confusion Matrix", 
        "Classification Report", 
        "ROC & Feature Importance"
    ])
    
    with tab1:
        st.subheader("Evaluasi Nilai K (Elbow Method)")
        st.write("Grafik ini menunjukkan akurasi KNN untuk berbagai nilai K (tetangga).")
        
        fig, ax = plt.subplots(figsize=(10, 5))
        ax.plot(models['k_range'], models['k_scores'], marker='o', linestyle='-', color='b')
        ax.set_title('Akurasi vs Nilai K')
        ax.set_xlabel('Nilai K')
        ax.set_ylabel('Akurasi')
        ax.set_xticks(models['k_range'])
        ax.grid(True)
        
        # Highlight best K
        best_k = models['best_k']
        best_score = max(models['k_scores'])
        ax.annotate(f'Best K={best_k}', xy=(best_k, best_score), xytext=(best_k, best_score - 0.05),
                    arrowprops=dict(facecolor='red', shrink=0.05))
        
        st.pyplot(fig)
        st.write(f"**Kesimpulan:** Nilai K yang dipilih secara otomatis adalah **{best_k}** karena memiliki akurasi tertinggi.")

    with tab2:
        col_cm1, col_cm2 = st.columns(2)
        with col_cm1:
            st.subheader("Confusion Matrix - KNN")
            cm_knn = confusion_matrix(y_test, y_pred_knn)
            fig, ax = plt.subplots()
            sns.heatmap(cm_knn, annot=True, fmt='d', cmap='Blues', ax=ax)
            st.pyplot(fig)
        with col_cm2:
            st.subheader("Confusion Matrix - RF")
            cm_rf = confusion_matrix(y_test, y_pred_rf)
            fig, ax = plt.subplots()
            sns.heatmap(cm_rf, annot=True, fmt='d', cmap='Greens', ax=ax)
            st.pyplot(fig)
            
    with tab3:
        col_cr1, col_cr2 = st.columns(2)
        with col_cr1:
            st.subheader("Report KNN")
            st.text(classification_report(y_test, y_pred_knn))
        with col_cr2:
            st.subheader("Report Random Forest")
            st.text(classification_report(y_test, y_pred_rf))
        
    with tab4:
        # ROC & Feature Importance
        y_test_bin = label_binarize(y_test, classes=[1, 2, 3])
        n_classes = y_test_bin.shape[1]
        
        y_score_knn = models['knn_model'].predict_proba(models['X_test_scaled'])
        y_score_rf = models['rf_model'].predict_proba(models['X_test_scaled'])
        
        def plot_roc(y_test_bin, y_score, title):
            fpr, tpr, roc_auc = {}, {}, {}
            fig, ax = plt.subplots()
            colors = ['blue', 'red', 'green']
            labels = ['Normal', 'Suspect', 'Pathological']
            
            for i in range(n_classes):
                fpr[i], tpr[i], _ = roc_curve(y_test_bin[:, i], y_score[:, i])
                roc_auc[i] = auc(fpr[i], tpr[i])
                ax.plot(fpr[i], tpr[i], color=colors[i], lw=2,
                        label=f'{labels[i]} (area = {roc_auc[i]:.2f})')
            
            ax.plot([0, 1], [0, 1], 'k--', lw=2)
            ax.set_xlim([0.0, 1.0])
            ax.set_ylim([0.0, 1.05])
            ax.set_title(title)
            ax.legend(loc="lower right")
            return fig

        col_roc1, col_roc2 = st.columns(2)
        with col_roc1:
            st.write("**ROC Curve - KNN**")
            st.pyplot(plot_roc(y_test_bin, y_score_knn, "ROC KNN"))
        with col_roc2:
            st.write("**ROC Curve - Random Forest**")
            st.pyplot(plot_roc(y_test_bin, y_score_rf, "ROC Random Forest"))
            
        st.divider()
        st.subheader("Feature Importance (Random Forest)")
        importances = models['rf_model'].feature_importances_
        features = df.drop('fetal_health', axis=1).columns
        indices = np.argsort(importances)[::-1]
        
        fig, ax = plt.subplots(figsize=(10, 6))
        ax.bar(range(len(features)), importances[indices], align="center", color='teal')
        ax.set_xticks(range(len(features)))
        ax.set_xticklabels([features[i] for i in indices], rotation=90)
        st.pyplot(fig)

# --- HALAMAN 4: SIMULASI PREDIKSI ---
elif menu == "Simulasi Prediksi":
    st.header("Simulasi Diagnosa Pasien Baru")
    st.write("Masukkan parameter CTG (Cardiotocogram) di bawah ini:")
    
    col1, col2, col3 = st.columns(3)
    input_data = {}
    features = df.drop('fetal_health', axis=1).columns
    
    for i, col in enumerate(features):
        with [col1, col2, col3][i % 3]:
            default_val = float(df[col].mean())
            input_data[col] = st.number_input(f"{col}", value=default_val, format="%.4f")
    
    st.divider()
    
    if st.button("🔍 Prediksi Kondisi Janin"):
        new_data = pd.DataFrame([input_data])
        scaler = models['scaler']
        new_data_scaled = scaler.transform(new_data)
        
        # Prediksi pakai Random Forest (karena biasanya performa lebih baik)
        rf_model = models['rf_model']
        prediction = rf_model.predict(new_data_scaled)[0]
        probabilities = rf_model.predict_proba(new_data_scaled)[0]
        
        st.subheader("Hasil Analisis:")
        mapping_hasil = {1.0: "NORMAL", 2.0: "SUSPECT", 3.0: "PATHOLOGICAL"}
        hasil_text = mapping_hasil.get(prediction, "Unknown")
        
        if prediction == 1.0:
            st.success(f"### Kategori: {hasil_text} 😊")
            st.write("Janin dalam kondisi sehat.")
        elif prediction == 2.0:
            st.warning(f"### Kategori: {hasil_text} ⚠️")
            st.write("Kondisi mencurigakan, perlu pemantauan lebih lanjut.")
        else:
            st.error(f"### Kategori: {hasil_text} 🚨")
            st.write("Terdeteksi potensi patologi, segera rujuk ke dokter spesialis.")
            
        st.write("Probabilitas Model:")
        prob_df = pd.DataFrame(probabilities, index=['Normal', 'Suspect', 'Pathological'], columns=['Probability'])
        st.bar_chart(prob_df)