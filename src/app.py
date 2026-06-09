"""
Student Performance & Dropout Risk Prediction
Sistem Prediksi Risiko Dropout Mahasiswa Berbasis Machine Learning

A production-ready Streamlit application for predicting student dropout risk
using Logistic Regression model with interactive EDA dashboard.
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from plotly.subplots import make_subplots
import joblib
import os
import warnings

warnings.filterwarnings("ignore")

# ──────────────────────────────────────────────
# PAGE CONFIG
# ──────────────────────────────────────────────
st.set_page_config(
    page_title="Student Dropout Prediction",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ──────────────────────────────────────────────
# MINIMAL CUSTOM CSS
# ──────────────────────────────────────────────
st.markdown("""
<style>
    /* Main metric cards */
    div[data-testid="stMetric"] {
        background: linear-gradient(135deg, #667eea11, #764ba211);
        border: 1px solid #667eea33;
        border-radius: 12px;
        padding: 16px 20px;
    }
    div[data-testid="stMetric"] label {
        font-size: 0.85rem !important;
        color: #888 !important;
    }
    div[data-testid="stMetric"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
    }

    /* Sidebar styling */
    section[data-testid="stSidebar"] {
        background: linear-gradient(180deg, #0e1117 0%, #1a1f2e 100%);
    }
    section[data-testid="stSidebar"] .stRadio label {
        font-size: 1rem;
    }

    /* Result boxes */
    .risk-low {
        background: linear-gradient(135deg, #00c85320, #00e67620);
        border: 2px solid #00c853;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        height: 100%;
    }
    .risk-high {
        background: linear-gradient(135deg, #ff173520, #ff523520);
        border: 2px solid #ff1744;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        height: 100%;
    }
    .risk-medium {
        background: linear-gradient(135deg, #ff910020, #ffab0020);
        border: 2px solid #ff9100;
        border-radius: 12px;
        padding: 24px;
        margin: 12px 0;
        height: 100%;
    }

    /* Page header */
    .page-header {
        text-align: center;
        margin-bottom: 2rem;
        width: 100%;
        display: flex;
        flex-direction: column;
        align-items: center;
        justify-content: center;
    }
    .page-header h1 {
        font-size: 2.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #667eea, #764ba2);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin-bottom: 0.5rem;
        line-height: 1.2;
    }
    .page-header p {
        font-size: 1.15rem;
        color: #aaa;
        opacity: 0.9;
        max-width: 800px;
        margin: 0 auto;
    }

    /* Info card */
    .info-card {
        background: #1a1f2e;
        border: 1px solid #2d3348;
        border-radius: 12px;
        padding: 20px;
        margin: 8px 0;
        height: 100%;
    }
    .info-card h4 {
        color: #667eea;
        margin-bottom: 8px;
        display: flex;
        align-items: center;
        gap: 8px;
    }

    /* Section divider */
    .section-divider {
        border: none;
        height: 1px;
        background: linear-gradient(90deg, transparent, #667eea44, transparent);
        margin: 24px 0;
    }
    
    /* Footer */
    .sidebar-footer {
        position: absolute;
        bottom: 0;
        width: 100%;
        padding: 16px;
        text-align: center;
        font-size: 0.8rem;
        color: #888;
        border-top: 1px solid #333;
        background: #0e1117;
    }
</style>
""", unsafe_allow_html=True)


# ──────────────────────────────────────────────
# DATA & MODEL LOADING (cached)
# ──────────────────────────────────────────────
@st.cache_data
def load_dataset():
    """Load the student dataset."""
    try:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "data", "dataset.csv"),
            os.path.join("data", "dataset.csv"),
            os.path.join("..", "data", "dataset.csv"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                df = pd.read_csv(path)
                return df
        st.error("❌ Dataset file not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading dataset: {e}")
        return None


@st.cache_resource
def load_model():
    """Load the trained model."""
    try:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "best_model.joblib"),
            os.path.join("models", "best_model.joblib"),
            os.path.join("..", "models", "best_model.joblib"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                return model
        st.error("❌ Model file not found.")
        return None
    except Exception as e:
        st.error(f"❌ Error loading model: {e}")
        return None


@st.cache_resource
def load_rf_model():
    """Load Random Forest model for feature importance."""
    try:
        possible_paths = [
            os.path.join(os.path.dirname(__file__), "..", "models", "random_forest.joblib"),
            os.path.join("models", "random_forest.joblib"),
            os.path.join("..", "models", "random_forest.joblib"),
        ]
        for path in possible_paths:
            if os.path.exists(path):
                model = joblib.load(path)
                return model
        return None
    except Exception:
        return None


# ──────────────────────────────────────────────
# FEATURE ORDER (must match training)
# ──────────────────────────────────────────────
FEATURE_ORDER = [
    'Marital status', 'Application mode', 'Application order', 'Course',
    'Daytime/evening attendance', 'Previous qualification', 'Nacionality',
    "Mother's qualification", "Father's qualification",
    "Mother's occupation", "Father's occupation",
    'Displaced', 'Educational special needs', 'Debtor',
    'Tuition fees up to date', 'Gender', 'Scholarship holder',
    'Age at enrollment', 'International',
    'Curricular units 1st sem (credited)',
    'Curricular units 1st sem (enrolled)',
    'Curricular units 1st sem (evaluations)',
    'Curricular units 1st sem (approved)',
    'Curricular units 1st sem (grade)',
    'Curricular units 1st sem (without evaluations)',
    'Unemployment rate', 'Inflation rate', 'GDP',
]

DEFAULT_VALUES = {
    'Marital status': 1.0, 'Application mode': 8.0, 'Application order': 1.0,
    'Course': 10.0, 'Daytime/evening attendance': 1.0, 'Previous qualification': 1.0,
    'Nacionality': 1.0, "Mother's qualification": 13.0, "Father's qualification": 14.0,
    "Mother's occupation": 6.0, "Father's occupation": 8.0, 'Displaced': 1.0,
    'Educational special needs': 0.0, 'Gender': 0.0, 'International': 0.0,
    'Curricular units 1st sem (credited)': 0.0, 'Curricular units 1st sem (enrolled)': 6.0,
    'Curricular units 1st sem (without evaluations)': 0.0, 'Unemployment rate': 11.1,
    'Inflation rate': 1.4, 'GDP': 0.32,
}


# ──────────────────────────────────────────────
# PLOTLY THEME
# ──────────────────────────────────────────────
PLOTLY_LAYOUT = dict(
    template="plotly_dark",
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family="Inter, sans-serif", color="#ccc"),
    margin=dict(l=40, r=40, t=50, b=40),
)

TARGET_COLORS = {"Tidak Dropout": "#667eea", "Dropout": "#ff5252"}

# Initialize session state for page routing
if 'page' not in st.session_state:
    st.session_state.page = "🏠 Beranda"

# ──────────────────────────────────────────────
# SIDEBAR NAVIGATION
# ──────────────────────────────────────────────
with st.sidebar:
    st.markdown("### 🎓 Student Dropout Prediction")
    st.markdown("────────────────────")

    page = st.radio(
        "Menu Utama",
        [
            "🏠 Beranda", 
            "📊 Dashboard EDA", 
            "🔍 Prediksi Risiko", 
            "ℹ️ Tentang Proyek"
        ],
        key="page",
        label_visibility="collapsed",
    )

    st.markdown("<br><br><br><br><br><br>", unsafe_allow_html=True)
    st.markdown("────────────────────")
    st.markdown("**Versi Aplikasi:** 1.0  \n**Raihan Azka Hidayat**")


# ══════════════════════════════════════════════
#  PAGE 1 — BERANDA
# ══════════════════════════════════════════════
if page == "🏠 Beranda":

    # Hero section
    st.markdown(
        "<div class='page-header'>"
        "<h1>Sistem Prediksi Risiko Dropout Mahasiswa</h1>"
        "<p>Menggunakan Machine Learning untuk membantu institusi pendidikan mengidentifikasi mahasiswa yang berisiko dropout lebih awal sehingga dapat dilakukan intervensi akademik maupun finansial secara tepat waktu.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    col_btn1, col_btn2, col_btn3 = st.columns([1, 1, 1])
    with col_btn2:
        if st.button("🚀 Mulai Prediksi", type="primary", use_container_width=True):
            st.info("Silakan pilih menu '🔍 Prediksi Risiko' di Sidebar untuk memulai prediksi.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Key Metrics ──
    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Jumlah Mahasiswa", "4.424")
    m2.metric("Jumlah Dropout", "1.421")
    m3.metric("Akurasi Model", "86%")
    m4.metric("Model Terbaik", "Logistic Reg.")

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Background ──
    col_bg, col_obj = st.columns(2)

    with col_bg:
        st.markdown(
            "<div class='info-card'>"
            "<h4>📋 Latar Belakang</h4>"
            "<p>Dropout mahasiswa merupakan masalah serius yang berdampak signifikan pada institusi pendidikan "
            "maupun mahasiswa itu sendiri. Tingkat dropout yang tinggi menyebabkan kerugian finansial bagi universitas, "
            "menurunkan peringkat institusi, dan menyia-nyiakan sumber daya publik. Bagi mahasiswa, dropout sering kali "
            "mengakibatkan terbatasnya peluang karir, hutang pinjaman pendidikan, dan berkurangnya pendapatan seumur hidup.</p>"
            "<p>Identifikasi dini terhadap mahasiswa berisiko memungkinkan institusi untuk memberikan intervensi tepat waktu "
            "seperti dukungan akademik, konseling bantuan keuangan, dan program bimbingan — yang pada akhirnya "
            "meningkatkan tingkat retensi dan kesuksesan mahasiswa.</p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_obj:
        st.markdown(
            "<div class='info-card'>"
            "<h4>🎯 Tujuan Proyek</h4>"
            "<p>Proyek ini bertujuan untuk membantu institusi pendidikan <strong>mengidentifikasi mahasiswa "
            "yang berisiko dropout pada tahap awal</strong> sehingga intervensi yang tepat "
            "dapat diimplementasikan.</p>"
            "<p>Dengan memanfaatkan machine learning pada data akademik dan sosial ekonomi, sistem ini memberikan:</p>"
            "<ul>"
            "<li><strong>Klasifikasi risiko</strong> (Rendah / Sedang / Tinggi)</li>"
            "<li><strong>Skor probabilitas dropout</strong></li>"
            "<li><strong>Rekomendasi tindakan</strong> untuk pembimbing akademik</li>"
            "</ul>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Dataset Info ──
    st.markdown("### 📁 Informasi Dataset")

    di1, di2, di3 = st.columns(3)
    with di1:
        st.markdown(
            "<div class='info-card'>"
            "<h4>Sumber</h4>"
            "<p><em>Higher Education Predictors of Student Retention</em><br>"
            "Kaggle Dataset</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with di2:
        st.markdown(
            "<div class='info-card'>"
            "<h4>Ukuran</h4>"
            "<p><strong>4.424</strong> data mahasiswa<br>"
            "<strong>35</strong> atribut</p>"
            "</div>",
            unsafe_allow_html=True,
        )
    with di3:
        st.markdown(
            "<div class='info-card'>"
            "<h4>Variabel Target</h4>"
            "<p>🟢 <strong>Tidak Dropout</strong> (Lulus / Aktif)<br>"
            "🔴 <strong>Dropout</strong></p>"
            "</div>",
            unsafe_allow_html=True,
        )


# ══════════════════════════════════════════════
#  PAGE 2 — DASHBOARD EDA
# ══════════════════════════════════════════════
elif page == "📊 Dashboard EDA":

    st.markdown(
        "<div class='page-header'>"
        "<h1>📊 Exploratory Data Analysis (EDA)</h1>"
        "<p>Visualisasi interaktif pola dropout mahasiswa</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    df = load_dataset()

    if df is not None:
        df_analysis = df.copy()
        df_analysis['Target_Binary'] = df_analysis['Target'].apply(
            lambda x: 'Dropout' if x == 'Dropout' else 'Tidak Dropout'
        )

        st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
        
        # Filter Interaktif
        st.markdown("#### ⚙️ Filter Analisis")
        category_filter = st.pills("Pilih Kategori Insight:", ["Semua", "Akademik", "Finansial", "Beasiswa", "Demografi"], default="Semua")
        st.markdown("<br>", unsafe_allow_html=True)

        # ── 1. Target Distribution (Demografi / Semua) ──
        if category_filter in ["Semua", "Demografi"]:
            st.markdown("### 📈 Distribusi Target")
            col_pie, col_bar = st.columns(2)

            target_counts = df_analysis['Target_Binary'].value_counts().reset_index()
            target_counts.columns = ['Status', 'Count']

            with col_pie:
                fig_pie = px.pie(
                    target_counts, values='Count', names='Status',
                    color='Status', color_discrete_map=TARGET_COLORS, hole=0.45,
                )
                fig_pie.update_layout(**PLOTLY_LAYOUT, title="Distribusi (Pie Chart)", height=380)
                fig_pie.update_traces(textinfo='percent+label', textfont_size=13)
                st.plotly_chart(fig_pie, use_container_width=True)

            with col_bar:
                fig_bar = px.bar(
                    target_counts, x='Status', y='Count', color='Status',
                    color_discrete_map=TARGET_COLORS, text='Count',
                )
                fig_bar.update_layout(**PLOTLY_LAYOUT, title="Distribusi (Bar Chart)", height=380, showlegend=False)
                fig_bar.update_traces(textposition='outside', textfont_size=14)
                st.plotly_chart(fig_bar, use_container_width=True)

            st.caption(
                "💡 **Insight:** Sekitar **32%** mahasiswa pada dataset mengalami dropout. Kondisi ini menunjukkan adanya "
                "ketidakseimbangan kelas sehingga model menggunakan `class_weight='balanced'` untuk meningkatkan kemampuan deteksi mahasiswa berisiko."
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── 2. Scholarship Impact (Beasiswa / Semua) ──
        if category_filter in ["Semua", "Beasiswa", "Finansial"]:
            st.markdown("### 🎓 Penerima Beasiswa vs Status Dropout")

            scholarship_data = df_analysis.groupby(['Scholarship holder', 'Target_Binary']).size().reset_index(name='Count')
            scholarship_data['Scholarship'] = scholarship_data['Scholarship holder'].map({0: 'Bukan Penerima Beasiswa', 1: 'Penerima Beasiswa'})

            fig_sch = px.bar(
                scholarship_data, x='Scholarship', y='Count', color='Target_Binary', barmode='group',
                color_discrete_map=TARGET_COLORS, text='Count',
            )
            fig_sch.update_layout(**PLOTLY_LAYOUT, title="Tingkat Dropout Berdasarkan Status Beasiswa", height=400, legend_title="Status")
            fig_sch.update_traces(textposition='outside')
            st.plotly_chart(fig_sch, use_container_width=True)

            st.caption(
                "💡 **Insight:** Mahasiswa penerima beasiswa menunjukkan tingkat dropout yang **jauh lebih rendah** "
                "dibandingkan mahasiswa non-penerima. Hal ini mengindikasikan bahwa dukungan finansial "
                "berperan penting dalam membantu mahasiswa menyelesaikan studinya."
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── 3. Tuition Fees (Finansial / Semua) ──
        if category_filter in ["Semua", "Finansial"]:
            st.markdown("### 💳 Status Pembayaran Kuliah vs Status Dropout")

            tuition_data = df_analysis.groupby(['Tuition fees up to date', 'Target_Binary']).size().reset_index(name='Count')
            tuition_data['Tuition Status'] = tuition_data['Tuition fees up to date'].map({0: 'Menunggak', 1: 'Lancar'})

            fig_tui = px.bar(
                tuition_data, x='Tuition Status', y='Count', color='Target_Binary', barmode='group',
                color_discrete_map=TARGET_COLORS, text='Count',
            )
            fig_tui.update_layout(**PLOTLY_LAYOUT, title="Tingkat Dropout Berdasarkan Status Pembayaran Kuliah", height=400, legend_title="Status")
            fig_tui.update_traces(textposition='outside')
            st.plotly_chart(fig_tui, use_container_width=True)

            st.caption(
                "💡 **Insight:** Mahasiswa yang memiliki **tunggakan biaya kuliah** menunjukkan tingkat dropout yang "
                "jauh lebih tinggi. Temuan ini menunjukkan bahwa status pembayaran kuliah merupakan salah satu "
                "faktor terkuat dalam memprediksi risiko dropout mahasiswa."
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── 4. Debtor Status (Finansial / Semua) ──
        if category_filter in ["Semua", "Finansial"]:
            st.markdown("### 💳 Status Debitur vs Status Dropout")

            debtor_data = df_analysis.groupby(['Debtor', 'Target_Binary']).size().reset_index(name='Count')
            debtor_data['Debtor Status'] = debtor_data['Debtor'].map({0: 'Bukan Debitur', 1: 'Debitur'})

            fig_deb = px.bar(
                debtor_data, x='Debtor Status', y='Count', color='Target_Binary', barmode='group',
                color_discrete_map=TARGET_COLORS, text='Count',
            )
            fig_deb.update_layout(**PLOTLY_LAYOUT, title="Tingkat Dropout Berdasarkan Status Debitur", height=400, legend_title="Status")
            fig_deb.update_traces(textposition='outside')
            st.plotly_chart(fig_deb, use_container_width=True)

            st.caption(
                "💡 **Insight:** Mahasiswa dengan status **debitur** menunjukkan proporsi dropout yang lebih tinggi, "
                "mengindikasikan kesulitan keuangan sebagai faktor penyebab."
            )
            st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

        # ── 5. Feature Importance (Akademik / Semua) ──
        if category_filter in ["Semua", "Akademik"]:
            st.markdown("### ⭐ 10 Fitur Terpenting")

            rf_model = load_rf_model()

            if rf_model is not None:
                drop_cols = [
                    'Curricular units 2nd sem (credited)', 'Curricular units 2nd sem (enrolled)',
                    'Curricular units 2nd sem (evaluations)', 'Curricular units 2nd sem (approved)',
                    'Curricular units 2nd sem (grade)', 'Curricular units 2nd sem (without evaluations)', 'Target',
                ]
                feature_cols = [c for c in df.columns if c not in drop_cols]
                importances = rf_model.feature_importances_
                feat_imp = pd.DataFrame({'Feature': feature_cols, 'Importance': importances}).sort_values('Importance', ascending=True).tail(10)

                fig_imp = px.bar(
                    feat_imp, x='Importance', y='Feature', orientation='h', color='Importance',
                    color_continuous_scale=['#667eea', '#f093fb'],
                )
                fig_imp.update_layout(**PLOTLY_LAYOUT, title="10 Fitur Paling Penting (Random Forest)", height=450, coloraxis_showscale=False)
                st.plotly_chart(fig_imp, use_container_width=True)
            else:
                st.warning("🟡 Model Random Forest tidak ditemukan. Feature importance tidak dapat ditampilkan.")

            st.caption(
                "💡 **Insight:** **Nilai semester 1**, **jumlah mata kuliah lulus**, dan "
                "**status pembayaran kuliah** mendominasi fitur teratas — mengonfirmasi bahwa performa "
                "akademik awal dan kondisi finansial adalah prediktor dropout terkuat."
            )


# ══════════════════════════════════════════════
#  PAGE 3 — DROPOUT PREDICTION
# ══════════════════════════════════════════════
elif page == "🔍 Prediksi Risiko":

    st.markdown(
        "<div class='page-header'>"
        "<h1>🔍 Prediksi Risiko Dropout</h1>"
        "<p>Masukkan profil, data akademik, dan kondisi finansial mahasiswa untuk melakukan prediksi risiko dropout.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    model = load_model()

    if model is not None:

        col_input1, col_input2, col_input3 = st.columns(3)

        with col_input1:
            with st.container(border=True):
                st.markdown("#### 👤 Profil Mahasiswa")
                age = st.slider(
                    "Usia Saat Masuk", min_value=17, max_value=60, value=20,
                    help="Usia mahasiswa saat pertama kali masuk.",
                )

        with col_input2:
            with st.container(border=True):
                st.markdown("#### 📚 Kondisi Akademik")
                sem1_grade = st.slider(
                    "Nilai Semester 1", min_value=0.0, max_value=20.0, value=12.0, step=0.5,
                    help="Rata-rata nilai pada semester pertama (skala 0–20).",
                )
                sem1_approved = st.number_input(
                    "Mata Kuliah Lulus Sem 1", min_value=0, max_value=30, value=5, step=1,
                    help="Jumlah mata kuliah yang berhasil diluluskan pada semester pertama.",
                )
                sem1_evaluations = st.number_input(
                    "Evaluasi Semester 1", min_value=0, max_value=50, value=8, step=1,
                    help="Jumlah evaluasi/ujian yang diambil pada semester pertama.",
                )

        with col_input3:
            with st.container(border=True):
                st.markdown("#### 💳 Kondisi Finansial")
                tuition = st.selectbox(
                    "Status Pembayaran Kuliah", options=[1, 0], format_func=lambda x: "Lancar" if x == 1 else "Menunggak",
                    help="Apakah pembayaran kuliah mahasiswa berstatus lancar atau menunggak.",
                )
                debtor = st.selectbox(
                    "Status Debitur", options=[0, 1], format_func=lambda x: "Ya (Ada Tunggakan)" if x == 1 else "Tidak",
                    help="Apakah mahasiswa memiliki status sebagai debitur (memiliki hutang/tunggakan lain).",
                )
                scholarship = st.selectbox(
                    "Penerima Beasiswa", options=[0, 1], format_func=lambda x: "Ya" if x == 1 else "Tidak",
                    help="Apakah mahasiswa menerima beasiswa.",
                )

        st.markdown("<br>", unsafe_allow_html=True)

        # ── Predict Button ──
        predict_col1, predict_col2, predict_col3 = st.columns([1, 2, 1])
        with predict_col2:
            predict_btn = st.button(
                "🚀 PREDIKSI RISIKO DROPOUT",
                use_container_width=True,
                type="primary",
            )

        if predict_btn:
            try:
                # Build feature vector with all 28 features
                input_data = DEFAULT_VALUES.copy()
                input_data.update({
                    'Age at enrollment': float(age),
                    'Tuition fees up to date': float(tuition),
                    'Scholarship holder': float(scholarship),
                    'Debtor': float(debtor),
                    'Curricular units 1st sem (approved)': float(sem1_approved),
                    'Curricular units 1st sem (grade)': float(sem1_grade),
                    'Curricular units 1st sem (evaluations)': float(sem1_evaluations),
                })

                input_df = pd.DataFrame([input_data])[FEATURE_ORDER]

                # Predict
                prediction = model.predict(input_df)[0]

                # Probability
                has_proba = hasattr(model, "predict_proba")
                if has_proba:
                    proba = model.predict_proba(input_df)[0]
                    dropout_prob = proba[1] * 100  # class 1 = Dropout

                # ── Evaluate Contributing Factors ──
                positive_factors = []
                risk_factors = []
                
                if sem1_grade >= 13: positive_factors.append("Nilai semester 1 tinggi")
                elif sem1_grade < 10: risk_factors.append("Nilai semester 1 relatif rendah")
                
                if scholarship == 1: positive_factors.append("Menerima beasiswa pendidikan")
                else: risk_factors.append("Tidak menerima beasiswa")
                
                if tuition == 1: positive_factors.append("Pembayaran kuliah lancar")
                else: risk_factors.append("Terdapat tunggakan pembayaran kuliah")
                
                if sem1_approved >= 5: positive_factors.append("Jumlah mata kuliah lulus cukup baik")
                elif sem1_approved < 3: risk_factors.append("Jumlah mata kuliah lulus masih sedikit")
                
                if debtor == 1: risk_factors.append("Memiliki status debitur (tunggakan lain)")

                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

                # ── Result Display (Separated Info & Action) ──
                if has_proba:
                    if dropout_prob <= 30:
                        risk_level = "Risiko Rendah"
                        risk_label = "RISIKO DROPOUT RENDAH"
                        risk_icon = "🟢"
                        risk_color = "#00c853"
                        risk_class = "risk-low"
                        risk_detail = "Mahasiswa saat ini diprediksi aman dan tidak akan dropout. Mahasiswa menunjukkan indikator akademik dan finansial yang baik."
                        action_title = "Rekomendasi Tindakan"
                        risk_desc = "<ul><li>Pertahankan performa akademik saat ini</li><li>Lakukan pemantauan rutin pada tiap semester</li><li>Dorong partisipasi aktif dalam kegiatan kemahasiswaan</li></ul>"
                    elif dropout_prob <= 60:
                        risk_level = "Risiko Sedang"
                        risk_label = "RISIKO DROPOUT SEDANG"
                        risk_icon = "🟡"
                        risk_color = "#ff9100"
                        risk_class = "risk-medium"
                        risk_detail = "Mahasiswa saat ini tidak diprediksi akan dropout secara langsung, namun menunjukkan beberapa indikator risiko yang perlu mendapatkan perhatian lebih lanjut."
                        action_title = "Rekomendasi Tindakan"
                        risk_desc = "<ul><li>Lakukan monitoring akademik berkala</li><li>Jadwalkan konsultasi rutin dengan dosen pembimbing</li><li>Siapkan intervensi dini jika performa akademik mulai menurun</li></ul>"
                    else:
                        risk_level = "Risiko Tinggi"
                        risk_label = "RISIKO DROPOUT TINGGI"
                        risk_icon = "🔴"
                        risk_color = "#ff1744"
                        risk_class = "risk-high"
                        risk_detail = "Peringatan! Mahasiswa ini diprediksi sangat berisiko untuk mengalami dropout akibat berbagai faktor dominan yang menghambat."
                        action_title = "Tindakan Darurat (Wajib)"
                        risk_desc = "<ul><li>Lakukan pendampingan akademik intensif segera</li><li>Evaluasi kondisi finansial mahasiswa secara mendalam</li><li>Prioritaskan program intervensi dan konseling</li><li>Berikan dukungan berkelanjutan</li></ul>"
                else:
                    if prediction == 0:
                        risk_label = "RISIKO DROPOUT RENDAH"
                        risk_icon = "🟢"
                        risk_color = "#00c853"
                        risk_class = "risk-low"
                        risk_detail = "Mahasiswa menunjukkan indikator akademik dan finansial yang baik serta memiliki risiko dropout yang rendah."
                        action_title = "Rekomendasi Tindakan"
                        risk_desc = "Lanjutkan pemantauan reguler."
                    else:
                        risk_label = "RISIKO DROPOUT TINGGI"
                        risk_icon = "🔴"
                        risk_color = "#ff1744"
                        risk_class = "risk-high"
                        risk_detail = "Mahasiswa menunjukkan beberapa faktor yang berkaitan dengan risiko dropout dan memerlukan perhatian lebih lanjut."
                        action_title = "Tindakan Darurat (Wajib)"
                        risk_desc = "Lakukan intervensi akademik dan finansial."

                res_col1, res_col2 = st.columns(2)

                # CARD KIRI: Hasil Analisis
                with res_col1:
                    st.markdown(
                        f"<div class='{risk_class}'>"
                        f"<div style='font-size:1.1rem; font-weight:600; color:#888; margin-bottom:8px;'>Hasil Prediksi Model</div>"
                        f"<h2 style='color:{risk_color}; margin:0; display:flex; align-items:center; justify-content:center; gap:8px;'>{risk_icon} {risk_label}</h2>"
                        f"<p style='font-size:1rem; margin-top:16px; color:#ddd;'>{risk_detail}</p>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )

                # CARD KANAN: Rekomendasi
                with res_col2:
                    st.markdown(
                        f"<div class='{risk_class}' style='text-align: left;'>"
                        f"<div style='font-size:1.1rem; font-weight:600; color:{risk_color}; margin-bottom:8px; display:flex; align-items:center; gap:8px;'>📋 {action_title}</div>"
                        f"<div style='font-size:1rem; color:#ddd;'>{risk_desc}</div>"
                        f"</div>",
                        unsafe_allow_html=True,
                    )
                    
                # ── Tampilkan Faktor Penyebab ──
                st.markdown("<br>", unsafe_allow_html=True)
                fact_col1, fact_col2 = st.columns(2)
                with fact_col1:
                    if risk_factors:
                        st.error("**Faktor Risiko Dominan:**\n" + "\n".join([f"- {r}" for r in risk_factors]), icon="🟡")
                with fact_col2:
                    if positive_factors:
                        st.success("**Faktor Positif Pendukung:**\n" + "\n".join([f"- {p}" for p in positive_factors]), icon="🟢")

                # ── Probability Gauge ──
                if has_proba:
                    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                    st.markdown("### 📊 Distribusi Probabilitas")

                    gauge_col1, gauge_col2 = st.columns(2)

                    with gauge_col1:
                        fig_gauge = go.Figure(go.Indicator(
                            mode="gauge+number", value=dropout_prob,
                            number={'suffix': '%', 'font': {'size': 42}},
                            title={'text': "Probabilitas Dropout", 'font': {'size': 16}},
                            gauge={
                                'axis': {'range': [0, 100], 'tickwidth': 1},
                                'bar': {'color': "#667eea"},
                                'steps': [
                                    {'range': [0, 30], 'color': "rgba(0,200,83,0.19)"},
                                    {'range': [30, 60], 'color': "rgba(255,145,0,0.19)"},
                                    {'range': [60, 100], 'color': "rgba(255,23,68,0.19)"},
                                ],
                                'threshold': {'line': {'color': "#ff1744", 'width': 3}, 'thickness': 0.8, 'value': dropout_prob},
                            },
                        ))
                        gauge_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != 'margin'}
                        fig_gauge.update_layout(**gauge_layout, height=300, margin=dict(l=30, r=30, t=60, b=20))
                        st.plotly_chart(fig_gauge, use_container_width=True)

                    with gauge_col2:
                        fig_prob = go.Figure(go.Bar(
                            x=[proba[0] * 100, proba[1] * 100], y=['Tidak Dropout', 'Dropout'],
                            orientation='h', marker_color=['#667eea', '#ff5252'],
                            text=[f'{proba[0]*100:.1f}%', f'{proba[1]*100:.1f}%'], textposition='auto',
                        ))
                        prob_layout = {k: v for k, v in PLOTLY_LAYOUT.items() if k != 'margin'}
                        fig_prob.update_layout(**prob_layout, height=300, title="Probabilitas Kelas", xaxis_title="Probabilitas (%)", margin=dict(l=30, r=30, t=60, b=40))
                        st.plotly_chart(fig_prob, use_container_width=True)

                # ── Input Summary ──
                st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)
                with st.expander("📄 Lihat Ringkasan Input", expanded=False):
                    summary_df = pd.DataFrame({
                        'Fitur': ['Usia Saat Masuk', 'Status Pembayaran Kuliah', 'Penerima Beasiswa', 'Status Debitur', 'Jumlah Mata Kuliah Lulus Semester 1', 'Nilai Semester 1', 'Jumlah Evaluasi Semester 1'],
                        'Nilai': [age, "Lancar" if tuition == 1 else "Menunggak", "Ya" if scholarship == 1 else "Tidak", "Ya" if debtor == 1 else "Tidak", sem1_approved, sem1_grade, sem1_evaluations],
                    })
                    st.dataframe(summary_df, use_container_width=True, hide_index=True)

            except Exception as e:
                st.error(f"🔴 Prediksi gagal: {e}")


# ══════════════════════════════════════════════
#  PAGE 4 — ABOUT
# ══════════════════════════════════════════════
elif page == "ℹ️ Tentang Proyek":

    st.markdown(
        "<div class='page-header'>"
        "<h1>ℹ️ Tentang Proyek Ini</h1>"
        "<p>Detail proyek, alat yang digunakan, dan evaluasi performa model.</p>"
        "</div>",
        unsafe_allow_html=True,
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    col_info, col_author = st.columns(2)

    with col_info:
        st.markdown(
            "<div class='info-card'>"
            "<h4>📋 Informasi Proyek</h4>"
            "<p><strong>Student Performance & Dropout Risk Prediction</strong></p>"
            "<p>Sistem peringatan dini berbasis machine learning yang memprediksi apakah seorang mahasiswa berisiko mengalami dropout berdasarkan indikator akademik dan sosial ekonomi.</p>"
            "<p><strong>Dataset:</strong> <em>Higher Education Predictors of Student Retention</em></p>"
            "<p><strong>Sumber:</strong> <a href='https://www.kaggle.com/datasets/thedevastator/higher-education-predictors-of-student-retention' target='_blank' style='color:#667eea;'>Kaggle Dataset</a></p>"
            "</div>",
            unsafe_allow_html=True,
        )

    with col_author:
        st.markdown(
            "<div class='info-card'>"
            "<h4>👤 Penulis</h4>"
            "<p style='font-size: 1.2rem; font-weight: bold;'>Raihan Azka Hidayat</p>"
            "<p><em>Proyek ini dibangun sebagai bagian dari proyek final akademik dan penjurian Data Science.</em></p>"
            "</div>",
            unsafe_allow_html=True,
        )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Why Logistic Regression ──
    st.markdown("### 🧠 Mengapa Memilih Logistic Regression?")
    st.markdown(
        """
        Model **Logistic Regression** dipilih karena memberikan performa terbaik berdasarkan hasil evaluasi F1-Score pada dataset ini. 
        
        Selain memiliki akurasi yang kompetitif, Logistic Regression juga **lebih mudah diinterpretasikan** dibandingkan model yang lebih kompleks (seperti *Neural Networks* atau *Ensemble Trees*). Hal ini sangat penting karena hasil prediksi akan digunakan sebagai sistem pendukung keputusan di lingkungan pendidikan, sehingga alasan di balik prediksi (seperti faktor dominan) harus dapat dijelaskan dengan mudah kepada pihak kampus maupun dosen pembimbing.
        
        Keunggulan lain dari Logistic Regression adalah proses inferensi yang sangat cepat, ukuran model yang ringan, serta sangat cocok digunakan pada aplikasi web interaktif berbasis Streamlit.
        """
    )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Tools Used ──
    st.markdown("### 🛠️ Alat & Teknologi")
    t1, t2, t3, t4, t5 = st.columns(5)
    tools = [
        ("Python", "Bahasa Utama"), ("Pandas", "Manipulasi Data"),
        ("Scikit-Learn", "Machine Learning"), ("Streamlit", "Aplikasi Web"),
        ("Plotly", "Visualisasi"),
    ]
    for col, (name, desc) in zip([t1, t2, t3, t4, t5], tools):
        with col:
            st.markdown(
                f"<div class='info-card' style='text-align:center;'>"
                f"<h4 style='margin:8px 0 4px; justify-content:center;'>{name}</h4>"
                f"<p style='font-size:0.85rem; color:#888;'>{desc}</p>"
                f"</div>",
                unsafe_allow_html=True,
            )

    st.markdown("<hr class='section-divider'>", unsafe_allow_html=True)

    # ── Model Performance ──
    st.markdown("### 🏆 Performa Model Machine Learning")
    st.markdown(
        "<div class='info-card'>"
        "<h4 style='justify-content:flex-start;'>Model Terbaik: Logistic Regression</h4>"
        "<p>Pipeline: StandardScaler → LogisticRegression (class_weight='balanced', max_iter=1000)</p>"
        "</div>",
        unsafe_allow_html=True,
    )
    st.markdown("<br>", unsafe_allow_html=True)

    pm1, pm2, pm3, pm4 = st.columns(4)
    pm1.metric("Akurasi", "86%", help="Akurasi klasifikasi keseluruhan pada data uji")
    pm2.metric("Presisi", "78%", help="Proporsi prediksi dropout yang benar-benar dropout")
    pm3.metric("Recall", "81%", help="Proporsi dropout aktual yang diidentifikasi dengan benar")
    pm4.metric("F1-Score", "79%", help="Rata-rata harmonik dari Presisi dan Recall")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown("#### 📊 Perbandingan Performa Antar Model")
    comparison_data = pd.DataFrame({
        'Model': ['Logistic Regression', 'Decision Tree', 'Random Forest'],
        'Accuracy': [0.86, 0.79, 0.84], 'Precision': [0.78, 0.68, 0.80],
        'Recall': [0.81, 0.72, 0.71], 'F1-Score': [0.79, 0.70, 0.75],
    })

    fig_comp = go.Figure()
    metrics_to_plot = ['Accuracy', 'Precision', 'Recall', 'F1-Score']
    colors = ['#667eea', '#f093fb', '#4facfe', '#43e97b']
    for metric, color in zip(metrics_to_plot, colors):
        fig_comp.add_trace(go.Bar(
            name=metric, x=comparison_data['Model'], y=comparison_data[metric],
            marker_color=color, text=[f'{v:.0%}' for v in comparison_data[metric]], textposition='outside',
        ))

    fig_comp.update_layout(**PLOTLY_LAYOUT, barmode='group', height=420, legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="right", x=1))
    st.plotly_chart(fig_comp, use_container_width=True)

    st.caption("💡 Logistic Regression dipilih sebagai model terbaik berdasarkan **F1-Score** tertinggi, yang menyeimbangkan presisi dan recall — sangat penting untuk mendeteksi mahasiswa berisiko sekaligus meminimalkan alarm palsu.")
