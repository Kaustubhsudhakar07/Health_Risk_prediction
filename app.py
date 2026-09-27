"""
CardioHealth AI: Real-Time Patient Health Risk Prediction & Stratification Dashboard.
Built with Streamlit, Scikit-Learn, XGBoost, CatBoost, LightGBM, SHAP, Plotly, and Google Gemini 3.7 Flash.
Designed with human-crafted, high-contrast Senior Data Science & Medical Informatics aesthetics.
"""

import json
import os
import sys
from datetime import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

import src.features
import src.models
from src.ai_assistant import ask_gemini_health_assistant
from src.config import CLASS_NAMES, METADATA_SAVE_PATH, SAMPLE_DATA_PATH
from src.predict import HealthRiskPredictor
from src.visualizations import (
    plot_cohort_bar_metrics,
    plot_cohort_donut,
)

# Page Configuration
st.set_page_config(
    page_title="Health Care Prediction System",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Human-Crafted, High-Contrast Clinical Informatics Theme
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&display=swap');

    /* Global Fonts & Background Setup */
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: #0b0f19;
        color: #f1f5f9;
    }

    /* Top Hero Header */
    .hero-container {
        background: linear-gradient(135deg, #131b2e 0%, #0f172a 100%);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 18px;
        padding: 1.8rem 2.2rem;
        margin-bottom: 1.5rem;
        box-shadow: 0 10px 30px -10px rgba(0, 0, 0, 0.5);
    }
    .hero-pill {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: rgba(99, 102, 241, 0.12);
        border: 1px solid rgba(99, 102, 241, 0.3);
        color: #a5b4fc;
        font-size: 0.75rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.08em;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        margin-bottom: 0.6rem;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.3rem;
        font-weight: 800;
        letter-spacing: -0.02em;
        color: #ffffff;
        margin: 0;
        line-height: 1.2;
    }
    .hero-subtitle {
        color: #94a3b8;
        font-size: 0.95rem;
        margin-top: 0.4rem;
        max-width: 900px;
        line-height: 1.5;
    }

    /* Content Cards */
    .metric-card {
        background: #131b2e;
        border: 1px solid rgba(255, 255, 255, 0.07);
        border-radius: 14px;
        padding: 1.1rem 1.2rem;
        margin-bottom: 0.8rem;
    }
    .metric-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.06em;
    }
    .metric-card-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.45rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.2rem;
    }

    /* Verdict Banners */
    .verdict-banner {
        border-radius: 16px;
        padding: 1.4rem 1.8rem;
        margin-bottom: 1.2rem;
    }
    .verdict-fit {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.7) 0%, rgba(6, 95, 70, 0.5) 100%);
        border: 1px solid #10b981;
    }
    .verdict-at-risk {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.7) 0%, rgba(146, 64, 14, 0.5) 100%);
        border: 1px solid #f59e0b;
    }
    .verdict-unhealthy {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.7) 0%, rgba(153, 27, 27, 0.5) 100%);
        border: 1px solid #ef4444;
    }
    .verdict-title {
        font-family: 'Outfit', sans-serif;
        font-size: 1.8rem;
        font-weight: 800;
        margin-bottom: 0.2rem;
    }
    .verdict-fit .verdict-title { color: #6ee7b7; }
    .verdict-at-risk .verdict-title { color: #fde047; }
    .verdict-unhealthy .verdict-title { color: #fca5a5; }

    /* AI Response Card */
    .ai-chat-box {
        background: #131b2e;
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 16px;
        padding: 1.4rem 1.6rem;
        margin-top: 1rem;
        line-height: 1.6;
    }
    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: #4f46e5;
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .context-pill {
        background: rgba(16, 185, 129, 0.12);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 0.8rem;
    }

    /* Buttons */
    div[data-testid="stFormSubmitButton"] > button, .stButton > button {
        background: #4f46e5 !important;
        color: #ffffff !important;
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-weight: 700 !important;
        font-size: 1rem !important;
        padding: 0.65rem 1.8rem !important;
        border-radius: 12px !important;
        border: 1px solid rgba(255, 255, 255, 0.15) !important;
        transition: all 0.2s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover, .stButton > button:hover {
        background: #4338ca !important;
        border-color: rgba(255, 255, 255, 0.3) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Plus Jakarta Sans', sans-serif !important;
        font-size: 0.95rem !important;
        font-weight: 600 !important;
        color: #94a3b8 !important;
        padding: 0.7rem 1.4rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 2px solid #6366f1 !important;
        font-weight: 700 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Cached initialization of HealthRiskPredictor."""
    return HealthRiskPredictor()


@st.cache_data
def get_population_data():
    """Loads and caches reference population dataset."""
    if SAMPLE_DATA_PATH.exists():
        try:
            return pd.read_csv(SAMPLE_DATA_PATH)
        except Exception:
            pass
    return pd.DataFrame()


predictor = get_predictor()
df_population = get_population_data()

# Hero Header
st.markdown("""
<div class="hero-container">
    <div class="hero-title">Health Care Prediction System</div>
</div>
""", unsafe_allow_html=True)

# Preset Selector
preset_col1, preset_col2 = st.columns([1.2, 3.8])
with preset_col1:
    st.markdown("##### ⚡ Quick Patient Preset:")
with preset_col2:
    selected_preset = st.radio(
        "Choose Preset:",
        ["Custom Input", "Healthy Athlete (Fit)", "Desk Worker (At-Risk)", "High Distress (Unhealthy)"],
        horizontal=True,
        label_visibility="collapsed",
    )

presets_data = {
    "Healthy Athlete (Fit)": {
        "sleep_duration": 8.2, "heart_rate": 58.0, "bmi": 21.5, "calorie_expenditure": 2800.0,
        "step_count": 14000.0, "exercise_duration": 65.0, "water_intake": 3.5, "diet_type": "veg",
        "stress_level": "low", "sleep_quality": "good", "physical_activity_level": "active",
        "smoking_alcohol": "no", "gender": "female",
    },
    "Desk Worker (At-Risk)": {
        "sleep_duration": 6.0, "heart_rate": 78.0, "bmi": 26.8, "calorie_expenditure": 1950.0,
        "step_count": 4200.0, "exercise_duration": 15.0, "water_intake": 1.5, "diet_type": "non-veg",
        "stress_level": "moderate", "sleep_quality": "average", "physical_activity_level": "moderate",
        "smoking_alcohol": "occasional", "gender": "male",
    },
    "High Distress (Unhealthy)": {
        "sleep_duration": 4.5, "heart_rate": 89.0, "bmi": 33.2, "calorie_expenditure": 2300.0,
        "step_count": 2100.0, "exercise_duration": 5.0, "water_intake": 1.0, "diet_type": "non-veg",
        "stress_level": "high", "sleep_quality": "poor", "physical_activity_level": "sedentary",
        "smoking_alcohol": "yes", "gender": "male",
    },
}

active_preset = presets_data.get(selected_preset, {})

# Tabs
tab1, tab2, tab3 = st.tabs([
    "🩺 Real-Time Patient Assessment",
    "🤖 AI Clinical Assistant (Gemini 3.7)",
    "📁 Population Cohort Screening",
])


# =============================================================
# TAB 1: Real-Time Patient Assessment & Physiological Profiling
# =============================================================
with tab1:
    with st.form(key="patient_intake_form"):
        col1, col2, col3 = st.columns(3, gap="medium")

        # PANEL 1: Biometrics
        with col1:
            st.markdown("##### 🧬 Physiological Vitals")
            bmi = st.number_input(
                "Body Mass Index (BMI)",
                min_value=12.0, max_value=55.0,
                value=float(active_preset.get("bmi", 25.5)),
                step=0.1,
                help="WHO: <18.5 Underweight, 18.5-24.9 Normal, 25-29.9 Overweight, ≥30 Obese",
            )
            heart_rate = st.number_input(
                "Resting Heart Rate (bpm)",
                min_value=40.0, max_value=140.0,
                value=float(active_preset.get("heart_rate", 72.0)),
                step=1.0,
                help="Healthy resting heart rate: 60-100 bpm.",
            )
            gender = st.selectbox(
                "Biological Gender",
                ["male", "female", "other"],
                index=["male", "female", "other"].index(active_preset.get("gender", "male")),
            )
            water_intake = st.slider(
                "Daily Hydration (Liters/day)",
                min_value=0.5, max_value=6.0,
                value=float(active_preset.get("water_intake", 2.2)),
                step=0.1,
            )

        # PANEL 2: Activity
        with col2:
            st.markdown("##### 🏃 Activity & Energy Burn")
            step_count = st.number_input(
                "Daily Step Count",
                min_value=500.0, max_value=30000.0,
                value=float(active_preset.get("step_count", 6500.0)),
                step=500.0,
            )
            exercise_duration = st.slider(
                "Exercise Duration (Minutes/day)",
                min_value=0.0, max_value=180.0,
                value=float(active_preset.get("exercise_duration", 30.0)),
                step=5.0,
            )
            calorie_exp = st.number_input(
                "Daily Caloric Burn (kcal/day)",
                min_value=1000.0, max_value=5000.0,
                value=float(active_preset.get("calorie_expenditure", 2150.0)),
                step=50.0,
            )
            activity_level = st.selectbox(
                "Physical Activity Level",
                ["sedentary", "moderate", "active"],
                index=["sedentary", "moderate", "active"].index(active_preset.get("physical_activity_level", "moderate")),
            )

        # PANEL 3: Lifestyle
        with col3:
            st.markdown("##### 🧘 Sleep & Lifestyle Habits")
            sleep_duration = st.slider(
                "Sleep Duration (Hours/night)",
                min_value=3.0, max_value=12.0,
                value=float(active_preset.get("sleep_duration", 7.0)),
                step=0.25,
            )
            sleep_quality = st.selectbox(
                "Sleep Quality",
                ["good", "average", "poor"],
                index=["good", "average", "poor"].index(active_preset.get("sleep_quality", "average")),
            )
            stress_level = st.selectbox(
                "Perceived Stress Level",
                ["low", "moderate", "high"],
                index=["low", "moderate", "high"].index(active_preset.get("stress_level", "moderate")),
            )
            diet_type = st.selectbox(
                "Dietary Pattern",
                ["veg", "non-veg", "vegan"],
                index=["veg", "non-veg", "vegan"].index(active_preset.get("diet_type", "non-veg")),
            )
            smoking_alcohol = st.selectbox(
                "Smoking & Alcohol Consumption",
                ["no", "occasional", "yes"],
                index=["no", "occasional", "yes"].index(active_preset.get("smoking_alcohol", "no")),
            )

        st.markdown("")
        analyze_btn = st.form_submit_button(
            "⚡ RUN ENSEMBLE CLINICAL DIAGNOSIS",
            use_container_width=True,
        )

    patient_payload = {
        "bmi": bmi,
        "heart_rate": heart_rate,
        "sleep_duration": sleep_duration,
        "calorie_expenditure": calorie_exp,
        "step_count": step_count,
        "exercise_duration": exercise_duration,
        "water_intake": water_intake,
        "diet_type": diet_type,
        "stress_level": stress_level,
        "sleep_quality": sleep_quality,
        "physical_activity_level": activity_level,
        "smoking_alcohol": smoking_alcohol,
        "gender": gender,
    }

    if analyze_btn or "last_analysis" not in st.session_state:
        st.session_state["last_analysis"] = predictor.predict_single(patient_payload)
        st.session_state["analyzed_payload"] = patient_payload

    if "last_analysis" in st.session_state:
        prediction = st.session_state["last_analysis"]
        payload = st.session_state["analyzed_payload"]
        condition = prediction["predicted_condition"]
        confidence = prediction["confidence"]
        probs = prediction["probabilities"]

        st.markdown("---")

        # 1. Hero Verdict Card
        verdict_class = f"verdict-{condition}"
        badge_text = "🟢 HEALTH STATUS: FIT" if condition == "fit" else ("🟡 HEALTH STATUS: AT-RISK" if condition == "at-risk" else "🔴 HEALTH STATUS: UNHEALTHY")
        desc_text = (
            "Patient demonstrates optimal cardiovascular parameters, restorative sleep patterns, and low metabolic risk."
            if condition == "fit"
            else (
                "Moderate physiological stress detected. Preventative lifestyle adjustments are recommended to mitigate long-term cardiovascular risks."
                if condition == "at-risk"
                else "Elevated cardiometabolic distress indicators observed. Clinical consultation and immediate lifestyle interventions advised."
            )
        )

        st.markdown(f"""
        <div class="verdict-banner {verdict_class}">
            <div class="verdict-title">{badge_text}</div>
            <div style="font-size: 1.05rem; color: #f8fafc; margin-bottom: 0.3rem;">
                <b>Model Consensus:</b> XGBoost + CatBoost + LightGBM
            </div>
            <div style="color: rgba(255,255,255,0.85); font-size: 0.9rem;">{desc_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Vitals Quick KPI Grid
        bmi_val = payload["bmi"]
        bmi_status = "Normal" if 18.5 <= bmi_val < 25 else ("Overweight" if 25 <= bmi_val < 30 else ("Obese" if bmi_val >= 30 else "Underweight"))
        hr_val = payload["heart_rate"]
        hr_status = "Optimal" if hr_val < 75 else ("Moderate" if hr_val <= 85 else "Elevated")
        sleep_val = payload["sleep_duration"]
        sleep_status = "Adequate" if sleep_val >= 7.0 else "Deficient"
        steps_val = payload["step_count"]
        step_status = "Active" if steps_val >= 8000 else ("Moderate" if steps_val >= 5000 else "Sedentary")

        sc1, sc2, sc3, sc4 = st.columns(4)
        with sc1:
            st.markdown(f'<div class="metric-card"><div class="metric-card-title">BMI Status</div><div class="metric-card-val">{bmi_val:.1f} <span style="font-size:0.85rem; color:#38bdf8">({bmi_status})</span></div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="metric-card"><div class="metric-card-title">Resting Heart Rate</div><div class="metric-card-val">{hr_val:.0f} bpm <span style="font-size:0.85rem; color:#a78bfa">({hr_status})</span></div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="metric-card"><div class="metric-card-title">Sleep Duration</div><div class="metric-card-val">{sleep_val:.1f} hrs <span style="font-size:0.85rem; color:#f472b6">({sleep_status})</span></div></div>', unsafe_allow_html=True)
        with sc4:
            st.markdown(f'<div class="metric-card"><div class="metric-card-title">Daily Steps</div><div class="metric-card-val">{steps_val:,.0f} <span style="font-size:0.85rem; color:#34d399">({step_status})</span></div></div>', unsafe_allow_html=True)




# =============================================================
# TAB 2: AI Clinical Assistant (Powered by Gemini 3.7 Flash)
# =============================================================
with tab2:
    st.markdown("### 🤖 AI Clinical Health Assistant (Google Gemini 3.7 Flash)")
    st.caption("Ask clinical questions, request personalized diet and exercise interventions, or explore biomarker interactions powered by Google Gemini 3.7 Flash.")

    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    with st.expander("🔑 Configure Google Gemini 3.7 Flash API Key", expanded=not bool(gemini_key)):
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            entered_key = st.text_input(
                "Gemini API Key:",
                value=st.session_state.get("gemini_key", gemini_key),
                type="password",
                placeholder="AIzaSy...",
                help="Get a free API key from Google AI Studio",
            )
            if entered_key:
                st.session_state["gemini_key"] = entered_key
        with col_k2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("[👉 Get Free API Key](https://aistudio.google.com/app/apikey)")

    active_api_key = st.session_state.get("gemini_key", gemini_key)

    if "last_analysis" in st.session_state:
        p_res = st.session_state["last_analysis"]
        p_load = st.session_state["analyzed_payload"]
        cond_str = p_res["predicted_condition"].upper()

        st.markdown(f"""
        <div class="context-pill">
            <span>🩺 <b>Active Patient Context:</b> Status <b>{cond_str}</b> (BMI: {p_load['bmi']}, HR: {p_load['heart_rate']} bpm, Sleep: {p_load['sleep_duration']}h, Steps: {p_load['step_count']})</span>
        </div>
        """, unsafe_allow_html=True)
        include_context = st.checkbox("Attach active patient biometrics to Gemini clinical prompt", value=True)
    else:
        include_context = False

    st.markdown("##### 💬 Clinical Inquiry Presets:")
    quick_prompts = [
        "🏃 Create a personalized 7-day workout & meal plan based on my vitals",
        "❤️ What are the most effective ways to lower my resting heart rate naturally?",
        "🥗 How does sleep deprivation directly impact metabolic health and BMI?",
        "🔍 Explain why my health status was classified this way and what to prioritize",
    ]

    selected_prompt = None
    qcol1, qcol2 = st.columns(2)
    for i, q in enumerate(quick_prompts):
        target_col = qcol1 if i % 2 == 0 else qcol2
        if target_col.button(q, key=f"quick_btn_{i}", use_container_width=True):
            selected_prompt = q

    st.markdown("---")
    user_query = st.text_area(
        "Ask a Clinical, Biomarker, or Lifestyle Question:",
        value=selected_prompt if selected_prompt else "",
        placeholder="e.g., How can I reduce cardiometabolic risk if I work a sedentary desk job 10 hours a day?",
        height=90,
    )

    if st.button("✨ Consult Gemini 3.7 Flash Assistant", use_container_width=True):
        if not user_query.strip():
            st.warning("Please enter a question first.")
        else:
            p_context = None
            if include_context and "last_analysis" in st.session_state:
                p_context = {
                    "predicted_condition": st.session_state["last_analysis"]["predicted_condition"],
                    "confidence": st.session_state["last_analysis"]["confidence"],
                    "payload": st.session_state["analyzed_payload"],
                }

            with st.spinner("🧠 Consulting Google Gemini 3.7 Flash Clinical Intelligence..."):
                ai_response = ask_gemini_health_assistant(
                    user_question=user_query,
                    api_key=active_api_key,
                    patient_context=p_context,
                )

            st.markdown(f"""
            <div class="ai-chat-box">
                <div class="ai-badge">✨ Google Gemini 3.7 Flash Clinical Intelligence</div>
                <div>{ai_response}</div>
            </div>
            """, unsafe_allow_html=True)


# =============================================================
# TAB 3: Batch Cohort Screening & Population Health
# =============================================================
with tab3:
    st.markdown("### 📁 Population Cohort Screening & Risk Analytics")
    st.caption("Screen multi-patient populations, compare biomarker averages across risk tiers, and export stratified clinical records.")

    c_dl, c_up = st.columns([1, 2])
    with c_dl:
        if SAMPLE_DATA_PATH.exists():
            with open(SAMPLE_DATA_PATH, "rb") as f:
                st.download_button(
                    "📥 Download Patient Cohort Template (CSV)",
                    data=f,
                    file_name="sample_patient_cohort.csv",
                    mime="text/csv",
                )

    with c_up:
        uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"], label_visibility="collapsed")

    # Load uploaded file or fallback to cached 3,000 reference cohort
    if uploaded_file is not None:
        cohort_raw = pd.read_csv(uploaded_file)
        with st.spinner("Executing Soft-Voting Ensemble on uploaded cohort..."):
            cohort_df = predictor.predict_batch(cohort_raw)
    elif not df_population.empty:
        cohort_df = df_population.copy()
        if "predicted_health_condition" not in cohort_df.columns and "health_condition" in cohort_df.columns:
            cohort_df["predicted_health_condition"] = cohort_df["health_condition"]
    else:
        cohort_df = pd.DataFrame()

    if not cohort_df.empty:
        # Top KPI Metrics
        total_patients = len(cohort_df)
        cond_col = "predicted_health_condition" if "predicted_health_condition" in cohort_df.columns else "health_condition"
        counts = cohort_df[cond_col].value_counts(normalize=True)

        k1, k2, k3, k4, k5 = st.columns(5)
        k1.metric("Screened Cohort", f"{total_patients:,} Patients")
        k2.metric("Fit Rate", f"{counts.get('fit', 0)*100:.1f}%")
        k3.metric("At-Risk Rate", f"{counts.get('at-risk', 0)*100:.1f}%")
        k4.metric("Unhealthy Rate", f"{counts.get('unhealthy', 0)*100:.1f}%")
        mean_bmi = cohort_df["bmi"].mean() if "bmi" in cohort_df.columns else 25.0
        k5.metric("Mean Cohort BMI", f"{mean_bmi:.1f}")

        st.markdown("---")

        # Row 1 Graphs: Donut + Bar Metrics
        cr1, cr2 = st.columns([1.1, 1.9], gap="medium")
        with cr1:
            fig_donut = plot_cohort_donut(cohort_df)
            st.plotly_chart(fig_donut, use_container_width=True)

        with cr2:
            fig_bars = plot_cohort_bar_metrics(cohort_df)
            st.plotly_chart(fig_bars, use_container_width=True)

        st.markdown("---")
        st.markdown("##### 📋 Screened Patient Records Table")
        st.dataframe(cohort_df.head(15), use_container_width=True)

        csv_data = cohort_df.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Complete Stratified Cohort CSV",
            data=csv_data,
            file_name="cohort_predictions.csv",
            mime="text/csv",
        )

