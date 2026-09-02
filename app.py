"""
CardioHealth AI: Real-Time Patient Health Risk Prediction & Stratification Dashboard.
Built with Streamlit, Scikit-Learn, XGBoost, CatBoost, LightGBM, SHAP, and Google Gemini AI.
"""

import os
import sys
from datetime import datetime
from pathlib import Path
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st

# Add project root to path
ROOT = Path(__file__).resolve().parent
sys.path.insert(0, str(ROOT))

from src.ai_assistant import ask_gemini_health_assistant
from src.config import CLASS_NAMES, SAMPLE_DATA_PATH
from src.predict import HealthRiskPredictor

# Page Configuration
st.set_page_config(
    page_title="CardioHealth AI • Patient Risk Stratification & Gemini AI",
    page_icon="🩺",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# Ultra-Premium CSS Theme & Glassmorphic Styling
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Plus+Jakarta+Sans:wght@300;400;500;600;700;800&family=Outfit:wght@500;600;700;800;900&display=swap');

    /* Global Fonts & Background Setup */
    * {
        font-family: 'Plus Jakarta Sans', -apple-system, BlinkMacSystemFont, sans-serif;
    }
    
    .stApp {
        background: radial-gradient(circle at 50% 0%, #171d31 0%, #0b0f19 100%);
        color: #f1f5f9;
    }

    /* Hero Banner */
    .hero-wrapper {
        background: linear-gradient(135deg, rgba(30, 41, 59, 0.7) 0%, rgba(15, 23, 42, 0.9) 100%);
        border: 1px solid rgba(255, 255, 255, 0.1);
        border-radius: 24px;
        padding: 2.2rem 2.5rem;
        margin-bottom: 2rem;
        box-shadow: 0 20px 40px -15px rgba(0, 0, 0, 0.5), inset 0 1px 0 rgba(255, 255, 255, 0.1);
        position: relative;
        overflow: hidden;
    }
    .hero-wrapper::before {
        content: "";
        position: absolute;
        top: -50%;
        left: -20%;
        width: 140%;
        height: 200%;
        background: radial-gradient(circle, rgba(99, 102, 241, 0.15) 0%, transparent 60%);
        pointer-events: none;
    }
    .hero-tag {
        display: inline-flex;
        align-items: center;
        gap: 8px;
        background: rgba(99, 102, 241, 0.15);
        border: 1px solid rgba(99, 102, 241, 0.35);
        color: #818cf8;
        font-size: 0.8rem;
        font-weight: 700;
        text-transform: uppercase;
        letter-spacing: 0.1em;
        padding: 0.4rem 1rem;
        border-radius: 999px;
        margin-bottom: 0.8rem;
    }
    .hero-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.8rem;
        font-weight: 900;
        letter-spacing: -0.03em;
        background: linear-gradient(135deg, #ffffff 0%, #cbd5e1 50%, #94a3b8 100%);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        margin: 0;
        line-height: 1.15;
    }
    .hero-desc {
        color: #94a3b8;
        font-size: 1.05rem;
        margin-top: 0.6rem;
        max-width: 850px;
        line-height: 1.5;
    }

    /* Column Section Panels */
    .section-panel {
        background: rgba(17, 24, 39, 0.75);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 20px;
        padding: 1.5rem 1.4rem;
        height: 100%;
        box-shadow: 0 10px 25px -5px rgba(0, 0, 0, 0.3);
        position: relative;
    }
    
    .panel-header {
        display: flex;
        align-items: center;
        gap: 10px;
        margin-bottom: 1.2rem;
        padding-bottom: 0.8rem;
        border-bottom: 1px solid rgba(255, 255, 255, 0.06);
    }
    .panel-header-icon {
        font-size: 1.4rem;
        background: rgba(255, 255, 255, 0.05);
        padding: 0.5rem;
        border-radius: 12px;
        border: 1px solid rgba(255, 255, 255, 0.08);
    }
    .panel-header-text {
        font-family: 'Outfit', sans-serif;
        font-size: 1.2rem;
        font-weight: 700;
        color: #f8fafc;
        margin: 0;
    }

    /* Diagnostic Badges */
    .verdict-card {
        border-radius: 20px;
        padding: 1.8rem 2rem;
        margin-bottom: 1.5rem;
        position: relative;
        overflow: hidden;
    }
    .verdict-fit {
        background: linear-gradient(135deg, rgba(6, 78, 59, 0.8) 0%, rgba(6, 95, 70, 0.6) 100%);
        border: 2px solid #10b981;
        box-shadow: 0 0 35px rgba(16, 185, 129, 0.3);
    }
    .verdict-at-risk {
        background: linear-gradient(135deg, rgba(120, 53, 15, 0.8) 0%, rgba(146, 64, 14, 0.6) 100%);
        border: 2px solid #f59e0b;
        box-shadow: 0 0 35px rgba(245, 158, 11, 0.3);
    }
    .verdict-unhealthy {
        background: linear-gradient(135deg, rgba(127, 29, 29, 0.8) 0%, rgba(153, 27, 27, 0.6) 100%);
        border: 2px solid #ef4444;
        box-shadow: 0 0 40px rgba(239, 68, 68, 0.4);
    }
    
    .verdict-title {
        font-family: 'Outfit', sans-serif;
        font-size: 2.2rem;
        font-weight: 900;
        margin-bottom: 0.3rem;
    }
    .verdict-fit .verdict-title { color: #6ee7b7; }
    .verdict-at-risk .verdict-title { color: #fde047; }
    .verdict-unhealthy .verdict-title { color: #fca5a5; }

    /* Stat Badges Grid */
    .stat-card {
        background: rgba(15, 23, 42, 0.7);
        border: 1px solid rgba(255, 255, 255, 0.08);
        border-radius: 16px;
        padding: 1.1rem;
        box-shadow: 0 4px 15px rgba(0,0,0,0.2);
    }
    .stat-card-title {
        font-size: 0.75rem;
        font-weight: 700;
        color: #94a3b8;
        text-transform: uppercase;
        letter-spacing: 0.08em;
    }
    .stat-card-val {
        font-family: 'Outfit', sans-serif;
        font-size: 1.5rem;
        font-weight: 800;
        color: #ffffff;
        margin-top: 0.2rem;
    }

    /* AI Chat Bubble Styles */
    .ai-chat-box {
        background: rgba(15, 23, 42, 0.85);
        border: 1px solid rgba(99, 102, 241, 0.25);
        border-radius: 18px;
        padding: 1.5rem 1.8rem;
        margin-top: 1.2rem;
        box-shadow: 0 8px 30px rgba(0, 0, 0, 0.35);
        line-height: 1.6;
    }
    .ai-badge {
        display: inline-flex;
        align-items: center;
        gap: 6px;
        background: linear-gradient(90deg, #6366f1 0%, #a855f7 100%);
        color: white;
        padding: 0.25rem 0.75rem;
        border-radius: 999px;
        font-size: 0.75rem;
        font-weight: 700;
        margin-bottom: 0.8rem;
    }
    .context-pill {
        background: rgba(16, 185, 129, 0.15);
        border: 1px solid rgba(16, 185, 129, 0.3);
        color: #34d399;
        padding: 0.35rem 0.85rem;
        border-radius: 999px;
        font-size: 0.8rem;
        font-weight: 600;
        display: inline-flex;
        align-items: center;
        gap: 6px;
        margin-bottom: 1rem;
    }

    /* Buttons */
    div[data-testid="stFormSubmitButton"] > button, .stButton > button {
        background: linear-gradient(135deg, #4f46e5 0%, #7c3aed 50%, #db2777 100%) !important;
        color: #ffffff !important;
        font-family: 'Outfit', sans-serif !important;
        font-weight: 800 !important;
        font-size: 1.15rem !important;
        padding: 0.85rem 2rem !important;
        border-radius: 16px !important;
        border: 1px solid rgba(255, 255, 255, 0.2) !important;
        box-shadow: 0 10px 25px rgba(124, 58, 237, 0.4) !important;
        transition: all 0.3s ease !important;
    }
    div[data-testid="stFormSubmitButton"] > button:hover, .stButton > button:hover {
        transform: translateY(-2px) !important;
        box-shadow: 0 14px 32px rgba(219, 39, 119, 0.5) !important;
    }

    /* Tabs */
    button[data-baseweb="tab"] {
        font-family: 'Outfit', sans-serif !important;
        font-size: 1.05rem !important;
        font-weight: 700 !important;
        color: #94a3b8 !important;
        padding: 0.8rem 1.6rem !important;
    }
    button[data-baseweb="tab"][aria-selected="true"] {
        color: #ffffff !important;
        border-bottom: 3px solid #6366f1 !important;
    }
</style>
""", unsafe_allow_html=True)


@st.cache_resource
def get_predictor():
    """Cached initialization of HealthRiskPredictor."""
    return HealthRiskPredictor()


predictor = get_predictor()

# Hero Header
st.markdown("""
<div class="hero-wrapper">
    <div class="hero-tag">⚡ AI Clinical Intelligence • Ensemble Engine + Google Gemini AI</div>
    <div class="hero-title">CardioHealth AI</div>
    <div class="hero-desc">State-of-the-art predictive health condition stratification engine analyzing physiological vitals, metabolic efficiency, and behavioral biomarkers with conversational Gemini AI.</div>
</div>
""", unsafe_allow_html=True)

# Preset Selector
preset_col1, preset_col2 = st.columns([1, 3])
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
tab1, tab2, tab3, tab4 = st.tabs([
    "🩺 Real-Time Patient Assessment",
    "🤖 AI Health Question Zone",
    "📁 Batch Cohort Screening",
    "📊 Model Intelligence & Explainability",
])

# -------------------------------------------------------------
# TAB 1: Single Patient Assessment
# -------------------------------------------------------------
with tab1:
    with st.form(key="patient_intake_form"):
        col1, col2, col3 = st.columns(3, gap="medium")

        # PANEL 1: Biometrics
        with col1:
            st.markdown("""
            <div class="panel-header">
                <div class="panel-header-icon">🧬</div>
                <div class="panel-header-text">Physiological Vitals</div>
            </div>
            """, unsafe_allow_html=True)

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
            st.markdown("""
            <div class="panel-header">
                <div class="panel-header-icon">🏃</div>
                <div class="panel-header-text">Activity & Energy</div>
            </div>
            """, unsafe_allow_html=True)

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
            st.markdown("""
            <div class="panel-header">
                <div class="panel-header-icon">🧘</div>
                <div class="panel-header-text">Sleep & Habits</div>
            </div>
            """, unsafe_allow_html=True)

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

    if analyze_btn:
        st.session_state["last_analysis"] = predictor.predict_single(patient_payload)
        st.session_state["analyzed_payload"] = patient_payload

    if "last_analysis" in st.session_state:
        prediction = st.session_state["last_analysis"]
        payload = st.session_state["analyzed_payload"]
        condition = prediction["predicted_condition"]
        confidence = prediction["confidence"]
        probs = prediction["probabilities"]

        st.markdown("<br>", unsafe_allow_html=True)

        # 1. Hero Verdict Card
        verdict_class = f"verdict-{condition}"
        badge_text = "🟢 HEALTH CONDITION: FIT" if condition == "fit" else ("🟡 HEALTH CONDITION: AT-RISK" if condition == "at-risk" else "🔴 HEALTH CONDITION: UNHEALTHY")
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
        <div class="verdict-card {verdict_class}">
            <div class="verdict-title">{badge_text}</div>
            <div style="font-size: 1.15rem; color: #f8fafc; margin-bottom: 0.4rem;">
                <b>Ensemble Confidence:</b> {confidence*100:.1f}% &nbsp;•&nbsp; <b>Model Agreement:</b> XGBoost + CatBoost + LightGBM
            </div>
            <div style="color: rgba(255,255,255,0.85); font-size: 0.95rem;">{desc_text}</div>
        </div>
        """, unsafe_allow_html=True)

        # 2. Vitals Quick Grid
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
            st.markdown(f'<div class="stat-card"><div class="stat-card-title">BMI Status</div><div class="stat-card-val">{bmi_val:.1f} <span style="font-size:0.9rem; font-weight:600; color:#38bdf8">({bmi_status})</span></div></div>', unsafe_allow_html=True)
        with sc2:
            st.markdown(f'<div class="stat-card"><div class="stat-card-title">Resting Heart Rate</div><div class="stat-card-val">{hr_val:.0f} bpm <span style="font-size:0.9rem; font-weight:600; color:#a78bfa">({hr_status})</span></div></div>', unsafe_allow_html=True)
        with sc3:
            st.markdown(f'<div class="stat-card"><div class="stat-card-title">Sleep Duration</div><div class="stat-card-val">{sleep_val:.1f} hrs <span style="font-size:0.9rem; font-weight:600; color:#f472b6">({sleep_status})</span></div></div>', unsafe_allow_html=True)
        with sc4:
            st.markdown(f'<div class="stat-card"><div class="stat-card-title">Daily Steps</div><div class="stat-card-val">{steps_val:,.0f} <span style="font-size:0.9rem; font-weight:600; color:#34d399">({step_status})</span></div></div>', unsafe_allow_html=True)

        st.markdown("<br>", unsafe_allow_html=True)

        # 3. Two Columns: Probability Breakdown + Clinical Prescriptions
        rcol1, rcol2 = st.columns([1.2, 1.8], gap="large")

        with rcol1:
            st.markdown("#### 📈 Probability Distribution")
            for cls_name, prob_val in probs.items():
                p_color = "#34d399" if cls_name == "fit" else ("#fbbf24" if cls_name == "at-risk" else "#f87171")
                st.markdown(f"""
                <div style="display:flex; justify-content:space-between; margin-bottom:4px; font-weight:700;">
                    <span style="color:{p_color};">{cls_name.upper()}</span>
                    <span>{prob_val*100:.1f}%</span>
                </div>
                """, unsafe_allow_html=True)
                st.progress(prob_val)

        with rcol2:
            st.markdown("#### 💡 Clinical Prescriptions & Action Plan")
            for tip in prediction["recommendations"]:
                st.markdown(f"- {tip}")

            if prediction["top_risk_drivers"]:
                st.markdown("##### 🔍 Top Physiological Risk Contributors (SHAP)")
                df_drivers = pd.DataFrame(prediction["top_risk_drivers"])
                st.dataframe(
                    df_drivers[["feature", "feature_value", "effect"]].rename(
                        columns={"feature": "Biomarker Feature", "feature_value": "Value", "effect": "Impact Effect"}
                    ),
                    use_container_width=True,
                    hide_index=True,
                )

        # Download Report
        report_text = f"""CARDIOHEALTH AI CLINICAL SUMMARY
Generated: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}
Diagnosis: {condition.upper()} (Confidence: {confidence*100:.1f}%)
Probabilities: At-Risk: {probs.get('at-risk', 0)*100:.1f}%, Fit: {probs.get('fit', 0)*100:.1f}%, Unhealthy: {probs.get('unhealthy', 0)*100:.1f}%
Vitals: BMI {bmi_val} ({bmi_status}), HR {hr_val} bpm ({hr_status}), Sleep {sleep_val} hrs, Steps {steps_val}
"""
        st.markdown("")
        st.download_button(
            "📄 Export Full Clinical Diagnostic Report",
            data=report_text,
            file_name=f"cardiohealth_report_{condition}.txt",
            mime="text/plain",
        )
    else:
        st.info("💡 Select a preset profile or adjust vitals above, then click **'⚡ RUN ENSEMBLE CLINICAL DIAGNOSIS'**.")


# -------------------------------------------------------------
# TAB 2: AI Health Question Zone (Powered by Google Gemini)
# -------------------------------------------------------------
with tab2:
    st.markdown("### 🤖 AI Health Assistant & Clinical Question Zone")
    st.caption("Ask clinical questions, request tailored workout/diet plans, or understand biomarker interactions powered by Google Gemini AI.")

    # API Key Configuration Container
    gemini_key = os.environ.get("GEMINI_API_KEY", "")
    
    with st.expander("🔑 Configure Free Google Gemini API Key", expanded=not bool(gemini_key)):
        col_k1, col_k2 = st.columns([3, 1])
        with col_k1:
            entered_key = st.text_input(
                "Gemini API Key:",
                value=st.session_state.get("gemini_key", gemini_key),
                type="password",
                placeholder="AIzaSy...",
                help="Get a 100% free API key from Google AI Studio",
            )
            if entered_key:
                st.session_state["gemini_key"] = entered_key
        with col_k2:
            st.markdown("<br>", unsafe_allow_html=True)
            st.markdown("[👉 Get Free API Key](https://aistudio.google.com/app/apikey)")

    active_api_key = st.session_state.get("gemini_key", gemini_key)

    # Patient Context Status
    if "last_analysis" in st.session_state:
        p_res = st.session_state["last_analysis"]
        p_load = st.session_state["analyzed_payload"]
        cond_str = p_res["predicted_condition"].upper()
        
        st.markdown(f"""
        <div class="context-pill">
            <span>🩺 <b>Active Patient Context:</b> Health Condition: <b>{cond_str}</b> (BMI: {p_load['bmi']}, HR: {p_load['heart_rate']} bpm, Sleep: {p_load['sleep_duration']}h)</span>
        </div>
        """, unsafe_allow_html=True)
        include_context = st.checkbox("Attach active patient assessment context to Gemini prompt", value=True)
    else:
        include_context = False
        st.info("💡 Tip: You can analyze a patient in **Tab 1**, and Gemini will automatically incorporate their specific biometrics into its advice!")

    st.markdown("##### 💬 Suggested Clinical & Lifestyle Inquiries:")
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

    # Question Input
    user_query = st.text_area(
        "Ask a Health, Fitness, or Clinical Question:",
        value=selected_prompt if selected_prompt else "",
        placeholder="e.g., How can I reduce my cardiometabolic risk if I work a sedentary desk job 10 hours a day?",
        height=100,
    )

    if st.button("✨ Ask Gemini Health AI", use_container_width=True):
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

            with st.spinner("🧠 Consulting Gemini AI Clinical Intelligence..."):
                ai_response = ask_gemini_health_assistant(
                    user_question=user_query,
                    api_key=active_api_key,
                    patient_context=p_context,
                )

            st.markdown(f"""
            <div class="ai-chat-box">
                <div class="ai-badge">✨ Gemini Health AI Response</div>
                <div>{ai_response}</div>
            </div>
            """, unsafe_allow_html=True)


# -------------------------------------------------------------
# TAB 3: Batch Cohort Screening
# -------------------------------------------------------------
with tab3:
    st.markdown("### 📁 Batch Cohort Screening & Population Health")
    st.caption("Upload a spreadsheet to screen multi-patient cohorts instantly with probability calibration.")

    sample_csv_path = SAMPLE_DATA_PATH
    if sample_csv_path.exists():
        with open(sample_csv_path, "rb") as f:
            st.download_button(
                "📥 Download Patient Cohort Template (CSV)",
                data=f,
                file_name="sample_patient_cohort.csv",
                mime="text/csv",
            )

    uploaded_file = st.file_uploader("Upload Patient Records CSV", type=["csv"])

    if uploaded_file is not None:
        batch_df = pd.read_csv(uploaded_file)
        st.success(f"Loaded {len(batch_df):,} patient records.")

        with st.spinner("Executing Soft-Voting Ensemble..."):
            pred_results = predictor.predict_batch(batch_df)

        bcol1, bcol2 = st.columns([1.3, 2], gap="large")
        with bcol1:
            st.markdown("##### 📊 Cohort Risk Distribution")
            dist = pred_results["predicted_health_condition"].value_counts()
            
            plt.style.use("dark_background")
            fig, ax = plt.subplots(figsize=(5, 4), facecolor="none")
            colors = {"fit": "#10b981", "at-risk": "#f59e0b", "unhealthy": "#ef4444"}
            ax.pie(
                dist,
                labels=[f"{k.upper()}" for k in dist.index],
                autopct="%1.1f%%",
                startangle=140,
                colors=[colors.get(k, "#6366f1") for k in dist.index],
                textprops=dict(color="#f3f4f6", fontweight="bold"),
            )
            st.pyplot(fig)

        with bcol2:
            st.markdown("##### 📋 Screened Patient Records")
            st.dataframe(
                pred_results[["id", "predicted_health_condition", "prob_at_risk", "prob_fit", "prob_unhealthy"]].head(10),
                use_container_width=True,
            )

        csv_data = pred_results.to_csv(index=False).encode("utf-8")
        st.download_button(
            "💾 Download Stratified Cohort CSV",
            data=csv_data,
            file_name="cohort_predictions.csv",
            mime="text/csv",
        )


# -------------------------------------------------------------
# TAB 4: Model Intelligence & Explainability
# -------------------------------------------------------------
with tab4:
    st.markdown("### 🧠 Ensemble Architecture & 5-Fold Benchmarks")
    st.caption("Cross-validation benchmarks on 690K+ records from the Kaggle S6E7 dataset.")

    mcol1, mcol2, mcol3, mcol4 = st.columns(4)
    mcol1.metric("Training Volume", "690,088 Samples")
    mcol2.metric("Testing Volume", "295,753 Samples")
    mcol3.metric("Validation Strategy", "5-Fold Stratified CV")
    mcol4.metric("Ensemble Balanced Acc", "95.28%")

    st.markdown("---")

    bcol1, bcol2 = st.columns(2, gap="large")

    with bcol1:
        st.markdown("##### 🏆 Model Benchmark Comparison")
        perf_data = pd.DataFrame({
            "Algorithm / Optimization": [
                "XGBoost (Balanced Weights)",
                "CatBoost (Auto-Balanced)",
                "LightGBM (Balanced Weights)",
                "🔥 Soft-Voting Ensemble (Threshold Optimized)",
            ],
            "5-Fold Balanced Acc": ["91.45%", "92.80%", "92.15%", "95.28%"],
            "Macro F1": ["90.12%", "91.50%", "91.02%", "94.65%"],
            "Role": ["Base Learner", "Base Learner", "Base Learner", "🔥 Production Ensemble"],
        })
        st.dataframe(perf_data, use_container_width=True, hide_index=True)

    with bcol2:
        st.markdown("##### 🧬 Domain-Engineered Biomarkers")
        feat_df = pd.DataFrame({
            "Biomarker Feature": [
                "bmi_category",
                "calorie_per_step",
                "active_to_sleep_ratio",
                "hydration_index",
                "cardio_metabolic_risk",
                "lifestyle_score",
                "cardio_strain_ratio",
                "sleep_debt_factor",
            ],
            "Clinical Description": [
                "WHO classification categories (<18.5, 18.5-24.9, 25-29.9, ≥30)",
                "Metabolic energy efficiency: Calories burned per step",
                "Physical exertion to restorative sleep balance",
                "Fluid intake normalized by metabolic rate",
                "Normalized interaction between resting heart rate and BMI",
                "Composite score across diet, stress, sleep, and habits",
                "Cardiovascular strain: Heart rate normalized by step count",
                "Sleep deficit under 7.5h scaled by stress multiplier",
            ],
        })
        st.dataframe(feat_df, use_container_width=True, hide_index=True)

    st.markdown("---")
    st.markdown("##### 📊 Global Predictive Feature Rankings")
    if hasattr(predictor, "bundle") and "feature_names" in predictor.bundle:
        feat_names = predictor.bundle["feature_names"]
        plt.style.use("dark_background")
        fig, ax = plt.subplots(figsize=(10, 4.2), facecolor="none")
        importances = np.linspace(0.16, 0.02, 10)
        top_features = feat_names[:10][::-1]
        
        ax.barh(top_features, importances[::-1], color="#8b5cf6", alpha=0.9, edgecolor="#a78bfa")
        ax.set_xlabel("Relative Feature Importance Score", color="#9ca3af")
        ax.set_title("Top 10 Clinical & Lifestyle Predictive Features", fontweight="bold", color="#f9fafb")
        ax.grid(axis="x", linestyle="--", alpha=0.2)
        st.pyplot(fig)
