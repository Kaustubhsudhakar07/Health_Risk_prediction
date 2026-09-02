"""
Gemini AI Clinical Assistant & Health Q&A Integration.
Provides context-aware conversational health insights, personalized medical explanations,
and lifestyle guidance powered by Google Gemini API.
"""

import os
import google.generativeai as genai

SYSTEM_INSTRUCTION = """
You are 'CardioHealth AI Assistant', an empathetic, highly knowledgeable, evidence-based clinical intelligence assistant specializing in preventive cardiology, metabolic health, lifestyle medicine, and biomarker analysis.

Your responsibilities:
1. Explain health metrics (BMI, resting heart rate, sleep duration, calorie expenditure, water intake, stress, smoking/alcohol risks).
2. If patient assessment data is provided, analyze their specific condition (Fit, At-Risk, Unhealthy), explain why certain indicators increase or decrease risk, and provide customized, actionable lifestyle prescriptions (dietary adjustments, exercise regimens, sleep hygiene, stress reduction).
3. Always maintain an encouraging, scientific, and professional tone. Use clear formatting with bullet points and bold highlights.
4. Include a standard medical disclaimer reminding users that you provide educational guidance and they should consult licensed healthcare professionals for definitive diagnoses.
"""


def ask_gemini_health_assistant(user_question: str, api_key: str = None, patient_context: dict = None) -> str:
    """
    Sends user query and optional patient context to Google Gemini API.
    """
    key = api_key or os.environ.get("GEMINI_API_KEY", "").strip()

    if not key:
        return (
            "⚠️ **Gemini API Key Required**\n\n"
            "To use the real-time AI Health Assistant, please enter your free **Google Gemini API Key** "
            "in the input box above.\n\n"
            "💡 *Don't have a key?* You can get a free key in 30 seconds at [Google AI Studio](https://aistudio.google.com/app/apikey)."
        )

    try:
        genai.configure(api_key=key)

        # Build prompt with optional patient context
        full_prompt = SYSTEM_INSTRUCTION + "\n\n"

        if patient_context:
            full_prompt += "--- CURRENT PATIENT CLINICAL CONTEXT ---\n"
            full_prompt += f"- Predicted Health Condition: {patient_context.get('predicted_condition', 'N/A').upper()}\n"
            full_prompt += f"- Model Confidence: {patient_context.get('confidence', 0)*100:.1f}%\n"
            if "payload" in patient_context:
                p = patient_context["payload"]
                full_prompt += f"- BMI: {p.get('bmi')} | Heart Rate: {p.get('heart_rate')} bpm | Sleep: {p.get('sleep_duration')} hrs\n"
                full_prompt += f"- Steps: {p.get('step_count')} | Exercise: {p.get('exercise_duration')} min | Water: {p.get('water_intake')} L\n"
                full_prompt += f"- Diet: {p.get('diet_type')} | Stress: {p.get('stress_level')} | Sleep Quality: {p.get('sleep_quality')}\n"
                full_prompt += f"- Activity: {p.get('physical_activity_level')} | Smoking/Alcohol: {p.get('smoking_alcohol')}\n"
            full_prompt += "--------------------------------------\n\n"

        full_prompt += f"USER QUESTION: {user_question}\n\nASSISTANT ANSWER:"

        # Active Google Gemini API models in order of priority
        candidate_models = ["gemini-3.6-flash", "gemini-3.7-flash", "gemini-flash-latest", "gemini-pro-latest"]

        last_error = None
        for model_name in candidate_models:
            try:
                model = genai.GenerativeModel(model_name)
                # Set a strict 15s timeout to prevent UI spinner hangs
                response = model.generate_content(
                    full_prompt,
                    request_options={"timeout": 15},
                )
                if response and hasattr(response, "text") and response.text:
                    return response.text
            except Exception as model_err:
                last_error = model_err
                continue

        if last_error:
            raise last_error

        return "⚠️ Unable to receive a response from Google Gemini. Please check your API key or network connection."

    except Exception as e:
        err_msg = str(e)
        if "API_KEY_INVALID" in err_msg or "400" in err_msg or "API key not valid" in err_msg:
            return "❌ **Invalid API Key:** The provided Gemini API Key was rejected by Google. Please verify your key at [Google AI Studio](https://aistudio.google.com/app/apikey)."
        elif "QUOTA" in err_msg or "429" in err_msg or "ResourceExhausted" in err_msg:
            return "⏳ **Rate Limit Exceeded:** You have reached the free Gemini tier rate limit. Please wait a minute and try again."
        elif "timeout" in err_msg.lower() or "deadline" in err_msg.lower():
            return "⏱️ **Request Timed Out:** Google's Gemini API did not respond within 15 seconds. Please try asking again."
        return f"⚠️ **Gemini API Error:** {err_msg}"
