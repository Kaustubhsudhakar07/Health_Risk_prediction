"""
Streamlined & High-Contrast Visualization Suite for CardioHealth AI.
Provides clean, collision-free, standard Machine Learning and Clinical graphs.
"""

import numpy as np
import pandas as pd
import plotly.graph_objects as go

# Palette Constants
PALETTE = {
    "fit": "#10b981",          # Emerald
    "at-risk": "#f59e0b",      # Amber
    "unhealthy": "#ef4444",    # Crimson / Rose
    "primary": "#6366f1",      # Indigo
    "sky": "#38bdf8",          # Sky Blue
    "purple": "#8b5cf6",       # Violet
    "bg_dark": "#0f172a",      # Deep Slate
    "card_bg": "#131b2e",      # Slate
    "grid_color": "rgba(255, 255, 255, 0.08)",
    "text_light": "#f8fafc",
    "text_muted": "#94a3b8",
}


# -------------------------------------------------------------
# TAB 1: PATIENT LEVEL VISUALIZATIONS
# -------------------------------------------------------------

def plot_patient_radar(patient_data: dict) -> go.Figure:
    """
    Renders an interactive radar chart comparing patient vitals against optimal healthy baseline.
    Legend is positioned cleanly on top to avoid any collisions.
    """
    categories = [
        "BMI (Norm)",
        "Resting Heart Rate",
        "Sleep Duration",
        "Daily Steps",
        "Hydration",
        "Calorie Expenditure",
    ]

    bmi = float(patient_data.get("bmi", 22.0))
    hr = float(patient_data.get("heart_rate", 70.0))
    sleep = float(patient_data.get("sleep_duration", 7.5))
    steps = float(patient_data.get("step_count", 8000.0))
    water = float(patient_data.get("water_intake", 2.5))
    cal = float(patient_data.get("calorie_expenditure", 2200.0))

    # Normalized score 0-100 where 100 is optimal
    bmi_score = max(0, min(100, 100 - abs(bmi - 21.7) * 6))
    hr_score = max(0, min(100, 100 - max(0, hr - 60) * 1.6))
    sleep_score = max(0, min(100, 100 - abs(sleep - 8.0) * 18))
    steps_score = max(0, min(100, (steps / 10000.0) * 100))
    water_score = max(0, min(100, (water / 3.0) * 100))
    cal_score = max(0, min(100, (cal / 2500.0) * 100))

    patient_scores = [bmi_score, hr_score, sleep_score, steps_score, water_score, cal_score]
    optimal_scores = [100, 100, 100, 100, 100, 100]

    categories_closed = categories + [categories[0]]
    patient_closed = patient_scores + [patient_scores[0]]
    optimal_closed = optimal_scores + [optimal_scores[0]]

    fig = go.Figure()

    fig.add_trace(go.Scatterpolar(
        r=optimal_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(16, 185, 129, 0.12)",
        line=dict(color="#10b981", width=1.5, dash="dash"),
        name="Healthy Baseline",
        hoverinfo="theta+r",
    ))

    fig.add_trace(go.Scatterpolar(
        r=patient_closed,
        theta=categories_closed,
        fill="toself",
        fillcolor="rgba(99, 102, 241, 0.35)",
        line=dict(color="#818cf8", width=2.5),
        name="Current Patient",
        hoverinfo="theta+r",
    ))

    fig.update_layout(
        polar=dict(
            radialaxis=dict(
                visible=True,
                range=[0, 100],
                tickfont=dict(size=9, color="#94a3b8"),
                gridcolor=PALETTE["grid_color"],
            ),
            angularaxis=dict(
                tickfont=dict(size=11, color="#f8fafc", family="Plus Jakarta Sans"),
                gridcolor=PALETTE["grid_color"],
            ),
            bgcolor="rgba(15, 23, 42, 0.6)",
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.05,
            xanchor="center",
            x=0.5,
            font=dict(color="#cbd5e1", size=11),
        ),
        margin=dict(l=35, r=35, t=45, b=25),
        height=320,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_risk_gauge(condition: str, confidence: float, probs: dict) -> go.Figure:
    """
    Renders an interactive clinical risk meter gauge (0-100%).
    """
    fit_p = probs.get("fit", 0.0)
    at_risk_p = probs.get("at-risk", 0.0)
    unhealthy_p = probs.get("unhealthy", 0.0)

    severity_score = (fit_p * 15.0) + (at_risk_p * 50.0) + (unhealthy_p * 90.0)
    bar_color = PALETTE["fit"] if condition == "fit" else (PALETTE["at-risk"] if condition == "at-risk" else PALETTE["unhealthy"])

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=severity_score,
        domain={"x": [0, 1], "y": [0, 1]},
        title={"text": f"<b>Cardiometabolic Risk Index</b><br><span style='font-size:0.8em;color:#94a3b8'>Confidence: {confidence*100:.1f}%</span>", "font": {"size": 14, "color": "#f8fafc"}},
        number={"suffix": "/100", "font": {"size": 26, "color": "#ffffff", "family": "Outfit"}},
        gauge={
            "axis": {"range": [0, 100], "tickwidth": 1, "tickcolor": "#94a3b8", "tickfont": {"size": 10, "color": "#94a3b8"}},
            "bar": {"color": bar_color, "thickness": 0.28},
            "bgcolor": "rgba(255,255,255,0.05)",
            "borderwidth": 1,
            "bordercolor": "rgba(255,255,255,0.1)",
            "steps": [
                {"range": [0, 35], "color": "rgba(16, 185, 129, 0.2)"},
                {"range": [35, 70], "color": "rgba(245, 158, 11, 0.2)"},
                {"range": [70, 100], "color": "rgba(239, 68, 68, 0.2)"},
            ],
            "threshold": {
                "line": {"color": "#ffffff", "width": 3},
                "thickness": 0.8,
                "value": severity_score,
            },
        },
    ))

    fig.update_layout(
        height=260,
        margin=dict(l=25, r=25, t=50, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
        font=dict(color="#f8fafc", family="Plus Jakarta Sans"),
    )
    return fig


def plot_local_shap_bars(shap_contributions: list) -> go.Figure:
    """
    Renders an interactive horizontal bar chart of local SHAP feature contributions for the current patient.
    """
    if not shap_contributions:
        fig = go.Figure()
        fig.add_annotation(text="SHAP local explanations unavailable", showarrow=False, font=dict(color="#94a3b8"))
        fig.update_layout(height=240, paper_bgcolor="rgba(0,0,0,0)")
        return fig

    df = pd.DataFrame(shap_contributions).sort_values(by="shap_impact", ascending=True)
    colors = [PALETTE["unhealthy"] if x > 0 else PALETTE["fit"] for x in df["shap_impact"]]

    fig = go.Figure(go.Bar(
        x=df["shap_impact"],
        y=[f.replace("_", " ").title() for f in df["feature"]],
        orientation="h",
        marker=dict(color=colors, line=dict(width=1, color="rgba(255,255,255,0.2)")),
        text=[f"{v:+.3f}" for v in df["shap_impact"]],
        textposition="outside",
        hovertemplate="<b>%{y}</b><br>SHAP Contribution: %{x:+.4f}<extra></extra>",
    ))

    fig.update_layout(
        title=dict(
            text="<b>Local SHAP Risk Attribution</b> (Red = Increases Risk, Green = Protective)",
            font=dict(size=13, color="#f8fafc"),
        ),
        xaxis=dict(
            title=dict(text="SHAP Impact on Prediction", font=dict(color="#cbd5e1", size=11)),
            gridcolor=PALETTE["grid_color"],
            zeroline=True,
            zerolinecolor="rgba(255,255,255,0.3)",
            zerolinewidth=1.5,
            tickfont=dict(color="#94a3b8", size=10),
        ),
        yaxis=dict(
            tickfont=dict(color="#f1f5f9", size=11),
        ),
        height=280,
        margin=dict(l=20, r=40, t=40, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    return fig


def plot_patient_population_overlay(df_pop: pd.DataFrame, patient_data: dict, selected_feature: str = "bmi") -> go.Figure:
    """
    Renders population density distribution with a vertical marker indicating patient's value.
    Legend is positioned cleanly at the top so it NEVER collides with the x-axis label.
    """
    fig = go.Figure()

    feature_labels = {
        "bmi": ("Body Mass Index (BMI)", "kg/m²"),
        "heart_rate": ("Resting Heart Rate", "bpm"),
        "sleep_duration": ("Sleep Duration", "hours"),
        "step_count": ("Daily Steps", "steps"),
        "water_intake": ("Hydration", "liters"),
        "calorie_expenditure": ("Caloric Burn", "kcal"),
    }

    label, unit = feature_labels.get(selected_feature, (selected_feature.replace("_", " ").title(), ""))

    for cond, color, name in [("fit", PALETTE["fit"], "Fit"), ("at-risk", PALETTE["at-risk"], "At-Risk"), ("unhealthy", PALETTE["unhealthy"], "Unhealthy")]:
        if "health_condition" in df_pop.columns and selected_feature in df_pop.columns:
            subset = df_pop[df_pop["health_condition"] == cond][selected_feature].dropna()
            fig.add_trace(go.Histogram(
                x=subset,
                name=f"{name} Cohort",
                opacity=0.45,
                marker_color=color,
                nbinsx=35,
                histnorm="probability density",
            ))

    patient_val = float(patient_data.get(selected_feature, 0.0))
    fig.add_vline(
        x=patient_val,
        line_width=3,
        line_dash="dash",
        line_color="#38bdf8",
        annotation_text=f"This Patient ({patient_val:.1f} {unit})",
        annotation_position="top right",
        annotation_font=dict(color="#38bdf8", size=11, family="Plus Jakarta Sans"),
    )

    fig.update_layout(
        barmode="overlay",
        title=dict(text=f"<b>Population Distribution vs. Patient: {label}</b>", font=dict(size=13, color="#f8fafc")),
        xaxis=dict(
            title=dict(text=f"{label} ({unit})", font=dict(color="#cbd5e1", size=11)),
            gridcolor=PALETTE["grid_color"],
            tickfont=dict(color="#94a3b8"),
        ),
        yaxis=dict(
            title=dict(text="Probability Density", font=dict(color="#cbd5e1", size=11)),
            gridcolor=PALETTE["grid_color"],
            tickfont=dict(color="#94a3b8"),
        ),
        showlegend=True,
        legend=dict(
            orientation="h",
            yanchor="bottom",
            y=1.06,
            xanchor="right",
            x=1.0,
            font=dict(color="#cbd5e1", size=10),
        ),
        height=320,
        margin=dict(l=35, r=25, t=65, b=45),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    return fig


# -------------------------------------------------------------
# TAB 3: POPULATION COHORT SCREENING (SIMPLE & INTUITIVE)
# -------------------------------------------------------------

def plot_cohort_donut(df_screened: pd.DataFrame) -> go.Figure:
    """
    Renders a clear, simple donut chart of the screened population breakdown.
    """
    col = "predicted_health_condition" if "predicted_health_condition" in df_screened.columns else "health_condition"
    counts = df_screened[col].value_counts()

    color_map = {
        "fit": PALETTE["fit"],
        "at-risk": PALETTE["at-risk"],
        "unhealthy": PALETTE["unhealthy"],
    }
    colors = [color_map.get(k, PALETTE["primary"]) for k in counts.index]

    fig = go.Figure(data=[go.Pie(
        labels=[k.upper() for k in counts.index],
        values=counts.values,
        hole=0.55,
        marker=dict(colors=colors, line=dict(color="#0f172a", width=2)),
        textinfo="label+percent",
        textfont=dict(size=12, color="#ffffff", family="Plus Jakarta Sans"),
        hovertemplate="<b>%{label}</b><br>Patients: %{value:,}<br>Proportion: %{percent}<extra></extra>",
    )])

    total = len(df_screened)
    fig.add_annotation(
        text=f"<b>{total:,}</b><br><span style='font-size:0.75em;color:#94a3b8'>Patients</span>",
        x=0.5, y=0.5,
        font=dict(size=17, color="#ffffff", family="Outfit"),
        showarrow=False,
    )

    fig.update_layout(
        title=dict(text="<b>Cohort Health Status Proportion</b>", font=dict(size=13, color="#f8fafc")),
        showlegend=False,
        height=300,
        margin=dict(l=20, r=20, t=40, b=20),
        paper_bgcolor="rgba(0,0,0,0)",
    )
    return fig


def plot_cohort_bar_metrics(df_screened: pd.DataFrame) -> go.Figure:
    """
    Renders an intuitive grouped bar chart comparing mean vitals across classes.
    """
    col = "predicted_health_condition" if "predicted_health_condition" in df_screened.columns else "health_condition"
    
    # Calculate group averages
    grp = df_screened.groupby(col).agg({
        "bmi": "mean",
        "heart_rate": "mean",
        "sleep_duration": "mean",
    }).reset_index()

    categories = [c.upper() for c in grp[col]]

    fig = go.Figure()
    fig.add_trace(go.Bar(
        x=categories,
        y=grp["bmi"].round(1),
        name="Avg BMI",
        marker_color="#38bdf8",
        text=grp["bmi"].round(1),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=grp["heart_rate"].round(1),
        name="Avg Heart Rate (bpm)",
        marker_color="#a855f7",
        text=grp["heart_rate"].round(0).astype(int),
        textposition="outside",
    ))
    fig.add_trace(go.Bar(
        x=categories,
        y=(grp["sleep_duration"] * 10).round(1),
        name="Avg Sleep (hrs x10)",
        marker_color="#34d399",
        text=grp["sleep_duration"].round(1),
        textposition="outside",
    ))

    fig.update_layout(
        barmode="group",
        title=dict(text="<b>Key Physiological Vitals by Risk Group</b>", font=dict(size=13, color="#f8fafc")),
        xaxis=dict(tickfont=dict(color="#f8fafc", size=11)),
        yaxis=dict(title=dict(text="Average Metric Value", font=dict(color="#cbd5e1", size=11)), gridcolor=PALETTE["grid_color"], tickfont=dict(color="#94a3b8")),
        legend=dict(orientation="h", yanchor="bottom", y=1.03, xanchor="center", x=0.5, font=dict(color="#cbd5e1", size=10)),
        height=300,
        margin=dict(l=30, r=20, t=50, b=30),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(15, 23, 42, 0.4)",
    )
    return fig

