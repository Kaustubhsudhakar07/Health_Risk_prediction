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

