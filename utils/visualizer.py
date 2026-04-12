import plotly.graph_objects as go
import plotly.express as px
import pandas as pd
import numpy as np

# ─── COLOR CONSTANTS ─────────────────────────────────────────────────────────
COLOR_REAL = "#22c55e"
COLOR_UNCERTAIN = "#f59e0b"
COLOR_FAKE = "#ef4444"
COLOR_BG = "rgba(0,0,0,0)"

# ✅ FIXED: Removed margin from here
PLOTLY_THEME = dict(
    paper_bgcolor=COLOR_BG,
    plot_bgcolor=COLOR_BG,
    font=dict(family="monospace", color="#e2e8f0", size=13)
)


# ═══════════════════════════════════════════════════════════════════════
# TIMELINE
# ═══════════════════════════════════════════════════════════════════════
def make_timeline_chart(frame_results):

    if not frame_results:
        return go.Figure()

    frames = [r["frame_number"] for r in frame_results]
    scores = [r["fake_score"] for r in frame_results]

    point_colors = [
        COLOR_REAL if s < 0.4 else COLOR_UNCERTAIN if s < 0.6 else COLOR_FAKE
        for s in scores
    ]

    fig = go.Figure()

    fig.add_trace(go.Scatter(
        x=frames, y=scores,
        mode='lines+markers',
        line=dict(color='#f87171', width=2.5),
        marker=dict(color=point_colors, size=8),
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        title="Fake Probability Timeline",
        height=350
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# GAUGE (FIXED)
# ═══════════════════════════════════════════════════════════════════════
def make_gauge_chart(verdict):

    score_pct = verdict.get("overall_percent", 50)

    if score_pct < 35:
        color = COLOR_REAL
    elif score_pct < 55:
        color = COLOR_UNCERTAIN
    else:
        color = COLOR_FAKE

    fig = go.Figure(go.Indicator(
        mode="gauge+number",
        value=score_pct,
        number={'suffix': "%"},
        gauge={
            'axis': {'range': [0, 100]},
            'bar': {'color': color},
            'steps': [
                {'range': [0, 35], 'color': "lightgreen"},
                {'range': [35, 55], 'color': "yellow"},
                {'range': [55, 100], 'color': "red"}
            ]
        }
    ))

    # ✅ margin only here (no conflict now)
    fig.update_layout(
        **PLOTLY_THEME,
        height=280,
        margin=dict(l=30, r=30, t=30, b=10)
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# DONUT
# ═══════════════════════════════════════════════════════════════════════
def make_frame_distribution_chart(verdict):

    values = [
        verdict.get("real_frames", 0),
        verdict.get("uncertain_frames", 0),
        verdict.get("fake_frames", 0)
    ]

    fig = go.Figure(go.Pie(
        values=values,
        hole=0.5
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        height=280
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# SIGNAL BAR
# ═══════════════════════════════════════════════════════════════════════
def make_signal_breakdown_chart(signal_pcts):

    if not signal_pcts:
        return go.Figure()

    fig = go.Figure(go.Bar(
        x=list(signal_pcts.values()),
        y=list(signal_pcts.keys()),
        orientation='h'
    ))

    fig.update_layout(
        **PLOTLY_THEME,
        height=280
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# HISTOGRAM
# ═══════════════════════════════════════════════════════════════════════
def make_score_distribution(frame_results):

    if not frame_results:
        return go.Figure()

    scores = [r["fake_score"] for r in frame_results]

    fig = go.Figure(go.Histogram(x=scores))

    fig.update_layout(
        **PLOTLY_THEME,
        height=260
    )

    return fig


# ═══════════════════════════════════════════════════════════════════════
# TEXT
# ═══════════════════════════════════════════════════════════════════════
def generate_verdict_text(verdict, frame_results):

    score = verdict.get("overall_percent", 0)

    if score < 35:
        return "Likely Real"
    elif score < 55:
        return "Uncertain"
    else:
        return "Likely Deepfake"
        