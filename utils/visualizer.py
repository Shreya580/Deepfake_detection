"""
visualizer.py — All Plotly charts

KEY FIX from your previous versions:
  Never put margin in the BASE dict AND in update_layout().
  BASE only contains paper/plot bgcolor and font.
  Each function sets its own margin explicitly.
"""

import plotly.graph_objects as go
import numpy as np

C_REAL   = "#00c870"
C_UNSURE = "#ffa000"
C_FAKE   = "#ff2850"
C_CYAN   = "#00c8ff"
C_MUTED  = "#3a5060"
C_GRID   = "rgba(255,255,255,0.05)"
MONO     = "IBM Plex Mono, monospace"
DISP     = "Bebas Neue, cursive"

# BASE: ONLY background + font. NO margin here — that caused your bug.
BASE = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(family=MONO, color="#b8ccd8", size=10),
)


def make_gauge_chart(verdict):
    pct   = verdict.get("overall_percent", 50)
    color = C_REAL if pct < 35 else C_UNSURE if pct < 55 else C_FAKE
    label = verdict.get("verdict", "").upper()

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(family=DISP, size=44, color=color)),
        title=dict(
            text=f"<span style='font-family:{MONO};font-size:9px;"
                 f"letter-spacing:0.18em;color:{C_MUTED};'>{label}</span>",
            font=dict(size=10)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0,
                      tickfont=dict(family=MONO, size=8, color=C_MUTED),
                      tickcolor="rgba(0,0,0,0)"),
            bar=dict(color=color, thickness=0.2),
            bgcolor="rgba(0,200,255,0.02)",
            borderwidth=1, bordercolor="rgba(255,255,255,0.06)",
            steps=[dict(range=[0,35],   color="rgba(0,200,112,0.05)"),
                   dict(range=[35,55],  color="rgba(255,160,0,0.05)"),
                   dict(range=[55,100], color="rgba(255,40,80,0.05)")]),
    ))
    fig.update_layout(
        **BASE,
        title=dict(text="OVERALL SCORE",
                   font=dict(family=MONO, size=10, color=C_MUTED), x=0.0),
        height=260,
        margin=dict(l=20, r=20, t=44, b=10),  # margin ONLY here, not in BASE
    )
    return fig


def make_timeline_chart(frame_results):
    if not frame_results:
        return go.Figure()

    frames = [r["frame_number"] for r in frame_results]
    scores = [r["fake_score"]   for r in frame_results]
    colors = [C_REAL if s<0.4 else C_UNSURE if s<0.6 else C_FAKE for s in scores]

    fig = go.Figure()
    fig.add_hrect(y0=0.0, y1=0.4,  fillcolor="rgba(0,200,112,0.05)",  line_width=0)
    fig.add_hrect(y0=0.4, y1=0.6,  fillcolor="rgba(255,160,0,0.05)",  line_width=0)
    fig.add_hrect(y0=0.6, y1=1.02, fillcolor="rgba(255,40,80,0.05)",  line_width=0)
    fig.add_hline(y=0.4, line_dash="dot", line_color=C_UNSURE, line_width=1, opacity=0.3)
    fig.add_hline(y=0.6, line_dash="dot", line_color=C_FAKE,   line_width=1, opacity=0.3)

    fig.add_trace(go.Scatter(
        x=frames, y=scores, fill="tozeroy",
        fillcolor="rgba(255,40,80,0.04)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"))

    fig.add_trace(go.Scatter(
        x=frames, y=scores, mode="lines+markers",
        line=dict(color=C_FAKE, width=2, shape="spline"),
        marker=dict(color=colors, size=7,
                    line=dict(color="rgba(0,0,0,0.5)", width=1)),
        hovertemplate="<b>FRAME %{x}</b><br>SCORE: %{y:.1%}<extra></extra>",
        showlegend=False))

    pk = int(np.argmax(scores))
    if scores[pk] > 0.5:
        fig.add_annotation(
            x=frames[pk], y=scores[pk],
            text=f"PEAK  {scores[pk]:.0%}",
            showarrow=True, arrowhead=2, arrowcolor=C_FAKE,
            ax=0, ay=-30, font=dict(family=MONO, color=C_FAKE, size=9),
            bgcolor="rgba(255,40,80,0.08)", borderpad=4)

    if frames:
        for yp, lbl, col in [(0.2,"REAL",C_REAL),(0.5,"UNCERTAIN",C_UNSURE),(0.8,"FAKE",C_FAKE)]:
            fig.add_annotation(x=max(frames), y=yp, text=lbl, showarrow=False,
                xanchor="left", xshift=8,
                font=dict(family=MONO, color=col, size=8), opacity=0.45)

    fig.update_layout(
        **BASE,
        title=dict(text="FRAME-BY-FRAME FAKE PROBABILITY",
                   font=dict(family=MONO, size=10, color=C_MUTED), x=0.0),
        xaxis=dict(title=dict(text="FRAME", font=dict(size=9, color=C_MUTED)),
                   showgrid=False, zeroline=False, color=C_MUTED,
                   tickfont=dict(size=9), linecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title=dict(text="FAKE PROB", font=dict(size=9, color=C_MUTED)),
                   range=[0, 1.05], tickformat=".0%",
                   showgrid=True, gridcolor=C_GRID,
                   zeroline=False, color=C_MUTED, tickfont=dict(size=9),
                   linecolor="rgba(255,255,255,0.06)"),
        height=320,
        margin=dict(l=12, r=64, t=44, b=32),
    )
    return fig


def make_frame_distribution_chart(verdict):
    real  = verdict.get("real_frames",      0)
    unc   = verdict.get("uncertain_frames", 0)
    fake  = verdict.get("fake_frames",      0)
    total = real + unc + fake

    fig = go.Figure(go.Pie(
        labels=["REAL","UNCERTAIN","FAKE"], values=[real, unc, fake],
        hole=0.62, pull=[0,0,0.05],
        marker=dict(colors=[C_REAL, C_UNSURE, C_FAKE],
                    line=dict(color="rgba(7,13,26,0.9)", width=3)),
        textfont=dict(family=MONO, size=8),
        hovertemplate="<b>%{label}</b>: %{value} frames (%{percent})<extra></extra>"))

    fig.add_annotation(
        text=f"<span style='font-family:{DISP};font-size:26px;color:#f0f8ff;'>"
             f"{total}</span><br>"
             f"<span style='font-family:{MONO};font-size:7px;"
             f"letter-spacing:0.18em;color:{C_MUTED};'>FRAMES</span>",
        x=0.5, y=0.5, showarrow=False, align="center")

    fig.update_layout(
        **BASE,
        title=dict(text="FRAME BREAKDOWN",
                   font=dict(family=MONO, size=10, color=C_MUTED), x=0.0),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                    font=dict(family=MONO, color=C_MUTED, size=8)),
        height=260,
        margin=dict(l=10, r=10, t=44, b=10),
    )
    return fig


def make_region_chart(region_scores):
    """
    Horizontal bar chart of per-facial-zone suspicion scores (0–100%).
    Only shown when Grad-CAM successfully computed region scores.
    """
    if not region_scores:
        return go.Figure()

    labels = [l.upper() for l in region_scores.keys()]
    values = list(region_scores.values())
    mx     = max(values) if values else 1

    colors = [C_FAKE if v/mx > 0.66 else C_UNSURE if v/mx > 0.33 else C_REAL
              for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.04)", width=0.5)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family=MONO, color=C_MUTED, size=9),
        hovertemplate="<b>%{y}</b>: %{x:.1f}% suspicion<extra></extra>"))

    fig.update_layout(
        **BASE,
        title=dict(text="FACIAL REGION ACTIVATION",
                   font=dict(family=MONO, size=10, color=C_MUTED), x=0.0),
        xaxis=dict(
            title=dict(text="GRAD-CAM ACTIVATION %", font=dict(size=9, color=C_MUTED)),
            range=[0, max(values+[1]) * 1.28],
            showgrid=True, gridcolor=C_GRID, zeroline=False,
            tickfont=dict(size=9, color=C_MUTED),
            linecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=False, color="#b8ccd8",
                   tickfont=dict(size=9, color=C_MUTED)),
        height=280, bargap=0.3,
        margin=dict(l=12, r=60, t=44, b=12),
    )
    return fig


def make_score_distribution(frame_results):
    """
    Histogram of fake scores across all frames.
    X = fake score (0–1), Y = number of frames (COUNT).
    Previous version had these axes swapped — now fixed.
    """
    if not frame_results:
        return go.Figure()

    scores = [r["fake_score"] for r in frame_results]

    fig = go.Figure(go.Histogram(
        x=scores,           # ← x = the variable (fake score 0-1)
        nbinsx=20,          # ← y is automatically computed as count
        marker=dict(color=C_CYAN, opacity=0.55,
                    line=dict(color="rgba(0,200,255,0.2)", width=0.5)),
        hovertemplate="SCORE: %{x:.2f}<br>FRAMES: %{y}<extra></extra>"))

    fig.add_vline(x=0.4, line_dash="dot", line_color=C_UNSURE, line_width=1, opacity=0.5)
    fig.add_vline(x=0.6, line_dash="dot", line_color=C_FAKE,   line_width=1, opacity=0.5)

    fig.update_layout(
        **BASE,
        title=dict(text="SCORE DISTRIBUTION",
                   font=dict(family=MONO, size=10, color=C_MUTED), x=0.0),
        xaxis=dict(
            title=dict(text="FAKE SCORE (0=real, 1=fake)",
                       font=dict(size=9, color=C_MUTED)),
            range=[0,1], showgrid=False, zeroline=False,
            tickfont=dict(size=9, color=C_MUTED),
            linecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(
            title=dict(text="FRAME COUNT", font=dict(size=9, color=C_MUTED)),
            showgrid=True, gridcolor=C_GRID, zeroline=False,
            tickfont=dict(size=9, color=C_MUTED),
            linecolor="rgba(255,255,255,0.05)"),
        height=260, showlegend=False,
        margin=dict(l=12, r=12, t=44, b=32),
    )
    return fig


def generate_verdict_text(verdict, frame_results):
    score  = verdict.get("overall_percent", 0)
    total  = verdict.get("total_frames", 1)
    fake_c = verdict.get("fake_frames", 0)
    scores = [r["fake_score"] for r in frame_results]

    if scores:
        pk  = int(np.argmax(scores))
        rng = f"frames {max(1,pk-1)}–{min(len(scores),pk+2)}"
    else:
        rng = "the analysed media"

    if score < 35:
        return (f"Analysis of {total} frame(s) yields a low manipulation "
                f"probability of {score}%. No significant artifacts detected. "
                f"This media appears authentic.")
    elif score < 55:
        return (f"Analysis of {total} frame(s) yields a moderate manipulation "
                f"probability of {score}%. Inconsistencies detected around {rng}. "
                f"{fake_c} of {total} frames crossed the threshold. "
                f"Independent verification recommended.")
    else:
        pf = round(fake_c/total*100) if total else 0
        return (f"Analysis of {total} frame(s) yields a high manipulation "
                f"probability of {score}%. Significant anomalies around {rng}. "
                f"{fake_c}/{total} frames ({pf}%) exceeded the fake threshold. "
                f"This media is likely deepfake or AI-generated.")