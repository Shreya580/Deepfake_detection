import plotly.graph_objects as go
import numpy as np

# ── Palette ───────────────────────────────────────────────────────────────────
C_REAL    = "#00c870"
C_UNSURE  = "#ffa000"
C_FAKE    = "#ff2850"
C_CYAN    = "#00c8ff"
C_TEXT    = "#b8ccd8"
C_MUTED   = "#3a5060"
C_BG      = "rgba(0,0,0,0)"
C_GRID    = "rgba(255,255,255,0.04)"
FONT_MONO = "IBM Plex Mono, monospace"
FONT_DISP = "Bebas Neue, cursive"

BASE = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    font=dict(family=FONT_MONO, color=C_TEXT, size=10),
    margin=dict(l=12, r=16, t=44, b=24),
)


def make_timeline_chart(frame_results):
    if not frame_results:
        return go.Figure()

    frames = [r["frame_number"] for r in frame_results]
    scores = [r["fake_score"]   for r in frame_results]
    colors = [C_REAL if s < 0.4 else C_UNSURE if s < 0.6 else C_FAKE for s in scores]

    fig = go.Figure()

    # Zone bands
    fig.add_hrect(y0=0.0, y1=0.4,  fillcolor="rgba(0,200,112,0.04)",  line_width=0)
    fig.add_hrect(y0=0.4, y1=0.6,  fillcolor="rgba(255,160,0,0.04)",  line_width=0)
    fig.add_hrect(y0=0.6, y1=1.02, fillcolor="rgba(255,40,80,0.04)",  line_width=0)

    # Threshold lines
    fig.add_hline(y=0.4, line_dash="dot", line_color=C_UNSURE, line_width=1, opacity=0.25)
    fig.add_hline(y=0.6, line_dash="dot", line_color=C_FAKE,   line_width=1, opacity=0.25)

    # Fill under line
    fig.add_trace(go.Scatter(
        x=frames, y=scores, fill="tozeroy",
        fillcolor="rgba(255,40,80,0.04)",
        line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"))

    # Main line
    fig.add_trace(go.Scatter(
        x=frames, y=scores, mode="lines+markers",
        line=dict(color=C_FAKE, width=1.8, shape="spline"),
        marker=dict(color=colors, size=6,
                    line=dict(color="rgba(0,0,0,0.6)", width=1)),
        hovertemplate="<b>FRAME %{x}</b><br>SCORE: %{y:.1%}<extra></extra>",
        showlegend=False))

    # Peak annotation
    pk = int(np.argmax(scores))
    if scores[pk] > 0.5:
        fig.add_annotation(
            x=frames[pk], y=scores[pk],
            text=f"PEAK  {scores[pk]:.0%}",
            showarrow=True, arrowhead=2, arrowcolor=C_FAKE,
            ax=0, ay=-30, font=dict(family=FONT_MONO, color=C_FAKE, size=9),
            bgcolor="rgba(255,40,80,0.08)", borderpad=4)

    # Zone labels
    if frames:
        for ypos, lbl, col in [(0.2,"REAL",C_REAL),(0.5,"UNCERTAIN",C_UNSURE),(0.8,"FAKE",C_FAKE)]:
            fig.add_annotation(x=max(frames), y=ypos, text=lbl,
                showarrow=False, xanchor="left", xshift=8,
                font=dict(family=FONT_MONO, color=col, size=8), opacity=0.45)

    fig.update_layout(**BASE,
        title=dict(text="FRAME-BY-FRAME FAKE PROBABILITY",
                   font=dict(family=FONT_MONO, size=10, color=C_MUTED), x=0.0),
        xaxis=dict(title=dict(text="FRAME", font=dict(size=9, color=C_MUTED)),
                   showgrid=False, zeroline=False, color=C_MUTED,
                   tickfont=dict(size=9), linecolor="rgba(255,255,255,0.06)"),
        yaxis=dict(title=dict(text="FAKE PROB", font=dict(size=9, color=C_MUTED)),
                   range=[0, 1.05], tickformat=".0%",
                   showgrid=True, gridcolor=C_GRID,
                   zeroline=False, color=C_MUTED, tickfont=dict(size=9),
                   linecolor="rgba(255,255,255,0.06)"),
        height=320, margin=dict(l=12, r=64, t=44, b=32))
    return fig


def make_gauge_chart(verdict):
    pct   = verdict.get("overall_percent", 50)
    color = C_REAL if pct < 35 else C_UNSURE if pct < 55 else C_FAKE
    label = verdict.get("verdict", "").upper()

    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(family=FONT_DISP, size=44, color=color)),
        title=dict(
            text=f"<span style='font-family:IBM Plex Mono,monospace;font-size:9px;"
                 f"letter-spacing:0.18em;color:{C_MUTED};'>{label}</span>",
            font=dict(size=10)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0,
                      tickfont=dict(family=FONT_MONO, size=8, color=C_MUTED),
                      tickcolor="rgba(0,0,0,0)"),
            bar=dict(color=color, thickness=0.2),
            bgcolor="rgba(0,200,255,0.02)",
            borderwidth=1, bordercolor="rgba(255,255,255,0.05)",
            steps=[dict(range=[0,35],   color="rgba(0,200,112,0.04)"),
                   dict(range=[35,55],  color="rgba(255,160,0,0.04)"),
                   dict(range=[55,100], color="rgba(255,40,80,0.04)")]),
    ))
    fig.update_layout(**BASE,
        title=dict(text="OVERALL SCORE",
                   font=dict(family=FONT_MONO, size=10, color=C_MUTED), x=0.0),
        height=260, margin=dict(l=20, r=20, t=44, b=10))
    return fig


def make_frame_distribution_chart(verdict):
    real = verdict.get("real_frames", 0)
    unc  = verdict.get("uncertain_frames", 0)
    fake = verdict.get("fake_frames", 0)
    total = real + unc + fake

    fig = go.Figure(go.Pie(
        labels=["REAL", "UNCERTAIN", "FAKE"],
        values=[real, unc, fake],
        hole=0.64, pull=[0, 0, 0.05],
        marker=dict(colors=[C_REAL, C_UNSURE, C_FAKE],
                    line=dict(color="rgba(7,13,26,0.9)", width=3)),
        textfont=dict(family=FONT_MONO, size=8),
        hovertemplate="<b>%{label}</b>: %{value} frames (%{percent})<extra></extra>"))

    fig.add_annotation(
        text=f"<span style='font-family:Bebas Neue,cursive;font-size:26px;"
             f"color:#f0f8ff;'>{total}</span><br>"
             f"<span style='font-family:IBM Plex Mono,monospace;font-size:7px;"
             f"letter-spacing:0.18em;color:{C_MUTED};'>FRAMES</span>",
        x=0.5, y=0.5, showarrow=False, align="center")

    fig.update_layout(**BASE,
        title=dict(text="FRAME BREAKDOWN",
                   font=dict(family=FONT_MONO, size=10, color=C_MUTED), x=0.0),
        showlegend=True,
        legend=dict(orientation="h", y=-0.1, x=0.5, xanchor="center",
                    font=dict(family=FONT_MONO, color=C_MUTED, size=8)),
        height=260, margin=dict(l=10, r=10, t=44, b=10))
    return fig


def make_signal_breakdown_chart(signal_pcts):
    """
    Horizontal bar chart of region suspicion percentages.
    Only shown when signal_pcts is non-empty (i.e. heatmaps ran successfully).
    """
    if not signal_pcts:
        return go.Figure()

    labels = [l.upper() for l in signal_pcts.keys()]
    values = list(signal_pcts.values())
    mx     = max(values) if values else 1

    colors = [C_FAKE if v/mx > 0.66 else C_UNSURE if v/mx > 0.33 else C_REAL
              for v in values]

    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.8,
                    line=dict(color="rgba(255,255,255,0.04)", width=0.5)),
        text=[f"{v:.1f}%" for v in values],
        textposition="outside",
        textfont=dict(family=FONT_MONO, color=C_MUTED, size=9),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>"))

    fig.update_layout(**BASE,
        title=dict(text="REGION ACTIVATION",
                   font=dict(family=FONT_MONO, size=10, color=C_MUTED), x=0.0),
        xaxis=dict(
            title=dict(text="ACTIVATION %", font=dict(size=9, color=C_MUTED)),
            range=[0, max(values+[1]) * 1.28],
            showgrid=True, gridcolor=C_GRID, zeroline=False,
            tickfont=dict(size=9, color=C_MUTED),
            linecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(showgrid=False, color=C_TEXT,
                   tickfont=dict(size=9, color=C_MUTED)),
        height=260, bargap=0.3,
        margin=dict(l=12, r=60, t=44, b=12))
    return fig


def make_score_distribution(frame_results):
    """
    FIX: Previous version had x and y axes swapped.
    Histogram should have: X = fake score (0-1), Y = number of frames.
    """
    if not frame_results:
        return go.Figure()

    scores = [r["fake_score"] for r in frame_results]

    fig = go.Figure(go.Histogram(
        # FIX: x = scores (the variable), y is automatically the count
        x=scores,
        nbinsx=20,
        marker=dict(
            color=C_CYAN, opacity=0.55,
            line=dict(color="rgba(0,200,255,0.2)", width=0.5)),
        hovertemplate="SCORE: %{x:.2f}<br>FRAMES: %{y}<extra></extra>"))

    # Threshold markers
    fig.add_vline(x=0.4, line_dash="dot", line_color=C_UNSURE,
                  line_width=1, opacity=0.4,
                  annotation_text="0.4", annotation_position="top",
                  annotation_font=dict(family=FONT_MONO, size=8, color=C_UNSURE))
    fig.add_vline(x=0.6, line_dash="dot", line_color=C_FAKE,
                  line_width=1, opacity=0.4,
                  annotation_text="0.6", annotation_position="top",
                  annotation_font=dict(family=FONT_MONO, size=8, color=C_FAKE))

    fig.update_layout(**BASE,
        title=dict(text="SCORE DISTRIBUTION",
                   font=dict(family=FONT_MONO, size=10, color=C_MUTED), x=0.0),
        xaxis=dict(
            title=dict(text="FAKE SCORE", font=dict(size=9, color=C_MUTED)),
            range=[0, 1], showgrid=False, zeroline=False,
            tickfont=dict(size=9, color=C_MUTED),
            linecolor="rgba(255,255,255,0.05)"),
        yaxis=dict(
            title=dict(text="FRAME COUNT", font=dict(size=9, color=C_MUTED)),
            showgrid=True, gridcolor=C_GRID, zeroline=False,
            tickfont=dict(size=9, color=C_MUTED),
            linecolor="rgba(255,255,255,0.05)"),
        height=260, showlegend=False)
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
        return (
            f"Analysis of {total} frame(s) yields a low manipulation probability of {score}%. "
            f"No significant facial artefacts detected. This media appears authentic."
        )
    elif score < 55:
        return (
            f"Analysis of {total} frame(s) yields a moderate manipulation probability of {score}%. "
            f"Inconsistencies were detected around {rng}. "
            f"{fake_c} of {total} frames crossed the detection threshold. "
            f"Independent verification recommended."
        )
    else:
        pf = round(fake_c / total * 100) if total else 0
        return (
            f"Analysis of {total} frame(s) yields a high manipulation probability of {score}%. "
            f"Significant anomalies concentrated around {rng}. "
            f"{fake_c} of {total} frames ({pf}%) exceeded the fake threshold. "
            f"This media is likely deepfake or AI-generated."
        )
