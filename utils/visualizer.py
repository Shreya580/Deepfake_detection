import plotly.graph_objects as go
import numpy as np

C_REAL      = "#00ff88"
C_UNCERTAIN = "#ffb800"
C_FAKE      = "#ff3366"
C_CYAN      = "#00ffc8"
C_TEXT      = "#c8d8e8"
C_MUTED     = "#4a6080"
C_BG        = "rgba(0,0,0,0)"
C_GRID      = "rgba(255,255,255,0.04)"
C_LINE      = "rgba(255,255,255,0.06)"
FONT        = "Space Mono, monospace"

BASE = dict(
    paper_bgcolor=C_BG,
    plot_bgcolor=C_BG,
    font=dict(family=FONT, color=C_TEXT, size=11),
    margin=dict(l=12, r=12, t=40, b=12),
)


def make_timeline_chart(frame_results):
    if not frame_results:
        return go.Figure()
    frames = [r["frame_number"] for r in frame_results]
    scores = [r["fake_score"]   for r in frame_results]
    pt_colors = [C_REAL if s < 0.4 else C_UNCERTAIN if s < 0.6 else C_FAKE for s in scores]
    fig = go.Figure()
    fig.add_hrect(y0=0.0, y1=0.4,  fillcolor="rgba(0,255,136,0.04)",  line_width=0)
    fig.add_hrect(y0=0.4, y1=0.6,  fillcolor="rgba(255,184,0,0.04)",  line_width=0)
    fig.add_hrect(y0=0.6, y1=1.02, fillcolor="rgba(255,51,102,0.04)", line_width=0)
    fig.add_hline(y=0.4, line_dash="dot", line_color=C_UNCERTAIN, line_width=1, opacity=0.3)
    fig.add_hline(y=0.6, line_dash="dot", line_color=C_FAKE,      line_width=1, opacity=0.3)
    fig.add_trace(go.Scatter(x=frames, y=scores, fill="tozeroy",
        fillcolor="rgba(255,51,102,0.04)", line=dict(color="rgba(0,0,0,0)"),
        showlegend=False, hoverinfo="skip"))
    fig.add_trace(go.Scatter(x=frames, y=scores, mode="lines+markers",
        line=dict(color=C_FAKE, width=2, shape="spline"),
        marker=dict(color=pt_colors, size=7, line=dict(color="rgba(0,0,0,0.5)", width=1)),
        hovertemplate="<b>FRAME %{x}</b><br>SCORE: %{y:.1%}<extra></extra>",
        showlegend=False))
    pk = int(np.argmax(scores))
    if scores[pk] > 0.5:
        fig.add_annotation(x=frames[pk], y=scores[pk], text=f"PEAK {scores[pk]:.0%}",
            showarrow=True, arrowhead=2, arrowcolor=C_FAKE, ax=0, ay=-32,
            font=dict(family=FONT, color=C_FAKE, size=10),
            bgcolor="rgba(255,51,102,0.1)", borderpad=4)
    for ypos, label, color in [(0.2,"REAL",C_REAL),(0.5,"UNCERTAIN",C_UNCERTAIN),(0.8,"FAKE",C_FAKE)]:
        fig.add_annotation(x=max(frames), y=ypos, text=label, showarrow=False,
            xanchor="left", xshift=8, font=dict(family=FONT, color=color, size=8), opacity=0.5)
    fig.update_layout(**BASE,
        title=dict(text="FRAME-BY-FRAME FAKE PROBABILITY",
                   font=dict(family=FONT, size=11, color=C_MUTED), x=0.0),
        xaxis=dict(title=dict(text="FRAME", font=dict(family=FONT, size=9, color=C_MUTED)),
                   showgrid=False, zeroline=False, color=C_MUTED,
                   tickfont=dict(family=FONT, size=9), linecolor=C_LINE),
        yaxis=dict(title=dict(text="FAKE PROB", font=dict(family=FONT, size=9, color=C_MUTED)),
                   range=[0,1.05], tickformat=".0%", showgrid=True, gridcolor=C_GRID,
                   zeroline=False, color=C_MUTED, tickfont=dict(family=FONT, size=9),
                   linecolor=C_LINE),
        height=320, margin=dict(l=12, r=60, t=44, b=32))
    return fig


def make_gauge_chart(verdict):
    pct   = verdict.get("overall_percent", 50)
    color = C_REAL if pct < 35 else C_UNCERTAIN if pct < 55 else C_FAKE
    label = verdict.get("verdict", "").upper()
    fig = go.Figure(go.Indicator(
        mode="gauge+number", value=pct,
        number=dict(suffix="%", font=dict(family="Syne, sans-serif", size=40, color=color)),
        title=dict(text=f"<span style='font-family:Space Mono,monospace;font-size:10px;"
                        f"letter-spacing:0.15em;color:{C_MUTED};'>{label}</span>",
                   font=dict(size=11)),
        gauge=dict(
            axis=dict(range=[0,100], tickwidth=0,
                      tickfont=dict(family=FONT, size=8, color=C_MUTED),
                      tickcolor="rgba(0,0,0,0)"),
            bar=dict(color=color, thickness=0.22),
            bgcolor="rgba(0,255,200,0.03)",
            borderwidth=1, bordercolor="rgba(255,255,255,0.06)",
            steps=[dict(range=[0,35],  color="rgba(0,255,136,0.05)"),
                   dict(range=[35,55], color="rgba(255,184,0,0.05)"),
                   dict(range=[55,100],color="rgba(255,51,102,0.05)")]),
    ))
    fig.update_layout(**BASE,
        title=dict(text="OVERALL SCORE", font=dict(family=FONT, size=11, color=C_MUTED), x=0.0),
        height=260, margin=dict(l=20, r=20, t=44, b=10))
    return fig


def make_frame_distribution_chart(verdict):
    real=verdict.get("real_frames",0); uncertain=verdict.get("uncertain_frames",0)
    fake=verdict.get("fake_frames",0); total=real+uncertain+fake
    fig = go.Figure(go.Pie(
        labels=["REAL","UNCERTAIN","FAKE"], values=[real,uncertain,fake],
        hole=0.62, pull=[0,0,0.06],
        marker=dict(colors=[C_REAL,C_UNCERTAIN,C_FAKE],
                    line=dict(color="rgba(0,8,18,0.8)", width=3)),
        textfont=dict(family=FONT, color="rgba(0,0,0,0.6)", size=9),
        hovertemplate="<b>%{label}</b>: %{value} frames (%{percent})<extra></extra>"))
    fig.add_annotation(
        text=f"<b style='font-family:Syne,sans-serif;font-size:22px;color:#f0f8ff;'>{total}</b><br>"
             f"<span style='font-family:Space Mono,monospace;font-size:8px;"
             f"letter-spacing:0.15em;color:{C_MUTED};'>FRAMES</span>",
        x=0.5, y=0.5, showarrow=False, align="center")
    fig.update_layout(**BASE,
        title=dict(text="FRAME BREAKDOWN", font=dict(family=FONT, size=11, color=C_MUTED), x=0.0),
        showlegend=True,
        legend=dict(orientation="h", y=-0.12, x=0.5, xanchor="center",
                    font=dict(family=FONT, color=C_MUTED, size=9)),
        height=260, margin=dict(l=10, r=10, t=44, b=10))
    return fig


def make_signal_breakdown_chart(signal_pcts):
    if not signal_pcts:
        return go.Figure()
    labels=[l.upper() for l in signal_pcts.keys()]; values=list(signal_pcts.values())
    mx=max(values) if values else 1
    colors=[C_FAKE if v/mx>0.66 else C_UNCERTAIN if v/mx>0.33 else C_REAL for v in values]
    fig = go.Figure(go.Bar(
        x=values, y=labels, orientation="h",
        marker=dict(color=colors, opacity=0.85,
                    line=dict(color="rgba(255,255,255,0.05)", width=0.5)),
        text=[f"{v:.1f}%" for v in values], textposition="outside",
        textfont=dict(family=FONT, color=C_MUTED, size=9),
        hovertemplate="<b>%{y}</b>: %{x:.1f}%<extra></extra>"))
    fig.update_layout(**BASE,
        title=dict(text="REGION SUSPICION SCORES",
                   font=dict(family=FONT, size=11, color=C_MUTED), x=0.0),
        xaxis=dict(title=dict(text="ACTIVATION %", font=dict(family=FONT, size=9, color=C_MUTED)),
                   range=[0, max(values+[1])*1.3], showgrid=True, gridcolor=C_GRID,
                   zeroline=False, tickfont=dict(family=FONT, size=9, color=C_MUTED),
                   linecolor=C_LINE),
        yaxis=dict(showgrid=False, color=C_TEXT,
                   tickfont=dict(family=FONT, size=9, color=C_MUTED)),
        height=280, bargap=0.32, margin=dict(l=12, r=60, t=44, b=12))
    return fig


def make_score_distribution(frame_results):
    if not frame_results:
        return go.Figure()
    scores=[r["fake_score"] for r in frame_results]
    fig = go.Figure(go.Histogram(
        x=scores, nbinsx=20,
        marker=dict(color=C_CYAN, opacity=0.6,
                    line=dict(color="rgba(0,255,200,0.2)", width=0.5)),
        hovertemplate="SCORE: %{x}<br>FRAMES: %{y}<extra></extra>"))
    fig.add_vline(x=0.4, line_dash="dot", line_color=C_UNCERTAIN, line_width=1, opacity=0.5)
    fig.add_vline(x=0.6, line_dash="dot", line_color=C_FAKE,      line_width=1, opacity=0.5)
    fig.update_layout(**BASE,
        title=dict(text="SCORE DISTRIBUTION",
                   font=dict(family=FONT, size=11, color=C_MUTED), x=0.0),
        xaxis=dict(title=dict(text="FAKE SCORE", font=dict(family=FONT, size=9, color=C_MUTED)),
                   range=[0,1], showgrid=False, zeroline=False,
                   tickfont=dict(family=FONT, size=9, color=C_MUTED), linecolor=C_LINE),
        yaxis=dict(title=dict(text="FRAMES", font=dict(family=FONT, size=9, color=C_MUTED)),
                   showgrid=True, gridcolor=C_GRID, zeroline=False,
                   tickfont=dict(family=FONT, size=9, color=C_MUTED), linecolor=C_LINE),
        height=280, showlegend=False, margin=dict(l=12, r=12, t=44, b=32))
    return fig


def generate_verdict_text(verdict, frame_results):
    score=verdict.get("overall_percent",0); total=verdict.get("total_frames",1)
    fake_c=verdict.get("fake_frames",0)
    scores=[r["fake_score"] for r in frame_results]
    if scores:
        pk=int(np.argmax(scores))
        rng=f"frames {max(1,pk-1)}-{min(len(scores),pk+2)}"
    else:
        rng="the analysed media"
    if score < 35:
        return (f"Analysis of {total} frames yields a low manipulation probability of {score}%. "
                f"No significant facial artefacts or frequency anomalies detected. "
                f"High confidence this media is authentic.")
    elif score < 55:
        return (f"Analysis of {total} frames yields a moderate manipulation probability of {score}%. "
                f"Inconsistencies detected around {rng}. {fake_c} of {total} frames "
                f"crossed the detection threshold. Independent verification recommended.")
    else:
        pf=round(fake_c/total*100) if total else 0
        return (f"Analysis of {total} frames yields a high manipulation probability of {score}%. "
                f"Significant anomalies concentrated around {rng}. "
                f"{fake_c} of {total} frames ({pf}%) exceeded the fake threshold. "
                f"This media is likely deepfake or AI-generated.")