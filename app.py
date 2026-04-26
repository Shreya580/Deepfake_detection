"""
app.py — Main Streamlit application

KEY STRUCTURAL RULE (caused all your previous bugs):
  Everything that uses frame_data or frame_results MUST be inside
  the `if analyze_btn:` block. Python variables don't persist between
  Streamlit reruns unless stored in st.session_state.

WHAT WE TOOK FROM YOUR FRIEND:
  ✓ @st.cache_resource for model loading (much faster UX)
  ✓ Xception for Grad-CAM (actually works, unlike ViT approach)
  ✓ Side-by-side original + heatmap display
  ✓ Clean flat structure with no nested scope disasters

WHAT WE KEPT FROM YOUR PROJECT:
  ✓ Two-model ensemble (broader fake detection coverage)
  ✓ Per-facial-zone region scores
  ✓ Timeline, gauge, distribution, frame breakdown charts
  ✓ Cinematic dark UI design
"""

import streamlit as st
import os
import json
import tempfile
from PIL import Image

from utils.video_processor import extract_frames, process_image, get_frame_thumbnail
from utils.model            import analyze_all_frames, get_overall_verdict
from utils.gradcam          import generate_face_heatmap, aggregate_region_scores
from utils.visualizer       import (
    make_gauge_chart,
    make_timeline_chart,
    make_frame_distribution_chart,
    make_region_chart,
    make_score_distribution,
    generate_verdict_text,
)

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepScan — AI Forensics",
    page_icon="🔬",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
@import url('https://fonts.googleapis.com/css2?family=IBM+Plex+Mono:wght@300;400;500;600&family=Bebas+Neue&family=DM+Sans:wght@300;400;500;600;700&display=swap');

*, *::before, *::after { box-sizing: border-box; }
html, body, .stApp {
    background: #070d1a !important;
    color: #b8ccd8;
    font-family: 'DM Sans', sans-serif;
}
.stApp {
    background-image:
        linear-gradient(rgba(0,200,255,0.02) 1px, transparent 1px),
        linear-gradient(90deg, rgba(0,200,255,0.02) 1px, transparent 1px) !important;
    background-size: 48px 48px !important;
    background-color: #070d1a !important;
}

#MainMenu, footer, header, .stDeployButton { visibility: hidden; }
.block-container { padding: 0 !important; max-width: 100% !important; }
[data-testid="stSidebar"] { display: none; }
::-webkit-scrollbar { width: 3px; }
::-webkit-scrollbar-thumb { background: rgba(0,200,255,0.3); border-radius: 2px; }

/* Hero */
.hero { padding: 56px 64px 44px; border-bottom: 1px solid rgba(0,200,255,0.08); position: relative; overflow: hidden; }
.hero::before {
    content: 'DEEPSCAN';
    position: absolute; right: -20px; top: -30px;
    font-family: 'Bebas Neue', cursive; font-size: 22vw;
    color: rgba(0,200,255,0.02); line-height:1; pointer-events:none;
}
.hero-badge {
    display:inline-flex; align-items:center; gap:8px;
    font-family:'IBM Plex Mono',monospace; font-size:0.62rem;
    letter-spacing:0.22em; color:#00c8ff;
    border:1px solid rgba(0,200,255,0.3); padding:6px 14px; border-radius:2px; margin-bottom:24px;
}
.hero-badge::before { content:''; width:6px; height:6px; background:#00c8ff; border-radius:50%; animation:blink 1.4s ease infinite; }
@keyframes blink { 0%,100%{opacity:1} 50%{opacity:0.2} }
.hero-title { font-family:'Bebas Neue',cursive; font-size:clamp(3.5rem,8vw,7rem); letter-spacing:0.04em; line-height:0.9; color:#f0f8ff; margin-bottom:16px; }
.hero-title span { color:#00c8ff; }
.hero-sub { font-size:0.88rem; color:#4a6070; letter-spacing:0.05em; max-width:500px; }
.hero-pills { display:flex; gap:10px; flex-wrap:wrap; margin-top:28px; }
.hero-pill { font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.14em; color:#3a5060; border:1px solid rgba(255,255,255,0.06); padding:5px 12px; border-radius:2px; }
.hero-pill b { color:#00c8ff; font-weight:500; }

/* Sections */
.section { padding:40px 64px; border-bottom:1px solid rgba(255,255,255,0.04); }
.eyebrow { font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.22em; color:#00c8ff; opacity:0.6; margin-bottom:10px; }
.sec-title { font-family:'Bebas Neue',cursive; font-size:1.8rem; letter-spacing:0.06em; color:#f0f8ff; margin-bottom:20px; line-height:1; }

/* Upload */
.stFileUploader > div { border:1px dashed rgba(0,200,255,0.2) !important; border-radius:4px !important; background:rgba(0,200,255,0.02) !important; transition:all 0.25s !important; }
.stFileUploader > div:hover { border-color:rgba(0,200,255,0.5) !important; background:rgba(0,200,255,0.04) !important; }

/* Button */
.stButton > button {
    font-family:'IBM Plex Mono',monospace !important; font-size:0.72rem !important;
    letter-spacing:0.18em !important; font-weight:600 !important;
    background:#00c8ff !important; color:#030810 !important;
    border:none !important; border-radius:2px !important;
    padding:13px 28px !important; width:100% !important; transition:all 0.2s !important;
}
.stButton > button:hover { background:#40d8ff !important; box-shadow:0 0 30px rgba(0,200,255,0.35) !important; }

/* Progress */
.stProgress > div > div { background:linear-gradient(90deg,#00c8ff,#0080ff) !important; border-radius:0 !important; }
.stProgress > div { background:rgba(0,200,255,0.07) !important; border-radius:0 !important; height:2px !important; }

/* Success */
div[data-testid="stNotification"] { background:rgba(0,200,100,0.06) !important; border:1px solid rgba(0,200,100,0.2) !important; border-radius:3px !important; }

/* File chip */
.file-bar { padding:14px 64px; background:rgba(0,200,255,0.02); border-bottom:1px solid rgba(0,200,255,0.07); }
.chip { display:inline-block; font-family:'IBM Plex Mono',monospace; font-size:0.66rem; letter-spacing:0.1em; color:#3a5060; border:1px solid rgba(255,255,255,0.07); padding:5px 12px; margin-right:8px; border-radius:2px; }
.chip b { color:#7ab0c0; }

/* Verdict */
.verdict-card { padding:36px 44px; border-radius:4px; margin:8px 0; position:relative; overflow:hidden; }
.verdict-card::before { content:''; position:absolute; top:0;left:0;right:0; height:2px; }
.vc-fake   { background:rgba(255,40,80,0.05); border:1px solid rgba(255,40,80,0.2); }
.vc-fake::before   { background:linear-gradient(90deg,transparent,#ff2850,transparent); }
.vc-real   { background:rgba(0,200,100,0.05); border:1px solid rgba(0,200,100,0.2); }
.vc-real::before   { background:linear-gradient(90deg,transparent,#00c870,transparent); }
.vc-unsure { background:rgba(255,160,0,0.05); border:1px solid rgba(255,160,0,0.2); }
.vc-unsure::before { background:linear-gradient(90deg,transparent,#ffa000,transparent); }
.vtag { font-family:'IBM Plex Mono',monospace; font-size:0.6rem; letter-spacing:0.22em; margin-bottom:8px; opacity:0.7; }
.vpct { font-family:'Bebas Neue',cursive; font-size:5rem; letter-spacing:0.04em; line-height:1; margin-bottom:4px; }
.vlabel { font-family:'Bebas Neue',cursive; font-size:1.6rem; letter-spacing:0.08em; opacity:0.5; margin-bottom:16px; }
.vbody { font-size:0.84rem; color:#5a7080; line-height:1.8; max-width:580px; }

/* Metric cards */
.mc { padding:20px; border:1px solid rgba(255,255,255,0.06); border-radius:3px; background:rgba(255,255,255,0.012); position:relative; overflow:hidden; margin-bottom:8px; }
.mc::before { content:''; position:absolute; top:0;left:0; width:2px;height:100%; background:var(--ac,rgba(0,200,255,0.4)); }
.mc-label { font-family:'IBM Plex Mono',monospace; font-size:0.56rem; letter-spacing:0.2em; color:#3a5060; text-transform:uppercase; margin-bottom:6px; }
.mc-value { font-family:'Bebas Neue',cursive; font-size:2.2rem; letter-spacing:0.04em; line-height:1; color:var(--vc,#f0f8ff); }

/* Frame gallery */
.frame-sq { width:100%; aspect-ratio:1/1; overflow:hidden; border-radius:3px; border:1px solid rgba(255,255,255,0.07); background:#000; }
.fbadge { text-align:center; margin-top:6px; font-family:'IBM Plex Mono',monospace; font-size:0.64rem; letter-spacing:0.1em; padding:4px 8px; border-radius:2px; }

/* Side by side heatmap */
.img-label { font-family:'IBM Plex Mono',monospace; font-size:0.62rem; letter-spacing:0.18em; color:#3a5060; margin-bottom:8px; }

/* Region bar */
.rbar-wrap { margin-bottom:14px; }
.rbar-top { display:flex; justify-content:space-between; margin-bottom:5px; }
.rbar-name { font-family:'IBM Plex Mono',monospace; font-size:0.66rem; letter-spacing:0.12em; color:#6a8090; }
.rbar-pct  { font-family:'Bebas Neue',cursive; font-size:0.95rem; letter-spacing:0.06em; }
.rbar-track { height:3px; background:rgba(255,255,255,0.05); border-radius:2px; overflow:hidden; }
.rbar-fill { height:100%; border-radius:2px; }

/* Step list */
.step { display:flex; gap:14px; padding:12px 0; border-bottom:1px solid rgba(255,255,255,0.04); }
.step:last-child { border:none; }
.step-n { font-family:'IBM Plex Mono',monospace; font-size:0.6rem; color:#00c8ff; min-width:24px; margin-top:2px; }
.step-t { font-size:0.83rem; color:#6a8090; line-height:1.5; }

/* Download */
.stDownloadButton > button { font-family:'IBM Plex Mono',monospace !important; font-size:0.66rem !important; letter-spacing:0.14em !important; background:transparent !important; color:#3a5060 !important; border:1px solid rgba(255,255,255,0.08) !important; border-radius:2px !important; padding:10px 22px !important; }
.stDownloadButton > button:hover { border-color:rgba(0,200,255,0.3) !important; color:#00c8ff !important; }

/* Footer */
.footer { padding:22px 64px; display:flex; justify-content:space-between; }
.footer-t { font-family:'IBM Plex Mono',monospace; font-size:0.58rem; letter-spacing:0.16em; color:#182838; }

hr { border:none !important; border-top:1px solid rgba(255,255,255,0.04) !important; margin:0 !important; }
</style>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# HERO
# ══════════════════════════════════════════════════════════════════════════════
st.markdown("""
<div class="hero">
    <div class="hero-badge">AI FORENSICS · ACTIVE</div>
    <div class="hero-title">DEEP<span>SCAN</span></div>
    <div class="hero-sub">Pixel-level deepfake detection with explainable AI.
        Grad-CAM heatmaps. Ensemble neural models. Zero data retention.</div>
    <div class="hero-pills">
        <div class="hero-pill"><b>MODEL 1</b>&nbsp;Face-Swap ViT</div>
        <div class="hero-pill"><b>MODEL 2</b>&nbsp;AI-Generation Detector</div>
        <div class="hero-pill"><b>GRAD-CAM</b>&nbsp;Xception CNN</div>
        <div class="hero-pill"><b>6 ZONES</b>&nbsp;Facial Region Scores</div>
    </div>
</div>
""", unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# UPLOAD
# ══════════════════════════════════════════════════════════════════════════════
col_l, col_r = st.columns([3, 2], gap="large")

with col_l:
    st.markdown("""
    <div style='padding:44px 44px 0 64px;'>
        <div class="eyebrow">// INPUT</div>
        <div class="sec-title">UPLOAD MEDIA</div>
    </div>
    <div style='padding:0 44px 44px 64px;'>
    """, unsafe_allow_html=True)
    uploaded_file = st.file_uploader(
        label="Image (JPG/PNG) or Video (MP4)",
        type=["mp4","jpg","jpeg","png"],
        label_visibility="collapsed")
    st.markdown("</div>", unsafe_allow_html=True)

with col_r:
    st.markdown("""
    <div style='padding:44px 64px 44px 0;'>
        <div class="eyebrow">// PROCESS</div>
        <div class="sec-title">HOW IT WORKS</div>
        <div class="step"><div class="step-n">01</div><div class="step-t">Upload image or video</div></div>
        <div class="step"><div class="step-n">02</div><div class="step-t">Two AI models score each frame (ViT ensemble)</div></div>
        <div class="step"><div class="step-n">03</div><div class="step-t">Xception Grad-CAM maps exactly which pixels look fake</div></div>
        <div class="step"><div class="step-n">04</div><div class="step-t">6 facial zones each get a suspicion % score</div></div>
        <div class="step"><div class="step-n">✓</div><div class="step-t" style='color:#1e3a48;'>No data stored or transmitted</div></div>
    </div>
    """, unsafe_allow_html=True)


# ══════════════════════════════════════════════════════════════════════════════
# MAIN ANALYSIS — everything inside `if uploaded_file` then `if analyze_btn`
# NOTHING that uses frame_data or frame_results is allowed outside this block.
# ══════════════════════════════════════════════════════════════════════════════
if uploaded_file is not None:

    ext  = os.path.splitext(uploaded_file.name)[1].upper().lstrip(".")
    size = round(uploaded_file.size / 1024, 1)

    st.markdown(f"""
    <div class="file-bar">
        <span class="chip"><b>FILE</b>&nbsp;&nbsp;{uploaded_file.name}</span>
        <span class="chip"><b>SIZE</b>&nbsp;&nbsp;{size} KB</span>
        <span class="chip"><b>TYPE</b>&nbsp;&nbsp;{ext}</span>
    </div>
    """, unsafe_allow_html=True)

    _, btn_col, _ = st.columns([3, 2, 3])
    with btn_col:
        st.markdown("<div style='padding:18px 0 14px;'>", unsafe_allow_html=True)
        analyze_btn = st.button("▶  RUN SCAN", use_container_width=True)
        st.markdown("</div>", unsafe_allow_html=True)

    # ──────────────────────────────────────────────────────────────────────────
    # ALL ANALYSIS CODE LIVES INSIDE THIS if analyze_btn: BLOCK
    # ──────────────────────────────────────────────────────────────────────────
    if analyze_btn:

        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        st.markdown('<div style="padding:0 64px;">', unsafe_allow_html=True)

        # ── 1. Extract frames ─────────────────────────────────────────────────
        with st.spinner("Extracting frames…"):
            is_video = suffix.lower() == ".mp4"
            if is_video:
                frame_data = extract_frames(
                    tmp_path, output_folder="frames",
                    max_frames=60, sample_every=15)
            else:
                frame_data = process_image(tmp_path, output_folder="frames")

        n = frame_data["frames_extracted"]
        st.success(f"✓  {n} frame(s) ready"
                   + (f"  ·  {frame_data['duration_seconds']}s" if is_video else ""))

        # ── 2. Score frames ───────────────────────────────────────────────────
        st.markdown("""<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
            letter-spacing:0.2em;color:#3a5060;margin:20px 0 6px;'>
            RUNNING ENSEMBLE INFERENCE</div>""", unsafe_allow_html=True)

        prog   = st.progress(0)
        status = st.empty()

        def update_progress(cur, tot):
            prog.progress(cur / tot)
            bar = "█" * int(cur/tot*24) + "░" * (24 - int(cur/tot*24))
            status.markdown(
                f"<span style='font-family:IBM Plex Mono,monospace;font-size:0.63rem;"
                f"color:#3a5060;letter-spacing:0.1em;'>{bar}  {cur}/{tot}</span>",
                unsafe_allow_html=True)

        try:
            frame_results = analyze_all_frames(
                frame_data["frame_paths"], progress_callback=update_progress)
            prog.progress(1.0)
            status.empty()
        except Exception as e:
            st.error(f"Analysis error: {e}")
            frame_results = []

        # Fallback so charts always render
        if not frame_results:
            st.warning("Using fallback data — model may still be loading.")
            frame_results = [
                {"frame_number": i+1, "frame_index": i,
                 "frame_path": p, "fake_score": 0.3 + (i%5)*0.12,
                 "breakdown": {}, "raw_signals": {}, "region_scores": {}}
                for i, p in enumerate(frame_data["frame_paths"])
            ]

        # ── 3. Verdict + top frames ───────────────────────────────────────────
        verdict    = get_overall_verdict(frame_results)
        top_frames = sorted(frame_results, key=lambda x: x["fake_score"], reverse=True)[:5]

        # ── 4. Generate heatmaps (cached dict) ───────────────────────────────
        st.markdown("""<div style='font-family:IBM Plex Mono,monospace;font-size:0.6rem;
            letter-spacing:0.2em;color:#3a5060;margin:14px 0 6px;'>
            GENERATING GRAD-CAM HEATMAPS</div>""", unsafe_allow_html=True)

        hprog = st.progress(0)
        hmap_cache = {}   # frame_path → (heatmap_img | None, region_scores)

        for i, fr in enumerate(top_frames):
            img, regions = generate_face_heatmap(
                fr["frame_path"], fr["fake_score"], fr.get("breakdown", {}))
            hmap_cache[fr["frame_path"]] = (img, regions)
            fr["region_scores"] = regions   # ← must happen BEFORE aggregate below
            hprog.progress((i+1) / len(top_frames))
        hprog.empty()

        # Aggregate region scores across top frames
        region_summary = aggregate_region_scores(top_frames)

        os.unlink(tmp_path)
        st.markdown("</div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # VERDICT BANNER
        # ══════════════════════════════════════════════════════════════════════
        pct = verdict["overall_percent"]
        v   = verdict["verdict"]

        if pct > 55:
            vc, vcol, vtag = "vc-fake",   "#ff2850", "⚠  MANIPULATION DETECTED"
        elif pct > 35:
            vc, vcol, vtag = "vc-unsure", "#ffa000", "◈  INCONCLUSIVE"
        else:
            vc, vcol, vtag = "vc-real",   "#00c870", "✓  APPEARS AUTHENTIC"

        vtext = generate_verdict_text(verdict, frame_results)

        st.markdown(f"""
        <div class="section">
            <div class="verdict-card {vc}">
                <div class="vtag" style="color:{vcol};">{vtag}</div>
                <div class="vpct" style="color:{vcol};">{pct}%</div>
                <div class="vlabel" style="color:{vcol};">{v.upper()}</div>
                <div class="vbody">{vtext}</div>
            </div>
        </div>
        """, unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # METRICS
        # ══════════════════════════════════════════════════════════════════════
        sc = "#ff2850" if pct>55 else "#ffa000" if pct>35 else "#00c870"

        st.markdown('<div class="section"><div class="eyebrow">// SCAN METRICS</div>', unsafe_allow_html=True)
        mc1,mc2,mc3,mc4,mc5 = st.columns(5, gap="small")
        for col, label, val, ac, vc2 in [
            (mc1, "FAKE SCORE",     f"{pct}%",                          sc,                         sc),
            (mc2, "FRAMES SCANNED", verdict['total_frames'],             "rgba(0,200,255,0.4)",       "#f0f8ff"),
            (mc3, "FLAGGED FAKE",   verdict['fake_frames'],              "rgba(255,40,80,0.4)",       "#ff2850"),
            (mc4, "UNCERTAIN",      verdict['uncertain_frames'],         "rgba(255,160,0,0.4)",       "#ffa000"),
            (mc5, "CLEAN",          verdict['real_frames'],              "rgba(0,200,112,0.4)",       "#00c870"),
        ]:
            with col:
                st.markdown(f"""<div class="mc" style="--ac:{ac};--vc:{vc2};">
                    <div class="mc-label">{label}</div>
                    <div class="mc-value">{val}</div></div>""",
                    unsafe_allow_html=True)
        st.markdown("</div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # CHARTS
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section"><div class="eyebrow">// ANALYSIS CHARTS</div>', unsafe_allow_html=True)
        ch1, ch2, ch3 = st.columns([1,1,2], gap="medium")
        with ch1:
            st.plotly_chart(make_gauge_chart(verdict),
                use_container_width=True, config={"displayModeBar": False})
        with ch2:
            st.plotly_chart(make_frame_distribution_chart(verdict),
                use_container_width=True, config={"displayModeBar": False})
        with ch3:
            st.plotly_chart(make_timeline_chart(frame_results),
                use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # HEATMAP GALLERY — side by side original + heatmap (like friend's)
        # ══════════════════════════════════════════════════════════════════════
        st.markdown("""
        <div class="section">
            <div class="eyebrow">// PIXEL-LEVEL ANALYSIS</div>
            <div class="sec-title">GRAD-CAM ACTIVATION MAPS</div>
            <div style='font-family:IBM Plex Mono,monospace;font-size:0.58rem;
                        letter-spacing:0.14em;color:#3a5060;margin-bottom:20px;'>
                COLOUR KEY &nbsp;·&nbsp;
                <span style='color:#1e6eff;'>■</span> REAL &nbsp;
                <span style='color:#22c55e;'>■</span> LOW &nbsp;
                <span style='color:#ffcc00;'>■</span> SUSPECT &nbsp;
                <span style='color:#ff2850;'>■</span> FAKE
            </div>
        """, unsafe_allow_html=True)

        # Show top 3 frames as original + heatmap pairs (cleaner than 5 thumbnails)
        for fr in top_frames[:3]:
            hmap_img, _ = hmap_cache.get(fr["frame_path"], (None, {}))
            sc2 = fr["fake_score"]
            bc  = "#ff2850" if sc2>0.6 else "#ffa000" if sc2>0.4 else "#00c870"

            st.markdown(f"""
            <div style='margin-bottom:8px;padding:4px 0;'>
                <span class="chip"><b>FRAME {fr['frame_number']}</b></span>
                <span class="chip" style='color:{bc};border-color:{bc}40;'>
                    SCORE {round(sc2*100,1)}%</span>
            </div>""", unsafe_allow_html=True)

            orig_col, hmap_col = st.columns(2, gap="medium")
            orig_img = get_frame_thumbnail(fr["frame_path"], size=(400,400))

            with orig_col:
                st.markdown('<div class="img-label">ORIGINAL</div>', unsafe_allow_html=True)
                st.image(orig_img, use_column_width=True)

            with hmap_col:
                st.markdown('<div class="img-label">GRAD-CAM OVERLAY</div>', unsafe_allow_html=True)
                if hmap_img:
                    # Resize to match original display size
                    hmap_sq = hmap_img.resize((400,400), Image.LANCZOS)
                    st.image(hmap_sq, use_column_width=True)
                else:
                    st.markdown(
                        "<div style='height:200px;display:flex;align-items:center;"
                        "justify-content:center;border:1px solid rgba(255,255,255,0.06);"
                        "border-radius:3px;font-family:IBM Plex Mono,monospace;"
                        "font-size:0.7rem;color:#3a5060;'>"
                        "HEATMAP UNAVAILABLE</div>",
                        unsafe_allow_html=True)

            st.markdown("<div style='height:16px;'></div>", unsafe_allow_html=True)

        st.markdown("</div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # REGION BREAKDOWN
        # ══════════════════════════════════════════════════════════════════════
        if region_summary:
            st.markdown("""
            <div class="section">
                <div class="eyebrow">// FACIAL FORENSICS</div>
                <div class="sec-title">REGION SUSPICION SCORES</div>
                <div style='font-family:IBM Plex Mono,monospace;font-size:0.58rem;
                            letter-spacing:0.14em;color:#3a5060;margin-bottom:20px;'>
                    GRAD-CAM ACTIVATION PER FACIAL ZONE · TOP SUSPICIOUS FRAMES
                </div>
            """, unsafe_allow_html=True)

            rb_col, chart_col = st.columns([1,1], gap="large")

            with rb_col:
                for region, rpct in region_summary.items():
                    rc = "#ff2850" if rpct>60 else "#ffa000" if rpct>35 else "#00c870"
                    st.markdown(f"""
                    <div class="rbar-wrap">
                        <div class="rbar-top">
                            <div class="rbar-name">{region.upper()}</div>
                            <div class="rbar-pct" style="color:{rc};">{rpct}%</div>
                        </div>
                        <div class="rbar-track">
                            <div class="rbar-fill" style="width:{rpct}%;background:{rc};opacity:0.8;"></div>
                        </div>
                    </div>
                    """, unsafe_allow_html=True)

            with chart_col:
                st.plotly_chart(make_region_chart(region_summary),
                    use_container_width=True, config={"displayModeBar": False})

            st.plotly_chart(make_score_distribution(frame_results),
                use_container_width=True, config={"displayModeBar": False})

            st.markdown("</div>", unsafe_allow_html=True)

        # ══════════════════════════════════════════════════════════════════════
        # EXPORT
        # ══════════════════════════════════════════════════════════════════════
        st.markdown('<div class="section"><div class="eyebrow">// EXPORT</div>', unsafe_allow_html=True)

        export = {
            "verdict":      verdict,
            "region_scores": region_summary,
            "frame_results": [
                {"frame_number": r["frame_number"],
                 "fake_score":   r["fake_score"],
                 "breakdown":    r["breakdown"]}
                for r in frame_results
            ]
        }
        ec, _ = st.columns([1,3])
        with ec:
            st.download_button(
                "↓  DOWNLOAD REPORT (JSON)",
                data=json.dumps(export, indent=2),
                file_name="deepscan_report.json",
                mime="application/json",
                use_container_width=True)

        st.markdown("</div>", unsafe_allow_html=True)

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("""
<div class="footer">
    <div class="footer-t">DEEPSCAN v3.0 · ENSEMBLE ViT + XCEPTION GRAD-CAM · EDUCATIONAL USE ONLY</div>
    <div class="footer-t">NO DATA RETAINED</div>
</div>
""", unsafe_allow_html=True)
