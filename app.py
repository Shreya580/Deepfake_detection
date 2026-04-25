import streamlit as st
import os
import json
import tempfile
from PIL import Image

from utils.video_processor import extract_frames, process_image, get_frame_thumbnail
from utils.model import analyze_all_frames, get_overall_verdict
from utils.visualizer import (
    make_timeline_chart,
    make_gauge_chart,
    make_frame_distribution_chart,
    make_signal_breakdown_chart,
    make_score_distribution,
    generate_verdict_text,
)
from utils.heatmap import generate_face_heatmap, generate_signal_heatmap_data

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="DeepScan — Deepfake Detector",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="collapsed",
)

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown("""
<style>
    @import url('https://fonts.googleapis.com/css2?family=Inter:wght@400;500;600;700&display=swap');

    html, body, [class*="css"] { font-family: 'Inter', sans-serif; }

    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1a2744 50%, #1e293b 100%);
        color: #e2e8f0;
    }

    /* Upload box */
    .stFileUploader > div {
        border: 2px dashed #334155 !important;
        border-radius: 14px !important;
        background: rgba(30,41,59,0.4) !important;
        transition: border-color 0.2s;
    }
    .stFileUploader > div:hover {
        border-color: #3b82f6 !important;
    }

    /* Metric cards */
    div[data-testid="metric-container"] {
        background: rgba(30,41,59,0.7);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 18px 20px !important;
        margin-bottom: 10px;
    }
    div[data-testid="metric-container"] label {
        color: #94a3b8 !important;
        font-size: 0.8rem !important;
    }
    div[data-testid="metric-container"] [data-testid="stMetricValue"] {
        font-size: 1.6rem !important;
        font-weight: 700 !important;
        color: #f1f5f9 !important;
    }

    /* Verdict box */
    .verdict-box {
        background: linear-gradient(135deg, rgba(30,41,59,0.95), rgba(15,23,42,0.95));
        border: 1px solid #1e3a5f;
        border-radius: 16px;
        padding: 28px;
        margin: 8px 0 20px 0;
        line-height: 2;
        font-size: 0.95rem;
    }

    /* Section cards */
    .section-card {
        background: rgba(30,41,59,0.5);
        border: 1px solid #1e3a5f;
        border-radius: 14px;
        padding: 20px;
        margin-bottom: 16px;
    }

    /* Region score tiles */
    .region-tile {
        text-align: center;
        padding: 16px 8px;
        background: rgba(30,41,59,0.7);
        border-radius: 12px;
        border: 1px solid #1e3a5f;
        transition: transform 0.15s;
    }
    .region-tile:hover { transform: translateY(-2px); }

    /* Headings */
    h1, h2, h3 { color: #f1f5f9 !important; }

    /* Progress bar */
    .stProgress > div > div { background: #3b82f6 !important; }

    /* Analyze button */
    .stButton > button {
        background: linear-gradient(90deg, #3b82f6, #6366f1);
        color: white;
        border: none;
        border-radius: 10px;
        padding: 12px 28px;
        font-size: 15px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
        transition: opacity 0.2s;
    }
    .stButton > button:hover { opacity: 0.88; }

    hr { border-color: #1e3a5f !important; }

    /* Gallery frame image */
    .frame-card {
        background: rgba(15,23,42,0.8);
        border: 1px solid #1e3a5f;
        border-radius: 10px;
        padding: 8px;
        margin-bottom: 6px;
    }
</style>
""", unsafe_allow_html=True)

# ── Header ────────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 36px 0 20px 0;'>
    <div style='font-size:2.6rem; margin-bottom:6px;'>🔍</div>
    <h1 style='font-size:2.6rem; font-weight:800; letter-spacing:-1.5px; margin:0;
               background:linear-gradient(90deg,#60a5fa,#818cf8,#a78bfa);
               -webkit-background-clip:text; -webkit-text-fill-color:transparent;'>
        DeepScan
    </h1>
    <p style='color:#64748b; font-size:1rem; margin-top:6px; letter-spacing:0.3px;'>
        Deepfake &amp; AI-Generated Image Detection · Explainable by Design
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ── Upload section ────────────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### 📤 Upload Media")
    uploaded_file = st.file_uploader(
        label="Drag and drop a video (MP4) or image (JPG / PNG)",
        type=["mp4", "jpg", "jpeg", "png"],
        help="Videos: analysed at 2 fps, max 60 frames. Images: single-frame analysis.",
    )

with col_info:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    <div style='color:#64748b; font-size:0.9rem; line-height:2;'>
    1. Upload your image or video<br>
    2. Two AI models score each frame<br>
    3. Grad-CAM shows exactly <em>where</em> the AI detected manipulation<br>
    4. Each face region gets a suspicion %<br><br>
    <b style='color:#60a5fa;'>Nothing is stored or sent to any server.</b>
    </div>
    """, unsafe_allow_html=True)

# ── Analysis logic ────────────────────────────────────────────────────────────
if uploaded_file is not None:

    st.markdown("---")
    col_info2, col_btn = st.columns([3, 1])

    with col_info2:
        st.markdown(
            f"**File:** `{uploaded_file.name}` &nbsp;·&nbsp; "
            f"{round(uploaded_file.size / 1024, 1)} KB",
            unsafe_allow_html=True
        )

    with col_btn:
        analyze_btn = st.button("🚀 Analyze Now", use_container_width=True)

    if analyze_btn:

        # Save upload to a real temp file (OpenCV needs a path, not bytes)
        suffix = os.path.splitext(uploaded_file.name)[1]
        with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as tmp:
            tmp.write(uploaded_file.read())
            tmp_path = tmp.name

        # ── Step 1: Extract frames ────────────────────────────────────────
        with st.spinner("🎬 Extracting frames..."):
            is_video = suffix.lower() == ".mp4"
            if is_video:
                frame_data = extract_frames(
                    tmp_path, output_folder="frames", max_frames=60, sample_every=15
                )
            else:
                frame_data = process_image(tmp_path, output_folder="frames")

            st.success(
                f"✅ {frame_data['frames_extracted']} frame(s) extracted"
                + (f" from {frame_data['duration_seconds']}s video" if is_video else "")
            )

        # ── Step 2: Score frames ─────────────────────────────────────────
        st.markdown("### 🤖 Running AI analysis...")
        progress_bar = st.progress(0)
        status_text  = st.empty()

        def update_progress(current, total):
            progress_bar.progress(current / total)
            status_text.markdown(
                f"<span style='color:#64748b;font-size:0.9rem'>"
                f"Scoring frame {current} of {total}…</span>",
                unsafe_allow_html=True,
            )

        frame_results = analyze_all_frames(
            frame_data["frame_paths"], progress_callback=update_progress
        )
        progress_bar.progress(1.0)
        status_text.empty()

        # ── Step 3: Verdict + heatmaps ───────────────────────────────────
        verdict = get_overall_verdict(frame_results)

        # SPEED FIX: Generate heatmaps ONCE for top 5 frames only.
        # The old code called generate_face_heatmap() twice per frame
        # (once in the gallery loop, once again for the region breakdown).
        # Now we call it once and cache results in a dict.
        top_frames = sorted(
            frame_results, key=lambda x: x["fake_score"], reverse=True
        )[:5]

        st.markdown(
            "<span style='color:#64748b;font-size:0.9rem'>"
            "Generating heatmaps for top frames…</span>",
            unsafe_allow_html=True,
        )
        heatmap_cache = {}   # frame_path → (PIL Image | None, region_scores dict)
        heatmap_bar = st.progress(0)
        for i, fr in enumerate(top_frames):
            img, regions = generate_face_heatmap(
                fr["frame_path"], fr["fake_score"], fr.get("breakdown", {})
            )
            heatmap_cache[fr["frame_path"]] = (img, regions)
            # Store region scores back into frame_results for signal chart
            fr["region_scores"] = regions
            heatmap_bar.progress((i + 1) / len(top_frames))
        heatmap_bar.empty()

        signal_pcts = generate_signal_heatmap_data(top_frames)

        os.unlink(tmp_path)   # Clean up temp upload file

        st.markdown("---")

        # ════════════════════════════════════════════════════════════════
        # DASHBOARD
        # ════════════════════════════════════════════════════════════════
        st.markdown("## 📊 Analysis Results")

        # Row 1: Gauge · Donut · Metrics
        col_g, col_d, col_m = st.columns([1.2, 1.2, 1])

        with col_g:
            st.plotly_chart(
                make_gauge_chart(verdict),
                use_container_width=True, config={"displayModeBar": False}
            )

        with col_d:
            st.plotly_chart(
                make_frame_distribution_chart(verdict),
                use_container_width=True, config={"displayModeBar": False}
            )

        with col_m:
            st.markdown("<div style='height:16px'></div>", unsafe_allow_html=True)
            pct = verdict["overall_percent"]
            color = "#ef4444" if pct > 55 else "#f59e0b" if pct > 35 else "#22c55e"
            st.markdown(
                f"<div style='background:rgba(30,41,59,0.7);border:1px solid #1e3a5f;"
                f"border-radius:14px;padding:18px 20px;margin-bottom:10px;text-align:center;'>"
                f"<div style='color:#94a3b8;font-size:0.8rem;margin-bottom:4px;'>Overall Fake Score</div>"
                f"<div style='font-size:2.2rem;font-weight:800;color:{color};'>{pct}%</div>"
                f"</div>",
                unsafe_allow_html=True
            )
            st.metric("Frames Analysed",  verdict["total_frames"])
            st.metric("Frames Flagged",   verdict["fake_frames"])
            st.metric("Peak Frame Score", f"{round(verdict['max_score']*100,1)}%")

        st.markdown("---")

        # Row 2: Timeline (full width)
        st.plotly_chart(
            make_timeline_chart(frame_results),
            use_container_width=True, config={"displayModeBar": False}
        )

        st.markdown("---")

        # Row 3: Signal breakdown + histogram
        col_s, col_h = st.columns([1.2, 1])
        with col_s:
            st.plotly_chart(
                make_signal_breakdown_chart(signal_pcts),
                use_container_width=True, config={"displayModeBar": False}
            )
        with col_h:
            st.plotly_chart(
                make_score_distribution(frame_results),
                use_container_width=True, config={"displayModeBar": False}
            )

        st.markdown("---")

        # Row 4: Suspicious frame gallery (uses cached heatmaps)
        st.markdown("### 🎞️ Most Suspicious Frames")
        st.markdown(
            "<p style='color:#64748b;font-size:0.875rem;margin-top:-8px;'>"
            "Top 5 frames by fake score. Colour overlay: "
            "<span style='color:#1e90ff;'>blue=real</span> → "
            "<span style='color:#22c55e;'>green</span> → "
            "<span style='color:#f59e0b;'>yellow</span> → "
            "<span style='color:#ef4444;'>red=fake</span>.</p>",
            unsafe_allow_html=True
        )

        gallery_cols = st.columns(len(top_frames))
        for col, fr in zip(gallery_cols, top_frames):
            with col:
                heatmap_img, _ = heatmap_cache.get(fr["frame_path"], (None, {}))
                score     = fr["fake_score"]
                frame_num = fr["frame_number"]

                if heatmap_img:
                    st.image(heatmap_img, use_column_width=True)
                else:
                    orig = get_frame_thumbnail(fr["frame_path"])
                    st.image(orig, use_column_width=True)

                badge_color = "#ef4444" if score > 0.6 else "#f59e0b" if score > 0.4 else "#22c55e"
                st.markdown(
                    f"<div style='text-align:center;margin-top:4px;"
                    f"background:{badge_color}18;border:1px solid {badge_color};"
                    f"border-radius:8px;padding:4px 6px;font-size:0.82rem;color:{badge_color};'>"
                    f"Frame {frame_num} · {round(score*100,1)}%</div>",
                    unsafe_allow_html=True
                )

        st.markdown("---")

        # Row 5: Region breakdown tiles (from worst frame's cached heatmap)
        worst_frame = top_frames[0]
        _, region_scores = heatmap_cache.get(worst_frame["frame_path"], (None, {}))

        if region_scores:
            st.markdown("### 🧩 Face Region Breakdown")
            st.markdown(
                "<p style='color:#64748b;font-size:0.875rem;margin-top:-8px;'>"
                "Average Grad-CAM activation per facial zone on the most suspicious frame. "
                "Higher % = model focused more on that region when predicting fake.</p>",
                unsafe_allow_html=True
            )
            region_cols = st.columns(len(region_scores))
            for col, (region, pct_r) in zip(region_cols, region_scores.items()):
                rc = "#ef4444" if pct_r > 60 else "#f59e0b" if pct_r > 35 else "#22c55e"
                with col:
                    st.markdown(
                        f"<div class='region-tile' style='border-color:{rc}30;'>"
                        f"<div style='font-size:1.5rem;font-weight:700;color:{rc};'>{pct_r}%</div>"
                        f"<div style='font-size:0.78rem;color:#94a3b8;margin-top:4px;'>{region}</div>"
                        f"</div>",
                        unsafe_allow_html=True
                    )

        st.markdown("---")

        # Row 6: AI verdict summary
        st.markdown("### 🧠 AI Analysis Summary")

        raw_text = generate_verdict_text(verdict, frame_results)
        # Convert **bold** markdown to HTML bold
        html_text = raw_text.replace("**", "<b>", 1)
        while "**" in html_text:
            html_text = html_text.replace("**", "</b>", 1)
            if "**" in html_text:
                html_text = html_text.replace("**", "<b>", 1)

        icon = "🚨" if verdict["overall_percent"] > 55 else \
               "⚠️" if verdict["overall_percent"] > 35 else "✅"
        v_color = "#ef4444" if verdict["overall_percent"] > 55 else \
                  "#f59e0b" if verdict["overall_percent"] > 35 else "#22c55e"

        st.markdown(
            f"<div class='verdict-box'>"
            f"<div style='font-size:1.5rem;margin-bottom:12px;'>{icon} "
            f"<span style='font-size:1.1rem;font-weight:700;color:{v_color};'>"
            f"Verdict: {verdict['verdict']}</span></div>"
            f"{html_text}"
            f"</div>",
            unsafe_allow_html=True
        )

        # Row 7: Export
        st.markdown("### 📥 Export Results")
        export_data = {
            "verdict": verdict,
            "region_scores_worst_frame": region_scores,
            "signal_breakdown": signal_pcts,
            "frame_results": [
                {
                    "frame_number": r["frame_number"],
                    "fake_score": r["fake_score"],
                    "breakdown": r["breakdown"],
                }
                for r in frame_results
            ],
        }
        st.download_button(
            label="📄 Download Full Report (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="deepscan_report.json",
            mime="application/json",
        )

# ── Footer ────────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center;color:#334155;font-size:0.82rem;'>"
    "DeepScan v2.0 · Ensemble ViT model · Grad-CAM++ explainability · "
    "For educational and research use only"
    "</p>",
    unsafe_allow_html=True,
)
