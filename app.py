import streamlit as st
import os
import json
import tempfile
import pandas as pd
from PIL import Image

# Import our custom modules
from utils.video_processor import extract_frames, process_image, get_frame_thumbnail
from utils.model import analyze_all_frames, get_overall_verdict
from utils.visualizer import (
    make_timeline_chart,
    make_gauge_chart,
    make_frame_distribution_chart,
    make_signal_breakdown_chart,
    make_score_distribution,
    generate_verdict_text
)
from utils.heatmap import generate_face_heatmap, generate_signal_heatmap_data

# ─── PAGE CONFIG ─────────────────────────────────────────────────────────────
# WHY: Must be the very first Streamlit call. Sets tab title, icon, layout.
st.set_page_config(
    page_title="DeepScan — Deepfake Detector",
    page_icon="🔍",
    layout="wide",           # Full browser width
    initial_sidebar_state="collapsed"
)

# ─── DARK THEME OVERRIDE ─────────────────────────────────────────────────────
# WHY: Our Plotly charts use a dark theme. The default Streamlit light
# background would clash. We inject custom CSS to make everything cohesive.
st.markdown("""
<style>  
    /* Dark background for main app */
    .stApp {
        background: linear-gradient(135deg, #0f172a 0%, #1e293b 100%);
        color: #e2e8f0;
    }
    
    /* Style the upload area */
    .stFileUploader > div {
        border: 2px dashed #334155 !important;
        border-radius: 12px !important;
        background: rgba(30, 41, 59, 0.5) !important;
    }
    
    /* Style metric boxes */
    div[data-testid="metric-container"] {
        background: rgba(30, 41, 59, 0.7);
        border: 1px solid #334155;
        border-radius: 12px;
        padding: 16px !important;
    }
    
    /* Verdict box */
    .verdict-box {
        background: rgba(30, 41, 59, 0.9);
        border: 1px solid #334155;
        border-radius: 16px;
        padding: 24px;
        margin: 16px 0;
        line-height: 1.8;
    }
    
    /* Header styling */
    h1, h2, h3 { color: #f1f5f9 !important; }
    
    /* Progress bar color */
    .stProgress > div > div { background: #3b82f6 !important; }
    
    /* Button styling */
    .stButton > button {
        background: #3b82f6;
        color: white;
        border: none;
        border-radius: 8px;
        padding: 12px 32px;
        font-size: 16px;
        font-weight: 600;
        width: 100%;
        cursor: pointer;
    }
    .stButton > button:hover {
        background: #2563eb;
    }
    
    /* Section dividers */
    hr { border-color: #334155 !important; }
</style>
""", unsafe_allow_html=True)

# ─── HEADER ──────────────────────────────────────────────────────────────────
st.markdown("""
<div style='text-align:center; padding: 32px 0 16px 0;'>
    <h1 style='font-size: 2.8rem; font-weight: 700; letter-spacing: -1px;
               background: linear-gradient(90deg, #60a5fa, #818cf8, #a78bfa);
               -webkit-background-clip: text; -webkit-text-fill-color: transparent;'>
        🔍 DeepScan
    </h1>
    <p style='color: #94a3b8; font-size: 1.1rem; margin-top: -8px;'>
        Deepfake Detection with Explainable Visualization
    </p>
</div>
""", unsafe_allow_html=True)

st.markdown("---")

# ─── FILE UPLOAD SECTION ─────────────────────────────────────────────────────
col_upload, col_info = st.columns([2, 1])

with col_upload:
    st.markdown("### 📤 Upload Media")
    uploaded_file = st.file_uploader(
        label="Drag and drop a video (MP4) or image (JPG/PNG)",
        type=["mp4", "jpg", "jpeg", "png"],
        help="Videos are capped at 60 seconds. Images are analyzed as a single frame."
    )

with col_info:
    st.markdown("### ℹ️ How it works")
    st.markdown("""
    <div style='color: #94a3b8; font-size: 0.9rem; line-height: 1.8;'>
    1. Upload video or image<br>
    2. AI extracts and scores each frame<br>
    3. Dashboard shows WHERE and WHY manipulation was detected<br><br>
    <b style='color: #60a5fa;'>No data is stored or uploaded to any server.</b>
    </div>
    """, unsafe_allow_html=True)

# ─── ANALYSIS BUTTON + LOGIC ─────────────────────────────────────────────────
if uploaded_file is not None:
    
    st.markdown("---")
    
    # Show file preview
    col_prev, col_btn = st.columns([3, 1])
    
    with col_prev:
        st.markdown(f"**File:** `{uploaded_file.name}` — {round(uploaded_file.size/1024, 1)} KB")
    
    with col_btn:
        analyze_btn = st.button("🚀 Analyze Now", use_container_width=True)
    
    if analyze_btn:
        
        # ── SAVE UPLOADED FILE TEMPORARILY ────────────────────────────────
        # WHY tempfile? Streamlit gives us file bytes, but OpenCV needs a
        # real file path. tempfile creates a real file on disk temporarily.
        
        with tempfile.NamedTemporaryFile(
            delete=False,
            suffix=os.path.splitext(uploaded_file.name)[1]
        ) as tmp_file:
            tmp_file.write(uploaded_file.read())
            tmp_path = tmp_file.name
        
        # ── STEP 1: EXTRACT FRAMES ─────────────────────────────────────────
        with st.spinner("🎬 Extracting frames..."):
            
            is_video = uploaded_file.name.lower().endswith(".mp4")
            
            if is_video:
                frame_data = extract_frames(
                    tmp_path,
                    output_folder="frames",
                    max_frames=60,
                    sample_every=15
                )
            else:
                frame_data = process_image(tmp_path, output_folder="frames")
            
            st.success(
                f"✅ Extracted {frame_data['frames_extracted']} frames"
                + (f" from {frame_data['duration_seconds']}s video" if is_video else "")
            )
        
        # ── STEP 2: RUN MODEL ON ALL FRAMES ───────────────────────────────
        st.markdown("### 🤖 Analyzing frames...")
        progress_bar = st.progress(0)
        status_text = st.empty()
        
        def update_progress(current, total):
            pct = current / total
            progress_bar.progress(pct)
            status_text.markdown(
                f"<span style='color:#94a3b8'>Processing frame {current} of {total}...</span>",
                unsafe_allow_html=True
            )
        
        frame_results = analyze_all_frames(
            frame_data["frame_paths"],
            progress_callback=update_progress
        )
        
        progress_bar.progress(1.0)
        status_text.empty()
        
        # ── STEP 3: COMPUTE VERDICT ────────────────────────────────────────
        verdict = get_overall_verdict(frame_results)
        signal_pcts = generate_signal_heatmap_data(frame_results)
        
        # ── CLEANUP TEMP FILE ──────────────────────────────────────────────
        os.unlink(tmp_path)
        
        st.markdown("---")
        
        # ════════════════════════════════════════════════════════════════
        # DASHBOARD LAYOUT
        # ════════════════════════════════════════════════════════════════
        
        st.markdown("## 📊 Analysis Results")
        
        # ── ROW 1: Gauge + Donut + Key Metrics ──────────────────────────
        col_gauge, col_donut, col_metrics = st.columns([1.2, 1.2, 1])
        
        with col_gauge:
            gauge_fig = make_gauge_chart(verdict)
            st.plotly_chart(gauge_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col_donut:
            donut_fig = make_frame_distribution_chart(verdict)
            st.plotly_chart(donut_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col_metrics:
            st.markdown("<div style='height:20px'></div>", unsafe_allow_html=True)
            
            score_color = "#ef4444" if verdict["overall_percent"] > 55 else \
                         "#f59e0b" if verdict["overall_percent"] > 35 else "#22c55e"
            
            st.metric("Overall Fake Score",
                      f"{verdict['overall_percent']}%",
                      delta=None)
            st.metric("Total Frames Analyzed", verdict["total_frames"])
            st.metric("Frames Flagged as Fake", verdict["fake_frames"])
            st.metric("Peak Frame Score",
                      f"{round(verdict['max_score'] * 100, 1)}%")
        
        st.markdown("---")
        
        # ── ROW 2: Timeline (full width) ──────────────────────────────────
        timeline_fig = make_timeline_chart(frame_results)
        st.plotly_chart(timeline_fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        # ── ROW 3: Signal Breakdown + Score Distribution ──────────────────
        col_signals, col_dist = st.columns([1.2, 1])
        
        with col_signals:
            signal_fig = make_signal_breakdown_chart(signal_pcts)
            st.plotly_chart(signal_fig, use_container_width=True, config={'displayModeBar': False})
        
        with col_dist:
            dist_fig = make_score_distribution(frame_results)
            st.plotly_chart(dist_fig, use_container_width=True, config={'displayModeBar': False})
        
        st.markdown("---")
        
        # ── ROW 4: Suspicious Frame Gallery ──────────────────────────────
        st.markdown("### 🎞️ Most Suspicious Frames")
        st.markdown(
            "<p style='color:#94a3b8;font-size:0.9rem'>Top 5 frames with highest fake scores. "
            "Heatmap overlay shows where the AI detected anomalies.</p>",
            unsafe_allow_html=True
        )
        
        # Sort frames by fake score, take top 5
        top_frames = sorted(frame_results, key=lambda x: x["fake_score"], reverse=True)[:5]
        
        gallery_cols = st.columns(5)
        
        for i, (col, frame_result) in enumerate(zip(gallery_cols, top_frames)):
            with col:
                frame_path = frame_result["frame_path"]
                score = frame_result["fake_score"]
                frame_num = frame_result["frame_number"]
                breakdown = frame_result.get("breakdown", {})
                
                # Generate heatmap for this frame
                heatmap_img = generate_face_heatmap(frame_path, score, breakdown)
                
                if heatmap_img:
                    st.image(heatmap_img, use_column_width=True, caption=f"Frame {frame_num}")
                else:
                    # Fallback: show original frame
                    orig_img = get_frame_thumbnail(frame_path)
                    st.image(orig_img, use_column_width=True, caption=f"Frame {frame_num}")
                
                # Score badge
                badge_color = "#ef4444" if score > 0.6 else \
                             "#f59e0b" if score > 0.4 else "#22c55e"
                st.markdown(
                    f"<div style='text-align:center; background:{badge_color}20; "
                    f"border:1px solid {badge_color}; border-radius:8px; "
                    f"padding:4px 8px; font-size:0.85rem; color:{badge_color};'>"
                    f"Score: {round(score * 100, 1)}%</div>",
                    unsafe_allow_html=True
                )
        
        st.markdown("---")
        
        # ── ROW 5: AI Verdict Summary ─────────────────────────────────────
        st.markdown("### 🧠 AI Analysis Summary")
        
        verdict_text = generate_verdict_text(verdict, frame_results)
        
        # Verdict icon
        icon = "🚨" if verdict["overall_percent"] > 55 else \
               "⚠️" if verdict["overall_percent"] > 35 else "✅"
        
        st.markdown(
            f"<div class='verdict-box'>"
            f"<span style='font-size:1.5rem'>{icon}</span>&nbsp;&nbsp;"
            f"<strong style='font-size:1.1rem; color:#f1f5f9;'>"
            f"Verdict: {verdict['verdict']}</strong><br><br>"
            f"{verdict_text.replace('**', '<b>').replace('**', '</b>')}"
            f"</div>",
            unsafe_allow_html=True
        )
        
        # ── ROW 6: Download Results JSON ─────────────────────────────────
        st.markdown("### 📥 Export Results")
        
        export_data = {
            "verdict": verdict,
            "signal_breakdown_pct": signal_pcts,
            "frame_results": [
                {
                    "frame_number": r["frame_number"],
                    "fake_score": r["fake_score"],
                    "breakdown": r["breakdown"]
                }
                for r in frame_results
            ]
        }
        
        st.download_button(
            label="📄 Download Full Analysis (JSON)",
            data=json.dumps(export_data, indent=2),
            file_name="deepscan_results.json",
            mime="application/json"
        )

# ─── FOOTER ──────────────────────────────────────────────────────────────────
st.markdown("---")
st.markdown(
    "<p style='text-align:center; color:#475569; font-size:0.85rem;'>"
    "DeepScan v1.0 — Built with Streamlit + OpenCV + DeepFace + Plotly | "
    "For educational and research purposes only"
    "</p>",
    unsafe_allow_html=True
)