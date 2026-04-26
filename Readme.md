# 🧠 Deepfake Detection System with Explainable AI Dashboard

An interactive deepfake detection system that combines **multi-signal analysis, machine learning concepts, and advanced visualizations** to detect manipulated media and provide clear, interpretable explanations.

---

## 🚀 Overview

Traditional deepfake detection systems typically provide a binary output:

> Fake / Real

This project extends beyond that by delivering a **comprehensive analytical pipeline** that evaluates media content at a granular level and explains the reasoning behind each prediction.

---

## 🎯 Key Features

### 🔍 Multi-Stage Processing Pipeline

* Supports both **image and video inputs**
* Extracts frames using OpenCV
* Performs per-frame analysis
* Aggregates results into a final verdict

---

### 🤖 Hybrid Detection Strategy

The system leverages multiple signals instead of relying on a single model:

* Face detection confidence
* Blur anomaly detection
* Color inconsistency analysis
* Frequency/noise-based artifacts

These signals are combined to compute a **frame-level and overall fake probability score**.

---

### 📊 Visualization Dashboard

The application includes an interactive dashboard built with Plotly:

* 📈 **Timeline Graph** – Frame-wise fake probability progression
* 🎯 **Gauge Chart** – Overall likelihood of manipulation
* 🥧 **Distribution Chart** – Ratio of suspicious vs normal frames
* 📉 **Histogram** – Score distribution across frames
* 🖼️ **Frame Inspection Panel** – Highlights key frames
* 🔥 **Heatmaps** – Visual indication of potential manipulation regions

---

### 🧠 Explainable AI (XAI)

The system emphasizes interpretability by providing insights into *why* a frame is classified as suspicious.

Example:

> “Detected high blur and inconsistent facial features across multiple frames.”

This improves transparency and supports better understanding of model behavior.

---

## 🏗️ Project Structure

```bash
Deepfake_detection/
│
├── app.py                  # Streamlit application entry point
├── utils/
│   ├── video_processor.py # Frame extraction and preprocessing
│   ├── model.py           # Detection logic and scoring
│   └── visualizer.py      # Visualization and plotting utilities
│
├── frames/                # Generated frames during runtime
├── requirements.txt
└── README.md
```

---

## ⚙️ Tech Stack

* **Frontend / UI:** Streamlit
* **Computer Vision:** OpenCV
* **Visualization:** Plotly
* **Analysis:** Heuristic-based signal processing + ML concepts
* **Language:** Python

---

## ▶️ Setup & Execution

### 1. Clone the repository

```bash
git clone https://github.com/Shreya580/Deepfake_detection.git
cd Deepfake_detection
```

### 2. Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate   # macOS/Linux
venv\Scripts\activate      # Windows
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

### 4. Run the application

```bash
streamlit run app.py
```

---

## 📌 System Workflow

```text
Input (Image/Video)
        ↓
Frame Extraction
        ↓
Per-frame Signal Analysis
        ↓
Score Aggregation
        ↓
Visualization Dashboard
        ↓
Final Verdict + Explanation
```

---

## ⚠️ Limitations

* Current detection relies on heuristic signals rather than fully trained deepfake-specific CNN models
* Heatmaps provide indicative regions, not pixel-accurate localization
* Performance may vary based on input quality and lighting conditions

---

## 🔮 Future Enhancements

* Integration with pretrained deepfake detection models (e.g., Xception, EfficientNet)
* Landmark-based and region-specific heatmap generation
* Audio-visual consistency analysis
* Real-time detection capabilities
* Model optimization for improved accuracy and performance

---

## 💡 Design Philosophy

This project focuses on:

> **Enhancing transparency in deepfake detection through explainable and visual analytics**

Rather than treating detection as a black-box problem, the system provides insight into the underlying decision-making process.

---
