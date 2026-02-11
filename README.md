# 🔍 Trade Shipment Anomaly Detective

An AI-powered 3-layer anomaly detection system for Indian export trade shipments.

**Live Demo:** [https://your-app-name.streamlit.app](https://your-app-name.streamlit.app)

---

## Architecture
```
Layer 1 (Rule-Based)     → Arithmetic & logic checks (5 rules)
Layer 2 (Statistical)    → Z-score outlier detection (6 checks)  
Layer 3 (LLM-Powered)    → Gemini 1.5 Flash for HS code validation & summary
```

## Setup (Local)

### Prerequisites
- Python 3.10+
- Free Gemini API key from [Google AI Studio](https://aistudio.google.com/app/apikey)

### Installation
```bash
# Clone repo
git clone https://github.com/YOUR_USERNAME/trade-anomaly-detective.git
cd trade-anomaly-detective

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Windows: venv\Scripts\activate

# Install dependencies
pip install -r requirements.txt

# Add your API key
echo "GEMINI_API_KEY=your_key_here" > .env
```

### Run
```bash
streamlit run src/app.py
```

Then click **"Run Analysis"** in the sidebar.

### Run Pipeline Manually (without UI)
```bash
python src/data_generator.py
python src/rule_engine.py
python src/statistical_detector.py
python src/llm_detector.py
```

---

## What Gets Detected

| Layer | Method | Checks |
|-------|--------|--------|
| Rule-Based | Arithmetic logic | FOB math, drawback on rejected, CIF/freight, insurance rate, payment status |
| Statistical | Z-scores (σ=2.5) | Price outliers, transit outliers, freight outliers, payment delays, volume spikes |
| LLM | Gemini 1.5 Flash | HS code classification, executive summary |

## Planted Anomalies

12 anomalies across 6 categories. See `data/planted_anomalies.json`.

## Submission Structure
```
├── data/               # Generated CSVs + planted_anomalies.json
├── src/                # All Python source files
├── output/             # Generated reports (JSON + MD)
├── DESIGN_DECISIONS.md # Architecture rationale
├── requirements.txt
└── README.md
```

## LLM Cost

Gemini 1.5 Flash free tier: **$0.00** for this workload (~3 API calls, ~2,500 tokens total).
