"""
app.py
Streamlit dashboard for Trade Shipment Anomaly Detective.
Run: streamlit run src/app.py
"""

import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import json
import sys
import datetime
import os
from pathlib import Path
from dotenv import load_dotenv
import streamlit as st
import os

# ✅ DEBUG: Check if secrets are accessible
st.write("### 🔍 DEBUG INFO")
try:
    if "GROQ_API_KEY" in st.secrets:
        key = st.secrets["GROQ_API_KEY"]
        st.write(f"✅ Secret found in Streamlit")
        st.write(f"Key length: {len(key)}")
        st.write(f"Key starts with: {key[:15]}...")
    else:
        st.write("❌ GROQ_API_KEY not in st.secrets")
        st.write(f"Available secrets: {list(st.secrets.keys())}")
except Exception as e:
    st.write(f"❌ Error accessing secrets: {e}")

# Also check environment
env_key = os.getenv("GROQ_API_KEY", "")
st.write(f"Environment variable: {'✅ Found' if env_key else '❌ Not found'}")
st.write("---")


# Load .env from project root (parent of src/)
env_path = Path(__file__).parent.parent / ".env"
load_dotenv(env_path)
# ── Path setup ──────────────────────────────────────────────────────────
ROOT_DIR   = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA_DIR   = os.path.join(ROOT_DIR, 'data')
OUTPUT_DIR = os.path.join(ROOT_DIR, 'output')
SRC_DIR    = os.path.join(ROOT_DIR, 'src')
sys.path.insert(0, SRC_DIR)
os.makedirs(DATA_DIR, exist_ok=True)
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ── Page config ──────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Trade Anomaly Detective | Liquidmind AI",
    page_icon="🔍",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ── Custom CSS ────────────────────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0f1117; }
    .stMetric { background: #1e2130; border-radius: 10px; padding: 15px; border-left: 4px solid #4f46e5; }
    .critical-badge { background: #dc2626; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .high-badge     { background: #ea580c; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .medium-badge   { background: #ca8a04; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    .low-badge      { background: #16a34a; color: white; padding: 2px 8px; border-radius: 4px; font-size: 12px; font-weight: bold; }
    div[data-testid="stExpander"] { border: 1px solid #2d3748; border-radius: 8px; }
    .stButton > button { background: #4f46e5; color: white; border-radius: 8px; border: none; padding: 10px 24px; font-weight: bold; }
    .stButton > button:hover { background: #4338ca; }
</style>
""", unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════
#  HELPER: Load data
# ═══════════════════════════════════════════════════════════════════════
@st.cache_data(ttl=30)
def load_shipments():
    path = os.path.join(DATA_DIR, 'shipments.csv')
    if os.path.exists(path):
        return pd.read_csv(path)
    return None

@st.cache_data(ttl=30)
def load_anomaly_report():
    path = os.path.join(OUTPUT_DIR, 'anomaly_report.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data(ttl=30)
def load_accuracy_report():
    path = os.path.join(OUTPUT_DIR, 'accuracy_report.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

@st.cache_data(ttl=30)
def load_executive_summary():
    path = os.path.join(OUTPUT_DIR, 'executive_summary.md')
    if os.path.exists(path):
        with open(path) as f:
            return f.read()
    return None

@st.cache_data(ttl=30)
def load_llm_usage():
    path = os.path.join(OUTPUT_DIR, 'llm_usage_report.json')
    if os.path.exists(path):
        with open(path) as f:
            return json.load(f)
    return None

def data_exists():
    return (
        os.path.exists(os.path.join(DATA_DIR, 'shipments.csv')) and
        os.path.exists(os.path.join(OUTPUT_DIR, 'anomaly_report.json'))
    )

def run_full_analysis():
    """Run the complete detection pipeline."""
    import importlib
    with st.spinner("🔧 Step 1/5: Generating synthetic data..."):
        from data_generator import (
            generate_product_catalog, generate_buyers,
            generate_routes, generate_shipments, save_planted_anomalies
        )
        products_df  = generate_product_catalog()
        buyers_df    = generate_buyers()
        routes_df    = generate_routes()
        shipments_df = generate_shipments(products_df, buyers_df, routes_df)
        save_planted_anomalies()
        st.success("✅ Data generated: 250 shipments, 12 planted anomalies")

    with st.spinner("⚙️ Step 2/5: Layer 1 — Rule-based detection..."):
        from rule_engine import run_rule_checks
        rule_anomalies = run_rule_checks(shipments_df)
        st.success(f"✅ Rule engine: {len(rule_anomalies)} anomalies found")

    with st.spinner("📊 Step 3/5: Layer 2 — Statistical detection..."):
        from statistical_detector import run_statistical_checks
        stat_anomalies = run_statistical_checks(
            shipments_df, products_df, routes_df, buyers_df
        )
        st.success(f"✅ Statistical: {len(stat_anomalies)} anomalies found")

    with st.spinner("🤖 Step 4/5: Layer 3 — LLM detection (Gemini)..."):
        from llm_detector import validate_hs_codes, generate_executive_summary, save_llm_usage_report
        llm_anomalies = validate_hs_codes(shipments_df)
        st.success(f"✅ LLM: {len(llm_anomalies)} HS code issues found")

    with st.spinner("📋 Step 5/5: Generating reports + executive summary..."):
        from report_generator import run_full_pipeline
        all_anomalies = rule_anomalies + stat_anomalies + llm_anomalies

        # Build a temp report for executive summary
        temp_report = {"total_shipments": len(shipments_df), "anomalies": all_anomalies}
        exec_summary = generate_executive_summary(temp_report)
        save_llm_usage_report()

        results = run_full_pipeline(
            rule_anomalies, stat_anomalies, llm_anomalies,
            shipments_df, exec_summary
        )
        st.success("✅ All reports generated!")

    # Clear cache
    st.cache_data.clear()
    return results


def severity_badge(sev: str) -> str:
    colors = {
        "critical": "#dc2626",
        "high": "#ea580c",
        "medium": "#ca8a04",
        "low": "#16a34a"
    }
    return f'<span style="background:{colors.get(sev,"#6b7280")};color:white;padding:2px 8px;border-radius:4px;font-size:12px;font-weight:bold;">{sev.upper()}</span>'


# ═══════════════════════════════════════════════════════════════════════
#  SIDEBAR
# ═══════════════════════════════════════════════════════════════════════
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/detective.png", width=64)
    st.title("Trade Anomaly\nDetective")
    st.caption("Powered by Liquidmind AI")
    st.divider()

    st.subheader("🔍 Analysis Controls")

    if st.button("🚀 Run Analysis", use_container_width=True, type="primary"):
        with st.status("Running full pipeline...", expanded=True):
            try:
                run_full_analysis()
                st.success("🎉 Analysis complete!")
            except Exception as e:
                st.error(f"Error: {str(e)}")
                st.exception(e)

    st.divider()
    st.subheader("📡 System Info")
    st.caption(f"Data: {'✅ Loaded' if data_exists() else '❌ Not generated'}")
    st.caption(f"LLM: Gemini 1.5 Flash")
    st.caption(f"Statistical: Z-scores (σ=2.5)")
    st.caption(f"Updated: {datetime.datetime.now().strftime('%H:%M:%S')}")

    if not data_exists():
        st.warning("👆 Click 'Run Analysis' to generate data and detect anomalies.")

    st.divider()
    # Severity legend
    st.subheader("🚦 Severity Guide")
    st.markdown("""
    🔴 **Critical** — Immediate action required  
    🟠 **High** — Resolve within 24 hours  
    🟡 **Medium** — Review this week  
    🟢 **Low** — Monitor only
    """)


# ═══════════════════════════════════════════════════════════════════════
#  MAIN CONTENT
# ═══════════════════════════════════════════════════════════════════════
st.title("🔍 Trade Shipment Anomaly Detective")
st.caption("AI-powered 3-layer detection: Rule-Based → Statistical → LLM | Indian Export Compliance")

if not data_exists():
    st.info("👈 Click **Run Analysis** in the sidebar to generate data and start detection.")
    st.image("https://img.icons8.com/fluency/200/analytics.png", width=200)
    st.stop()

# ── Load data ─────────────────────────────────────────────────────────
shipments_df    = load_shipments()
anomaly_report  = load_anomaly_report()
accuracy_report = load_accuracy_report()
exec_summary    = load_executive_summary()
llm_usage       = load_llm_usage()

if not anomaly_report:
    st.warning("Run analysis first.")
    st.stop()

anomalies = anomaly_report.get("anomalies", [])
adf = pd.DataFrame(anomalies) if anomalies else pd.DataFrame()


# ═══════════════════════════════════════════════════════════════════════
#  TABS
# ═══════════════════════════════════════════════════════════════════════
tab1, tab2, tab3, tab4, tab5 = st.tabs([
    "📊 Dashboard",
    "🔎 Anomaly Table",
    "📋 Executive Summary",
    "🎯 Detection Accuracy",
    "🤖 LLM Usage"
])


# ═══ TAB 1: DASHBOARD ═══════════════════════════════════════════════════
with tab1:
    # KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    with col1:
        st.metric("Total Shipments", anomaly_report.get("total_shipments", 0))
    with col2:
        st.metric("Anomalies Found", anomaly_report.get("total_anomalies", 0),
                  delta=f"+{anomaly_report.get('total_anomalies', 0)}", delta_color="inverse")
    with col3:
        penalty_usd = anomaly_report.get("total_estimated_penalty_usd", 0)
        st.metric("Est. Penalty Risk", f"${penalty_usd:,.0f}")
    with col4:
        penalty_inr = penalty_usd * 83
        st.metric("In INR", f"₹{penalty_inr/100000:.1f}L")
    with col5:
        if accuracy_report:
            st.metric("Detection F1 Score", f"{accuracy_report.get('f1_score', 0):.1%}")

    st.divider()

    col_left, col_right = st.columns(2)

    with col_left:
        # Anomalies by category
        by_cat = anomaly_report.get("anomalies_by_category", {})
        if by_cat:
            fig = px.bar(
                x=list(by_cat.keys()),
                y=list(by_cat.values()),
                title="🗂️ Anomalies by Category",
                color=list(by_cat.values()),
                color_continuous_scale="Reds",
                labels={"x": "Category", "y": "Count"}
            )
            fig.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white',
                showlegend=False
            )
            st.plotly_chart(fig, use_container_width=True)

    with col_right:
        # Anomalies by severity
        by_sev = anomaly_report.get("anomalies_by_severity", {})
        if by_sev:
            sev_colors = {
                "critical": "#dc2626",
                "high": "#ea580c",
                "medium": "#ca8a04",
                "low": "#16a34a"
            }
            fig2 = px.pie(
                values=list(by_sev.values()),
                names=list(by_sev.keys()),
                title="🚦 Anomalies by Severity",
                color=list(by_sev.keys()),
                color_discrete_map=sev_colors,
                hole=0.45
            )
            fig2.update_layout(
                plot_bgcolor='rgba(0,0,0,0)',
                paper_bgcolor='rgba(0,0,0,0)',
                font_color='white'
            )
            st.plotly_chart(fig2, use_container_width=True)

    # Timeline of anomalies
    if shipments_df is not None and anomalies:
        st.subheader("📅 Shipment Timeline with Anomaly Flags")
        df_plot = shipments_df.copy()
        df_plot['date'] = pd.to_datetime(df_plot['date'])
        df_plot['month'] = df_plot['date'].dt.to_period('M').astype(str)

        anomaly_ids = {a['shipment_id'] for a in anomalies}
        df_plot['is_anomaly'] = df_plot['shipment_id'].isin(anomaly_ids)

        monthly = df_plot.groupby('month').agg(
            total=('shipment_id', 'count'),
            anomalies=('is_anomaly', 'sum')
        ).reset_index()

        fig3 = go.Figure()
        fig3.add_trace(go.Bar(
            x=monthly['month'], y=monthly['total'],
            name='Total Shipments', marker_color='#4f46e5', opacity=0.7
        ))
        fig3.add_trace(go.Bar(
            x=monthly['month'], y=monthly['anomalies'],
            name='Anomalies', marker_color='#dc2626'
        ))
        fig3.update_layout(
            title="Monthly Shipments vs Anomalies",
            barmode='overlay',
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            legend=dict(orientation="h")
        )
        st.plotly_chart(fig3, use_container_width=True)

    # Detection layer breakdown
    if anomalies:
        st.subheader("🏗️ Detection by Layer")
        by_layer = {}
        for a in anomalies:
            layer = a.get('layer', 'unknown')
            by_layer[layer] = by_layer.get(layer, 0) + 1

        c1, c2, c3 = st.columns(3)
        layer_info = [
            ("rule_based", "⚙️ Rule-Based", "#4f46e5"),
            ("statistical", "📊 Statistical", "#0891b2"),
            ("llm", "🤖 LLM-Powered", "#7c3aed"),
        ]
        cols = [c1, c2, c3]
        for col, (key, label, color) in zip(cols, layer_info):
            with col:
                count = by_layer.get(key, 0)
                st.metric(label, count)


# ═══ TAB 2: ANOMALY TABLE ════════════════════════════════════════════════
with tab2:
    st.subheader("🔎 All Detected Anomalies")

    if not adf.empty:
        # Filters
        col_f1, col_f2, col_f3 = st.columns(3)
        with col_f1:
            sev_filter = st.multiselect(
                "Filter by Severity",
                options=["critical", "high", "medium", "low"],
                default=["critical", "high", "medium", "low"]
            )
        with col_f2:
            cat_options = adf['category'].unique().tolist() if 'category' in adf.columns else []
            cat_filter = st.multiselect("Filter by Category", options=cat_options, default=cat_options)
        with col_f3:
            layer_options = adf['layer'].unique().tolist() if 'layer' in adf.columns else []
            layer_filter = st.multiselect("Filter by Layer", options=layer_options, default=layer_options)

        # Apply filters
        filtered = adf.copy()
        if sev_filter and 'severity' in filtered.columns:
            filtered = filtered[filtered['severity'].isin(sev_filter)]
        if cat_filter and 'category' in filtered.columns:
            filtered = filtered[filtered['category'].isin(cat_filter)]
        if layer_filter and 'layer' in filtered.columns:
            filtered = filtered[filtered['layer'].isin(layer_filter)]

        st.caption(f"Showing {len(filtered)} of {len(adf)} anomalies")

        # Display table
        display_cols = ['anomaly_id', 'shipment_id', 'severity', 'category',
                        'sub_type', 'estimated_penalty_usd']
        display_cols = [c for c in display_cols if c in filtered.columns]

        if 'description' in filtered.columns:
            filtered['desc_short'] = filtered['description'].str[:100] + "..."
            display_cols.append('desc_short')

        st.dataframe(
            filtered[display_cols].rename(columns={'desc_short': 'description'}),
            use_container_width=True,
            hide_index=True,
            column_config={
                "estimated_penalty_usd": st.column_config.NumberColumn(
                    "Penalty Risk ($)", format="$%d"
                ),
                "severity": st.column_config.TextColumn("Severity", width="small"),
            }
        )

        # Anomaly detail expanders
        st.subheader("📌 Anomaly Details (Click to Expand)")
        for _, row in filtered.iterrows():
            sev = row.get('severity', 'low')
            icon = {"critical": "🔴", "high": "🟠", "medium": "🟡", "low": "🟢"}.get(sev, "⚪")

            with st.expander(
                f"{icon} [{row.get('anomaly_id', '')}] {row.get('shipment_id', '')} — "
                f"{str(row.get('description', ''))[:80]}..."
            ):
                c1, c2 = st.columns([2, 1])
                with c1:
                    st.markdown(f"**Description:** {row.get('description', '')}")
                    st.markdown(f"**Layer:** `{row.get('layer', '')}` | "
                                f"**Category:** `{row.get('category', '')}` | "
                                f"**Sub-type:** `{row.get('sub_type', '')}`")
                    st.markdown(f"**Detection Method:** {row.get('detection_method', '')}")
                    if row.get('recommendation'):
                        st.markdown(f"**✅ Recommendation:** {row.get('recommendation', '')}")
                with c2:
                    sev_color = {"critical": "#dc2626", "high": "#ea580c",
                                 "medium": "#ca8a04", "low": "#16a34a"}.get(sev, "#6b7280")
                    st.markdown(f"""
                    <div style="background:{sev_color};padding:10px;border-radius:8px;text-align:center">
                        <h3 style="color:white;margin:0">{sev.upper()}</h3>
                        <p style="color:white;margin:5px 0">Penalty Risk</p>
                        <h2 style="color:white;margin:0">
                            ${row.get('estimated_penalty_usd', 0):,.0f}
                        </h2>
                        <p style="color:white;margin:5px 0">
                            ₹{row.get('estimated_penalty_usd', 0) * 83:,.0f}
                        </p>
                    </div>
                    """, unsafe_allow_html=True)

                # Evidence section
                if row.get('evidence'):
                    st.markdown("**📎 Evidence:**")
                    ev = row['evidence']
                    if isinstance(ev, str):
                        try:
                            ev = json.loads(ev)
                        except:
                            pass
                    if isinstance(ev, dict):
                        ev_df = pd.DataFrame(
                            list(ev.items()), columns=["Field", "Value"]
                        )
                        st.dataframe(ev_df, hide_index=True, use_container_width=True)


# ═══ TAB 3: EXECUTIVE SUMMARY ═══════════════════════════════════════════
with tab3:
    st.subheader("📋 Executive Summary")
    st.caption("AI-generated summary for the Operations Head. Non-technical language.")

    if exec_summary:
        st.markdown(exec_summary)
        # Download button
        st.download_button(
            label="📥 Download Summary (Markdown)",
            data=exec_summary,
            file_name="executive_summary.md",
            mime="text/markdown"
        )
    else:
        st.info("Executive summary will appear here after running analysis.")


# ═══ TAB 4: DETECTION ACCURACY ═══════════════════════════════════════════
with tab4:
    st.subheader("🎯 Detection Accuracy vs Planted Anomalies")

    if accuracy_report:
        # Metrics
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Planted Anomalies", accuracy_report.get("planted_anomalies", 0))
        with c2:
            detected = accuracy_report.get("detected_correctly", 0)
            planted  = accuracy_report.get("planted_anomalies", 1)
            st.metric("Detected Correctly", detected,
                      delta=f"{detected/planted:.0%} of planted")
        with c3:
            st.metric("False Positives", accuracy_report.get("false_positives", 0))
        with c4:
            st.metric("Missed", accuracy_report.get("missed", 0))

        c5, c6, c7 = st.columns(3)
        with c5:
            st.metric("Precision", f"{accuracy_report.get('precision', 0):.1%}")
        with c6:
            st.metric("Recall", f"{accuracy_report.get('recall', 0):.1%}")
        with c7:
            st.metric("F1 Score", f"{accuracy_report.get('f1_score', 0):.1%}")

        # Gauge chart for F1
        f1 = accuracy_report.get('f1_score', 0)
        fig_gauge = go.Figure(go.Indicator(
            mode="gauge+number",
            value=f1 * 100,
            title={'text': "F1 Score (%)"},
            gauge={
                'axis': {'range': [0, 100]},
                'bar': {'color': "#4f46e5"},
                'steps': [
                    {'range': [0, 50], 'color': "#dc2626"},
                    {'range': [50, 75], 'color': "#ca8a04"},
                    {'range': [75, 100], 'color': "#16a34a"},
                ],
                'threshold': {
                    'line': {'color': "white", 'width': 4},
                    'thickness': 0.75,
                    'value': f1 * 100
                }
            }
        ))
        fig_gauge.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='white',
            height=300
        )
        st.plotly_chart(fig_gauge, use_container_width=True)

        # Missed anomalies
        if accuracy_report.get("missed_details"):
            st.subheader("❌ Missed Anomalies")
            st.caption("These planted anomalies were not detected by the system.")
            for missed in accuracy_report["missed_details"]:
                st.warning(
                    f"**{missed['anomaly_id']}** ({missed['shipment_id']}): "
                    f"{missed['description']}"
                )

        # False positives
        if accuracy_report.get("false_positive_details"):
            st.subheader("⚠️ False Positives")
            st.caption("These were flagged as anomalies but are actually fine.")
            for fp in accuracy_report["false_positive_details"]:
                with st.expander(f"{fp.get('anomaly_id', '')} — {fp.get('shipment_id', '')}"):
                    st.write(f"**Why flagged:** {fp.get('why_flagged', '')}")
                    st.write(f"**Why it's fine:** {fp.get('why_its_actually_fine', '')}")


# ═══ TAB 5: LLM USAGE ═══════════════════════════════════════════════════
with tab5:
    st.subheader("🤖 LLM API Usage Report")

    if llm_usage:
        c1, c2, c3, c4 = st.columns(4)
        with c1:
            st.metric("Total API Calls", llm_usage.get("total_calls", 0))
        with c2:
            total_tokens = llm_usage.get("total_tokens", {}).get("total", 0)
            st.metric("Total Tokens", f"{total_tokens:,}")
        with c3:
            st.metric("Est. Cost", f"${llm_usage.get('estimated_cost_usd', 0):.4f}")
        with c4:
            st.metric("Avg Latency", f"{llm_usage.get('avg_latency_ms', 0)}ms")

        st.info(f"**Model:** {llm_usage.get('model', 'N/A')} | "
                f"**Provider:** {llm_usage.get('provider', 'N/A')}")
        st.caption(llm_usage.get('notes', ''))

        # Breakdown by task
        breakdown = llm_usage.get("breakdown_by_task", {})
        if breakdown:
            st.subheader("📊 Breakdown by Task")
            task_data = []
            for task, info in breakdown.items():
                task_data.append({
                    "Task": task.replace("_", " ").title(),
                    "API Calls": info.get("calls", 0),
                    "Tokens Used": info.get("tokens", 0),
                    "Description": info.get("description", "")
                })
            task_df = pd.DataFrame(task_data)
            st.dataframe(task_df, use_container_width=True, hide_index=True)

        st.subheader("⚡ Why This LLM Strategy is Efficient")
        st.markdown("""
        | What we send to LLM | What we DON'T send |
        |---|---|
        | Unique (HS code, product) combos only | All 250 raw shipment rows |
        | Pre-aggregated anomaly summary for exec summary | Full JSON anomaly report |
        | Targeted HS code validation questions | Payment data (handled by rules/stats) |

        **Result:** ~3-5 API calls total vs 500+ if we sent every row.
        Layer 1 (rules) handles math. Layer 2 (stats) handles patterns. LLM handles *reasoning*.
        """)

    st.download_button(
        label="📥 Download LLM Usage Report",
        data=json.dumps(llm_usage, indent=2) if llm_usage else "{}",
        file_name="llm_usage_report.json",
        mime="application/json"
    )
