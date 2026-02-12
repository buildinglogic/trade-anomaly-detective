"""
llm_detector.py
Layer 3: LLM-powered detection using Google Gemini Flash
"""

import os
import json
import time
import datetime
import pandas as pd
from dotenv import load_dotenv

load_dotenv()

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GEMINI_API_KEY = ""

try:
    import streamlit as st
    if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
        GEMINI_API_KEY = st.secrets["GEMINI_API_KEY"]
except:
    pass

if not GEMINI_API_KEY:
    GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")

client = None
if GEMINI_API_KEY:
    try:
        import google.generativeai as genai
        genai.configure(api_key=GEMINI_API_KEY)
        client = genai.GenerativeModel("gemini-1.5-flash")
        print("✅ Gemini client configured")
    except Exception as e:
        print(f"⚠️ Gemini config error: {e}")

usage_log = {
    "provider": "Google AI Studio",
    "model": "gemini-1.5-flash",
    "total_calls": 0,
    "total_tokens": {"input": 0, "output": 0, "total": 0},
    "estimated_cost_usd": 0.0,
    "breakdown_by_task": {},
    "avg_latency_ms": 0,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "notes": "Gemini 1.5 Flash - free tier"
}
latencies = []


def call_gemini(prompt: str, task_name: str) -> str:
    """Call Gemini API."""
    if task_name not in usage_log["breakdown_by_task"]:
        usage_log["breakdown_by_task"][task_name] = {
            "calls": 0,
            "tokens": 0,
            "description": ""
        }
    
    if not client:
        return "[LLM UNAVAILABLE]"

    try:
        start = time.time()
        response = client.generate_content(prompt)
        latency_ms = int((time.time() - start) * 1000)
        latencies.append(latency_ms)
        
        text = response.text if response else ""
        
        input_tokens = len(prompt.split()) * 4 // 3
        output_tokens = len(text.split()) * 4 // 3
        
        usage_log["total_calls"] += 1
        usage_log["total_tokens"]["input"] += input_tokens
        usage_log["total_tokens"]["output"] += output_tokens
        usage_log["total_tokens"]["total"] += input_tokens + output_tokens
        usage_log["breakdown_by_task"][task_name]["calls"] += 1
        usage_log["breakdown_by_task"][task_name]["tokens"] += input_tokens + output_tokens
        
        return text
    except Exception as e:
        print(f"⚠️ LLM error: {e}")
        return "[LLM ERROR]"


def validate_hs_codes(shipments_df: pd.DataFrame) -> list:
    """Check HS code vs product description using LLM."""
    anomalies = []
    counter = [0]
    
    unique_combos = shipments_df[
        ['shipment_id', 'hs_code', 'product_description']
    ].drop_duplicates(subset=['hs_code', 'product_description'])
    
    print(f"   LLM: Validating {len(unique_combos)} unique HS code combinations...")
    
    combos_text = "\n".join([
        f"- ID:{row['shipment_id']} | HS:{row['hs_code']} | Product: {row['product_description']}"
        for _, row in unique_combos.iterrows()
    ])
    
    prompt = f"""You are an Indian customs classification expert. Review these HS code + product pairs.

HS code classification rules:
- Chapter 61: Knitted/crocheted clothing (T-shirts, sweaters)
- Chapter 62: Woven clothing, sarees
- Chapter 84: Machinery, computers (84713000 = laptops)
- Chapter 10: Cereals (rice = 1006xxxx)
- Chapter 09: Spices (pepper = 0904xxxx)
- Chapter 42: Leather articles

Entries to check:
{combos_text}

For each entry, respond ONLY as valid JSON array (no markdown):
[{{"shipment_id": "...", "hs_code": "...", "is_correct": true/false, "reason": "..."}}]"""
    
    response = call_gemini(prompt, task_name="hs_code_validation")
    usage_log["breakdown_by_task"]["hs_code_validation"]["description"] = "HS code validation"
    
    if "[LLM" in response:
        return anomalies
    
    try:
        clean = response.strip()
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                if part.strip().startswith("["):
                    clean = part.strip()
                    break
        
        results = json.loads(clean)
        
        for item in results:
            if not item.get("is_correct", True):
                counter[0] += 1
                affected = shipments_df[
                    (shipments_df['hs_code'] == item['hs_code']) &
                    (shipments_df['product_description'] == item.get('product', ''))
                ]
                for _, row in affected.iterrows():
                    anomalies.append({
                        "anomaly_id": f"LLM-{counter[0]:03d}",
                        "layer": "llm",
                        "shipment_id": row['shipment_id'],
                        "category": "compliance",
                        "sub_type": "hs_code_mismatch",
                        "description": f"HS code {item['hs_code']} doesn't match '{item.get('product', '')}'. {item.get('reason', '')}",
                        "evidence": {
                            "hs_code": item['hs_code'],
                            "product": item.get('product', ''),
                            "llm_reason": item.get('reason', '')
                        },
                        "severity": "critical",
                        "recommendation": "Re-classify to correct HS chapter. File amendment with customs.",
                        "estimated_penalty_usd": 6000,
                        "detection_method": "LLM: Gemini HS code validation"
                    })
    except:
        pass
    
    print(f"   LLM: {len(anomalies)} HS mismatches found")
    return anomalies


def generate_executive_summary(anomaly_report: dict) -> str:
    """Generate executive summary using LLM."""
    total = len(anomaly_report.get("anomalies", []))
    by_severity = {}
    by_category = {}
    total_penalty = 0
    
    for a in anomaly_report.get("anomalies", []):
        sev = a.get("severity", "unknown")
        cat = a.get("category", "unknown")
        penalty = a.get("estimated_penalty_usd", 0)
        by_severity[sev] = by_severity.get(sev, 0) + 1
        by_category[cat] = by_category.get(cat, 0) + 1
        total_penalty += penalty
    
    top_anomalies = sorted(
        anomaly_report.get("anomalies", []),
        key=lambda x: x.get("estimated_penalty_usd", 0),
        reverse=True
    )[:5]
    
    top_desc = "\n".join([
        f"- [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:100]}"
        for a in top_anomalies
    ])
    
    prompt = f"""You are a trade compliance expert. Write executive summary for Operations Head.

Total shipments: {anomaly_report.get('total_shipments', 0)}
Total anomalies: {total}
Penalty risk: ${total_penalty:,.0f} (≈₹{total_penalty * 83:,.0f})

By severity: {json.dumps(by_severity)}
By category: {json.dumps(by_category)}

Top issues:
{top_desc}

Write 300 words covering:
1. Overview
2. Top 3 critical risks
3. Immediate actions"""
    
    summary = call_gemini(prompt, task_name="executive_summary")
    usage_log["breakdown_by_task"]["executive_summary"]["description"] = "Executive summary"
    
    if "[LLM" in summary:
        return "## Executive Summary\n\n⚠️ LLM unavailable. Set GEMINI_API_KEY."
    
    return summary


def save_llm_usage_report():
    """Save LLM usage report."""
    if latencies:
        usage_log["avg_latency_ms"] = int(sum(latencies) / len(latencies))
    
    path = os.path.join(OUTPUT_DIR, 'llm_usage_report.json')
    with open(path, 'w') as f:
        json.dump(usage_log, f, indent=2)
    print(f"   ✅ llm_usage_report.json saved")
    return usage_log


def run_llm_detector(shipments_df: pd.DataFrame) -> list:
    """Run LLM detection layer."""
    print("\n🤖 LAYER 3: LLM-Powered Detection")
    print("─" * 50)
    
    all_anomalies = []
    all_anomalies.extend(validate_hs_codes(shipments_df))
    
    print(f"\n✅ Layer 3 complete: {len(all_anomalies)} anomalies detected")
    return all_anomalies