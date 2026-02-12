"""
llm_detector.py
Layer 3: LLM detection using Google Gemini 1.5 Flash (google-genai v0.2.2).
"""

import os
import json
import time
import datetime
import pandas as pd
from pathlib import Path

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "gemini-1.5-flash"

usage_log = {
    "provider": "Google AI Studio",
    "model": MODEL_NAME,
    "total_calls": 0,
    "total_tokens": {"input": 0, "output": 0, "total": 0},
    "estimated_cost_usd": 0.0,
    "breakdown_by_task": {},
    "avg_latency_ms": 0,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "notes": "Gemini 1.5 Flash - free tier. Cost: $0.00"
}
latencies = []

_client = None
_api_key_loaded = False


def _get_api_key():
    """Lazy load API key from Streamlit secrets or .env"""
    global _api_key_loaded
    
    if _api_key_loaded:
        return os.environ.get("GEMINI_API_KEY", "")
    
    api_key = ""
    
    # Try Streamlit secrets first
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "GEMINI_API_KEY" in st.secrets:
            api_key = st.secrets["GEMINI_API_KEY"]
            os.environ["GEMINI_API_KEY"] = api_key
            print(f"✅ Loaded API key from Streamlit secrets (length: {len(api_key)})")
            _api_key_loaded = True
            return api_key
    except Exception as e:
        print(f"⚠️ Could not load from Streamlit secrets: {e}")
    
    # Try .env file
    try:
        from dotenv import load_dotenv
        load_dotenv(override=True)
        api_key = os.getenv("GEMINI_API_KEY", "")
        if api_key:
            print(f"✅ Loaded API key from .env (length: {len(api_key)})")
            _api_key_loaded = True
            return api_key
    except Exception as e:
        print(f"⚠️ Could not load from .env: {e}")
    
    print("❌ No API key found")
    return ""


def _get_client():
    """Lazy initialize Gemini client"""
    global _client
    
    if _client is not None:
        return _client
    
    api_key = _get_api_key()
    if not api_key:
        return None
    
    try:
        from google import genai
        _client = genai.Client(api_key=api_key)
        print(f"✅ Gemini client initialized")
        return _client
    except Exception as e:
        print(f"❌ Error initializing Gemini: {e}")
        return None


def _ensure_task_exists(task_name: str):
    if task_name not in usage_log["breakdown_by_task"]:
        usage_log["breakdown_by_task"][task_name] = {
            "calls": 0,
            "tokens": 0,
            "description": ""
        }


def call_gemini(prompt: str, task_name: str, max_retries: int = 3) -> str:
    """Call Gemini API"""
    _ensure_task_exists(task_name)
    
    client = _get_client()
    if not client:
        return "[LLM UNAVAILABLE - Set GEMINI_API_KEY in Streamlit Secrets]"

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = client.models.generate_content(
                model=MODEL_NAME,
                contents=prompt
            )
            latency_ms = int((time.time() - start) * 1000)
            latencies.append(latency_ms)

            text = response.text
            input_tokens  = len(prompt.split()) * 4 // 3
            output_tokens = len(text.split()) * 4 // 3

            usage_log["total_calls"] += 1
            usage_log["total_tokens"]["input"]  += input_tokens
            usage_log["total_tokens"]["output"] += output_tokens
            usage_log["total_tokens"]["total"]  += (input_tokens + output_tokens)
            usage_log["breakdown_by_task"][task_name]["calls"]  += 1
            usage_log["breakdown_by_task"][task_name]["tokens"] += (input_tokens + output_tokens)

            print(f"✅ LLM call {usage_log['total_calls']} OK ({latency_ms}ms)")
            return text

        except Exception as e:
            print(f"⚠️ Attempt {attempt+1} failed: {str(e)[:80]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[LLM ERROR: {str(e)[:100]}]"

    return "[LLM MAX RETRIES EXCEEDED]"


def validate_hs_codes(shipments_df: pd.DataFrame) -> list:
    anomalies = []
    counter = [0]

    unique_combos = shipments_df[
        ['shipment_id', 'hs_code', 'product_description']
    ].drop_duplicates(subset=['hs_code', 'product_description'])

    print(f"   LLM: Validating {len(unique_combos)} unique HS combinations...")

    combos_text = "\n".join([
        f"- ID:{row['shipment_id']} | HS:{row['hs_code']} | Product: {row['product_description']}"
        for _, row in unique_combos.iterrows()
    ])

    prompt = f"""Indian customs expert: Review HS codes.

Rules:
- Ch 61: Knitted clothing (61091000 = T-shirts)
- Ch 62: Woven clothing (62114900 = sarees)
- Ch 84: Machinery (84713000 = laptops)
- Ch 87: Auto parts (87083010 = brake pads)
- Ch 42: Leather (42021200 = wallets, 42031000 = apparel accessories)
- Ch 09: Spices (09041100 = black pepper, 09042110 = chili/capsicum)
- Ch 10: Cereals (10063020 = rice)

Check:
{combos_text}

Return ONLY valid JSON array:
[{{"shipment_id":"...","hs_code":"...","product":"...","is_correct":true/false,"reason":"...","correct_hs_chapter":"..."}}]

No markdown, no backticks."""

    response = call_gemini(prompt, "hs_code_validation")
    usage_log["breakdown_by_task"]["hs_code_validation"]["description"] = "HS code validation"

    if response.startswith("[LLM"):
        print(f"   ⚠️ Skipped: {response}")
        return anomalies

    try:
        clean = response.strip()
        if "```" in clean:
            clean = clean.split("```")[1].replace("json", "").strip()
        
        results = json.loads(clean)
        if not isinstance(results, list):
            results = [results]

        for item in results:
            if not item.get("is_correct", True):
                counter[0] += 1
                affected = shipments_df[
                    (shipments_df['hs_code'] == item['hs_code']) &
                    (shipments_df['product_description'] == item['product'])
                ]
                for _, row in affected.iterrows():
                    anomalies.append({
                        "anomaly_id": f"LLM-{counter[0]:03d}",
                        "layer": "llm",
                        "shipment_id": row['shipment_id'],
                        "category": "compliance",
                        "sub_type": "hs_code_mismatch",
                        "description": f"HS {item['hs_code']} wrong for '{item['product']}'. {item['reason']}",
                        "evidence": {
                            "hs_code_used": item['hs_code'],
                            "product": item['product'],
                            "llm_verdict": "INCORRECT",
                            "correct_chapter": item.get('correct_hs_chapter', 'Unknown'),
                            "llm_reason": item['reason']
                        },
                        "severity": "critical",
                        "recommendation": f"Re-classify: {item.get('correct_hs_chapter', 'see above')}. Penalty: ₹50K-2L.",
                        "estimated_penalty_usd": 6000,
                        "detection_method": "LLM: Gemini 1.5 Flash HS check"
                    })
    except Exception as e:
        print(f"   ⚠️ Parse error: {e}")

    print(f"   LLM: {len(anomalies)} HS mismatches found")
    return anomalies


def generate_executive_summary(anomaly_report: dict) -> str:
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
        f"- [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:120]}"
        for a in top_anomalies
    ])

    prompt = f"""Trade compliance consultant: Write executive summary for Operations Head.

Data:
- Shipments: {anomaly_report.get('total_shipments', 0)}
- Anomalies: {total}
- Penalty risk: ${total_penalty:,.0f}
- By severity: {json.dumps(by_severity)}
- By category: {json.dumps(by_category)}

Top 5 issues:
{top_desc}

Write 300-400 words:
1. Executive Overview (2-3 sentences)
2. Top 3 Urgent Issues (shipment IDs, INR impact where 1 USD = ₹83)
3. Trends
4. Financial Exposure
5. Immediate Actions (3-4 bullets)

Professional, non-technical."""

    summary = call_gemini(prompt, "executive_summary")
    usage_log["breakdown_by_task"]["executive_summary"]["description"] = "Executive summary"

    if summary.startswith("[LLM"):
        return f"## Executive Summary\n\n⚠️ LLM unavailable.\n\n{summary}"

    return summary


def save_llm_usage_report():
    if latencies:
        usage_log["avg_latency_ms"] = int(sum(latencies) / len(latencies))
    usage_log["estimated_cost_usd"] = 0.0
    usage_log["notes"] += f" | {usage_log['total_calls']} total calls made."

    path = os.path.join(OUTPUT_DIR, 'llm_usage_report.json')
    with open(path, 'w') as f:
        json.dump(usage_log, f, indent=2)
    print("   ✅ llm_usage_report.json saved")
    return usage_log
