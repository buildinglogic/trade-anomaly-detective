"""
llm_detector.py
Layer 3: LLM-powered detection using OpenRouter API (Aurora Alpha - free).
"""

import os
import json
import time
import datetime
import pandas as pd
import requests
from pathlib import Path

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

MODEL_NAME = "openrouter/aurora-alpha"
API_URL = "https://openrouter.ai/api/v1/chat/completions"

usage_log = {
    "provider": "OpenRouter",
    "model": MODEL_NAME,
    "total_calls": 0,
    "total_tokens": {"input": 0, "output": 0, "total": 0},
    "estimated_cost_usd": 0.0,
    "breakdown_by_task": {},
    "avg_latency_ms": 0,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "notes": "OpenRouter Aurora Alpha - free tier. Cost: $0.00"
}
latencies = []

_api_key = None


def _get_api_key():
    """Get OpenRouter API key"""
    global _api_key
    
    if _api_key:
        return _api_key
    
    # Try Streamlit secrets
    try:
        import streamlit as st
        if hasattr(st, 'secrets') and "OPENROUTER_API_KEY" in st.secrets:
            _api_key = st.secrets["OPENROUTER_API_KEY"]
            print(f"✅ API key from Streamlit secrets")
            return _api_key
    except:
        pass
    
    # Try .env
    try:
        from dotenv import load_dotenv
        env_path = Path(__file__).parent.parent / ".env"
        if env_path.exists():
            load_dotenv(dotenv_path=env_path, override=True)
            _api_key = os.getenv("OPENROUTER_API_KEY")
            if _api_key:
                print(f"✅ API key from .env")
                return _api_key
    except:
        pass
    
    # Try environment
    _api_key = os.environ.get("OPENROUTER_API_KEY")
    if _api_key:
        print(f"✅ API key from environment")
        return _api_key
    
    print(f"❌ No OPENROUTER_API_KEY found")
    return None


def _ensure_task_exists(task_name: str):
    if task_name not in usage_log["breakdown_by_task"]:
        usage_log["breakdown_by_task"][task_name] = {
            "calls": 0,
            "tokens": 0,
            "description": ""
        }


def call_openrouter(prompt: str, task_name: str, max_retries: int = 3) -> str:
    """Call OpenRouter API"""
    _ensure_task_exists(task_name)
    
    api_key = _get_api_key()
    if not api_key:
        return "[LLM UNAVAILABLE - Set OPENROUTER_API_KEY in .env or Streamlit Secrets]"

    headers = {
        "Authorization": f"Bearer {api_key}",
        "Content-Type": "application/json",
        "HTTP-Referer": "https://github.com/buildinglogic/trade-anomaly-detective",
    }

    for attempt in range(max_retries):
        try:
            start = time.time()
            
            data = {
                "model": MODEL_NAME,
                "messages": [
                    {"role": "user", "content": prompt}
                ],
                "temperature": 0.3,
                "max_tokens": 2000
            }
            
            response = requests.post(API_URL, headers=headers, json=data, timeout=30)
            response.raise_for_status()
            
            latency_ms = int((time.time() - start) * 1000)
            latencies.append(latency_ms)

            result = response.json()
            text = result["choices"][0]["message"]["content"]
            
            # Token counts
            input_tokens = result.get("usage", {}).get("prompt_tokens", len(prompt.split()) * 4 // 3)
            output_tokens = result.get("usage", {}).get("completion_tokens", len(text.split()) * 4 // 3)

            usage_log["total_calls"] += 1
            usage_log["total_tokens"]["input"] += input_tokens
            usage_log["total_tokens"]["output"] += output_tokens
            usage_log["total_tokens"]["total"] += (input_tokens + output_tokens)
            usage_log["breakdown_by_task"][task_name]["calls"] += 1
            usage_log["breakdown_by_task"][task_name]["tokens"] += (input_tokens + output_tokens)

            print(f"✅ LLM call #{usage_log['total_calls']} ({latency_ms}ms, {input_tokens + output_tokens} tokens)")
            return text

        except requests.exceptions.HTTPError as e:
            if response.status_code == 429:
                print(f"⚠️ Rate limit, waiting...")
                time.sleep(5)
            else:
                print(f"⚠️ HTTP {response.status_code}")
            if attempt == max_retries - 1:
                return f"[LLM ERROR: HTTP {response.status_code}]"
        except Exception as e:
            print(f"⚠️ Attempt {attempt+1}: {str(e)[:60]}")
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
            else:
                return f"[LLM ERROR: {str(e)[:80]}]"

    return "[LLM MAX RETRIES EXCEEDED]"


def validate_hs_codes(shipments_df: pd.DataFrame) -> list:
    """Validate HS codes using LLM"""
    anomalies = []
    counter = [0]

    unique_combos = shipments_df[
        ['shipment_id', 'hs_code', 'product_description']
    ].drop_duplicates(subset=['hs_code', 'product_description'])

    print(f"   LLM: Validating {len(unique_combos)} unique HS codes...")

    combos_text = "\n".join([
        f"- {row['shipment_id']}: HS {row['hs_code']} = '{row['product_description']}'"
        for _, row in unique_combos.iterrows()
    ])

    prompt = f"""You are an Indian customs HS code expert. Check if these HS codes correctly classify the products.

HS Code Reference:
- 61091000 = Knitted T-shirts
- 62114900 = Sarees (other women's garments)
- 84713000 = Laptops/computers
- 84137000 = Centrifugal pumps
- 87083010 = Brake pads
- 42021200 = Leather wallets
- 09041100 = Black pepper
- 10063020 = Basmati rice
- 30049099 = Pharmaceutical tablets
- 39021000 = Polypropylene plastic
- 73239300 = Stainless steel utensils
- 83062910 = Brass figurines
- 94054090 = LED lights

Check these entries:
{combos_text}

Return ONLY valid JSON array (no markdown, no code blocks):
[{{"shipment_id":"SHP-XXX","hs_code":"XXXXX","product":"...","is_correct":true/false,"reason":"...","correct_hs_chapter":"XX - name"}}]"""

    response = call_openrouter(prompt, "hs_code_validation")
    usage_log["breakdown_by_task"]["hs_code_validation"]["description"] = "HS code validation"

    if response.startswith("[LLM"):
        print(f"   ⚠️ Skipped: {response}")
        return anomalies

    try:
        clean = response.strip()
        
        # Remove markdown
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    clean = part
                    break
        
        results = json.loads(clean)
        if not isinstance(results, list):
            results = [results]

        for item in results:
            if not item.get("is_correct", True):
                counter[0] += 1
                affected = shipments_df[
                    (shipments_df['hs_code'] == item.get('hs_code', '')) &
                    (shipments_df['product_description'] == item.get('product', ''))
                ]
                
                for _, row in affected.iterrows():
                    anomalies.append({
                        "anomaly_id": f"LLM-{counter[0]:03d}",
                        "layer": "llm",
                        "shipment_id": row['shipment_id'],
                        "category": "compliance",
                        "sub_type": "hs_code_mismatch",
                        "description": f"HS {item.get('hs_code')} incorrect for '{item.get('product')}'. {item.get('reason', '')}",
                        "evidence": {
                            "hs_code_used": item.get('hs_code'),
                            "product": item.get('product'),
                            "llm_verdict": "INCORRECT",
                            "correct_chapter": item.get('correct_hs_chapter', 'Unknown'),
                            "llm_reason": item.get('reason', '')
                        },
                        "severity": "critical",
                        "recommendation": f"Reclassify under {item.get('correct_hs_chapter')}. File amendment. Penalty: ₹50K-₹2L.",
                        "estimated_penalty_usd": 6000,
                        "detection_method": "LLM: OpenRouter Aurora Alpha"
                    })

    except Exception as e:
        print(f"   ⚠️ Parse error: {e}")

    print(f"   LLM: {len(anomalies)} HS issues found")
    return anomalies


def generate_executive_summary(anomaly_report: dict) -> str:
    """Generate executive summary using LLM"""
    _ensure_task_exists("executive_summary")
    
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
        f"- [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:90]}"
        for a in top_anomalies
    ])

    prompt = f"""You are a senior trade compliance consultant. Write a professional executive summary for the Operations Head of an Indian export company.

ANALYSIS RESULTS:
- Total shipments: {anomaly_report.get('total_shipments', 0)}
- Total anomalies: {total}
- Total penalty risk: ${total_penalty:,}
- By severity: {json.dumps(by_severity)}
- By category: {json.dumps(by_category)}

TOP 5 HIGHEST-RISK ISSUES:
{top_desc}

Write 300-400 words with these sections:
1. Executive Overview (2-3 sentences)
2. Top 3 Most Urgent Issues (with shipment IDs and impact in INR where 1 USD = ₹83)
3. Identified Trends
4. Estimated Financial Exposure
5. Recommended Immediate Actions (3-4 bullet points)

Keep it professional, non-technical, and action-oriented."""

    summary = call_openrouter(prompt, "executive_summary")
    usage_log["breakdown_by_task"]["executive_summary"]["description"] = "Executive summary"

    if summary.startswith("[LLM"):
        return f"## Executive Summary\n\n⚠️ LLM unavailable.\n\n{summary}"

    return summary


def save_llm_usage_report():
    """Save LLM usage report"""
    if latencies:
        usage_log["avg_latency_ms"] = int(sum(latencies) / len(latencies))
    
    usage_log["estimated_cost_usd"] = 0.0
    usage_log["notes"] += f" | {usage_log['total_calls']} calls made."

    path = os.path.join(OUTPUT_DIR, 'llm_usage_report.json')
    with open(path, 'w') as f:
        json.dump(usage_log, f, indent=2)
    
    print(f"   ✅ llm_usage_report.json saved")
    return usage_log