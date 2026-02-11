"""
llm_detector.py
Layer 3: LLM-powered detection using Google Gemini 1.5 Flash (FREE tier).
"""

import os
import json
import time
import datetime
import pandas as pd
import google.generativeai as genai
from dotenv import load_dotenv

load_dotenv()

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
if GEMINI_API_KEY:
    genai.configure(api_key=GEMINI_API_KEY)

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
    "notes": "Gemini 1.5 Flash - free tier. Cost effectively $0."
}
latencies = []


def _ensure_task_key(task_name: str):
    """Make sure the task key exists in breakdown_by_task before accessing it."""
    if task_name not in usage_log["breakdown_by_task"]:
        usage_log["breakdown_by_task"][task_name] = {
            "calls": 0,
            "tokens": 0,
            "description": ""
        }


def call_gemini(prompt: str, task_name: str, max_retries: int = 3) -> str:
    """Call Gemini API with retry logic and usage tracking."""
    _ensure_task_key(task_name)

    if not GEMINI_API_KEY:
        return "[LLM UNAVAILABLE - Set GEMINI_API_KEY in Streamlit Secrets]"

    model = genai.GenerativeModel(MODEL_NAME)

    for attempt in range(max_retries):
        try:
            start = time.time()
            response = model.generate_content(prompt)
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

            return text

        except Exception as e:
            if attempt < max_retries - 1:
                time.sleep(2 ** attempt)
                continue
            return f"[LLM ERROR: {str(e)}]"

    return "[LLM MAX RETRIES EXCEEDED]"


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

    prompt = f"""You are an Indian customs classification expert. Review these HS code + product description pairs.

For each entry, check if the HS code correctly classifies the product.
HS code classification rules:
- Chapter 61: Knitted/crocheted clothing (T-shirts, sweaters)
- Chapter 62: Woven clothing and sarees
- Chapter 84: Machinery, computers (84713000 = laptops/computers)
- Chapter 85: Electronics
- Chapter 87: Vehicles and parts
- Chapter 30: Pharmaceuticals
- Chapter 10: Cereals (rice = 1006xxxx)
- Chapter 09: Spices (pepper = 0904xxxx)
- Chapter 39: Plastics
- Chapter 42: Leather articles
- Chapter 73: Iron/Steel articles
- Chapter 83: Miscellaneous metal articles
- Chapter 94: Furniture, lamps, lighting
- Chapter 71: Precious stones

Entries to check:
{combos_text}

Respond ONLY as a valid JSON array. For each entry:
{{
  "shipment_id": "...",
  "hs_code": "...",
  "product": "...",
  "is_correct": true or false,
  "reason": "brief explanation",
  "correct_hs_chapter": "XX - chapter name"
}}

Return ONLY the JSON array with no other text, no markdown, no backticks."""

    response = call_gemini(prompt, task_name="hs_code_validation")

    # Always safe to set description now because _ensure_task_key was called inside call_gemini
    usage_log["breakdown_by_task"]["hs_code_validation"]["description"] = (
        "Batch validation of unique HS code + product description combinations"
    )

    if response.startswith("[LLM"):
        print(f"   LLM skipped: {response}")
        return anomalies

    try:
        clean = response.strip()
        # Strip markdown code fences if present
        if "```" in clean:
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("[") or part.startswith("{"):
                    clean = part
                    break

        results = json.loads(clean)

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
                        "description": (
                            f"HS code {item['hs_code']} does not match "
                            f"'{item['product']}'. {item['reason']}"
                        ),
                        "evidence": {
                            "hs_code_used": item['hs_code'],
                            "product": item['product'],
                            "llm_verdict": "INCORRECT",
                            "correct_chapter": item.get('correct_hs_chapter', 'Unknown'),
                            "llm_reason": item['reason']
                        },
                        "severity": "critical",
                        "recommendation": (
                            f"Re-classify under correct HS chapter: "
                            f"{item.get('correct_hs_chapter', 'see above')}. "
                            "File amendment with customs. Penalty: Rs 50K-2L."
                        ),
                        "estimated_penalty_usd": 6000,
                        "detection_method": "LLM: Gemini 1.5 Flash HS classification check"
                    })

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"   LLM response parsing error: {e}")

    print(f"   LLM: {len(anomalies)} HS code mismatches found")
    return anomalies


def generate_executive_summary(anomaly_report: dict) -> str:
    """Generate a 1-page executive summary."""
    _ensure_task_key("executive_summary")

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

    prompt = f"""You are a senior trade compliance consultant. Write a professional executive summary for the Operations Head of an Indian export company.

ANALYSIS RESULTS:
Total shipments analyzed: {anomaly_report.get('total_shipments', 0)}
Total anomalies detected: {total}
Total estimated penalty risk: ${total_penalty:,.0f}

BY SEVERITY: {json.dumps(by_severity)}
BY CATEGORY: {json.dumps(by_category)}

TOP 5 HIGHEST-RISK ISSUES:
{top_desc}

Write a 400-500 word executive summary with these sections:
1. **Executive Overview** (2-3 sentences)
2. **Top 3 Most Urgent Issues** (with specific shipment IDs and impact in INR, 1 USD = Rs 83)
3. **Identified Trends** (payment patterns, volume anomalies)
4. **Estimated Financial Exposure** (penalties, working capital at risk)
5. **Recommended Immediate Actions** (3-4 bullet points)

Tone: Professional, non-technical, action-oriented."""

    summary = call_gemini(prompt, task_name="executive_summary")

    usage_log["breakdown_by_task"]["executive_summary"]["description"] = (
        "One-page executive summary for Operations Head"
    )

    if summary.startswith("[LLM"):
        return "## Executive Summary\n\nLLM unavailable. Please set GEMINI_API_KEY in Streamlit Secrets to generate the executive summary."

    return summary


def save_llm_usage_report():
    """Save the LLM usage tracking report."""
    if latencies:
        usage_log["avg_latency_ms"] = int(sum(latencies) / len(latencies))
    usage_log["estimated_cost_usd"] = 0.0
    usage_log["notes"] += f" | {usage_log['total_calls']} total calls made."

    path = os.path.join(OUTPUT_DIR, 'llm_usage_report.json')
    with open(path, 'w') as f:
        json.dump(usage_log, f, indent=2)
    print(f"   LLM usage report saved")
    return usage_log
