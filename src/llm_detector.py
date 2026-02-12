"""
llm_detector.py
Layer 3: LLM-powered detection using Groq Llama 3.3 70B.
FIXED: Now properly loads .env file before checking for API key.
"""

import os
import json
import time
import datetime
import pandas as pd
from pathlib import Path

# ✅ CRITICAL FIX: Load .env file FIRST before checking environment
from dotenv import load_dotenv

# Get project root (parent directory of src/)
PROJECT_ROOT = Path(__file__).parent.parent
env_path = PROJECT_ROOT / ".env"

# Load the .env file into environment variables
load_dotenv(env_path, override=True)

# NOW get the API key from environment
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

# Debug output (shows in Streamlit console)
print("=" * 70)
print("🔍 llm_detector.py - API Key Loading:")
print(f"   Project root: {PROJECT_ROOT}")
print(f"   .env path: {env_path}")
print(f"   .env exists: {env_path.exists()}")
print(f"   API key loaded: {bool(GROQ_API_KEY)}")
if GROQ_API_KEY:
    print(f"   Key length: {len(GROQ_API_KEY)}")
    print(f"   Key preview: {GROQ_API_KEY[:20]}...")
    print("   ✅ API KEY READY!")
else:
    print("   ❌ NO API KEY - Check .env file contains GROQ_API_KEY=...")
print("=" * 70)

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)

# ─── Client Initialization ──────────────────────────────────────────────────
client = None
if GROQ_API_KEY:
    try:
        from groq import Groq
        client = Groq(api_key=GROQ_API_KEY)
        print("✅ Groq client configured successfully")
    except ImportError:
        print("❌ groq package not installed")
        print("   → Run: pip install groq")
    except Exception as e:
        print(f"❌ Error configuring Groq: {e}")
else:
    print("❌ Cannot initialize client: No API key")

MODEL_NAME = "llama-3.3-70b-versatile"

usage_log = {
    "provider": "Groq API",
    "model": MODEL_NAME,
    "total_calls": 0,
    "total_tokens": {"input": 0, "output": 0, "total": 0},
    "estimated_cost_usd": 0.0,
    "breakdown_by_task": {},
    "avg_latency_ms": 0,
    "timestamp": datetime.datetime.utcnow().isoformat() + "Z",
    "notes": "Groq - Free tier, unlimited API calls. Cost: $0.00"
}
latencies = []

# ... rest of your file stays exactly the same ...

def _ensure_task_exists(task_name: str):
    """Ensure task key exists in breakdown_by_task."""
    if task_name not in usage_log["breakdown_by_task"]:
        usage_log["breakdown_by_task"][task_name] = {
            "calls": 0,
            "tokens": 0,
            "description": ""
        }


def call_groq(prompt: str, task_name: str, max_retries: int = 3) -> str:
    """Call Groq API with error handling."""
    _ensure_task_exists(task_name)
    
    if not client:
        return "[LLM UNAVAILABLE - Set GROQ_API_KEY in .env]"

    for attempt in range(max_retries):
        try:
            start = time.time()
            
            # Call the model
            response = client.chat.completions.create(
                model=MODEL_NAME,
                messages=[
                    {"role": "user", "content": prompt}
                ],
                temperature=0.3,
                max_tokens=2000
            )
            
            latency_ms = int((time.time() - start) * 1000)
            latencies.append(latency_ms)

            text = response.choices[0].message.content

            # Token counts from response
            input_tokens  = response.usage.prompt_tokens
            output_tokens = response.usage.completion_tokens

            # Update usage log
            usage_log["total_calls"] += 1
            usage_log["total_tokens"]["input"]  += input_tokens
            usage_log["total_tokens"]["output"] += output_tokens
            usage_log["total_tokens"]["total"]  += (input_tokens + output_tokens)
            usage_log["breakdown_by_task"][task_name]["calls"]  += 1
            usage_log["breakdown_by_task"][task_name]["tokens"] += (input_tokens + output_tokens)

            print(f"✅ LLM call {usage_log['total_calls']} successful ({latency_ms}ms, {input_tokens+output_tokens} tokens)")
            return text

        except Exception as e:
            error_msg = str(e)
            print(f"⚠️  API call attempt {attempt+1} failed: {error_msg[:80]}")
            
            # Don't retry on API key errors
            if "API key" in error_msg or "authentication" in error_msg.lower():
                return f"[LLM ERROR: Invalid API key. Check GROQ_API_KEY in .env]"
            
            if attempt < max_retries - 1:
                wait_time = 2 ** attempt
                print(f"   Retrying in {wait_time}s...")
                time.sleep(wait_time)
            else:
                return f"[LLM ERROR: {error_msg[:100]}]"

    return "[LLM MAX RETRIES EXCEEDED]"


def validate_hs_codes(shipments_df: pd.DataFrame) -> list:
    """Check HS code vs product description using LLM."""
    anomalies = []
    counter = [0]

    unique_combos = shipments_df[
        ['shipment_id', 'hs_code', 'product_description']
    ].drop_duplicates(subset=['hs_code', 'product_description'])

    print(f"   LLM: Validating {len(unique_combos)} unique HS code combinations...")

    if len(unique_combos) == 0:
        print(f"   ⚠️  No unique combinations to validate")
        return anomalies

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

Entries to check:
{combos_text}

Respond ONLY as a valid JSON array with this format:
[
  {{"shipment_id": "...", "hs_code": "...", "product": "...", "is_correct": true/false, "reason": "...", "correct_hs_chapter": "..."}}
]

IMPORTANT: Return ONLY valid JSON. No markdown, no backticks, no explanation."""

    response = call_groq(prompt, task_name="hs_code_validation")
    
    usage_log["breakdown_by_task"]["hs_code_validation"]["description"] = (
        "Batch validation of unique HS code + product description combinations"
    )

    if response.startswith("[LLM"):
        print(f"   ⚠️  LLM skipped: {response}")
        return anomalies

    try:
        clean = response.strip()
        
        # Clean up markdown formatting if present
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
                        "description": (
                            f"HS code {item.get('hs_code')} does not match "
                            f"'{item.get('product')}'. {item.get('reason', '')}"
                        ),
                        "evidence": {
                            "hs_code_used": item.get('hs_code'),
                            "product": item.get('product'),
                            "llm_verdict": "INCORRECT",
                            "correct_chapter": item.get('correct_hs_chapter', 'Unknown'),
                            "llm_reason": item.get('reason')
                        },
                        "severity": "critical",
                        "recommendation": (
                            f"Re-classify under correct HS chapter: "
                            f"{item.get('correct_hs_chapter', 'see above')}. "
                            "File amendment with customs. Penalty: ₹50K-₹2L."
                        ),
                        "estimated_penalty_usd": 6000,
                        "detection_method": "LLM: Groq Mixtral HS classification check"
                    })

    except (json.JSONDecodeError, KeyError, Exception) as e:
        print(f"   ⚠️  LLM response parsing error: {e}")

    print(f"   LLM: {len(anomalies)} HS code mismatches found")
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
        f"- [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:120]}"
        for a in top_anomalies
    ])

    prompt = f"""You are a senior trade compliance consultant. Write a professional executive summary.

ANALYSIS RESULTS:
Total shipments: {anomaly_report.get('total_shipments', 0)}
Total anomalies: {total}
Total penalty risk: ${total_penalty:,.0f}

BY SEVERITY: {json.dumps(by_severity)}
BY CATEGORY: {json.dumps(by_category)}

TOP 5 HIGHEST-RISK ISSUES:
{top_desc}

Write a 300-400 word executive summary with:
1. Executive Overview (2-3 sentences)
2. Top 3 Most Urgent Issues (with shipment IDs and impact in INR where 1 USD = ₹83)
3. Identified Trends
4. Financial Exposure
5. Immediate Actions (3-4 bullet points)

Keep it professional and non-technical for Operations Head."""

    summary = call_groq(prompt, task_name="executive_summary")
    
    usage_log["breakdown_by_task"]["executive_summary"]["description"] = (
        "One-page executive summary for Operations Head"
    )

    if summary.startswith("[LLM"):
        return f"## Executive Summary\n\n⚠️ LLM unavailable.\n\n{summary}"

    return summary


def save_llm_usage_report():
    """Save LLM usage report to JSON."""
    if latencies:
        usage_log["avg_latency_ms"] = int(sum(latencies) / len(latencies))
    usage_log["estimated_cost_usd"] = 0.0
    usage_log["notes"] += f" | {usage_log['total_calls']} total calls made."

    path = os.path.join(OUTPUT_DIR, 'llm_usage_report.json')
    with open(path, 'w') as f:
        json.dump(usage_log, f, indent=2)
    print(f"   ✅ llm_usage_report.json saved")
    return usage_log
