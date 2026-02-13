"""
llm_detector.py
Layer 3: LLM-powered detection using OpenRouter API (Aurora Alpha - free).
"""

import os
import json
import re
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
                "temperature": 0.1,
                "max_tokens": 3000
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


def extract_json_from_response(response: str) -> list:
    """Extract and parse JSON from LLM response with multiple fallback strategies"""
    
    # Strategy 1: Remove markdown code blocks
    clean = response.strip()
    if "```" in clean:
        # Extract content between ```json and ``` or just between ```
        pattern = r'```(?:json)?\s*(\[.*?\])\s*```'
        matches = re.findall(pattern, clean, re.DOTALL)
        if matches:
            clean = matches[0]
        else:
            # Try to get anything between ```
            parts = clean.split("```")
            for part in parts:
                part = part.strip()
                if part.startswith("json"):
                    part = part[4:].strip()
                if part.startswith("["):
                    clean = part
                    break
    
    # Strategy 2: Find JSON array in text
    if not clean.startswith("["):
        match = re.search(r'\[.*\]', clean, re.DOTALL)
        if match:
            clean = match.group(0)
    
    # Strategy 3: Try to parse as-is
    try:
        results = json.loads(clean)
        if isinstance(results, dict):
            results = [results]
        print(f"   ✅ Parsed {len(results)} entries from JSON")
        return results
    except json.JSONDecodeError as e:
        print(f"   ⚠️ JSON parse failed: {str(e)[:100]}")
        
        # Strategy 4: Try to fix common JSON issues
        try:
            # Fix unescaped quotes in strings
            fixed = re.sub(r'(?<!\\)"(?=([^"\\]*(\\.[^"\\]*)*)"[^"]*$)', r'\"', clean)
            results = json.loads(fixed)
            if isinstance(results, dict):
                results = [results]
            print(f"   ✅ Fixed with quote escaping: {len(results)} entries")
            return results
        except:
            pass
        
        # Strategy 5: Manual parsing for simple cases
        try:
            # Look for is_correct: false entries
            incorrect_entries = []
            
            # More lenient patterns
            shipment_pattern = r'"shipment_id"\s*:\s*"([^"]+)"'
            hs_pattern = r'"hs_code"\s*:\s*"([^"]+)"'
            product_pattern = r'"product"\s*:\s*"([^"]+?)"(?=,|\})'
            correct_pattern = r'"is_correct"\s*:\s*(false|true)'
            reason_pattern = r'"reason"\s*:\s*"([^"]+?)"(?=,|\})'
            chapter_pattern = r'"correct_hs_chapter"\s*:\s*"([^"]+?)"(?=,|\})'
            
            # Split by likely record boundaries
            records = re.split(r'\}\s*,?\s*\{', clean)
            
            for record in records:
                # Add back braces if needed
                if not record.startswith('{'):
                    record = '{' + record
                if not record.endswith('}'):
                    record = record + '}'
                
                correct_match = re.search(correct_pattern, record, re.DOTALL)
                if correct_match and correct_match.group(1) == 'false':
                    shipment_match = re.search(shipment_pattern, record)
                    hs_match = re.search(hs_pattern, record)
                    product_match = re.search(product_pattern, record, re.DOTALL)
                    reason_match = re.search(reason_pattern, record, re.DOTALL)
                    chapter_match = re.search(chapter_pattern, record, re.DOTALL)
                    
                    if shipment_match and hs_match and product_match:
                        entry = {
                            "shipment_id": shipment_match.group(1),
                            "hs_code": hs_match.group(1),
                            "product": product_match.group(1).strip(),
                            "is_correct": False,
                            "reason": reason_match.group(1).strip() if reason_match else "HS code mismatch",
                            "correct_hs_chapter": chapter_match.group(1).strip() if chapter_match else "Unknown"
                        }
                        incorrect_entries.append(entry)
            
            if incorrect_entries:
                print(f"   ✅ Manually extracted {len(incorrect_entries)} entries")
                return incorrect_entries
        except Exception as e:
            print(f"   ⚠️ Manual parsing failed: {str(e)[:100]}")
        
        return []


def validate_hs_codes(shipments_df: pd.DataFrame) -> list:
    """Validate HS codes using LLM"""
    anomalies = []
    counter = [0]

    unique_combos = shipments_df[
        ['shipment_id', 'hs_code', 'product_description']
    ].drop_duplicates(subset=['hs_code', 'product_description'])

    print(f"   LLM: Validating {len(unique_combos)} unique HS codes...")

    combos_text = "\n".join([
        f"{row['shipment_id']}: HS_{row['hs_code']} -> {row['product_description']}"
        for _, row in unique_combos.iterrows()
    ])

    prompt = f"""You are an HS code auditor. Find MISMATCHED HS codes where the product and code are from DIFFERENT CHAPTERS.

KEY CHAPTERS:
Chapter 84 = COMPUTERS/MACHINERY (84713000 = laptops/processors/computers)
Chapter 61 = KNITTED TEXTILES (61091000 = T-shirts/cotton knits)
Chapter 62 = WOVEN TEXTILES (62046200 = trousers)
Chapter 87 = VEHICLES (87083010 = brake pads)
Chapter 42 = LEATHER (42021200 = wallets)
Chapter 09 = SPICES (09041100 = pepper)

SHIPMENTS TO CHECK:
{combos_text}

EXAMPLE ERRORS:
✗ HS_84713000 (computers Ch.84) for "Cotton T-shirts" = WRONG (textiles Ch.61)
✗ HS_61091000 (textiles Ch.61) for "Laptop computers" = WRONG (computers Ch.84)
✓ HS_61091000 (textiles Ch.61) for "Cotton T-shirts" = CORRECT

YOUR TASK:
Find entries where HS code chapter does NOT match product type.

Return ONLY JSON array (NO markdown, NO ```):
[
  {{"shipment_id":"SHP-2025-0089","hs_code":"84713000","product":"Cotton T-shirts 100% knitted","is_correct":false,"reason":"Textiles Chapter 61, not computers Chapter 84","correct_hs_chapter":"61 - Knitted textiles"}}
]

Include ONLY entries with is_correct: false. Be strict - if chapters don't match, mark false."""

    response = call_openrouter(prompt, "hs_code_validation")
    usage_log["breakdown_by_task"]["hs_code_validation"]["description"] = "HS code validation"

    if response.startswith("[LLM"):
        print(f"   ⚠️ Skipped: {response}")
        return anomalies

    results = extract_json_from_response(response)
    
    if not results:
        print(f"   ⚠️ Could not parse any results from LLM response")
        return anomalies

    for item in results:
        if not item.get("is_correct", True):
            counter[0] += 1
            
            # Get the HS code and product from LLM response
            hs_code = str(item.get('hs_code', '')).strip()
            product = str(item.get('product', '')).strip()
            
            # Find matching rows - try exact match first
            affected = shipments_df[
                (shipments_df['hs_code'].astype(str).str.strip() == hs_code) &
                (shipments_df['product_description'].astype(str).str.strip() == product)
            ]
            
            # If no exact match, try matching just by HS code (since product might have minor differences)
            if len(affected) == 0:
                print(f"   🔍 No exact match, trying HS code only for: {hs_code}")
                affected = shipments_df[shipments_df['hs_code'].astype(str).str.strip() == hs_code]
            
            print(f"   🎯 Found {len(affected)} affected shipments for HS {hs_code}")
            
            if len(affected) == 0:
                # If still no match, create anomaly using the shipment_id from LLM
                shipment_id = item.get('shipment_id', 'UNKNOWN')
                anomalies.append({
                    "anomaly_id": f"LLM-{counter[0]:03d}",
                    "layer": "llm",
                    "shipment_id": shipment_id,
                    "category": "compliance",
                    "sub_type": "hs_code_mismatch",
                    "description": f"HS {hs_code} incorrect for '{product}'. {item.get('reason', '')}",
                    "evidence": {
                        "hs_code_used": hs_code,
                        "product": product,
                        "llm_verdict": "INCORRECT",
                        "correct_chapter": item.get('correct_hs_chapter', 'Unknown'),
                        "llm_reason": item.get('reason', '')
                    },
                    "severity": "critical",
                    "recommendation": f"Reclassify under {item.get('correct_hs_chapter')}. File amendment. Penalty: ₹50K-₹2L.",
                    "estimated_penalty_usd": 6000,
                    "detection_method": "LLM: OpenRouter Aurora Alpha"
                })
            else:
                # Create anomaly for each affected shipment
                for _, row in affected.iterrows():
                    anomalies.append({
                        "anomaly_id": f"LLM-{counter[0]:03d}",
                        "layer": "llm",
                        "shipment_id": row['shipment_id'],
                        "category": "compliance",
                        "sub_type": "hs_code_mismatch",
                        "description": f"HS {hs_code} incorrect for '{product}'. {item.get('reason', '')}",
                        "evidence": {
                            "hs_code_used": hs_code,
                            "product": product,
                            "llm_verdict": "INCORRECT",
                            "correct_chapter": item.get('correct_hs_chapter', 'Unknown'),
                            "llm_reason": item.get('reason', '')
                        },
                        "severity": "critical",
                        "recommendation": f"Reclassify under {item.get('correct_hs_chapter')}. File amendment. Penalty: ₹50K-₹2L.",
                        "estimated_penalty_usd": 6000,
                        "detection_method": "LLM: OpenRouter Aurora Alpha"
                    })

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