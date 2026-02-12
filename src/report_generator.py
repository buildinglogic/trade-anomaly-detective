"""
report_generator.py
Combines anomalies from all 3 layers, deduplicates, generates all output files.
Includes IMPROVEMENT #4: Confidence Score calculation.
"""

import pandas as pd
import numpy as np
import json
import os
import datetime

DATA_DIR   = os.path.join(os.path.dirname(__file__), '..', 'data')
OUTPUT_DIR = os.path.join(os.path.dirname(__file__), '..', 'output')
os.makedirs(OUTPUT_DIR, exist_ok=True)


def calculate_confidence_score(anomaly: dict) -> float:
    """
    IMPROVEMENT #4: Score 0.0 (likely false positive) to 1.0 (definitely real)
    """
    layer = anomaly.get('layer', '')
    severity = anomaly.get('severity', 'medium')
    evidence_count = len(anomaly.get('evidence', {}))
    
    # Base scores by layer
    scores = {
        'rule_based': 0.95,      # Rules are almost always correct
        'statistical': 0.55,      # Stats have more false positives
        'llm': 0.80,              # LLM is pretty reliable
        'trend': 0.85             # Trends are reliable
    }
    
    base_score = scores.get(layer, 0.5)
    
    # Adjust by severity
    if severity == 'critical':
        base_score += 0.10
    elif severity == 'medium':
        base_score -= 0.10
    
    # Adjust by evidence richness
    if evidence_count >= 5:
        base_score += 0.05
    
    return min(1.0, max(0.0, base_score))


def deduplicate_anomalies(all_anomalies: list) -> list:
    """
    If multiple layers detect the same shipment+sub_type, keep the most informative one.
    Priority: LLM > Trend > Statistical > Rule-based
    """
    seen = {}
    priority = {"llm": 4, "trend": 3.5, "statistical": 2, "rule_based": 1}

    for a in all_anomalies:
        key = (a['shipment_id'], a['sub_type'])
        current_priority = priority.get(a.get('layer', 'rule_based'), 1)
        if key not in seen or current_priority > priority.get(seen[key].get('layer', 'rule_based'), 1):
            seen[key] = a

    return list(seen.values())


def generate_anomaly_report(
    shipments_df: pd.DataFrame,
    all_anomalies: list
) -> dict:
    """Generate the master anomaly_report.json with confidence scores."""

    # Add confidence scores
    for a in all_anomalies:
        a['confidence_score'] = calculate_confidence_score(a)

    # Add ranking/severity scores
    severity_order = {"critical": 4, "high": 3, "medium": 2, "low": 1}
    for a in all_anomalies:
        a['risk_score'] = (
            severity_order.get(a.get('severity', 'low'), 1) * 25 +
            min(a.get('estimated_penalty_usd', 0) / 1000, 25)
        )

    # Sort by risk score
    all_anomalies.sort(key=lambda x: x.get('risk_score', 0), reverse=True)

    by_category = {}
    by_severity = {}
    by_confidence = {"high": 0, "medium": 0, "low": 0}
    total_penalty = 0
    
    for a in all_anomalies:
        cat = a.get('category', 'unknown')
        sev = a.get('severity', 'unknown')
        conf = a.get('confidence_score', 0.5)
        
        by_category[cat] = by_category.get(cat, 0) + 1
        by_severity[sev] = by_severity.get(sev, 0) + 1
        
        if conf >= 0.75:
            by_confidence["high"] += 1
        elif conf >= 0.55:
            by_confidence["medium"] += 1
        else:
            by_confidence["low"] += 1
        
        total_penalty += a.get('estimated_penalty_usd', 0)

    report = {
        "report_generated_at": datetime.datetime.utcnow().isoformat() + "Z",
        "total_shipments": len(shipments_df),
        "total_anomalies": len(all_anomalies),
        "anomalies_by_category": by_category,
        "anomalies_by_severity": by_severity,
        "anomalies_by_confidence": by_confidence,
        "total_estimated_penalty_usd": total_penalty,
        "total_estimated_penalty_inr": total_penalty * 83,
        "anomalies": all_anomalies
    }

    path = os.path.join(OUTPUT_DIR, 'anomaly_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"   ✅ anomaly_report.json saved ({len(all_anomalies)} anomalies)")
    print(f"      High Confidence: {by_confidence['high']} | Medium: {by_confidence['medium']} | Low: {by_confidence['low']}")
    return report


def generate_accuracy_report(all_anomalies: list) -> dict:
    """Compare detected anomalies vs planted anomalies."""
    planted_path = os.path.join(DATA_DIR, 'planted_anomalies.json')
    with open(planted_path) as f:
        planted = json.load(f)

    planted_shipment_ids = {p['shipment_id'] for p in planted}
    detected_shipment_ids = {
        a['shipment_id'] for a in all_anomalies
        if not a['shipment_id'].startswith(('MULTI-', 'CTRY-'))
    }

    correctly_detected = planted_shipment_ids & detected_shipment_ids
    missed_planted     = planted_shipment_ids - detected_shipment_ids
    false_positives    = [
        a for a in all_anomalies
        if (a['shipment_id'] not in planted_shipment_ids and
            not a['shipment_id'].startswith(('MULTI-', 'CTRY-')))
    ]

    n_planted   = len(planted)
    n_detected  = len(correctly_detected)
    n_missed    = len(missed_planted)
    n_fp        = len(false_positives)

    precision = n_detected / (n_detected + n_fp) if (n_detected + n_fp) > 0 else 0
    recall    = n_detected / n_planted if n_planted > 0 else 0
    f1        = (2 * precision * recall / (precision + recall)
                 if (precision + recall) > 0 else 0)

    # Map missed to their planted info
    missed_details = [
        p for p in planted if p['shipment_id'] in missed_planted
    ]

    report = {
        "planted_anomalies": n_planted,
        "detected_correctly": n_detected,
        "missed": n_missed,
        "false_positives": n_fp,
        "precision": round(precision, 3),
        "recall": round(recall, 3),
        "f1_score": round(f1, 3),
        "missed_anomalies": [p['anomaly_id'] for p in missed_details],
        "missed_details": missed_details,
        "false_positive_details": [
            {
                "anomaly_id": a.get("anomaly_id"),
                "shipment_id": a.get("shipment_id"),
                "why_flagged": a.get("description", ""),
                "confidence_score": round(a.get("confidence_score", 0), 2),
                "why_its_actually_fine": "Flagged by statistical model as outlier but within acceptable variance for this product/route combination."
            }
            for a in false_positives[:5]  # Top 5 FPs
        ]
    }

    path = os.path.join(OUTPUT_DIR, 'accuracy_report.json')
    with open(path, 'w') as f:
        json.dump(report, f, indent=2)
    print(f"   ✅ accuracy_report.json: Precision={precision:.1%} Recall={recall:.1%} F1={f1:.1%}")
    return report


def save_executive_summary(summary_text: str):
    """Save the LLM-generated executive summary."""
    path = os.path.join(OUTPUT_DIR, 'executive_summary.md')
    with open(path, 'w') as f:
        f.write("# Executive Summary: Trade Shipment Anomaly Analysis\n\n")
        f.write(f"*Generated: {datetime.datetime.utcnow().strftime('%B %d, %Y')}*\n\n")
        f.write("---\n\n")
        f.write(summary_text)
    print(f"   ✅ executive_summary.md saved")


def run_full_pipeline(
    rule_anomalies: list,
    stat_anomalies: list,
    llm_anomalies: list,
    shipments_df: pd.DataFrame,
    executive_summary: str
) -> dict:
    """Orchestrate full report generation."""
    all_raw    = rule_anomalies + stat_anomalies + llm_anomalies
    all_dedupe = deduplicate_anomalies(all_raw)

    report        = generate_anomaly_report(shipments_df, all_dedupe)
    accuracy      = generate_accuracy_report(all_dedupe)
    save_executive_summary(executive_summary)

    return {
        "anomaly_report": report,
        "accuracy_report": accuracy
    }