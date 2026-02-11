"""
rule_engine.py
Layer 1: Hard-coded rule-based anomaly detection.
No ML, no LLM — just math and logic checks.
"""

import pandas as pd
import numpy as np
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def run_rule_checks(shipments_df: pd.DataFrame) -> list:
    """
    Run all rule-based checks on shipments DataFrame.
    Returns list of anomaly dicts.
    """
    anomalies = []
    counter = [0]

    def make_anomaly(shipment_id, category, sub_type, description,
                     evidence, severity, recommendation, estimated_penalty_usd=0):
        counter[0] += 1
        return {
            "anomaly_id": f"RULE-{counter[0]:03d}",
            "layer": "rule_based",
            "shipment_id": shipment_id,
            "category": category,
            "sub_type": sub_type,
            "description": description,
            "evidence": evidence,
            "severity": severity,
            "recommendation": recommendation,
            "estimated_penalty_usd": estimated_penalty_usd,
            "detection_method": "Rule-based arithmetic/logic check"
        }

    df = shipments_df.copy()

    # ── CHECK 1: FOB ≠ qty × unit_price ─────────────────────────────────
    df['expected_fob'] = (df['quantity'] * df['unit_price_usd']).round(2)
    df['fob_diff'] = (df['total_fob_usd'] - df['expected_fob']).abs()
    fob_errors = df[df['fob_diff'] > 0.05]  # allow $0.05 rounding tolerance
    for _, row in fob_errors.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="pricing",
            sub_type="fob_math_error",
            description=f"FOB mismatch: reported ${row['total_fob_usd']:,.2f} ≠ calculated ${row['expected_fob']:,.2f}",
            evidence={
                "reported_fob": float(row['total_fob_usd']),
                "quantity": int(row['quantity']),
                "unit_price": float(row['unit_price_usd']),
                "calculated_fob": float(row['expected_fob']),
                "difference": float(row['fob_diff'])
            },
            severity="critical",
            recommendation="Verify invoice with buyer. Correct FOB before drawback claim submission.",
            estimated_penalty_usd=5000
        ))

    # ── CHECK 2: Drawback claimed on rejected shipment ────────────────────
    drawback_on_rejected = df[
        (df['customs_status'] == 'rejected') &
        (df['drawback_amount_usd'] > 0)
    ]
    for _, row in drawback_on_rejected.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="compliance",
            sub_type="drawback_on_rejected",
            description=f"Drawback of ${row['drawback_amount_usd']:,.2f} claimed but customs_status is REJECTED.",
            evidence={
                "customs_status": row['customs_status'],
                "drawback_amount": float(row['drawback_amount_usd']),
                "drawback_rate_pct": float(row['drawback_rate_pct'])
            },
            severity="critical",
            recommendation="Reverse drawback claim immediately. File amendment with DGFT.",
            estimated_penalty_usd=int(row['drawback_amount_usd'] * 3)
        ))

    # ── CHECK 3: Payment received but days_to_payment is null ────────────
    received_null = df[
        (df['payment_status'] == 'received') &
        (df['days_to_payment'].isnull())
    ]
    for _, row in received_null.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="payment",
            sub_type="received_null_days",
            description="Payment status = 'received' but days_to_payment is NULL. Contradictory record.",
            evidence={
                "payment_status": row['payment_status'],
                "days_to_payment": None,
                "buyer": row['buyer_name']
            },
            severity="medium",
            recommendation="Investigate with accounts team. Update payment date in ERP.",
            estimated_penalty_usd=500
        ))

    # ── CHECK 4: CIF incoterm but freight = 0 ────────────────────────────
    cif_no_freight = df[
        (df['incoterm'] == 'CIF') &
        (df['freight_cost_usd'] == 0)
    ]
    for _, row in cif_no_freight.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="cross_field",
            sub_type="cif_zero_freight",
            description="Incoterm is CIF (seller pays freight+insurance) but freight_cost_usd = 0.",
            evidence={
                "incoterm": row['incoterm'],
                "freight_cost_usd": float(row['freight_cost_usd']),
                "total_fob": float(row['total_fob_usd'])
            },
            severity="high",
            recommendation="Check if freight was invoiced separately. Update freight_cost_usd or change incoterm.",
            estimated_penalty_usd=2500
        ))

    # ── CHECK 5: Insurance rate anomaly (should be ~0.1%-0.4% of FOB) ────
    df_valid_fob = df[df['total_fob_usd'] > 0].copy()
    df_valid_fob['insurance_rate'] = (
        df_valid_fob['insurance_usd'] / df_valid_fob['total_fob_usd'] * 100
    )
    # Flag if insurance rate > 0.8% or < 0.05%
    insurance_anomalies = df_valid_fob[
        (df_valid_fob['insurance_rate'] > 0.8) |
        ((df_valid_fob['insurance_rate'] < 0.05) & (df_valid_fob['insurance_usd'] > 0))
    ]
    for _, row in insurance_anomalies.iterrows():
        direction = "OVERCHARGED" if row['insurance_rate'] > 0.8 else "SUSPICIOUSLY LOW"
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="cross_field",
            sub_type="insurance_rate_error",
            description=f"Insurance rate = {row['insurance_rate']:.3f}% of FOB. Normal is 0.1-0.4%. {direction}.",
            evidence={
                "insurance_usd": float(row['insurance_usd']),
                "total_fob_usd": float(row['total_fob_usd']),
                "calculated_rate_pct": round(float(row['insurance_rate']), 4),
                "expected_range": "0.1% - 0.4%"
            },
            severity="medium" if direction == "OVERCHARGED" else "low",
            recommendation="Verify insurance policy. Standard marine cargo insurance = 0.1-0.3% of CIF value.",
            estimated_penalty_usd=500
        ))

    print(f"   Layer 1 (Rule-based): {len(anomalies)} anomalies found")
    return anomalies


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'shipments.csv'))
    results = run_rule_checks(df)
    print(f"\nRule-based check complete. Found {len(results)} anomalies.")
    for a in results:
        print(f"  [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:80]}")"""
rule_engine.py
Layer 1: Hard-coded rule-based anomaly detection.
No ML, no LLM — just math and logic checks.
"""

import pandas as pd
import numpy as np
import os
import json

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def run_rule_checks(shipments_df: pd.DataFrame) -> list:
    """
    Run all rule-based checks on shipments DataFrame.
    Returns list of anomaly dicts.
    """
    anomalies = []
    counter = [0]

    def make_anomaly(shipment_id, category, sub_type, description,
                     evidence, severity, recommendation, estimated_penalty_usd=0):
        counter[0] += 1
        return {
            "anomaly_id": f"RULE-{counter[0]:03d}",
            "layer": "rule_based",
            "shipment_id": shipment_id,
            "category": category,
            "sub_type": sub_type,
            "description": description,
            "evidence": evidence,
            "severity": severity,
            "recommendation": recommendation,
            "estimated_penalty_usd": estimated_penalty_usd,
            "detection_method": "Rule-based arithmetic/logic check"
        }

    df = shipments_df.copy()

    # ── CHECK 1: FOB ≠ qty × unit_price ─────────────────────────────────
    df['expected_fob'] = (df['quantity'] * df['unit_price_usd']).round(2)
    df['fob_diff'] = (df['total_fob_usd'] - df['expected_fob']).abs()
    fob_errors = df[df['fob_diff'] > 0.05]  # allow $0.05 rounding tolerance
    for _, row in fob_errors.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="pricing",
            sub_type="fob_math_error",
            description=f"FOB mismatch: reported ${row['total_fob_usd']:,.2f} ≠ calculated ${row['expected_fob']:,.2f}",
            evidence={
                "reported_fob": float(row['total_fob_usd']),
                "quantity": int(row['quantity']),
                "unit_price": float(row['unit_price_usd']),
                "calculated_fob": float(row['expected_fob']),
                "difference": float(row['fob_diff'])
            },
            severity="critical",
            recommendation="Verify invoice with buyer. Correct FOB before drawback claim submission.",
            estimated_penalty_usd=5000
        ))

    # ── CHECK 2: Drawback claimed on rejected shipment ────────────────────
    drawback_on_rejected = df[
        (df['customs_status'] == 'rejected') &
        (df['drawback_amount_usd'] > 0)
    ]
    for _, row in drawback_on_rejected.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="compliance",
            sub_type="drawback_on_rejected",
            description=f"Drawback of ${row['drawback_amount_usd']:,.2f} claimed but customs_status is REJECTED.",
            evidence={
                "customs_status": row['customs_status'],
                "drawback_amount": float(row['drawback_amount_usd']),
                "drawback_rate_pct": float(row['drawback_rate_pct'])
            },
            severity="critical",
            recommendation="Reverse drawback claim immediately. File amendment with DGFT.",
            estimated_penalty_usd=int(row['drawback_amount_usd'] * 3)
        ))

    # ── CHECK 3: Payment received but days_to_payment is null ────────────
    received_null = df[
        (df['payment_status'] == 'received') &
        (df['days_to_payment'].isnull())
    ]
    for _, row in received_null.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="payment",
            sub_type="received_null_days",
            description="Payment status = 'received' but days_to_payment is NULL. Contradictory record.",
            evidence={
                "payment_status": row['payment_status'],
                "days_to_payment": None,
                "buyer": row['buyer_name']
            },
            severity="medium",
            recommendation="Investigate with accounts team. Update payment date in ERP.",
            estimated_penalty_usd=500
        ))

    # ── CHECK 4: CIF incoterm but freight = 0 ────────────────────────────
    cif_no_freight = df[
        (df['incoterm'] == 'CIF') &
        (df['freight_cost_usd'] == 0)
    ]
    for _, row in cif_no_freight.iterrows():
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="cross_field",
            sub_type="cif_zero_freight",
            description="Incoterm is CIF (seller pays freight+insurance) but freight_cost_usd = 0.",
            evidence={
                "incoterm": row['incoterm'],
                "freight_cost_usd": float(row['freight_cost_usd']),
                "total_fob": float(row['total_fob_usd'])
            },
            severity="high",
            recommendation="Check if freight was invoiced separately. Update freight_cost_usd or change incoterm.",
            estimated_penalty_usd=2500
        ))

    # ── CHECK 5: Insurance rate anomaly (should be ~0.1%-0.4% of FOB) ────
    df_valid_fob = df[df['total_fob_usd'] > 0].copy()
    df_valid_fob['insurance_rate'] = (
        df_valid_fob['insurance_usd'] / df_valid_fob['total_fob_usd'] * 100
    )
    # Flag if insurance rate > 0.8% or < 0.05%
    insurance_anomalies = df_valid_fob[
        (df_valid_fob['insurance_rate'] > 0.8) |
        ((df_valid_fob['insurance_rate'] < 0.05) & (df_valid_fob['insurance_usd'] > 0))
    ]
    for _, row in insurance_anomalies.iterrows():
        direction = "OVERCHARGED" if row['insurance_rate'] > 0.8 else "SUSPICIOUSLY LOW"
        anomalies.append(make_anomaly(
            shipment_id=row['shipment_id'],
            category="cross_field",
            sub_type="insurance_rate_error",
            description=f"Insurance rate = {row['insurance_rate']:.3f}% of FOB. Normal is 0.1-0.4%. {direction}.",
            evidence={
                "insurance_usd": float(row['insurance_usd']),
                "total_fob_usd": float(row['total_fob_usd']),
                "calculated_rate_pct": round(float(row['insurance_rate']), 4),
                "expected_range": "0.1% - 0.4%"
            },
            severity="medium" if direction == "OVERCHARGED" else "low",
            recommendation="Verify insurance policy. Standard marine cargo insurance = 0.1-0.3% of CIF value.",
            estimated_penalty_usd=500
        ))

    print(f"   Layer 1 (Rule-based): {len(anomalies)} anomalies found")
    return anomalies


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'shipments.csv'))
    results = run_rule_checks(df)
    print(f"\nRule-based check complete. Found {len(results)} anomalies.")
    for a in results:
        print(f"  [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:80]}")
