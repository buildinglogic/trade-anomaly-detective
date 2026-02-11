"""
statistical_detector.py
Layer 2: Statistical anomaly detection using Z-scores.
Choice rationale in DESIGN_DECISIONS.md.
"""

import pandas as pd
import numpy as np
import os

DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')


def zscore(series: pd.Series) -> pd.Series:
    """Return Z-scores. Safe against zero std-dev."""
    std = series.std()
    if std == 0 or pd.isna(std):
        return pd.Series(np.zeros(len(series)), index=series.index)
    return (series - series.mean()) / std


def run_statistical_checks(
    shipments_df: pd.DataFrame,
    products_df: pd.DataFrame,
    routes_df: pd.DataFrame,
    buyers_df: pd.DataFrame,
    z_threshold: float = 2.5
) -> list:
    """
    Run statistical anomaly detection.
    Returns list of anomaly dicts.
    """
    anomalies = []
    counter = [0]

    def make_anomaly(shipment_id, category, sub_type, description,
                     evidence, severity, recommendation, estimated_penalty_usd=0):
        counter[0] += 1
        return {
            "anomaly_id": f"STAT-{counter[0]:03d}",
            "layer": "statistical",
            "shipment_id": shipment_id,
            "category": category,
            "sub_type": sub_type,
            "description": description,
            "evidence": evidence,
            "severity": severity,
            "recommendation": recommendation,
            "estimated_penalty_usd": estimated_penalty_usd,
            "detection_method": f"Z-score (threshold={z_threshold})"
        }

    df = shipments_df.copy()

    # ── STAT-1: Price outliers per product ───────────────────────────────
    prod_lookup = {
        r['product_description']: r
        for r in products_df.to_dict('records')
    }
    for prod_desc, group in df.groupby('product_description'):
        if len(group) < 3:
            continue
        prod_info = prod_lookup.get(prod_desc, {})
        zscores = zscore(group['unit_price_usd'])
        outliers = group[zscores.abs() > z_threshold]
        for _, row in outliers.iterrows():
            z = zscores[row.name]
            direction = "HIGH" if z > 0 else "LOW"
            anomalies.append(make_anomaly(
                shipment_id=row['shipment_id'],
                category="pricing",
                sub_type="price_outlier",
                description=(
                    f"{prod_desc}: unit_price ${row['unit_price_usd']:.2f} is "
                    f"{abs(z):.1f}σ {direction} from mean ${group['unit_price_usd'].mean():.2f}."
                ),
                evidence={
                    "unit_price": float(row['unit_price_usd']),
                    "product_mean": round(float(group['unit_price_usd'].mean()), 2),
                    "product_std": round(float(group['unit_price_usd'].std()), 2),
                    "z_score": round(float(z), 2),
                    "catalog_range": f"${prod_info.get('price_range_min','?')} - ${prod_info.get('price_range_max','?')}",
                    "buyer": row['buyer_name']
                },
                severity="critical" if abs(z) > 4 else "high",
                recommendation=f"Review pricing with {row['buyer_name']}. Check for under/over-invoicing.",
                estimated_penalty_usd=8000 if direction == "LOW" else 2000
            ))

    # ── STAT-2: Transit time outliers per route ──────────────────────────
    route_groups = df.groupby(['port_of_loading', 'port_of_discharge'])
    for (pol, pod), group in route_groups:
        if len(group) < 3:
            continue
        zscores = zscore(group['transit_days'])
        outliers = group[zscores.abs() > z_threshold]
        for _, row in outliers.iterrows():
            z = zscores[row.name]
            anomalies.append(make_anomaly(
                shipment_id=row['shipment_id'],
                category="route_logistics",
                sub_type="transit_days_outlier",
                description=(
                    f"Route {pol}→{pod}: transit {row['transit_days']} days is "
                    f"{abs(z):.1f}σ from route mean {group['transit_days'].mean():.0f} days."
                ),
                evidence={
                    "transit_days": int(row['transit_days']),
                    "route_mean": round(float(group['transit_days'].mean()), 1),
                    "route_std": round(float(group['transit_days'].std()), 1),
                    "z_score": round(float(z), 2),
                    "route": f"{pol} → {pod}"
                },
                severity="high" if abs(z) > 4 else "medium",
                recommendation="Contact freight forwarder. Verify vessel tracking. Check for detention/demurrage.",
                estimated_penalty_usd=3000
            ))

    # ── STAT-3: Freight cost outliers per route + container type ─────────
    for (pol, pod, ctype), group in df.groupby(
        ['port_of_loading', 'port_of_discharge', 'container_type']
    ):
        if len(group) < 3:
            continue
        valid = group[group['freight_cost_usd'] > 0]
        if len(valid) < 3:
            continue
        zscores = zscore(valid['freight_cost_usd'])
        outliers = valid[zscores.abs() > z_threshold]
        for _, row in outliers.iterrows():
            z = zscores[row.name]
            direction = "HIGH" if z > 0 else "LOW"
            anomalies.append(make_anomaly(
                shipment_id=row['shipment_id'],
                category="route_logistics",
                sub_type="freight_cost_outlier",
                description=(
                    f"Freight cost ${row['freight_cost_usd']:,.0f} for {pol}→{pod} ({ctype}) "
                    f"is {abs(z):.1f}σ {direction} from route avg ${valid['freight_cost_usd'].mean():,.0f}."
                ),
                evidence={
                    "freight_cost": float(row['freight_cost_usd']),
                    "route_avg_freight": round(float(valid['freight_cost_usd'].mean()), 2),
                    "route_std": round(float(valid['freight_cost_usd'].std()), 2),
                    "z_score": round(float(z), 2),
                    "route": f"{pol} → {pod}",
                    "container_type": ctype
                },
                severity="high" if z > 3 else "medium",
                recommendation="Verify with freight forwarder. Get 2 competitive quotes. Check for kickback arrangements.",
                estimated_penalty_usd=5000 if direction == "HIGH" else 0
            ))

    # ── STAT-4: Payment behavior change per buyer ────────────────────────
    buyers_lookup = {
        b['buyer_name']: b for b in buyers_df.to_dict('records')
    }
    paid_df = df[df['days_to_payment'].notna()].copy()
    paid_df['days_to_payment'] = paid_df['days_to_payment'].astype(float)

    for buyer, group in paid_df.groupby('buyer_name'):
        if len(group) < 3:
            continue
        buyer_info = buyers_lookup.get(buyer, {})
        historical_avg = float(buyer_info.get('avg_payment_days', group['days_to_payment'].mean()))
        zscores = zscore(group['days_to_payment'])
        outliers = group[zscores.abs() > z_threshold]
        for _, row in outliers.iterrows():
            z = zscores[row.name]
            if z > 0:  # Only flag when payment is SLOWER
                anomalies.append(make_anomaly(
                    shipment_id=row['shipment_id'],
                    category="payment",
                    sub_type="payment_delay",
                    description=(
                        f"{buyer} paid in {row['days_to_payment']:.0f} days — "
                        f"{abs(z):.1f}σ above their avg of {historical_avg:.0f} days."
                    ),
                    evidence={
                        "days_to_payment": float(row['days_to_payment']),
                        "buyer_historical_avg": historical_avg,
                        "z_score": round(float(z), 2),
                        "buyer": buyer,
                        "credit_rating": buyer_info.get('credit_rating', 'N/A')
                    },
                    severity="high" if z > 3.5 else "medium",
                    recommendation=f"Flag {buyer} for credit review. Consider LC instead of Open Account.",
                    estimated_penalty_usd=2000
                ))

    # ── STAT-5: Volume spikes per buyer ──────────────────────────────────
    buyer_qty = df.groupby('buyer_name')['total_fob_usd'].sum().reset_index()
    # Month-level check
    df['year_month'] = pd.to_datetime(df['date']).dt.to_period('M')
    buyer_monthly = df.groupby(['buyer_name', 'year_month'])['total_fob_usd'].sum().reset_index()

    for buyer, group in buyer_monthly.groupby('buyer_name'):
        if len(group) < 3:
            continue
        zscores = zscore(group['total_fob_usd'])
        outliers = group[zscores.abs() > z_threshold]
        for _, row in outliers.iterrows():
            z = zscores[row.name]
            if z > 0:
                anomalies.append(make_anomaly(
                    shipment_id=f"MULTI-{buyer[:10]}",
                    category="volume",
                    sub_type="buyer_volume_spike",
                    description=(
                        f"{buyer} in {str(row['year_month'])}: ${row['total_fob_usd']:,.0f} FOB — "
                        f"{abs(z):.1f}σ above their monthly average."
                    ),
                    evidence={
                        "buyer": buyer,
                        "month": str(row['year_month']),
                        "month_fob": float(row['total_fob_usd']),
                        "buyer_avg_monthly": round(float(group['total_fob_usd'].mean()), 2),
                        "z_score": round(float(z), 2)
                    },
                    severity="critical" if z > 4 else "high",
                    recommendation=f"Request end-user certificate from {buyer}. Verify business justification.",
                    estimated_penalty_usd=10000
                ))

    # ── STAT-6: Country monthly volume spike ─────────────────────────────
    country_monthly = df.groupby(
        ['buyer_country', 'year_month']
    )['total_fob_usd'].sum().reset_index()

    for country, group in country_monthly.groupby('buyer_country'):
        if len(group) < 3:
            continue
        zscores = zscore(group['total_fob_usd'])
        outliers = group[zscores.abs() > z_threshold]
        for _, row in outliers.iterrows():
            z = zscores[row.name]
            if z > 0:
                anomalies.append(make_anomaly(
                    shipment_id=f"CTRY-{country[:5]}-{str(row['year_month'])}",
                    category="volume",
                    sub_type="country_volume_spike",
                    description=(
                        f"Exports to {country} in {str(row['year_month'])}: "
                        f"${row['total_fob_usd']:,.0f} — {abs(z):.1f}σ above monthly average."
                    ),
                    evidence={
                        "country": country,
                        "month": str(row['year_month']),
                        "month_fob": float(row['total_fob_usd']),
                        "country_avg_monthly": round(float(group['total_fob_usd'].mean()), 2),
                        "z_score": round(float(z), 2)
                    },
                    severity="high",
                    recommendation=f"Review all {country} shipments in this month. Check for re-export patterns.",
                    estimated_penalty_usd=5000
                ))

    print(f"   Layer 2 (Statistical): {len(anomalies)} anomalies found")
    return anomalies


if __name__ == "__main__":
    df = pd.read_csv(os.path.join(DATA_DIR, 'shipments.csv'))
    products_df = pd.read_csv(os.path.join(DATA_DIR, 'product_catalog.csv'))
    routes_df   = pd.read_csv(os.path.join(DATA_DIR, 'routes.csv'))
    buyers_df   = pd.read_csv(os.path.join(DATA_DIR, 'buyers.csv'))
    results = run_statistical_checks(df, products_df, routes_df, buyers_df)
    print(f"\nStatistical check complete. Found {len(results)} anomalies.")
    for a in results:
        print(f"  [{a['severity'].upper()}] {a['shipment_id']}: {a['description'][:80]}")
