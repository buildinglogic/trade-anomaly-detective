"""
statistical_detector.py
Layer 2: Statistical anomaly detection using Z-scores + context-aware filtering.
Improved to reduce false positives while maintaining high recall.
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


def detect_payment_trend(df: pd.DataFrame, buyers_df: pd.DataFrame, window: int = 3) -> list:
    """
    IMPROVEMENT #2: Detect if a buyer's payment behavior is DETERIORATING over time.
    Only flag if last N shipments show consistent slowdown.
    This catches real patterns, not single anomalies.
    """
    anomalies = []
    counter = [0]
    
    buyers_lookup = {b['buyer_name']: b for b in buyers_df.to_dict('records')}
    
    paid_df = df[df['days_to_payment'].notna()].copy()
    paid_df['date'] = pd.to_datetime(paid_df['date'])
    paid_df = paid_df.sort_values('date')
    
    for buyer, group in paid_df.groupby('buyer_name'):
        if len(group) < window:
            continue
            
        buyer_info = buyers_lookup.get(buyer, {})
        historical_avg = float(buyer_info.get('avg_payment_days', group['days_to_payment'].mean()))
        
        # Get last N shipments
        recent = group.tail(window)
        recent_avg = recent['days_to_payment'].mean()
        
        # Calculate trend: is it getting worse?
        older = group.iloc[:-window]
        older_avg = older['days_to_payment'].mean() if len(older) > 0 else historical_avg
        
        slowdown = recent_avg - older_avg
        slowdown_pct = (slowdown / historical_avg) * 100 if historical_avg > 0 else 0
        
        # FLAG ONLY IF:
        # 1. Recent average is consistently 2+ weeks above historical
        # 2. Trend is getting worse (recent > older)
        # 3. Slowdown is significant (>40%)
        if slowdown >= 14 and slowdown_pct > 40 and recent_avg > historical_avg * 1.5:
            counter[0] += 1
            anomalies.append({
                "anomaly_id": f"TREND-{counter[0]:03d}",
                "layer": "statistical",
                "shipment_id": f"MULTI-{buyer[:10]}",
                "category": "payment",
                "sub_type": "payment_deterioration_trend",
                "description": (
                    f"{buyer}: Payment behavior DETERIORATING. "
                    f"Last {window} shipments avg {recent_avg:.0f} days vs "
                    f"historical {historical_avg:.0f} days (+{slowdown_pct:.0f}% slower)."
                ),
                "evidence": {
                    "buyer": buyer,
                    "historical_avg_days": historical_avg,
                    "recent_avg_days": round(recent_avg, 1),
                    "trend_slowdown_days": round(slowdown, 1),
                    "trend_slowdown_pct": round(slowdown_pct, 1),
                    "window_shipments": window,
                    "credit_rating": buyer_info.get('credit_rating', 'N/A')
                },
                "severity": "critical" if buyer_info.get('credit_rating') == 'C' else "high",
                "recommendation": (
                    f"URGENT: {buyer} showing deteriorating payment pattern. "
                    f"Recommend: (1) Reduce credit terms to LC instead of Open Account, "
                    f"(2) Reduce shipment sizes, (3) Request payment guarantee."
                ),
                "estimated_penalty_usd": 5000,
                "detection_method": "Trend analysis: recent payment pattern vs historical"
            })
    
    return anomalies


def detect_volume_anomalies_smart(df: pd.DataFrame, products_df: pd.DataFrame) -> list:
    """
    IMPROVEMENT #3: Volume spike detection that considers:
    1. Single shipment unusually large (outlier within buyer history)
    2. Buyer's typical order size
    3. Product type (rice to Gulf = more suspicious than textiles)
    """
    anomalies = []
    counter = [0]
    
    # Mark products by "suspicious export risk"
    suspicious_products = {
        'Basmati Rice Premium Grade': 3,        # Food to certain countries = re-export risk
        'Pharmaceutical Tablets Generic': 4,    # Highest risk (restricted countries)
        'Polypropylene Granules Industrial': 2  # Chemicals, moderate risk
    }
    
    for _, row in df.iterrows():
        product = row['product_description']
        buyer_fob = row['total_fob_usd']
        buyer_country = row['buyer_country']
        quantity = row['quantity']
        
        # Get buyer's typical order value
        buyer_orders = df[df['buyer_name'] == row['buyer_name']]['total_fob_usd']
        buyer_avg = buyer_orders.mean()
        buyer_max = buyer_orders.quantile(0.95)  # 95th percentile (normal max)
        
        # Calculate spike ratio
        spike_ratio = buyer_fob / buyer_avg if buyer_avg > 0 else 1
        risk_factor = suspicious_products.get(product, 1)
        
        # FLAG IF:
        # 1. Order is >8x buyer's average AND >$100K
        # 2. OR: High-risk product AND >5x average
        if (spike_ratio > 8 and buyer_fob > 100000) or (risk_factor >= 3 and spike_ratio > 5):
            counter[0] += 1
            anomalies.append({
                "anomaly_id": f"VOL-{counter[0]:03d}",
                "layer": "statistical",
                "shipment_id": row['shipment_id'],
                "category": "volume",
                "sub_type": "suspicious_large_order",
                "description": (
                    f"{row['buyer_name']} ({buyer_country}): "
                    f"Order of ${buyer_fob:,.0f} ({quantity:,} units) is {spike_ratio:.1f}x their average. "
                    f"Product: {product} (risk=HIGH)."
                ),
                "evidence": {
                    "buyer": row['buyer_name'],
                    "buyer_country": buyer_country,
                    "order_value": float(buyer_fob),
                    "buyer_typical_avg": round(buyer_avg, 0),
                    "buyer_typical_max": round(buyer_max, 0),
                    "spike_ratio": round(spike_ratio, 1),
                    "quantity": int(quantity),
                    "product": product,
                    "risk_factor": risk_factor
                },
                "severity": "critical",
                "recommendation": (
                    f"URGENT: Request end-use certificate from {row['buyer_name']}. "
                    f"Verify final destination (not re-export). "
                    f"Consider: (1) Reducing order size, (2) Requesting advance payment, "
                    f"(3) Checking for sanctions on buyer."
                ),
                "estimated_penalty_usd": 20000,
                "detection_method": "Volume spike + product risk analysis"
            })
    
    return anomalies


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

    # ── STAT-4: Payment behavior change per buyer (IMPROVED #1) ──────────
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
            credit_rating = buyer_info.get('credit_rating', 'B')
            
            # IMPROVEMENT #1: Higher threshold for A-rated buyers (lower risk)
            threshold = 4.5 if credit_rating == 'A' else 3.5 if credit_rating == 'B' else 2.5
            
            # Only flag if BOTH z-score is high AND payment is significantly late
            days_above_avg = row['days_to_payment'] - historical_avg
            
            if z > threshold and days_above_avg > 30:  # 30-day buffer
                anomalies.append(make_anomaly(
                    shipment_id=row['shipment_id'],
                    category="payment",
                    sub_type="payment_delay",
                    description=(
                        f"{buyer} (Rating: {credit_rating}) paid in {row['days_to_payment']:.0f} days — "
                        f"{days_above_avg:.0f} days above their avg of {historical_avg:.0f}. "
                        f"Pattern suggests working capital stress."
                    ),
                    evidence={
                        "days_to_payment": float(row['days_to_payment']),
                        "buyer_historical_avg": historical_avg,
                        "days_above_avg": days_above_avg,
                        "z_score": round(float(z), 2),
                        "buyer": buyer,
                        "credit_rating": credit_rating,
                        "threshold_applied": threshold
                    },
                    severity="high" if credit_rating in ['B', 'C'] else "medium",
                    recommendation=f"Flag {buyer} (Rating {credit_rating}) for credit review. Monitor next 2-3 shipments for trend.",
                    estimated_penalty_usd=2000 if credit_rating == 'A' else 3000
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
    
    # ── ADD IMPROVEMENTS #2 and #3 ───────────────────────────────────────
    trend_anomalies = detect_payment_trend(df, buyers_df, window=3)
    anomalies.extend(trend_anomalies)
    print(f"   Layer 2 (Trend Analysis): {len(trend_anomalies)} trend anomalies found")
    
    volume_anomalies = detect_volume_anomalies_smart(df, products_df)
    anomalies.extend(volume_anomalies)
    print(f"   Layer 2 (Smart Volume): {len(volume_anomalies)} volume anomalies found")
    
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