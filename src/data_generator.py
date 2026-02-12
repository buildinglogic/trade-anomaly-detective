"""
data_generator.py
Generates synthetic Indian export trade data with 12 planted anomalies.
Run: python src/data_generator.py
"""

import pandas as pd
import numpy as np
import json
import os
import random
from datetime import datetime, timedelta

# ─── Seed for reproducibility ───────────────────────────────────────────────
random.seed(42)
np.random.seed(42)

# ─── Output directory ───────────────────────────────────────────────────────
DATA_DIR = os.path.join(os.path.dirname(__file__), '..', 'data')
os.makedirs(DATA_DIR, exist_ok=True)


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 3 — PRODUCT CATALOG (12 products)
# ═══════════════════════════════════════════════════════════════════════════
def generate_product_catalog():
    products = [
        {
            "product_id": "PROD-001",
            "product_description": "Cotton T-shirts 100% knitted",
            "hs_code": "61091000",
            "hs_chapter": "61",
            "category": "Textiles",
            "avg_unit_price_usd": 4.50,
            "price_range_min": 3.00,
            "price_range_max": 6.50,
            "drawback_rate_pct": 2.0,
            "weight_per_unit_kg": 0.25,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-002",
            "product_description": "Basmati Rice Premium Grade",
            "hs_code": "10063020",
            "hs_chapter": "10",
            "category": "Food Products",
            "avg_unit_price_usd": 1.20,
            "price_range_min": 0.90,
            "price_range_max": 1.80,
            "drawback_rate_pct": 1.0,
            "weight_per_unit_kg": 1.0,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-003",
            "product_description": "Automobile Brake Pads",
            "hs_code": "87083010",
            "hs_chapter": "87",
            "category": "Auto Parts",
            "avg_unit_price_usd": 18.00,
            "price_range_min": 12.00,
            "price_range_max": 28.00,
            "drawback_rate_pct": 3.0,
            "weight_per_unit_kg": 0.80,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-004",
            "product_description": "Pharmaceutical Tablets Generic",
            "hs_code": "30049099",
            "hs_chapter": "30",
            "category": "Chemicals",
            "avg_unit_price_usd": 0.08,
            "price_range_min": 0.05,
            "price_range_max": 0.15,
            "drawback_rate_pct": 2.5,
            "weight_per_unit_kg": 0.001,
            "restricted_countries": "North Korea,Iran"
        },
        {
            "product_id": "PROD-005",
            "product_description": "Handcrafted Brass Figurines",
            "hs_code": "83062910",
            "hs_chapter": "83",
            "category": "Handicrafts",
            "avg_unit_price_usd": 12.00,
            "price_range_min": 8.00,
            "price_range_max": 18.00,
            "drawback_rate_pct": 1.5,
            "weight_per_unit_kg": 0.60,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-006",
            "product_description": "Industrial Centrifugal Pump",
            "hs_code": "84137000",
            "hs_chapter": "84",
            "category": "Machinery",
            "avg_unit_price_usd": 450.00,
            "price_range_min": 320.00,
            "price_range_max": 650.00,
            "drawback_rate_pct": 3.5,
            "weight_per_unit_kg": 45.0,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-007",
            "product_description": "Leather Wallet Genuine Cow Hide",
            "hs_code": "42021200",
            "hs_chapter": "42",
            "category": "Leather",
            "avg_unit_price_usd": 8.50,
            "price_range_min": 6.00,
            "price_range_max": 12.00,
            "drawback_rate_pct": 1.8,
            "weight_per_unit_kg": 0.15,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-008",
            "product_description": "Black Pepper Ground Organic",
            "hs_code": "09041100",
            "hs_chapter": "09",
            "category": "Food Products",
            "avg_unit_price_usd": 6.00,
            "price_range_min": 4.50,
            "price_range_max": 8.50,
            "drawback_rate_pct": 1.0,
            "weight_per_unit_kg": 1.0,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-009",
            "product_description": "Stainless Steel Kitchen Utensils",
            "hs_code": "73239300",
            "hs_chapter": "73",
            "category": "Metal Products",
            "avg_unit_price_usd": 3.20,
            "price_range_min": 2.20,
            "price_range_max": 4.80,
            "drawback_rate_pct": 2.0,
            "weight_per_unit_kg": 0.35,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-010",
            "product_description": "Polypropylene Granules Industrial",
            "hs_code": "39021000",
            "hs_chapter": "39",
            "category": "Chemicals",
            "avg_unit_price_usd": 1.10,
            "price_range_min": 0.85,
            "price_range_max": 1.50,
            "drawback_rate_pct": 1.5,
            "weight_per_unit_kg": 1.0,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-011",
            "product_description": "Embroidered Saree Silk",
            "hs_code": "62114900",
            "hs_chapter": "62",
            "category": "Textiles",
            "avg_unit_price_usd": 35.00,
            "price_range_min": 22.00,
            "price_range_max": 55.00,
            "drawback_rate_pct": 2.0,
            "weight_per_unit_kg": 0.50,
            "restricted_countries": ""
        },
        {
            "product_id": "PROD-012",
            "product_description": "LED Street Light Fixtures",
            "hs_code": "94054090",
            "hs_chapter": "94",
            "category": "Machinery",
            "avg_unit_price_usd": 85.00,
            "price_range_min": 60.00,
            "price_range_max": 120.00,
            "drawback_rate_pct": 3.0,
            "weight_per_unit_kg": 4.50,
            "restricted_countries": ""
        },
    ]
    df = pd.DataFrame(products)
    df.to_csv(os.path.join(DATA_DIR, 'product_catalog.csv'), index=False)
    print(f"✅ product_catalog.csv: {len(df)} products")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 2 — BUYERS (8 buyers)
# ═══════════════════════════════════════════════════════════════════════════
def generate_buyers():
    buyers = [
        {
            "buyer_name": "Global Mart Inc",
            "buyer_country": "USA",
            "relationship_since": "2019-04-10",
            "avg_order_value_usd": 35000.00,
            "avg_payment_days": 38,
            "credit_rating": "A",
            "total_shipments_historical": 210
        },
        {
            "buyer_name": "Euro Trade GmbH",
            "buyer_country": "Germany",
            "relationship_since": "2020-07-22",
            "avg_order_value_usd": 48000.00,
            "avg_payment_days": 32,
            "credit_rating": "A",
            "total_shipments_historical": 185
        },
        {
            "buyer_name": "Gulf Distributors LLC",
            "buyer_country": "UAE",
            "relationship_since": "2018-11-05",
            "avg_order_value_usd": 22000.00,
            "avg_payment_days": 28,
            "credit_rating": "A",
            "total_shipments_historical": 298
        },
        {
            "buyer_name": "Pacific Traders Pty",
            "buyer_country": "Australia",
            "relationship_since": "2021-03-14",
            "avg_order_value_usd": 19000.00,
            "avg_payment_days": 42,
            "credit_rating": "B",
            "total_shipments_historical": 96
        },
        {
            "buyer_name": "SG Merchants Pte",
            "buyer_country": "Singapore",
            "relationship_since": "2020-01-18",
            "avg_order_value_usd": 28000.00,
            "avg_payment_days": 25,
            "credit_rating": "A",
            "total_shipments_historical": 143
        },
        {
            "buyer_name": "London Imports Ltd",
            "buyer_country": "UK",
            "relationship_since": "2022-06-30",
            "avg_order_value_usd": 31000.00,
            "avg_payment_days": 45,
            "credit_rating": "B",
            "total_shipments_historical": 78
        },
        {
            "buyer_name": "Nippon Commerce KK",
            "buyer_country": "Japan",
            "relationship_since": "2021-09-11",
            "avg_order_value_usd": 55000.00,
            "avg_payment_days": 35,
            "credit_rating": "A",
            "total_shipments_historical": 112
        },
        {
            "buyer_name": "African Goods Co",
            "buyer_country": "South Africa",
            "relationship_since": "2023-02-20",
            "avg_order_value_usd": 14000.00,
            "avg_payment_days": 52,
            "credit_rating": "C",
            "total_shipments_historical": 34
        },
    ]
    df = pd.DataFrame(buyers)
    df.to_csv(os.path.join(DATA_DIR, 'buyers.csv'), index=False)
    print(f"✅ buyers.csv: {len(df)} buyers")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 4 — ROUTES (15 routes)
# ═══════════════════════════════════════════════════════════════════════════
def generate_routes():
    routes = [
        {"port_of_loading": "INMUN1", "port_of_discharge": "USLAX",
         "avg_transit_days": 28, "transit_range_min": 24, "transit_range_max": 34,
         "avg_freight_20ft_usd": 1400, "avg_freight_40ft_usd": 2200, "avg_freight_40hc_usd": 2400},
        {"port_of_loading": "INMUN1", "port_of_discharge": "DEHAM",
         "avg_transit_days": 25, "transit_range_min": 21, "transit_range_max": 30,
         "avg_freight_20ft_usd": 1100, "avg_freight_40ft_usd": 1800, "avg_freight_40hc_usd": 2000},
        {"port_of_loading": "INMUN1", "port_of_discharge": "GBFXT",
         "avg_transit_days": 24, "transit_range_min": 20, "transit_range_max": 29,
         "avg_freight_20ft_usd": 1200, "avg_freight_40ft_usd": 1900, "avg_freight_40hc_usd": 2100},
        {"port_of_loading": "INMUN1", "port_of_discharge": "AEJEA",
         "avg_transit_days": 8, "transit_range_min": 6, "transit_range_max": 11,
         "avg_freight_20ft_usd": 400, "avg_freight_40ft_usd": 700, "avg_freight_40hc_usd": 800},
        {"port_of_loading": "INMUN1", "port_of_discharge": "AUMEL",
         "avg_transit_days": 22, "transit_range_min": 18, "transit_range_max": 27,
         "avg_freight_20ft_usd": 1000, "avg_freight_40ft_usd": 1600, "avg_freight_40hc_usd": 1800},
        {"port_of_loading": "INMUN1", "port_of_discharge": "SGSIN",
         "avg_transit_days": 13, "transit_range_min": 10, "transit_range_max": 17,
         "avg_freight_20ft_usd": 600, "avg_freight_40ft_usd": 950, "avg_freight_40hc_usd": 1100},
        {"port_of_loading": "INMUN1", "port_of_discharge": "JPTYO",
         "avg_transit_days": 18, "transit_range_min": 15, "transit_range_max": 22,
         "avg_freight_20ft_usd": 900, "avg_freight_40ft_usd": 1450, "avg_freight_40hc_usd": 1600},
        {"port_of_loading": "INMUN1", "port_of_discharge": "ZACPT",
         "avg_transit_days": 20, "transit_range_min": 16, "transit_range_max": 25,
         "avg_freight_20ft_usd": 950, "avg_freight_40ft_usd": 1500, "avg_freight_40hc_usd": 1700},
        {"port_of_loading": "INNSA1", "port_of_discharge": "USLAX",
         "avg_transit_days": 30, "transit_range_min": 25, "transit_range_max": 36,
         "avg_freight_20ft_usd": 1450, "avg_freight_40ft_usd": 2300, "avg_freight_40hc_usd": 2500},
        {"port_of_loading": "INNSA1", "port_of_discharge": "DEHAM",
         "avg_transit_days": 23, "transit_range_min": 19, "transit_range_max": 28,
         "avg_freight_20ft_usd": 1150, "avg_freight_40ft_usd": 1850, "avg_freight_40hc_usd": 2050},
        {"port_of_loading": "INMAA1", "port_of_discharge": "LKCMB",
         "avg_transit_days": 3, "transit_range_min": 2, "transit_range_max": 5,
         "avg_freight_20ft_usd": 200, "avg_freight_40ft_usd": 350, "avg_freight_40hc_usd": 400},
        {"port_of_loading": "INMAA1", "port_of_discharge": "SGSIN",
         "avg_transit_days": 10, "transit_range_min": 8, "transit_range_max": 13,
         "avg_freight_20ft_usd": 550, "avg_freight_40ft_usd": 880, "avg_freight_40hc_usd": 980},
        {"port_of_loading": "INBLR4", "port_of_discharge": "AEJEA",
         "avg_transit_days": 12, "transit_range_min": 9, "transit_range_max": 16,
         "avg_freight_20ft_usd": 700, "avg_freight_40ft_usd": 1100, "avg_freight_40hc_usd": 1250},
        {"port_of_loading": "INBLR4", "port_of_discharge": "USLAX",
         "avg_transit_days": 32, "transit_range_min": 27, "transit_range_max": 38,
         "avg_freight_20ft_usd": 1600, "avg_freight_40ft_usd": 2500, "avg_freight_40hc_usd": 2700},
        {"port_of_loading": "INMUN1", "port_of_discharge": "BRSSZ",
         "avg_transit_days": 35, "transit_range_min": 29, "transit_range_max": 42,
         "avg_freight_20ft_usd": 1800, "avg_freight_40ft_usd": 2800, "avg_freight_40hc_usd": 3100},
    ]
    df = pd.DataFrame(routes)
    df.to_csv(os.path.join(DATA_DIR, 'routes.csv'), index=False)
    print(f"✅ routes.csv: {len(df)} routes")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  TABLE 1 — SHIPMENTS (250 rows with 12 planted anomalies)
# ═══════════════════════════════════════════════════════════════════════════
def generate_shipments(products_df, buyers_df, routes_df):
    prod_by_id   = {r['product_id']: r for r in products_df.to_dict('records')}
    route_lookup = {}
    for r in routes_df.to_dict('records'):
        route_lookup[(r['port_of_loading'], r['port_of_discharge'])] = r

    buyer_to_country = {b['buyer_name']: b['buyer_country'] for b in buyers_df.to_dict('records')}

    buyer_route_map = {
        "Global Mart Inc":      ("INMUN1",  "USLAX"),
        "Euro Trade GmbH":      ("INMUN1",  "DEHAM"),
        "Gulf Distributors LLC":("INMUN1",  "AEJEA"),
        "Pacific Traders Pty":  ("INMUN1",  "AUMEL"),
        "SG Merchants Pte":     ("INMAA1",  "SGSIN"),
        "London Imports Ltd":   ("INNSA1",  "GBFXT"),
        "Nippon Commerce KK":   ("INMUN1",  "JPTYO"),
        "African Goods Co":     ("INMUN1",  "ZACPT"),
    }

    shipping_lines    = ["Maersk", "MSC", "CMA CGM", "Hapag-Lloyd", "ONE"]
    container_types   = ["20ft", "40ft", "40ft HC"]
    payment_terms_map = {
        "Global Mart Inc": "LC 60 days",
        "Euro Trade GmbH": "LC 30 days",
        "Gulf Distributors LLC": "Advance",
        "Pacific Traders Pty": "Open Account 45 days",
        "SG Merchants Pte": "LC 30 days",
        "London Imports Ltd": "LC 60 days",
        "Nippon Commerce KK": "LC 30 days",
        "African Goods Co": "LC 90 days",
    }
    vessel_names = [
        "MSC Aurora", "Maersk Taurus", "CMA CGM Libra", "ONE Cosmos",
        "Hapag Express", "Ever Given II", "MSC Gulsun", "Maersk Elba",
        "CMA CGM Marco Polo", "Pacific Carrier", "Atlantic Voyager", "Indian Ocean Star"
    ]
    freight_forwarders = ["DHL Global", "Kuehne Nagel", "DB Schenker", "Expeditors", "Allcargo"]
    cha_names = ["ABC Customs", "XYZ Clearing", "TradeEase CHA"]

    buyers_list    = buyers_df['buyer_name'].tolist()
    products_list  = products_df['product_id'].tolist()

    start_date = datetime(2025, 9, 1)
    end_date   = datetime(2026, 2, 28)
    total_days = (end_date - start_date).days

    shipments = []
    TOTAL     = 250

    PLANTED_IDS = {
        "SHP-2025-0034",   # P-001 pricing math
        "SHP-2025-0067",   # P-002 price below range
        "SHP-2025-0089",   # P-003 HS code mismatch
        "SHP-2025-0115",   # P-004 drawback on rejected
        "SHP-2025-0127",   # P-005 transit days spike
        "SHP-2025-0156",   # P-006 freight cost 4x
        "SHP-2025-0187",   # P-007 buyer payment delay
        "SHP-2025-0199",   # P-008 payment received but days=null
        "SHP-2025-0212",   # P-009 volume spike
        "SHP-2025-0230",   # P-010 country volume spike
        "SHP-2025-0241",   # P-011 insurance 2%
        "SHP-2025-0248",   # P-012 CIF but freight=0
    }

    def random_shipment(shipment_id, date=None):
        """Generate a clean shipment row."""
        buyer  = random.choice(buyers_list)
        prod_id = random.choice(products_list)
        prod   = prod_by_id[prod_id]
        pol, pod = buyer_route_map.get(buyer, ("INMUN1", "USLAX"))
        route  = route_lookup.get((pol, pod), route_lookup[("INMUN1", "USLAX")])

        if date is None:
            date = start_date + timedelta(days=random.randint(0, total_days))

        qty = random.randint(100, 8000)
        unit_price = round(
            random.uniform(prod['price_range_min'], prod['price_range_max']), 2
        )
        total_fob = round(qty * unit_price, 2)

        ctype = random.choice(container_types)
        freight_key_map = {"20ft": "avg_freight_20ft_usd",
                           "40ft": "avg_freight_40ft_usd",
                           "40ft HC": "avg_freight_40hc_usd"}
        freight_cost = round(
            route[freight_key_map[ctype]] * random.uniform(0.85, 1.15), 2
        )

        insurance = round(total_fob * 0.002, 2)  # 0.2% of FOB

        incoterm = random.choices(
            ["FOB", "CIF", "EXW", "CFR"], weights=[50, 25, 15, 10]
        )[0]

        transit_days = random.randint(route['transit_range_min'], route['transit_range_max'])

        cstatus = random.choices(
            ["approved", "rejected", "pending"], weights=[85, 5, 10]
        )[0]

        drawback_rate = prod['drawback_rate_pct']
        drawback_amount = round(total_fob * drawback_rate / 100, 2)
        if cstatus == "rejected":
            drawback_amount = 0.0

        buyer_info = buyers_df[buyers_df['buyer_name'] == buyer].iloc[0]
        avg_pay_days = int(buyer_info['avg_payment_days'])
        pstatus = random.choices(
            ["received", "partial", "pending", "overdue"],
            weights=[70, 10, 15, 5]
        )[0]
        if pstatus in ("received", "partial"):
            days_to_payment = random.randint(
                max(1, avg_pay_days - 10), avg_pay_days + 20
            )
        elif pstatus == "overdue":
            payment_terms_days = {"Advance": 0, "LC 30 days": 30, "LC 60 days": 60,
                                  "LC 90 days": 90, "Open Account 45 days": 45}
            terms = payment_terms_map.get(buyer, "LC 60 days")
            days_to_payment = payment_terms_days.get(terms, 60) + random.randint(31, 60)
        else:
            days_to_payment = None

        return {
            "shipment_id": shipment_id,
            "date": date.strftime("%Y-%m-%d"),
            "buyer_name": buyer,
            "buyer_country": buyer_to_country[buyer],
            "product_description": prod['product_description'],
            "hs_code": prod['hs_code'],
            "quantity": qty,
            "unit_price_usd": unit_price,
            "total_fob_usd": total_fob,
            "currency": "USD",
            "freight_cost_usd": freight_cost,
            "insurance_usd": insurance,
            "incoterm": incoterm,
            "port_of_loading": pol,
            "port_of_discharge": pod,
            "shipping_line": random.choice(shipping_lines),
            "container_type": ctype,
            "transit_days": transit_days,
            "vessel_name": random.choice(vessel_names),
            "customs_status": cstatus,
            "drawback_rate_pct": drawback_rate,
            "drawback_amount_usd": drawback_amount,
            "payment_terms": payment_terms_map.get(buyer, "LC 60 days"),
            "payment_status": pstatus,
            "days_to_payment": days_to_payment,
            "freight_forwarder": random.choice(freight_forwarders),
            "cha_name": random.choice(cha_names),
        }

    # Generate 238 clean rows (250 - 12 planted)
    for i in range(1, TOTAL + 1):
        sid = f"SHP-2025-{i:04d}"
        if sid not in PLANTED_IDS:
            d = start_date + timedelta(days=random.randint(0, total_days))
            shipments.append(random_shipment(sid, date=d))

    # ────── PLANT 12 ANOMALIES ──────────────────────────────────────────

    # PLANTED-001: FOB math error
    s = random_shipment("SHP-2025-0034", date=datetime(2025, 10, 5))
    s["quantity"] = 2000
    s["unit_price_usd"] = 4.50
    s["total_fob_usd"] = 10800.00  # Wrong! Should be 9000
    s["buyer_name"] = "Global Mart Inc"
    s["product_description"] = "Cotton T-shirts 100% knitted"
    s["hs_code"] = "61091000"
    shipments.append(s)

    # PLANTED-002: Price dumping
    s = random_shipment("SHP-2025-0067", date=datetime(2025, 10, 18))
    s["product_description"] = "Cotton T-shirts 100% knitted"
    s["hs_code"] = "61091000"
    s["quantity"] = 5000
    s["unit_price_usd"] = 0.80  # Range min is 3.00!
    s["total_fob_usd"] = round(5000 * 0.80, 2)
    s["buyer_name"] = "Gulf Distributors LLC"
    shipments.append(s)

    # PLANTED-003: HS code mismatch
    s = random_shipment("SHP-2025-0089", date=datetime(2025, 11, 3))
    s["product_description"] = "Cotton T-shirts 100% knitted"
    s["hs_code"] = "84713000"  # Wrong! This is computers
    s["buyer_name"] = "Euro Trade GmbH"
    s["quantity"] = 3000
    shipments.append(s)

    # PLANTED-004: Drawback on rejected
    s = random_shipment("SHP-2025-0115", date=datetime(2025, 11, 20))
    s["customs_status"] = "rejected"
    s["drawback_amount_usd"] = 850.00  # Should be 0!
    s["buyer_name"] = "London Imports Ltd"
    shipments.append(s)

    # PLANTED-005: Transit spike
    s = random_shipment("SHP-2025-0127", date=datetime(2025, 12, 1))
    s["port_of_loading"] = "INMUN1"
    s["port_of_discharge"] = "AEJEA"
    s["buyer_name"] = "Gulf Distributors LLC"
    s["transit_days"] = 45  # Normal max: 11 days!
    shipments.append(s)

    # PLANTED-006: Freight 4x
    s = random_shipment("SHP-2025-0156", date=datetime(2025, 12, 15))
    s["port_of_loading"] = "INMUN1"
    s["port_of_discharge"] = "DEHAM"
    s["container_type"] = "40ft"
    s["buyer_name"] = "Euro Trade GmbH"
    s["freight_cost_usd"] = 7200.00  # Avg is 1800!
    shipments.append(s)

    # PLANTED-007: Payment delay
    s = random_shipment("SHP-2025-0187", date=datetime(2026, 1, 5))
    s["buyer_name"] = "Euro Trade GmbH"
    s["payment_status"] = "received"
    s["days_to_payment"] = 110  # Their avg: 32!
    shipments.append(s)

    # PLANTED-008: Received but null days
    s = random_shipment("SHP-2025-0199", date=datetime(2026, 1, 12))
    s["payment_status"] = "received"
    s["days_to_payment"] = None  # Contradiction!
    shipments.append(s)

    # PLANTED-009: Volume spike
    s = random_shipment("SHP-2025-0212", date=datetime(2026, 1, 18))
    s["buyer_name"] = "African Goods Co"
    s["product_description"] = "Cotton T-shirts 100% knitted"
    s["hs_code"] = "61091000"
    s["quantity"] = 80000  # 40x their usual!
    s["unit_price_usd"] = 4.50
    s["total_fob_usd"] = round(80000 * 4.50, 2)
    shipments.append(s)

    # PLANTED-010: Country volume spike
    s = random_shipment("SHP-2025-0230", date=datetime(2025, 10, 28))
    s["buyer_name"] = "Gulf Distributors LLC"
    s["quantity"] = 95000
    s["unit_price_usd"] = 1.20
    s["product_description"] = "Basmati Rice Premium Grade"
    s["hs_code"] = "10063020"
    s["total_fob_usd"] = round(95000 * 1.20, 2)
    shipments.append(s)

    # PLANTED-011: Insurance 2% not 0.2%
    s = random_shipment("SHP-2025-0241", date=datetime(2026, 1, 25))
    s["quantity"] = 1000
    s["unit_price_usd"] = 4.50
    s["total_fob_usd"] = round(1000 * 4.50, 2)
    s["insurance_usd"] = round(s["total_fob_usd"] * 0.02, 2)  # 2%!
    s["product_description"] = "Cotton T-shirts 100% knitted"
    s["hs_code"] = "61091000"
    shipments.append(s)

    # PLANTED-012: CIF but freight = 0
    s = random_shipment("SHP-2025-0248", date=datetime(2026, 2, 5))
    s["incoterm"] = "CIF"
    s["freight_cost_usd"] = 0.0  # Wrong! CIF seller pays!
    shipments.append(s)

    random.shuffle(shipments)
    df = pd.DataFrame(shipments)
    df = df.sort_values("date").reset_index(drop=True)
    df.to_csv(os.path.join(DATA_DIR, 'shipments.csv'), index=False)
    print(f"✅ shipments.csv: {len(df)} rows with 12 planted anomalies")
    return df


# ═══════════════════════════════════════════════════════════════════════════
#  PLANTED ANOMALIES MANIFEST
# ═══════════════════════════════════════════════════════════════════════════
def save_planted_anomalies():
    anomalies = [
        {
            "anomaly_id": "PLANTED-001",
            "shipment_id": "SHP-2025-0034",
            "category": "pricing",
            "sub_type": "fob_math_error",
            "description": "total_fob_usd ($10,800) ≠ quantity (2000) × unit_price ($4.50 = $9,000). Inflated by $1,800.",
            "why_this_matters": "Inflated FOB increases drawback claim. Customs penalty: ₹1-5 lakh.",
            "estimated_penalty_usd": 5000,
            "severity": "critical"
        },
        {
            "anomaly_id": "PLANTED-002",
            "shipment_id": "SHP-2025-0067",
            "category": "pricing",
            "sub_type": "price_below_range",
            "description": "Cotton T-shirts unit price $0.80 is 73% below minimum of $3.00. Suspicious discounting.",
            "why_this_matters": "Under-invoicing to reduce duties. FEMA violation risk: ₹2-10 lakh.",
            "estimated_penalty_usd": 8000,
            "severity": "critical"
        },
        {
            "anomaly_id": "PLANTED-003",
            "shipment_id": "SHP-2025-0089",
            "category": "compliance",
            "sub_type": "hs_code_mismatch",
            "description": "Cotton T-shirts (Chapter 61) exported under HS code 84713000 (Laptops - Chapter 84).",
            "why_this_matters": "Wrong duty rate applied. Penalty: ₹50K-₹2L + goods seized.",
            "estimated_penalty_usd": 6000,
            "severity": "critical"
        },
        {
            "anomaly_id": "PLANTED-004",
            "shipment_id": "SHP-2025-0115",
            "category": "compliance",
            "sub_type": "drawback_on_rejected",
            "description": "Drawback of $850 claimed on rejected shipment. Drawback ineligible for rejected cargo.",
            "why_this_matters": "Fraudulent drawback claim. Recovery + penalty up to 200%. Risk: ₹1.5-3 lakh.",
            "estimated_penalty_usd": 4000,
            "severity": "high"
        },
        {
            "anomaly_id": "PLANTED-005",
            "shipment_id": "SHP-2025-0127",
            "category": "route_logistics",
            "sub_type": "transit_days_spike",
            "description": "INMUN1→AEJEA normal range 6-11 days. This shipment: 45 days (4x maximum).",
            "why_this_matters": "Unexplained delay suggests port hold or re-routing. Demurrage risk: ₹80K.",
            "estimated_penalty_usd": 3000,
            "severity": "high"
        },
        {
            "anomaly_id": "PLANTED-006",
            "shipment_id": "SHP-2025-0156",
            "category": "route_logistics",
            "sub_type": "freight_cost_spike",
            "description": "INMUN1→DEHAM 40ft avg freight $1,800. This shows $7,200 (4x average).",
            "why_this_matters": "Inflated freight increases CIF value. Margin loss: $5,400 per shipment.",
            "estimated_penalty_usd": 5400,
            "severity": "high"
        },
        {
            "anomaly_id": "PLANTED-007",
            "shipment_id": "SHP-2025-0187",
            "category": "payment",
            "sub_type": "payment_behavior_change",
            "description": "Euro Trade GmbH avg_payment_days = 32. This shipment: 110 days (3.4x average).",
            "why_this_matters": "Buyer financial stress. Working capital blocked extra 78 days. Bad debt risk.",
            "estimated_penalty_usd": 2000,
            "severity": "high"
        },
        {
            "anomaly_id": "PLANTED-008",
            "shipment_id": "SHP-2025-0199",
            "category": "payment",
            "sub_type": "received_null_days",
            "description": "payment_status = 'received' but days_to_payment = NULL. Contradictory data.",
            "why_this_matters": "Data integrity issue. Accounting audit risk.",
            "estimated_penalty_usd": 1000,
            "severity": "medium"
        },
        {
            "anomaly_id": "PLANTED-009",
            "shipment_id": "SHP-2025-0212",
            "category": "volume",
            "sub_type": "buyer_volume_spike",
            "description": "African Goods Co avg order $14,000. This order: 80,000 units = $360,000 (26x average).",
            "why_this_matters": "Possible fictitious transaction or money laundering. FEMA scrutiny. Risk: ₹5-20 lakh.",
            "estimated_penalty_usd": 15000,
            "severity": "critical"
        },
        {
            "anomaly_id": "PLANTED-010",
            "shipment_id": "SHP-2025-0230",
            "category": "volume",
            "sub_type": "country_volume_spike",
            "description": "UAE single October shipment: 95,000 units Basmati Rice ($114,000) — anomalous spike.",
            "why_this_matters": "Possible re-export to sanctioned countries. Embargo compliance risk.",
            "estimated_penalty_usd": 10000,
            "severity": "critical"
        },
        {
            "anomaly_id": "PLANTED-011",
            "shipment_id": "SHP-2025-0241",
            "category": "cross_field",
            "sub_type": "insurance_rate_error",
            "description": "Insurance = $90 on FOB $4,500 = 2.0% rate. Normal is ~0.2%. 10x overcharge.",
            "why_this_matters": "Inflated CIF value. Unnecessary cost: ₹5K-₹40K.",
            "estimated_penalty_usd": 500,
            "severity": "medium"
        },
        {
            "anomaly_id": "PLANTED-012",
            "shipment_id": "SHP-2025-0248",
            "category": "cross_field",
            "sub_type": "cif_zero_freight",
            "description": "incoterm = CIF but freight_cost_usd = 0. CIF requires seller to pay freight.",
            "why_this_matters": "Contract violation or hidden cost splitting. Delivery responsibility dispute.",
            "estimated_penalty_usd": 2500,
            "severity": "high"
        },
    ]
    with open(os.path.join(DATA_DIR, 'planted_anomalies.json'), 'w') as f:
        json.dump(anomalies, f, indent=2)
    print(f"✅ planted_anomalies.json: {len(anomalies)} anomalies planted")
    return anomalies


# ═══════════════════════════════════════════════════════════════════════════
#  MAIN
# ═══════════════════════════════════════════════════════════════════════════
if __name__ == "__main__":
    print("\n🔧 Generating synthetic trade data...\n")
    products_df = generate_product_catalog()
    buyers_df   = generate_buyers()
    routes_df   = generate_routes()
    shipments_df = generate_shipments(products_df, buyers_df, routes_df)
    save_planted_anomalies()
    print(f"\n✅ All data generated in /data folder")
    print(f"   Shipments: {len(shipments_df)}")
    print(f"   Planted anomalies: 12 across 6 categories\n")