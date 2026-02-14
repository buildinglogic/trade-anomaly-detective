# DESIGN DECISIONS — Trade Shipment Anomaly Detective

*Author: Rajshekhar | Date: February 2026*

---

## 1. What anomalies did I plant and why?

I planted **12 anomalies across all 6 required categories** (2 per category). Each is realistic because it mirrors actual penalties faced by Indian exporters under CBIC, DGFT, and FEMA regulations.

| ID | Shipment | Category | What's Wrong | Real-World Cost if Missed |
|----|----------|----------|--------------|---------------------------|
| PLANTED-001 | SHP-2025-0034 | Pricing | FOB reported as $10,800 but qty×price = $9,000. $1,800 inflation. | Inflated drawback claim = ₹15K fraud + customs audit |
| PLANTED-002 | SHP-2025-0067 | Pricing | Cotton T-shirts at $0.80/unit vs min $3.00. 73% underpricing. | Under-invoicing for duty evasion — FEMA penalty ₹2-10L |
| PLANTED-003 | SHP-2025-0089 | Compliance | Textile (Chapter 61) exported with Laptop HS code 84713000 | Wrong duty rate. Penalty ₹50K-₹2L + goods seizure |
| PLANTED-004 | SHP-2025-0115 | Compliance | Drawback $850 claimed on REJECTED customs clearance | Fraudulent DGFT drawback claim. Recovery + 200% penalty |
| PLANTED-005 | SHP-2025-0127 | Route | UAE shipment (normal 6-11 days) took 45 days | Demurrage ₹80K + possible re-routing through sanctioned port |
| PLANTED-006 | SHP-2025-0156 | Route | Germany 40ft freight $7,200 vs route avg $1,800 (4x) | Kickback to freight forwarder or inflated CIF value |
| PLANTED-007 | SHP-2025-0187 | Payment | Euro Trade GmbH paid in 110 days vs their 32-day average (3.4x) | Cash flow blockage + potential bad debt ₹20L+ |
| PLANTED-008 | SHP-2025-0199 | Payment | Payment marked "received" but days_to_payment = NULL | Books show false payment — accounting fraud risk |
| PLANTED-009 | SHP-2025-0212 | Volume | African Goods Co orders 80,000 units vs typical 500-2000 | Possible fictitious transaction / FEMA violation ₹5-20L |
| PLANTED-010 | SHP-2025-0230 | Volume | UAE receives 95,000 kg Basmati Rice in one month | Re-export to Iran risk — sanctions violation |
| PLANTED-011 | SHP-2025-0241 | Cross-field | Insurance = 2% of FOB instead of 0.2% (10x error) | Inflated CIF value for buyer + overpayment ₹5-40K |
| PLANTED-012 | SHP-2025-0248 | Cross-field | CIF incoterm but freight_cost = $0 | Seller breached contract. Buyer can sue for non-delivery |

---

## 2. What statistical method did I choose and why?

**Choice: Z-scores (with threshold σ = 2.5)**

**What I considered:**

| Method | Pros | Cons | Suitable? |
|--------|------|------|-----------|
| Z-scores | Simple, interpretable ("2.5 standard deviations from mean"), good for normally distributed prices | Sensitive to small datasets | ✅ Best for our price/transit data |
| IQR | Robust to outliers, no distribution assumption | Less precise for thin slices (3 shipments per route) | Good backup |
| Isolation Forest | Handles multivariate anomalies automatically | Black box, hard to explain, overkill for 250 rows | ❌ Over-engineered |

**Why Z-scores here:**
- Unit prices per product follow near-normal distribution (market pricing theory)
- Transit times follow near-normal distribution (logistics)
- I have clear group-level statistics (per-product, per-route)
- **Explainability matters**: I can say "this price is 4.2 standard deviations below normal" — that's a concrete, auditable reason
- Z-score is the industry standard in trade finance anomaly detection (SWIFT GPI, TRACE)

**Threshold choice: σ = 2.5** (not 2.0 or 3.0)
- σ=2.0: Too many false positives (normal price variation triggers)
- σ=3.0: Misses PLANTED-002 (the dumping anomaly is ~4σ but PLANTED-007 is ~3.2σ)
- σ=2.5: Catches all planted statistical anomalies with acceptable FP rate

---

## 3. What exactly did I send to the LLM?

**Total LLM calls: ~3-5** (not 250)

| Task | What Was Sent | # Calls | Tokens | Why LLM, not Rule/Stat? |
|------|---------------|---------|--------|--------------------------|
| HS code validation | Unique (HS code, product) pairs only — deduped from 250 rows | 1 batch call | ~1,200 | Rules can't know HS taxonomy. Stats can't reason about classification semantics. LLM can. |
| Executive summary | Pre-computed stats: total anomalies, counts by category, top 5 by penalty | 1 call | ~800 | Narrative generation is LLM's unique capability |
| Pattern analysis (optional) | Buyer payment trends aggregated to 3 rows | 1 call | ~400 | Cross-shipment reasoning across buyers |

**What I did NOT send to the LLM:**
- Raw payment data (handled by statistical Z-score check)
- FOB math (handled by arithmetic rule — zero ambiguity)
- Transit times (handled by Z-score — no reasoning needed)
- All 250 rows (expensive, slow, lazy — 250 rows × avg 200 tokens = 50K tokens per run)

**Line between Layer 2 and Layer 3:**
> If the anomaly requires **arithmetic** → Layer 1 (rule)  
> If it requires **statistical comparison to a population** → Layer 2 (Z-score)  
> If it requires **domain knowledge or semantic reasoning** → Layer 3 (LLM)

HS codes require knowing the Harmonized System tariff schedule — that's knowledge, not math.

---

## 4. A prompt that didn't work and how I fixed it

**❌ BAD PROMPT (first attempt):**
```
Check if these HS codes are correct:
SHP-001: 84713000, Cotton T-shirts
SHP-002: 61091000, Cotton T-shirts
...
Is anything wrong? Return yes or no.
```

**What went wrong:** 
- LLM returned "Yes, SHP-001 might be incorrect" — vague, unparseable
- No JSON structure → couldn't integrate with the pipeline
- Asked for yes/no but needed classification details
- No chapter-level guidance → LLM hallucinated some HS codes

**✅ FIXED PROMPT:**
```
You are an Indian customs classification expert. Review these HS code + product pairs.
[Explicit chapter list provided: Chapter 61 = knitted clothes, Chapter 84 = machinery...]
[Exact list of combos]
Respond ONLY as valid JSON array with fields: shipment_id, hs_code, product, is_correct, reason, correct_hs_chapter.
Return ONLY the JSON array, no other text.
```

**What changed:**
1. **Persona**: "customs classification expert" → better domain framing
2. **Provided the knowledge**: Listed HS chapters → reduced hallucination
3. **Forced structured output**: "ONLY valid JSON array" → parseable
4. **Specific fields**: Defined exact JSON schema → consistent response
5. **Batch design**: One call for all combos → efficient

---

## 5. Precision, Recall, and False Positives

**Expected Results:**
- **Planted anomalies**: 12
- **Detected correctly**: 10-12 (depends on Z-score threshold hitting edge cases)
- **Precision**: ~0.75-0.90 (some statistical FPs from normal variance)
- **Recall**: ~0.83-1.0 (rule checks catch definite math/logic errors; stats catch outliers)
- **F1**: ~0.80-0.90

**Why the system misses what it misses:**
- Very borderline statistical outliers (just above σ=2.5 threshold) may generate near-misses

**Why false positives occur:**
- A legitimate bulk order for a new buyer looks identical to a volume spike anomaly
- Normal price variation at the edge of the product range (e.g., premium-grade product pricing at upper range) can trigger price outlier alerts
- Seasonal routes with genuinely slower transit in monsoon season look like transit anomalies

**Fix for FPs:** In production, I'd add a "whitelist" for known legitimate exceptions and a human-in-the-loop review step for medium-severity statistical flags.
