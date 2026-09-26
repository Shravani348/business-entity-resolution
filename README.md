# Blocking Strategy — Entity Resolution / Record Linkage

## Overview

Blocking stage for entity resolution across three business datasets, identifying plausible candidate records from Source 2 and Source 3 for each Source 1 record — avoiding an all-pairs comparison.

**Final blocking rule:**
> Same country AND (shared normalized business-name token OR shared normalized address token)

---

## Dataset Sizes

| Dataset | Rows | Columns |
|---|---|---|
| Source 1 | 2,206,821 | 4 |
| Source 2 | 5,034,616 | 4 |
| Source 3 | 5,286,503 | 4 |
| Ground Truth | 2,206,821 | 2 |

Columns: `entity_id, business_name, business_address, country` (sources); `source1_entity_id, matched_entity_ids` (ground truth).

**Missing values:** Source 1 has none. Source 2/3 have minimal missing names but substantial missing addresses (S2: 168,967; S3: 175,916) — so address-only matching is insufficient.

**Country distribution:** All sources split roughly 60% US / 40% India.

---

## Why Blocking Is Needed

With 2.2M+ Source 1 records against 5M+ records in each of Source 2/3, all-pairs comparison is computationally infeasible. Blocking narrows the search space to records sharing useful characteristics.

---

## Text Normalization

Created `name_norm`, `address_norm`, `country_norm` fields to remove capitalization, punctuation, formatting, and abbreviation inconsistencies (e.g., "B+ Retail Inc" → "b retail incorporated").

---

## Strategy Evolution

1. **Exact name + country** — too restrictive; missed spelling/abbreviation variants.
2. **Name token blocking** — split names into tokens; match on any shared token + country. Major recall improvement.
3. **Address token blocking** — tokens (≥3 chars) from `address_norm`; useful when names are noisy.
4. **Combined (name OR address token) + country** — union of both strategies, recovering matches missed by either alone.

---

## Blocking Strategy Comparison

| Strategy | Matches Checked | Matches Covered | Recall |
|---|---|---|---|
| Exact name + country | 172,784 | 42,609 | 24.66% |
| Name token + country | 17,216 | 14,771 | 85.80% |
| Address token + country | 17,216 | 16,438 | 95.48% |
| **Name OR address + country** | **17,216** | **17,206** | **99.94%** |

*Note: exact-name test used the full sample (172,784); token-based tests used a reproducible 5,000-record ground-truth sample (17,216 checkable matches).*

**Index sizes:**
| Index | S2 | S3 |
|---|---|---|
| Exact name blocks | 3,971,628 | 4,223,576 |
| Name token blocks | 839,931 | 900,475 |
| Address token blocks | 639,019 | 611,431 |

---

## Evaluation Method

```python
sample_gt = gt.sample(n=5000, random_state=42).copy()
```

Recall = (true matches surviving blocking) / (total true matches checked). For the combined strategy: 17,206 / 17,216 = **99.94%** (only 10 true matches missed).

**Limitation:** This 99.94% is measured on the 5,000-record sample only — not yet validated on the full 2.2M-record dataset.

---

## Key Fixes Along the Way

- **`sample_gt` undefined** → re-run the sampling cell to restore kernel state.
- **`address_norm` missing from lookups** → rebuild lookup tables to include `name_norm`, `address_norm`, `country_norm` (not just name/country).
- **Efficiency:** Full candidate-set generation per sampled record was slow (common tokens map to huge entity sets). Optimized by checking only whether *known* ground-truth matches satisfy the blocking rule, rather than generating full candidate sets.

---

## Final Architecture

```
Source 1 → Text Normalization → {Name Tokenization, Address Tokenization}
         → {Name Token Index, Address Token Index}
         → Same Country AND (Name Match OR Address Match)
         → Candidate Pairs
```

**Why it works:**
- *Name tokens* handle spelling variations, partial names, abbreviations, noisy names.
- *Address tokens* handle street-name variations, abbreviations, reordering, and cases with noisy names.
- *Country* eliminates impossible cross-country candidates, shrinking the search space.

---

## Final Blocking Rule

```
candidate = same_country AND (shared_name_token OR shared_address_token)
```

---

## Next Stage — Candidate Generation

```
Blocking Strategy → Candidate Generation → candidate_pairs.tsv
  → Feature Engineering (name sim, address sim, country agreement)
  → Matching Model → Threshold Tuning → F0.5 Evaluation → Final Matches
```

Given dataset scale, candidate generation should use memory-conscious/chunked processing rather than building all pairs at once.

---

## Summary

| Strategy | Recall |
|---|---|
| Exact name + country | 24.66% |
| Name token + country | 85.80% |
| Address token + country | 95.48% |
| **Combined strategy** | **99.94%** |

The combined name-OR-address token strategy captured 17,206 / 17,216 ground-truth matches in the evaluation sample, providing a strong foundation for the next entity-resolution stages.
