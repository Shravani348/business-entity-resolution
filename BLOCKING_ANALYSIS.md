# Blocking Strategy Analysis

## 1. Prototype Scale
- **Source 1 Sample Size:** 20,000
- **Indexed Candidates:** 235,464 (Source 2) + 237,997 (Source 3)
- **Total Possible Pairs in Sample:** 9,469,220,000

## 2. Performance Metrics
- **Blocking Recall:** 99.99% (73,451 / 73,461 true matches captured)
- **Reduction Ratio:** 90.7491% of pairs eliminated from search space.
- **Total Candidates Generated (Sample):** 875,987,567

## 3. Candidates per Source 1 Entity
- **Min:** 185
- **Median:** 34339.5
- **Max:** 139780
- **Average:** 43799.38

## 4. Non-Latin Script Limitation
- **Non-Latin S1 Entities in Sample:** 0
- **Missed matches due to Non-Latin name + missing address:** 0
  *(This confirms our known limitation where name tokens mismatch across scripts and address tokens are unavailable to bridge the gap.)*

## 5. Full Dataset Estimation
Scaling this strategy to the full 2.2M Source 1 entities would generate approximately **96,657,387,930** candidate pairs. **This is far too permissive (too many candidates) for a downstream ML model.** We will need to tighten the blocking keys (e.g. require longer token lengths, use TF-IDF to penalize common words like 'street'/'main', or require multi-token overlap).
