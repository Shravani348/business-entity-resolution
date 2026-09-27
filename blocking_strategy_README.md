# Blocking Strategy --- Entity Resolution / Record Linkage

## 1. Overview

This project performs entity resolution / record linkage across three
large business datasets.

The objective of the blocking stage is to efficiently identify plausible
candidate records from Source 2 and Source 3 for each Source 1 business
record, instead of comparing every possible pair.

The final blocking rule is:

> **Same country AND (shared normalized business-name token OR shared
> normalized address token)**

------------------------------------------------------------------------

## 2. Dataset Sizes

  Dataset               Rows   Columns
  -------------- ----------- ---------
  Source 1         2,206,821         4
  Source 2         5,034,616         4
  Source 3         5,286,503         4
  Ground Truth     2,206,821         2

### Source columns

All three source datasets contain:

``` text
entity_id
business_name
business_address
country
```

Ground Truth contains:

``` text
source1_entity_id
matched_entity_ids
```

`matched_entity_ids` contains the corresponding Source 2 and/or Source 3
entity IDs.

------------------------------------------------------------------------

## 3. Data Exploration

The datasets were inspected for:

-   row counts
-   column names
-   data types
-   missing values
-   sample records
-   country distribution
-   ground-truth structure

### Data types

Source datasets:

``` text
entity_id          object
business_name      object
business_address   object
country            object
```

Ground Truth:

``` text
source1_entity_id    object
matched_entity_ids   object
```

------------------------------------------------------------------------

## 4. Missing Values

### Source 1

``` text
entity_id          0
business_name      0
business_address   0
country            0
```

### Source 2

``` text
entity_id             0
business_name         2
business_address   168,967
country               0
```

### Source 3

``` text
entity_id             0
business_name        13
business_address   175,916
country               0
```

Source 2 and Source 3 contain substantial numbers of missing addresses,
so address-only matching would not be sufficient.

------------------------------------------------------------------------

## 5. Country Distribution

### Source 1

``` text
US       1,323,633
India      883,188
```

### Source 2

``` text
US       3,016,817
India    2,017,799
```

### Source 3

``` text
US       3,170,056
India    2,115,547
```

Country is used as a blocking condition to avoid generating candidates
across different countries.

------------------------------------------------------------------------

# 6. Why Blocking Is Needed

The datasets are too large for an all-pairs comparison.

Source 1 has more than 2.2 million records, while Source 2 and Source 3
have more than 5 million records each.

Comparing every Source 1 record against every Source 2 and Source 3
record would produce an impractically large number of comparisons.

Blocking reduces this search space by retaining only records that share
useful characteristics.

------------------------------------------------------------------------

# 7. Text Normalization

The following normalized fields were created:

``` text
name_norm
address_norm
country_norm
```

Normalization reduces differences caused by:

-   capitalization
-   punctuation
-   formatting
-   common textual variations
-   abbreviations

Example:

``` text
Original:
B+ Retail Inc
1712 Montebello Avenue, Phoenix, AZ
US

Normalized:
b retail incorporated
1712 montebello avenue phoenix az
us
```

------------------------------------------------------------------------

# 8. Initial Strategy --- Exact Name + Country

The first strategy required:

``` text
same normalized business name
AND
same country
```

Lookup tables were created using:

``` python
.groupby(["country_norm", "name_norm"])["entity_id"].agg(list)
```

### Number of exact-name blocks

``` text
S2 name blocks: 3,971,628
S3 name blocks: 4,223,576
```

### Recall result

The initial ground-truth evaluation produced:

``` text
Actual matches in sample: 172,784
Matches found by name blocking: 42,609
Candidate recall: 24.66%
```

Therefore, exact normalized name + country was too restrictive.

------------------------------------------------------------------------

# 9. Name Token Blocking

The next strategy split normalized business names into tokens.

For example:

``` text
dahlia power reliable scientific llc
```

becomes tokens such as:

``` text
dahlia
power
reliable
scientific
llc
```

The rule becomes:

``` text
same country
AND
at least one shared name token
```

This allows records with spelling, punctuation, abbreviation, or
partial-name differences to become candidates.

### Token index sizes

``` text
S2 token blocks: 839,931
S3 token blocks: 900,475
```

The token index uses:

``` text
(country, token) -> entity IDs
```

### Name-token recall

Using a reproducible 5,000-record ground-truth sample:

``` text
Actual matches checked: 17,216
Matches covered:        14,771
Token blocking recall:  85.80%
```

This was a major improvement over the 24.66% exact-name result.

------------------------------------------------------------------------

# 10. Address Token Blocking

Address information was added because it can remain useful when business
names are noisy or different.

Address tokens were generated from `address_norm`.

``` python
def get_address_tokens(address):
    if not address:
        return []

    tokens = address.split()

    return [
        token
        for token in tokens
        if len(token) >= 3
    ]
```

The address blocking rule is:

``` text
same country
AND
at least one shared address token
```

### Address index sizes

``` text
S2 address blocks: 639,019
S3 address blocks: 611,431
```

### Address recall

``` text
Actual matches checked: 17,216
Matches covered:        16,438
Address blocking recall: 95.48%
```

------------------------------------------------------------------------

# 11. Combined Blocking Strategy

The two strategies were combined:

``` text
same country
AND
(
    shared name token
    OR
    shared address token
)
```

Conceptually:

``` text
                    Source 1
                       |
                Same Country
                       |
              +--------+--------+
              |                 |
       Name Token Match   Address Token Match
              |                 |
              +--------+--------+
                       |
                 Candidate Pair
```

This union strategy allows a true match missed by one blocking rule to
potentially be recovered by the other.

------------------------------------------------------------------------

# 12. Combined Blocking Recall

The combined strategy was evaluated using the same 5,000-record
ground-truth sample.

Results:

``` text
Actual matches checked: 17,216
Matches covered:        17,206
Combined recall:         99.94%
```

Only:

``` text
10
```

of the 17,216 checked true matches were missed.

------------------------------------------------------------------------

# 13. Blocking Strategy Comparison

  Strategy                          Matches Checked   Matches Covered       Recall
  ------------------------------- ----------------- ----------------- ------------
  Exact name + country                      172,784            42,609       24.66%
  Name token + country                       17,216            14,771       85.80%
  Address token + country                    17,216            16,438       95.48%
  **Name OR address + country**          **17,216**        **17,206**   **99.94%**

------------------------------------------------------------------------

# 14. Evaluation Method

The faster diagnostic evaluation used:

``` python
sample_gt = gt.sample(
    n=5000,
    random_state=42
).copy()
```

This makes the evaluation reproducible.

Blocking recall is calculated as:

``` text
Blocking Recall =
True matches surviving blocking
--------------------------------
Total true matches checked
```

For the final combined result:

``` text
17,206 / 17,216 = 99.94%
```

### Important limitation

The 99.94% result is measured on the 5,000-record evaluation sample.

It should therefore be reported as:

> The combined name-token and address-token blocking strategy achieved
> 99.94% recall on a 5,000-record ground-truth evaluation sample.

It should not yet be presented as the measured recall of the complete
2.2 million-record dataset.

------------------------------------------------------------------------

# 15. Errors and Fixes

## Error 1 --- `sample_gt` not defined

Error:

``` text
NameError: name 'sample_gt' is not defined
```

Cause: the evaluation sample was not present in the current Jupyter
kernel state.

Fix:

``` python
sample_gt = gt.sample(
    n=5000,
    random_state=42
).copy()
```

## Error 2 --- `address_norm` not found

Error:

``` text
KeyError: 'address_norm'
```

Cause: the lookup tables initially contained only:

``` text
name_norm
country_norm
```

Fix:

``` python
s1_lookup = s1.set_index("entity_id")[
    ["name_norm", "address_norm", "country_norm"]
]

s2_lookup = s2.set_index("entity_id")[
    ["name_norm", "address_norm", "country_norm"]
]

s3_lookup = s3.set_index("entity_id")[
    ["name_norm", "address_norm", "country_norm"]
]
```

------------------------------------------------------------------------

# 16. Efficiency Considerations

An initial recall test attempted to construct complete candidate sets
for every sampled Source 1 record.

This was slow because common tokens can map to very large numbers of
entities.

The recall test was therefore optimized to inspect the known
ground-truth matches directly and check whether those true matches
satisfy the blocking rule.

This avoids generating huge candidate sets during the diagnostic recall
calculation.

------------------------------------------------------------------------

# 17. Current Blocking Architecture

``` text
                    Source 1
                       |
                Text Normalization
                       |
              +--------+--------+
              |                 |
        Name Tokenization   Address Tokenization
              |                 |
              v                 v
       Name Token Index    Address Token Index
              |                 |
              +--------+--------+
                       |
                  Same Country
                       |
             Name OR Address Match
                       |
                       v
                Candidate Pairs
```

------------------------------------------------------------------------

# 18. Current Blocking Results

``` text
Source 1 records:          2,206,821
Source 2 records:          5,034,616
Source 3 records:          5,286,503

S2 name blocks:            3,971,628
S3 name blocks:            4,223,576

S2 token blocks:             839,931
S3 token blocks:             900,475

S2 address blocks:           639,019
S3 address blocks:           611,431

Exact name recall:              24.66%
Name token recall:              85.80%
Address token recall:           95.48%
Combined recall:                99.94%
```

------------------------------------------------------------------------

# 19. Final Blocking Rule

The validated blocking rule is:

> **Country must match AND at least one normalized business-name token
> OR one normalized business-address token must match.**

In logical form:

``` text
candidate =
    same_country
    AND
    (
        shared_name_token
        OR
        shared_address_token
    )
```

------------------------------------------------------------------------

# 20. Why This Strategy Works

### Name-token blocking helps with:

-   spelling variations
-   partial names
-   punctuation differences
-   abbreviations
-   noisy business names

### Address-token blocking helps with:

-   street-name variations
-   address abbreviations
-   reordered address components
-   businesses with noisy names
-   geographic information

### Country blocking helps with:

-   eliminating obviously impossible cross-country candidates
-   reducing the candidate search space
-   avoiding unnecessary comparisons

The combination provides complementary candidate-generation signals.

------------------------------------------------------------------------

# 21. Next Stage --- Candidate Generation

The blocking stage is now ready for full candidate generation.

The next workflow is:

``` text
Blocking Strategy
       |
       v
Candidate Generation
       |
       v
candidate_pairs.tsv
       |
       v
Feature Engineering
       |
       +-- Name similarity
       +-- Address similarity
       +-- Country agreement
       |
       v
Matching Model
       |
       v
Threshold tuning
       |
       v
F0.5 evaluation
       |
       v
Final entity matches
```

Because the source datasets contain millions of records, candidate
generation should be implemented using memory-conscious/chunked
processing rather than creating all possible pair combinations at once.

------------------------------------------------------------------------

# 22. Final Summary

The blocking strategy evolved from a restrictive exact-name rule to a
combined token-based approach.

### Results

``` text
Exact name + country       -> 24.66%
Name token + country       -> 85.80%
Address token + country    -> 95.48%
Combined strategy          -> 99.94%
```

The final combined strategy captured:

``` text
17,206 / 17,216
```

ground-truth matches in the 5,000-record evaluation sample.

This provides a strong candidate-generation foundation for the next
stages of the entity-resolution pipeline.
