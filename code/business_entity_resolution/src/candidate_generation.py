import pandas as pd

from normalize import (
    add_normalized_columns,
    get_name_tokens,
    get_address_tokens,
)


MIN_TOKEN_LEN = 3
DEFAULT_MAX_BLOCK_SIZE = 2000


def _explode_tokens(df, token_col, entity_id_col="entity_id"):
    """
    Convert token lists into one row per (country, token, entity_id).
    """
    rows = []

    for _, row in df.iterrows():
        country = row["country_norm"]
        entity_id = row[entity_id_col]
        tokens = row[token_col]

        for token in tokens:
            rows.append(
                {
                    "country_norm": country,
                    "token": token,
                    "entity_id": entity_id,
                }
            )

    if not rows:
        return pd.DataFrame(
            columns=["country_norm", "token", "entity_id"]
        )

    return pd.DataFrame(rows)


def _prune_common_tokens(
    source1_tokens,
    source2_tokens,
    max_block_size=DEFAULT_MAX_BLOCK_SIZE,
):
    """
    Remove extremely common blocks.

    A block is defined by (country_norm, token).
    If the block is too large on either side, it is removed.
    """

    s1_sizes = (
        source1_tokens
        .groupby(["country_norm", "token"])
        .size()
        .rename("s1_block_size")
        .reset_index()
    )

    s2_sizes = (
        source2_tokens
        .groupby(["country_norm", "token"])
        .size()
        .rename("s2_block_size")
        .reset_index()
    )

    block_sizes = s1_sizes.merge(
        s2_sizes,
        on=["country_norm", "token"],
        how="outer",
    )

    block_sizes["s1_block_size"] = (
        block_sizes["s1_block_size"].fillna(0)
    )

    block_sizes["s2_block_size"] = (
        block_sizes["s2_block_size"].fillna(0)
    )

    valid_blocks = block_sizes[
        (block_sizes["s1_block_size"] <= max_block_size)
        & (block_sizes["s2_block_size"] <= max_block_size)
    ][["country_norm", "token"]]

    source1_tokens = source1_tokens.merge(
        valid_blocks,
        on=["country_norm", "token"],
        how="inner",
    )

    source2_tokens = source2_tokens.merge(
        valid_blocks,
        on=["country_norm", "token"],
        how="inner",
    )

    return source1_tokens, source2_tokens


def _candidate_pairs_from_tokens(
    source1_tokens,
    source2_tokens,
):
    """
    Generate candidate pairs by joining on country + token.
    """

    if source1_tokens.empty or source2_tokens.empty:
        return pd.DataFrame(
            columns=[
                "source1_entity_id",
                "candidate_entity_id",
            ]
        )

    pairs = source1_tokens.merge(
        source2_tokens,
        on=["country_norm", "token"],
        how="inner",
        suffixes=("_s1", "_candidate"),
    )

    pairs = pairs.rename(
        columns={
            "entity_id_s1": "source1_entity_id",
            "entity_id_candidate": "candidate_entity_id",
        }
    )

    pairs = pairs[
        [
            "source1_entity_id",
            "candidate_entity_id",
        ]
    ].drop_duplicates()

    return pairs


def block_on_name(
    source1,
    candidate_source,
    max_block_size=DEFAULT_MAX_BLOCK_SIZE,
):
    """
    Generate candidates using:

        country + shared normalized business-name token
    """

    s1 = source1.copy()
    candidate = candidate_source.copy()

    s1["name_tokens"] = s1["name_norm"].apply(get_name_tokens)
    candidate["name_tokens"] = candidate["name_norm"].apply(
        get_name_tokens
    )

    s1_tokens = _explode_tokens(
        s1,
        token_col="name_tokens",
    )

    candidate_tokens = _explode_tokens(
        candidate,
        token_col="name_tokens",
    )

    s1_tokens, candidate_tokens = _prune_common_tokens(
        s1_tokens,
        candidate_tokens,
        max_block_size=max_block_size,
    )

    pairs = _candidate_pairs_from_tokens(
        s1_tokens,
        candidate_tokens,
    )

    pairs["matched_on_name"] = True

    return pairs


def block_on_address(
    source1,
    candidate_source,
    max_block_size=DEFAULT_MAX_BLOCK_SIZE,
):
    """
    Generate candidates using:

        country + shared normalized address token
    """

    s1 = source1.copy()
    candidate = candidate_source.copy()

    s1["address_tokens"] = s1["address_norm"].apply(
        get_address_tokens
    )

    candidate["address_tokens"] = candidate["address_norm"].apply(
        get_address_tokens
    )

    s1_tokens = _explode_tokens(
        s1,
        token_col="address_tokens",
    )

    candidate_tokens = _explode_tokens(
        candidate,
        token_col="address_tokens",
    )

    s1_tokens, candidate_tokens = _prune_common_tokens(
        s1_tokens,
        candidate_tokens,
        max_block_size=max_block_size,
    )

    pairs = _candidate_pairs_from_tokens(
        s1_tokens,
        candidate_tokens,
    )

    pairs["matched_on_address"] = True

    return pairs


def generate_candidates(
    source1,
    source2,
    source3,
    max_block_size=DEFAULT_MAX_BLOCK_SIZE,
):
    """
    Generate candidate pairs from Source 1 against Source 2 and Source 3.

    Blocking strategy:

        country + (shared name token OR shared address token)

    Output columns:

        source1_entity_id
        candidate_entity_id
        candidate_source
        matched_on_name
        matched_on_address
    """

    # ---------------------------------------------------------
    # 1. Normalize all sources using the shared normalization
    # ---------------------------------------------------------

    source1 = add_normalized_columns(source1)
    source2 = add_normalized_columns(source2)
    source3 = add_normalized_columns(source3)

    all_results = []

    # ---------------------------------------------------------
    # 2. Generate candidates for S1 -> S2 and S1 -> S3
    # ---------------------------------------------------------

    for candidate_source_name, candidate_source in [
        ("S2", source2),
        ("S3", source3),
    ]:

        # -------------------------
        # Name blocking
        # -------------------------

        name_pairs = block_on_name(
            source1,
            candidate_source,
            max_block_size=max_block_size,
        )

        # -------------------------
        # Address blocking
        # -------------------------

        address_pairs = block_on_address(
            source1,
            candidate_source,
            max_block_size=max_block_size,
        )

        # -----------------------------------------------------
        # 3. UNION name + address candidates
        # -----------------------------------------------------

        name_pairs = name_pairs.assign(
            matched_on_address=False
        )

        address_pairs = address_pairs.assign(
            matched_on_name=False
        )

        combined = pd.concat(
            [
                name_pairs,
                address_pairs,
            ],
            ignore_index=True,
        )

        # -----------------------------------------------------
        # 4. Merge duplicate candidate pairs
        #
        # If the same pair was found through both blocks:
        #
        # matched_on_name    = True
        # matched_on_address = True
        # -----------------------------------------------------

        combined = (
            combined
            .groupby(
                [
                    "source1_entity_id",
                    "candidate_entity_id",
                ],
                as_index=False,
            )
            .agg(
                matched_on_name=("matched_on_name", "max"),
                matched_on_address=("matched_on_address", "max"),
            )
        )

        combined["candidate_source"] = candidate_source_name

        all_results.append(combined)

    # ---------------------------------------------------------
    # 5. Combine S2 and S3 candidates
    # ---------------------------------------------------------

    candidates = pd.concat(
        all_results,
        ignore_index=True,
    )

    # ---------------------------------------------------------
    # 6. Final safety deduplication
    # ---------------------------------------------------------

    candidates = candidates.drop_duplicates(
        subset=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
        ]
    )

    # ---------------------------------------------------------
    # 7. Clean column order
    # ---------------------------------------------------------

    candidates = candidates[
        [
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "matched_on_name",
            "matched_on_address",
        ]
    ]

    return candidates.reset_index(drop=True)


def evaluate_recall(
    candidates,
    ground_truth,
):
    """
    Evaluate whether every known ground-truth match appears
    in the generated candidate set.

    ground_truth must contain:

        source1_entity_id
        matched_entity_ids

    where matched_entity_ids is a comma-separated list.
    """

    candidate_lookup = {}

    for _, row in candidates.iterrows():
        s1_id = row["source1_entity_id"]
        candidate_id = row["candidate_entity_id"]

        if s1_id not in candidate_lookup:
            candidate_lookup[s1_id] = set()

        candidate_lookup[s1_id].add(candidate_id)

    total_matches = 0
    found_matches = 0

    for _, row in ground_truth.iterrows():

        s1_id = row["source1_entity_id"]
        matched_ids = row["matched_entity_ids"]

        if pd.isna(matched_ids):
            continue

        matched_ids = {
            x.strip()
            for x in str(matched_ids).split(",")
            if x.strip()
        }

        total_matches += len(matched_ids)

        generated_ids = candidate_lookup.get(
            s1_id,
            set(),
        )

        found_matches += len(
            matched_ids & generated_ids
        )

    if total_matches == 0:
        return 0.0

    return found_matches / total_matches