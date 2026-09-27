import pandas as pd

from normalize import (
    add_normalized_columns,
    get_name_tokens,
    get_address_tokens,
)
from blocking import TokenBlocker


def generate_candidates(
    source1,
    source2,
    source3,
    max_block_size=10000,
):
    """
    Generate candidate pairs from Source 1 against Source 2 and Source 3.

    Blocking strategy:
        country + (shared name token OR shared address token)
    """
    # 1. Normalize all sources
    source1 = add_normalized_columns(source1)
    source2 = add_normalized_columns(source2)
    source3 = add_normalized_columns(source3)

    source1["name_tokens"] = source1["name_norm"].apply(get_name_tokens)
    source1["address_tokens"] = source1["address_norm"].apply(get_address_tokens)

    all_results = []

    # 2. Generate candidates for S1 -> S2 and S1 -> S3
    for candidate_source_name, candidate_source in [
        ("S2", source2),
        ("S3", source3),
    ]:
        if candidate_source.empty:
            continue

        candidate_source["name_tokens"] = candidate_source["name_norm"].apply(get_name_tokens)
        candidate_source["address_tokens"] = candidate_source["address_norm"].apply(get_address_tokens)

        # We instantiate one TokenBlocker but we will hack it to serve our dual purpose
        # by passing max_freq into get_candidates
        name_blocker = TokenBlocker()
        address_blocker = TokenBlocker()

        # TokenBlocker expects strings (norm_name, norm_address) not lists
        for country, name, eid in zip(candidate_source["country_norm"], candidate_source["name_norm"], candidate_source["entity_id"]):
            name_blocker.index_record(eid, country, str(name) if pd.notna(name) else "", "")
            
        for country, addr, eid in zip(candidate_source["country_norm"], candidate_source["address_norm"], candidate_source["entity_id"]):
            address_blocker.index_record(eid, country, "", str(addr) if pd.notna(addr) else "")

        rows = []
        for country, name, addr, s1_id in zip(
            source1["country_norm"], source1["name_norm"], source1["address_norm"], source1["entity_id"]
        ):
            c_str = str(country)
            name_str = str(name) if pd.notna(name) else ""
            addr_str = str(addr) if pd.notna(addr) else ""
            
            # Pass max_freq=10000, min_overlap=1, and top_k=5 to get_candidates
            name_cands = name_blocker.get_candidates(c_str, name_str, "", max_freq=max_block_size, min_overlap=1, top_k=5)
            addr_cands = address_blocker.get_candidates(c_str, "", addr_str, max_freq=max_block_size, min_overlap=1, top_k=5)
            
            all_cands = set(name_cands) | set(addr_cands)
            for c_id in all_cands:
                rows.append({
                    "source1_entity_id": s1_id,
                    "candidate_entity_id": c_id,
                    "candidate_source": candidate_source_name,
                    "matched_on_name": c_id in name_cands,
                    "matched_on_address": c_id in addr_cands
                })

        if rows:
            all_results.append(pd.DataFrame(rows))

    if not all_results:
        return pd.DataFrame(columns=[
            "source1_entity_id",
            "candidate_entity_id",
            "candidate_source",
            "matched_on_name",
            "matched_on_address",
        ])

    candidates = pd.concat(all_results, ignore_index=True)
    return candidates


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