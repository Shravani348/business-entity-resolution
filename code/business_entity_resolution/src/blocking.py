from collections import defaultdict


class TokenBlocker:
    def __init__(self):
        # Maps (country, token) -> set of entity_ids
        self.name_index = defaultdict(set)
        self.address_index = defaultdict(set)

        self.stop_words = {
            'the', 'and', 'of', 'for',
            'in', 'a', 'an', 'on'
        }

    def tokenize(self, text):
        if not text:
            return set()

        tokens = text.split()

        # Remove small stop words or extremely short meaningless tokens
        return {
            t for t in tokens
            if len(t) > 1 and t not in self.stop_words
        }

    def index_record(
        self,
        entity_id,
        country,
        norm_name,
        norm_address
    ):
        """Index a single Source 2 or Source 3 record."""

        if not country:
            country = "UNKNOWN"

        c_key = str(country).lower().strip()

        # Index name tokens
        name_tokens = self.tokenize(norm_name)

        for token in name_tokens:
            self.name_index[(c_key, token)].add(entity_id)

        # Index address tokens
        address_tokens = self.tokenize(norm_address)

        for token in address_tokens:
            self.address_index[(c_key, token)].add(entity_id)

    def get_candidates(
        self,
        country,
        norm_name,
        norm_address
    ):
        """Retrieve candidates for a given Source 1 record."""

        if not country:
            country = "UNKNOWN"

        c_key = str(country).lower().strip()

        # ---------------------------------------------------------
        # 1. Gather all tokens
        # ---------------------------------------------------------
        name_tokens = self.tokenize(norm_name)
        address_tokens = self.tokenize(norm_address)

        all_keys = []

        for token in name_tokens:
            all_keys.append(
                ('name', (c_key, token))
            )

        for token in address_tokens:
            all_keys.append(
                ('address', (c_key, token))
            )

        # ---------------------------------------------------------
        # 2. Document Frequency Filter
        # ---------------------------------------------------------
        # Experimental safe version:
        # Allow buckets up to 50,000 records.
        #
        # This is intentionally higher than the original 10,000
        # threshold so that useful common tokens are not discarded.
        # ---------------------------------------------------------
        MAX_FREQ = 50000

        valid_buckets = []

        for t_type, key in all_keys:

            if t_type == 'name' and key in self.name_index:
                bucket = self.name_index[key]

                if len(bucket) <= MAX_FREQ:
                    valid_buckets.append(bucket)

            elif t_type == 'address' and key in self.address_index:
                bucket = self.address_index[key]

                if len(bucket) <= MAX_FREQ:
                    valid_buckets.append(bucket)

        # ---------------------------------------------------------
        # 3. Build candidate set
        # ---------------------------------------------------------
        candidates = set()

        if len(valid_buckets) == 0:
            return candidates

        elif len(valid_buckets) == 1:
            candidates.update(valid_buckets[0])

        else:
            from collections import Counter

            counts = Counter()

            for bucket in valid_buckets:
                for entity_id in bucket:
                    counts[entity_id] += 1

            # Experimental safe rule:
            # A candidate only needs to appear in ONE usable bucket.
            #
            # This is deliberately permissive for the A/B test.
            candidates = {
                entity_id
                for entity_id, count in counts.items()
                if count >= 1
            }

        return candidates