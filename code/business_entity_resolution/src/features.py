from typing import Dict, Optional

from rapidfuzz.fuzz import ratio
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from normalize import normalize_name, normalize_address


# This will be configured once using the prototype dataset.
_tfidf_vectorizer: Optional[TfidfVectorizer] = None


def configure_tfidf(names):
    """
    Fit the TF-IDF vectorizer on the names from the prototype dataset.
    """
    global _tfidf_vectorizer

    _tfidf_vectorizer = TfidfVectorizer(
        analyzer="word",
        lowercase=False,
        token_pattern=r"(?u)\b\w+\b"
    )

    _tfidf_vectorizer.fit(names)


def token_jaccard(text1, text2):
    """
    Calculate Jaccard similarity between the tokens of two strings.
    """
    tokens1 = set(text1.split())
    tokens2 = set(text2.split())

    if not tokens1 and not tokens2:
        return 1.0

    if not tokens1 or not tokens2:
        return 0.0

    return len(tokens1 & tokens2) / len(tokens1 | tokens2)


def levenshtein_similarity(text1, text2):
    """
    Return normalized Levenshtein-style similarity from 0 to 1.
    RapidFuzz ratio is based on edit similarity.
    """
    if not text1 and not text2:
        return 1.0

    if not text1 or not text2:
        return 0.0

    return ratio(text1, text2) / 100.0


def tfidf_cosine(text1, text2):
    """
    Calculate cosine similarity between two normalized names
    using the globally configured TF-IDF vectorizer.
    """
    if _tfidf_vectorizer is None:
        raise RuntimeError(
            "TF-IDF vectorizer is not configured. "
            "Call configure_tfidf() before extracting features."
        )

    vectors = _tfidf_vectorizer.transform([text1, text2])

    return float(cosine_similarity(vectors[0], vectors[1])[0][0])


def extract_features(
    name1,
    address1,
    name2,
    address2,
    country
) -> Dict[str, float]:
    """
    Extract similarity features for one Source1 vs Source2/3 pair.
    """

    # Reuse the project's existing normalization functions.
    normalized_name1 = normalize_name(name1)
    normalized_name2 = normalize_name(name2)

    normalized_address1 = normalize_address(address1)
    normalized_address2 = normalize_address(address2)

    features = {
        "name_token_jaccard": token_jaccard(
            normalized_name1,
            normalized_name2
        ),

        "name_levenshtein": levenshtein_similarity(
            normalized_name1,
            normalized_name2
        ),

        "name_tfidf_cosine": tfidf_cosine(
            normalized_name1,
            normalized_name2
        ),

        "address_token_jaccard": token_jaccard(
            normalized_address1,
            normalized_address2
        ),

        "address_levenshtein": levenshtein_similarity(
            normalized_address1,
            normalized_address2
        ),

        "exact_name_match": int(
            normalized_name1 == normalized_name2
            and normalized_name1 != ""
        ),

        "missing_address": int(
            normalized_address1 == "" or normalized_address2 == ""
        ),

        "country_match": int(
            str(country).strip().lower() != ""
        ),
    }

    return features