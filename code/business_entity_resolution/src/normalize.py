import re
import unicodedata

import pandas as pd


MIN_TOKEN_LEN = 3


NAME_REPLACEMENTS = {
    "pvt": "private",
    "ltd": "limited",
    "corp": "corporation",
    "co": "company",
    "inc": "incorporated",
}


ADDRESS_REPLACEMENTS = {
    "rd": "road",
    "st": "street",
    "ave": "avenue",
    "av": "avenue",
    "blvd": "boulevard",
    "ln": "lane",
    "dr": "drive",
    "hwy": "highway",
}


def normalize_text(text):
    """
    Lowercase, Unicode-normalize, remove punctuation,
    and collapse whitespace.
    """
    if pd.isna(text):
        return ""

    text = str(text).strip().lower()
    text = unicodedata.normalize("NFKC", text)
    text = re.sub(r"[^\w\s]", " ", text, flags=re.UNICODE)
    text = re.sub(r"\s+", " ", text).strip()

    return text


def normalize_name(text):
    """Normalize business name."""
    text = normalize_text(text)
    words = [NAME_REPLACEMENTS.get(w, w) for w in text.split()]
    return " ".join(words)


def normalize_address(text):
    """Normalize business address."""
    text = normalize_text(text)
    words = [ADDRESS_REPLACEMENTS.get(w, w) for w in text.split()]
    return " ".join(words)


def normalize_country(text):
    """Normalize country text."""
    if pd.isna(text):
        return ""

    return str(text).strip().lower()


def add_normalized_columns(
    df,
    name_col="business_name",
    address_col="business_address",
    country_col="country",
):
    """
    Return a copy of the dataframe with normalized columns.

    Existing normalized columns are reused if already present.
    """
    df = df.copy()

    if "name_norm" not in df.columns:
        df["name_norm"] = df[name_col].apply(normalize_name)

    if "address_norm" not in df.columns:
        df["address_norm"] = df[address_col].apply(normalize_address)

    if "country_norm" not in df.columns:
        df["country_norm"] = df[country_col].apply(normalize_country)

    return df


def tokenize(text, min_len=MIN_TOKEN_LEN):
    """Return tokens with length >= min_len."""
    if not text:
        return []

    return [token for token in str(text).split() if len(token) >= min_len]


def get_name_tokens(text):
    return tokenize(text)


def get_address_tokens(text):
    return tokenize(text)