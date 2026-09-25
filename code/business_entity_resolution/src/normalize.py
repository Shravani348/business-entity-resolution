import re
import pandas as pd

def clean_text(text):
    if pd.isna(text) or text is None:
        return ""
    text = str(text).lower()
    # Keep only alphanumeric and spaces
    text = re.sub(r'[^a-z0-9\s]', ' ', text)
    # Remove extra spaces
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def remove_suffixes(name):
    if not name:
        return ""
    # Suffixes padded with word boundaries
    suffixes = [
        r'\binc\b', r'\bincorporated\b',
        r'\bllc\b', r'\bl l c\b',
        r'\bcorp\b', r'\bcorporation\b',
        r'\bltd\b', r'\blimited\b',
        r'\bpvt\b', r'\bprivate\b',
        r'\bllp\b', r'\bco\b', r'\bcompany\b'
    ]
    pattern = '|'.join(suffixes)
    # Remove suffixes
    cleaned = re.sub(pattern, '', name)
    # Cleanup extra spaces left behind
    return re.sub(r'\s+', ' ', cleaned).strip()

def normalize_name(name):
    return remove_suffixes(clean_text(name))

def normalize_address(address):
    return clean_text(address)

def is_non_latin(text):
    if not text:
        return False
    # Check if text contains non-ascii characters (heuristically non-Latin for this dataset)
    # A more precise check for non-Latin scripts (Tamil, Hindi, etc.):
    return bool(re.search(r'[^\x00-\x7F]', text))
