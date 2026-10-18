"""Convert text into tokens for BM25."""

import string

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase tokens."""
    text = text.lower()
    for char in string.punctuation:
        if char != "_":
            text = text.replace(char, " ")
    return text.split()
