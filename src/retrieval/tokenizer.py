"""Convert text into tokens for BM25."""

import string

def tokenize(text: str) -> list[str]:
    """Convert text into lowercase tokens."""
    text = text.lower()
    for char in string.punctuation:
        if char != "_":
            text = text.replace(char, " ")

    tokens: list[str] = []
    for token in text.split():
        tokens.append(token)
        if "_" in token:
            tokens.extend(part for part in token.split("_") if part)
    return tokens
