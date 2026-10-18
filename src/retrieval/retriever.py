"""Retrieve relevant source chunks with BM25."""

from rank_bm25 import BM25Okapi

from src.retrieval.storage import Storage
from src.retrieval.tokenizer import tokenize
from src.schemas import MinimalSource, SourceChunk


class Retriever:
    """Search a stored BM25 index."""

    def __init__(self, output_path: str) -> None:
        """Load the stored chunks and BM25 index."""
        self.chunks: list[SourceChunk] = []
        self.bm25: BM25Okapi | None = None

        data = Storage().load(output_path)
        if data is not None:
            self.chunks, self.bm25 = data

    def search(self, query: str, k: int) -> list[MinimalSource]:
        """Return the top-k sources for a query."""
        if not query.strip() or k < 1 or self.bm25 is None:
            return []

        tokens = tokenize(query)
        if not tokens:
            return []

        scores = self.bm25.get_scores(tokens)
        indexes = sorted(
            range(len(scores)),
            key=lambda index: (-scores[index], index),
        )[:k]
        return [
            MinimalSource(
                file_path=self.chunks[index].file_path,
                first_character_index=self.chunks[index].first_character_index,
                last_character_index=self.chunks[index].last_character_index,
            )
            for index in indexes
        ]
