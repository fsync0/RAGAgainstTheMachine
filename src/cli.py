"""Command-line interface for the RAG application."""

import fire

from src.indexer import Indexer
from src.retrieval.retriever import Retriever


class RagCli:
    """Commands for indexing, searching, and answering questions."""

    @staticmethod
    def _get_int(value: str) -> int | None:
        """Convert a user value to a positive integer when possible."""
        try:
            number = int(value)
        except (TypeError, ValueError):
            return None
        if number < 1:
            return None
        return number

    def index(
        self,
        max_chunk_size: str = "2000",
        raw_path: str = "data/raw",
        output_path: str = "data/processed",
    ) -> str:
        """Build an index from source files."""
        chunk_size = self._get_int(max_chunk_size)
        if chunk_size is None:
            return "max_chunk_size must be a positive integer."
        chunk_size = min(chunk_size, 2000)
        indexed_chunk_count = Indexer().index(
            raw_path,
            output_path,
            chunk_size,
        )
        if indexed_chunk_count == 0:
            return "Indexing failed or no supported files were found."
        return (
            "Ingestion complete! "
            f"Indexed {indexed_chunk_count} chunks under {output_path}/"
        )

    def search(
        self,
        query: str,
        k: str = "5",
        output_path: str = "data/processed",
    ) -> str:
        """Return the top-k sources for a query."""
        result_count = self._get_int(k)
        if result_count is None:
            return "k must be a positive integer."

        sources = Retriever(output_path).search(query, result_count)
        if not sources:
            return "No sources found."
        return "\n".join(
            (
                f"{source.file_path} "
                f"[{source.first_character_index}:"
                f"{source.last_character_index}]"
            )
            for source in sources
        )


def main() -> None:
    """Start the command-line interface."""
    fire.Fire(RagCli)
