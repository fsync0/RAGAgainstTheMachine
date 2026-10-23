"""Command-line interface for the RAG application."""

from pathlib import Path

import fire
from tqdm import tqdm

from src.dataset_io import load_dataset, save_json
from src.evaluation import evaluate_results
from src.generation.generator import Generator
from src.indexer import Indexer
from src.retrieval.retriever import Retriever
from src.schemas import MinimalSearchResults, StudentSearchResults


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

    def search_dataset(
        self,
        dataset_path: str,
        k: str = "5",
        save_directory: str = "data/output/search_results",
        index_path: str = "data/processed",
    ) -> str:
        """Search every question in a dataset."""
        result_count = self._get_int(k)
        if result_count is None:
            return "k must be a positive integer."

        dataset = load_dataset(dataset_path)
        if dataset is None:
            return "Could not load the dataset."

        retriever = Retriever(index_path)
        search_results: list[MinimalSearchResults] = []
        for question in tqdm(
            dataset.rag_questions,
            desc="Searching",
            unit="question",
        ):
            sources = retriever.search(question.question, result_count)
            search_results.append(
                MinimalSearchResults(
                    question_id=question.question_id,
                    question=question.question,
                    retrieved_sources=sources,
                )
            )

        results = StudentSearchResults(
            search_results=search_results,
            k=result_count,
        )
        output_path = Path(save_directory) / Path(dataset_path).name
        if not save_json(results, str(output_path)):
            return "Could not save the search results."
        return f"Saved student_search_results to {output_path}"

    def answer(
        self,
        query: str,
        k: str = "5",
        index_path: str = "data/processed",
        max_new_tokens: str = "256",
    ) -> str:
        """Generate an answer from retrieved source chunks."""
        result_count = self._get_int(k)
        if result_count is None:
            return "k must be a positive integer."

        token_count = self._get_int(max_new_tokens)
        if token_count is None:
            return "max_new_tokens must be a positive integer."

        chunks = Retriever(index_path).search_chunks(query, result_count)
        return Generator().answer(query, chunks, token_count)

    def evaluate(
        self,
        student_search_results_path: str,
        dataset_path: str,
    ) -> str:
        """Report recall scores for student search results."""
        scores = evaluate_results(
            student_search_results_path,
            dataset_path,
        )
        if scores is None:
            return "Could not load the evaluation files."
        return "\n".join(
            f"Recall@{k}: {score:.3f} ({score:.1%})"
            for k, score in scores.items()
        )


def main() -> None:
    """Start the command-line interface."""
    fire.Fire(RagCli)
