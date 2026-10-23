"""Measure retrieval recall against answered questions."""

import json
from pathlib import Path

from pydantic import ValidationError

from src.dataset_io import load_dataset
from src.schemas import (
    AnsweredQuestion,
    MinimalSource,
    RagDataset,
    StudentSearchResults,
)

K_VALUES = [1, 3, 5, 10]


def load_search_results(path: str) -> StudentSearchResults | None:
    """Load and validate student search results."""
    try:
        with Path(path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        return StudentSearchResults.model_validate(data)
    except (OSError, TypeError, ValidationError, json.JSONDecodeError):
        return None


def overlap_score(first: MinimalSource, second: MinimalSource) -> float:
    """Return the intersection over union of two source ranges."""
    start = max(first.first_character_index, second.first_character_index)
    end = min(first.last_character_index, second.last_character_index)
    intersection = max(0, end - start)

    start = min(first.first_character_index, second.first_character_index)
    end = max(first.last_character_index, second.last_character_index)
    union = end - start
    if union < 1:
        return 0.0
    return intersection / union


def source_matches(first: MinimalSource, second: MinimalSource) -> bool:
    """Return whether two sources point to the same content."""
    return (
        first.file_path == second.file_path
        and overlap_score(first, second) > 0.05
    )


def recall_at_k(
    student_results: StudentSearchResults,
    dataset: RagDataset,
    k: int,
) -> float:
    """Calculate average source recall at k."""
    if k < 1:
        return 0.0

    results_by_id = {
        result.question_id: result
        for result in student_results.search_results
    }
    recalls: list[float] = []

    for question in dataset.rag_questions:
        if not isinstance(question, AnsweredQuestion) or not question.sources:
            continue

        result = results_by_id.get(question.question_id)
        sources = [] if result is None else result.retrieved_sources[:k]
        found_count = sum(
            any(source_matches(source, expected) for source in sources)
            for expected in question.sources
        )
        recalls.append(found_count / len(question.sources))

    if not recalls:
        return 0.0
    return sum(recalls) / len(recalls)


def evaluate_results(
    student_path: str,
    dataset_path: str,
) -> dict[int, float] | None:
    """Return recall scores for the standard k values."""
    student_results = load_search_results(student_path)
    dataset = load_dataset(dataset_path)
    if student_results is None or dataset is None:
        return None
    return {
        k: recall_at_k(student_results, dataset, k)
        for k in K_VALUES
    }
