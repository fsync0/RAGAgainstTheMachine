"""Read and write RAG dataset JSON files."""

import json
from pathlib import Path

from pydantic import BaseModel, ValidationError

from src.schemas import RagDataset


def load_dataset(dataset_path: str) -> RagDataset | None:
    """Load and validate a RAG dataset JSON file."""
    try:
        with Path(dataset_path).open("r", encoding="utf-8") as file:
            data = json.load(file)
        return RagDataset.model_validate(data)
    except (OSError, TypeError, ValidationError, json.JSONDecodeError):
        return None


def save_json(data: BaseModel, output_path: str) -> bool:
    """Save a Pydantic model as a JSON file."""
    try:
        path = Path(output_path)
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8") as file:
            file.write(data.model_dump_json(indent=2))
    except (OSError, TypeError, ValueError):
        return False
    return True
