"""Tests for dataset JSON input and output."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.dataset_io import load_dataset, save_json
from src.schemas import RagDataset, UnansweredQuestion


class DatasetIoTest(unittest.TestCase):
    """Check dataset loading and saving."""

    def create_dataset(self) -> RagDataset:
        """Create a small dataset for testing."""
        return RagDataset(
            rag_questions=[
                UnansweredQuestion(
                    question_id="question-1",
                    question="How does indexing work?",
                )
            ]
        )

    def test_save_and_load_dataset(self) -> None:
        """Save a dataset and load the same validated data."""
        with TemporaryDirectory() as folder:
            path = Path(folder) / "nested" / "dataset.json"
            dataset = self.create_dataset()

            self.assertTrue(save_json(dataset, str(path)))
            self.assertEqual(load_dataset(str(path)), dataset)

    def test_load_dataset_returns_none_for_invalid_files(self) -> None:
        """Return None for missing and malformed dataset files."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            invalid_path = root / "invalid.json"
            invalid_path.write_text("not valid JSON", encoding="utf-8")

            self.assertIsNone(load_dataset(str(root / "missing.json")))
            self.assertIsNone(load_dataset(str(invalid_path)))
