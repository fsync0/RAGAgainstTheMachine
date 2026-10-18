"""Tests for source chunking."""

import unittest

from src.chunking.chunker import Chunker, MAX_CHUNK_SIZE
from src.schemas import SourceChunk, SourceFile


class ChunkerTest(unittest.TestCase):
    """Check source splitting and character indexes."""

    def assert_valid_chunks(
        self,
        source_file: SourceFile,
        chunks: list[SourceChunk],
        max_chunk_size: int,
    ) -> None:
        """Check that chunks match their original source text."""
        self.assertTrue(chunks)
        for chunk in chunks:
            self.assertEqual(chunk.file_path, source_file.file_path)
            self.assertEqual(
                source_file.text[
                    chunk.first_character_index:chunk.last_character_index
                ],
                chunk.text,
            )
            self.assertLessEqual(len(chunk.text), max_chunk_size)

    def test_chunk_splits_python_and_markdown(self) -> None:
        """Use the correct strategy for Python and Markdown files."""
        python_file = SourceFile(
            file_path="demo.py",
            text=(
                "def first():\n"
                "    return 1\n\n"
                "def second():\n"
                "    return 2\n"
            ),
        )
        markdown_file = SourceFile(
            file_path="guide.md",
            text=(
                "# First\n\n"
                "First paragraph.\n\n"
                "# Second\n\n"
                "Second paragraph.\n"
            ),
        )

        chunks = Chunker().chunk([python_file, markdown_file], 30)
        python_chunks = [chunk for chunk in chunks if chunk.file_path == "demo.py"]
        markdown_chunks = [
            chunk for chunk in chunks if chunk.file_path == "guide.md"
        ]

        self.assert_valid_chunks(python_file, python_chunks, 30)
        self.assert_valid_chunks(markdown_file, markdown_chunks, 30)

    def test_chunk_returns_empty_list_for_invalid_input(self) -> None:
        """Return no chunks for empty text or an invalid size."""
        empty_file = SourceFile(file_path="empty.py", text="")
        source_file = SourceFile(file_path="demo.py", text="print('hello')")

        self.assertEqual(Chunker().chunk([empty_file], 100), [])
        self.assertEqual(Chunker().chunk([source_file], 0), [])

    def test_chunk_caps_size_at_the_allowed_limit(self) -> None:
        """Never create a chunk larger than the assignment limit."""
        source_file = SourceFile(
            file_path="large.txt",
            text="a" * (MAX_CHUNK_SIZE + 100),
        )

        chunks = Chunker().chunk([source_file], MAX_CHUNK_SIZE + 1)

        self.assert_valid_chunks(source_file, chunks, MAX_CHUNK_SIZE)
