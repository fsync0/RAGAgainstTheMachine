"""Tests for source loading."""

from pathlib import Path
from tempfile import TemporaryDirectory
import unittest

from src.loader import Loader


class LoaderTest(unittest.TestCase):
    """Check file discovery and reading."""

    def write_file(self, file_path: Path, text: str) -> None:
        """Create a temporary test file."""
        file_path.parent.mkdir(parents=True, exist_ok=True)
        file_path.write_text(text, encoding="utf-8")

    def test_find_files_returns_sorted_supported_files(self) -> None:
        """Find Python and text files but skip other suffixes."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_file(root / "b.py", "")
            self.write_file(root / "a.md", "")
            self.write_file(root / "nested" / "c.TXT", "")
            self.write_file(root / "skip.json", "{}")

            files = Loader().find_files(str(root))

            self.assertEqual(
                files,
                [root / "a.md", root / "b.py", root / "nested" / "c.TXT"],
            )

    def test_read_file_returns_source_file(self) -> None:
        """Read one file into a source record."""
        with TemporaryDirectory() as folder:
            file_path = Path(folder) / "hello.py"
            self.write_file(file_path, "print('hello')")

            source = Loader().read_file(file_path)

            self.assertIsNotNone(source)
            assert source is not None
            self.assertEqual(source.file_path, str(file_path))
            self.assertEqual(source.text, "print('hello')")

    def test_read_file_returns_none_for_missing_file(self) -> None:
        """Return None when a file cannot be opened."""
        file_path = Path("missing-file.py")

        self.assertIsNone(Loader().read_file(file_path))

    def test_load_reads_all_supported_files(self) -> None:
        """Load every supported file in sorted order."""
        with TemporaryDirectory() as folder:
            root = Path(folder)
            self.write_file(root / "b.py", "second")
            self.write_file(root / "a.md", "first")
            self.write_file(root / "skip.json", "{}")

            sources = Loader().load(str(root))

            self.assertEqual([source.text for source in sources], ["first", "second"])
