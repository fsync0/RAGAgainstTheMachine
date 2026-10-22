"""Load source files from a folder."""

from pathlib import Path

from src.schemas import SourceFile

ALLOWED_SUFFIXES = {".py", ".md", ".txt"}


class Loader:
    """Find and read source files."""

    def find_files(self, folder_path: str) -> list[Path]:
        """Return supported files in a stable order."""
        folder = Path(folder_path)
        if not folder.is_dir():
            return []
        try:
            files = [
                file_path
                for file_path in folder.rglob("*")
                if file_path.is_file()
                and file_path.suffix.lower() in ALLOWED_SUFFIXES
            ]
        except OSError:
            return []
        return sorted(files, key=str)

    def read_file(self, file_path: Path) -> SourceFile | None:
        """Read one file and return its source record."""
        try:
            with file_path.open(
                "r",
                encoding="utf-8",
                errors="replace",
            ) as file:
                text = file.read()
        except OSError:
            return None

        return SourceFile(
            file_path=str(file_path),
            text=text,
        )

    def load(self, folder_path: str) -> list[SourceFile]:
        """Load every supported file in a folder."""
        files = self.find_files(folder_path)
        sources: list[SourceFile] = []

        for file_path in files:
            source = self.read_file(file_path)
            if source is not None:
                sources.append(source)
        return sources
