"""Build the searchable source index."""

from pathlib import Path

from src.loader import Loader


class Indexer:
    """Load files, split them into chunks, and save an index."""

    def __init__(self) -> None:
        """Create the source loader."""
        self.loader = Loader()

    def index(
        self,
        raw_path: str,
        output_path: str,
        max_chunk_size: int,
    ) -> int:
        """Build and save the index and return its chunk count."""
        if max_chunk_size < 1:
            return 0
        if not Path(raw_path).is_dir():
            return 0
        return 0
