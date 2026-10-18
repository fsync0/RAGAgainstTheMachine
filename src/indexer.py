"""Build the searchable source index."""

from pathlib import Path

from src.chunking.chunker import Chunker
from src.loader import Loader
from src.retrieval.storage import Storage


class Indexer:
    """Load files, split them into chunks, and save an index."""

    def __init__(self) -> None:
        """Create the indexing services."""
        self.loader = Loader()
        self.chunker = Chunker()
        self.storage = Storage()

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

        source_files = self.loader.load(raw_path)
        if not source_files:
            return 0

        chunks = self.chunker.chunk(source_files, max_chunk_size)
        if not chunks:
            return 0

        bm25 = self.storage.build(chunks)
        if bm25 is None:
            return 0
        if not self.storage.save(chunks, bm25, output_path):
            return 0
        return len(chunks)
