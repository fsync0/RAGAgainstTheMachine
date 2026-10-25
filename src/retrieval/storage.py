"""Build and save the BM25 index."""

from pathlib import Path
import pickle

from rank_bm25 import BM25Okapi
from tqdm import tqdm

from src.retrieval.tokenizer import tokenize
from src.schemas import SourceChunk

INDEX_FILE = "bm25.pkl"


class Storage:
    """Build, save, and load a BM25 index."""

    def build(self, chunks: list[SourceChunk]) -> BM25Okapi | None:
        """Build a BM25 index from source chunks."""
        if not chunks:
            return None

        tokens = [
            tokenize(chunk.file_path) + tokenize(chunk.text)
            for chunk in tqdm(chunks, desc="Tokenizing", unit="chunk")
        ]
        if not any(tokens):
            return None
        return BM25Okapi(tokens)

    def save(
        self,
        chunks: list[SourceChunk],
        bm25: BM25Okapi,
        output_path: str,
    ) -> bool:
        """Save chunks and the BM25 index."""
        try:
            folder = Path(output_path)
            folder.mkdir(parents=True, exist_ok=True)
            data = {
                "chunks": [chunk.model_dump() for chunk in chunks],
                "bm25": bm25,
            }
            with (folder / INDEX_FILE).open("wb") as file:
                pickle.dump(data, file)
        except (OSError, pickle.PickleError, TypeError):
            return False
        return True

    def load(
        self,
        output_path: str,
    ) -> tuple[list[SourceChunk], BM25Okapi] | None:
        """Load chunks and the BM25 index."""
        try:
            with (Path(output_path) / INDEX_FILE).open("rb") as file:
                data = pickle.load(file)
            chunks = [
                SourceChunk.model_validate(item)
                for item in data["chunks"]
            ]
            bm25 = data["bm25"]
        except (
            OSError,
            EOFError,
            KeyError,
            TypeError,
            ValueError,
            pickle.UnpicklingError,
        ):
            return None

        if not isinstance(bm25, BM25Okapi):
            return None
        return chunks, bm25
