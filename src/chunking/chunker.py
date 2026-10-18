"""Split source files into searchable chunks."""

from langchain_text_splitters import Language, RecursiveCharacterTextSplitter

from src.schemas import SourceChunk, SourceFile

MAX_CHUNK_SIZE = 2000


class Chunker:
    """Split Python and text files with language-aware rules."""

    def create_splitter(
        self,
        language: Language,
        max_chunk_size: int,
        keep_separator: bool,
    ) -> RecursiveCharacterTextSplitter:
        """Create a splitter for one language."""
        return RecursiveCharacterTextSplitter.from_language(
            language=language,
            chunk_size=min(max_chunk_size, MAX_CHUNK_SIZE),
            chunk_overlap=0,
            keep_separator=keep_separator,
            add_start_index=True,
            strip_whitespace=False,
        )

    def chunk(
        self,
        source_files: list[SourceFile],
        max_chunk_size: int,
    ) -> list[SourceChunk]:
        """Split every source file into chunks."""
        if max_chunk_size < 1:
            return []

        chunks: list[SourceChunk] = []
        for source_file in source_files:
            if source_file.file_path.endswith(".py"):
                chunks.extend(
                    self.chunk_python(source_file, max_chunk_size)
                )
            else:
                chunks.extend(
                    self.chunk_text(source_file, max_chunk_size)
                )
        return chunks

    def chunk_file(
        self,
        source_file: SourceFile,
        language: Language,
        max_chunk_size: int,
        keep_separator: bool,
    ) -> list[SourceChunk]:
        """Split one file and keep its character indexes."""
        if not source_file.text or max_chunk_size < 1:
            return []

        splitter = self.create_splitter(
            language,
            max_chunk_size,
            keep_separator,
        )
        parts = splitter.create_documents([source_file.text])
        chunks: list[SourceChunk] = []

        for part in parts:
            start_index = int(part.metadata["start_index"])
            end_index = start_index + len(part.page_content)
            if source_file.text[start_index:end_index] != part.page_content:
                return []
            chunks.append(
                SourceChunk(
                    file_path=source_file.file_path,
                    text=part.page_content,
                    first_character_index=start_index,
                    last_character_index=end_index,
                )
            )
        return chunks

    def chunk_python(
        self,
        source_file: SourceFile,
        max_chunk_size: int,
    ) -> list[SourceChunk]:
        """Split one Python file."""
        return self.chunk_file(
            source_file,
            Language.PYTHON,
            max_chunk_size,
            True,
        )

    def chunk_text(
        self,
        source_file: SourceFile,
        max_chunk_size: int,
    ) -> list[SourceChunk]:
        """Split one Markdown or text file."""
        return self.chunk_file(
            source_file,
            Language.MARKDOWN,
            max_chunk_size,
            True,
        )
