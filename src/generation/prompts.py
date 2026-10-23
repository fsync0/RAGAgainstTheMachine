"""Build grounded prompts for answer generation."""

from src.schemas import SourceChunk


def build_prompt(question: str, sources: list[SourceChunk]) -> str:
    """Build a prompt from a question and retrieved source chunks."""
    context = "\n\n".join(
        (
            f"File: {source.file_path} "
            f"[{source.first_character_index}:"
            f"{source.last_character_index}]\n"
            f"{source.text}"
        )
        for source in sources
    )
    if not context:
        context = "No sources were retrieved."

    return (
        "Answer the question using only the provided sources. "
        "If the sources do not contain the answer, say so.\n\n"
        f"Sources:\n{context}\n\n"
        f"Question: {question}\n"
        "Answer:"
    )
