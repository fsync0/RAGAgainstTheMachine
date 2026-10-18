"""Data records shared by the application."""

from typing import List, Union
from uuid import uuid4

from pydantic import BaseModel, Field


class SourceFile(BaseModel):
    """Original source text and its project-relative path."""

    file_path: str
    text: str


class SourceChunk(BaseModel):
    """A chunk of text and its location in a source file."""

    file_path: str
    text: str
    first_character_index: int
    last_character_index: int


class MinimalSource(BaseModel):
    """A location in a source file."""

    file_path: str
    first_character_index: int
    last_character_index: int


class UnansweredQuestion(BaseModel):
    """A question waiting for an answer."""

    question_id: str = Field(default_factory=lambda: str(uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    """A question with its sources and generated answer."""

    sources: List[MinimalSource]
    answer: str


class RagDataset(BaseModel):
    """A group of answered or unanswered questions."""

    rag_questions: List[Union[AnsweredQuestion, UnansweredQuestion]]


class MinimalSearchResults(BaseModel):
    """The retrieved sources for one question."""

    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    """The retrieved sources and answer for one question."""

    answer: str


class StudentSearchResults(BaseModel):
    """Search results written for a dataset."""

    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    """Search results with answers written for a dataset."""

    search_results: List[MinimalAnswer]
    k: int
