# RAG against the machine

*Will you answer my questions?*

**Summary:** Retrieval-Augmented Generation, that's it. That's the goal of this project.

**Version:** 2.0

---

## Table of Contents

- [Overview](#overview)
- [Introduction](#introduction)
- [Common Instructions](#common-instructions)
  - [General Rules](#general-rules)
  - [Makefile](#makefile)
  - [Additional Guidelines](#additional-guidelines)
  - [Additional Requirements](#additional-requirements)
- [Mandatory Part](#mandatory-part)
  - [Indexing the codebase](#indexing-the-codebase)
  - [Retrieval](#retrieval)
  - [Answer generation](#answer-generation)
  - [Data Models](#data-models)
  - [Output](#output)
  - [Command-Line Interface](#command-line-interface)
  - [End-to-end walkthrough](#end-to-end-walkthrough)
- [Evaluation](#evaluation)
  - [Recall@k calculation](#recallk-calculation)
  - [Performance requirements](#performance-requirements)
- [Bonus Part](#bonus-part)
- [Submission](#submission)

---

## Overview

The pipeline you will build: **ingest and index a codebase**, **retrieve** the most relevant snippets for a question, hand them to a **small local model** to generate a grounded answer, and **measure retrieval quality with recall@k**.

```
Indexing                                    Retrieval and Generation
─────────────────────────────────────       ──────────────────────────────────────────────────────
Corpus ──► Chunking ──► Index ────────────► Retrieve ──► Augment ──► Generate ──► Answer (JSON)
(data/raw/  (code |     (BM25 / TF-IDF,     top-k        (build       (Qwen3-0.6B)
vllm-0.10.1) markdown)  data/processed/)    sources       context)
                                              │
                                              └╌╌ scoring ╌╌► Evaluate: recall@k (IoU > 0.05)
```

---

## Introduction

We met function calling in `call_me_maybe`; this time the topic is RAG. Before looking at what RAG is, focus on what it does.

A model only "knows" what it was trained on, and retraining it to add knowledge is slow and costly. RAG takes another route: instead of putting knowledge into the model, you give it access to an external source of your choosing and let it pull from there at answer time.

In practice, RAG has four stages:

- **Indexing:** organise the data so it can be searched.
- **Retrieving:** match a question against the index and pull the most relevant snippets.
- **Augmenting:** filter those snippets and place them in the model's context window.
- **Generating:** read that context and produce the answer.

> **Note:** Read the whole document before writing any code: each stage feeds the next.

---

## Common Instructions

### General Rules

- Your project must be written in **Python 3.10 or later**.
- Your project must adhere to the **flake8** coding standard.
- Your functions should handle exceptions gracefully to avoid crashes. Use `try-except` blocks to manage potential errors. Prefer context managers for resources like files or connections to ensure automatic cleanup. **If your program crashes due to unhandled exceptions during the review, it will be considered non-functional.**
- All resources (e.g., file handles, network connections) must be properly managed to prevent leaks. Use context managers where possible for automatic handling.
- Your code must include **type hints** for function parameters, return types, and variables where applicable (using the `typing` module). Use `mypy` for static type checking. All functions must pass mypy without errors.
- Include **docstrings** in functions and classes following PEP 257 (e.g., Google or NumPy style) to document purpose, parameters, and returns.

### Makefile

Include a `Makefile` in your project to automate common tasks. It must contain the following rules (mandatory `lint` implies the specified flags; it is strongly recommended to try `--strict` for enhanced checking):

| Rule | Description |
|------|-------------|
| `install` | Install project dependencies using `pip`, `uv`, `pipx`, or any other package manager of your choice. |
| `run` | Execute the main script of your project (e.g., via Python interpreter). |
| `debug` | Run the main script in debug mode using Python's built-in debugger (e.g., `pdb`). |
| `clean` | Remove temporary files or caches (e.g., `__pycache__`, `.mypy_cache`) to keep the project environment clean. |
| `lint` | Execute the commands `flake8 .` and `mypy . --warn-return-any --warn-unused-ignores --ignore-missing-imports --disallow-untyped-defs --check-untyped-defs` |
| `lint-strict` (optional) | Execute the commands `flake8 .` and `mypy . --strict` |

### Additional Guidelines

- Create test programs to verify project functionality (not submitted or graded). Use frameworks like `pytest` or `unittest` for unit tests, covering edge cases.
- Include a `.gitignore` file to exclude Python artifacts.
- It is recommended to use virtual environments (e.g., venv or conda) for dependency isolation during development.

### Additional Requirements

In addition to the common Python rules above, this project requires:

- Your **data models** must use `pydantic` for validation and type safety: the structures you exchange between stages, such as those in the [Data Models](#data-models) section. Service or orchestration classes (indexer, retriever, pipeline, etc.) do not have to be pydantic.
- The **flake8** standard applies to bonus files as well.
- You must use `uv` as your project and package manager: the reviewer and the moulinette only run `uv sync`.
- Your system must expose a Command-Line Interface (CLI) built with **Python Fire**, and progress bars (`tqdm`) for long-running operations.
- You need to use the following model:
  - **`Qwen/Qwen3-0.6B`** (default)
  - You may use other models as long as everything still works with `Qwen/Qwen3-0.6B`.
- Beyond the tools required above, you may use any library you like.

---

## Mandatory Part

You will build a **Retrieval-Augmented Generation system** that answers questions about a codebase. You ingest the provided vLLM repository into a searchable index, retrieve the most relevant snippets for a question, generate an answer from them with `Qwen/Qwen3-0.6B`, and measure retrieval quality with recall@k. Your system is judged on whether it retrieves the right source locations and produces answers grounded in them.

### Indexing the codebase

Everything starts with the index. Read the files you judge useful from the vLLM repository shipped in the attachments, split each one into chunks, and persist an index that retrieval can query in milliseconds. Indexing the whole corpus must take **at most 5 minutes**.

A Python file and a Markdown page do not break apart the same way, so your program must implement **two distinct chunking strategies**:

- Python code chunking,
- Markdown / text chunking.

For retrieval itself, implement **at least one** of the two classic lexical methods. The choice is yours:

- TF-IDF,
- BM25.

You may explore other methods on top, as long as one of these two is implemented.

> **Warning:** Chunk size is configurable through a CLI argument (`--max_chunk_size`), with a default of **2000 characters**. Do not go above it: the moulinette rejects any retrieved source longer than 2000 characters (its `max_context_length`), and a single over-long source makes your whole output invalid. Smaller chunks are fine; report the effect on your recall@k.

Indexing a target corpus: the indexer chunks every file and persists the index under `data/processed/`.

```
$ uv run python -m src index --max_chunk_size 2000
Chunking:   100%|##########| 1965/1965 [00:00<00:00, 16710 file/s]
Tokenizing: 100%|##########| 13466/13466 [00:01<00:00, 10668 chunk/s]
Ingestion complete! Indexed 13466 chunks under data/processed/
```

> **Hint:** Questions and source code rarely use the same words: a question may paraphrase an idea or quote an identifier verbatim. What you keep at indexing time decides which of the two you can still match.

### Retrieval

With the index built, you can search it. Given a question, your system returns the **top-k** most relevant snippets. Each result is a source location: a `file_path` and the character range (`first_character_index`, `last_character_index`) it covers, at most **2000 characters** wide.

> **Warning:** `file_path` must match the corpus path exactly (e.g. `data/raw/vllm-0.10.1/docs/features/lora.md`). The grader compares paths verbatim, so a path with a different prefix never matches.

Retrieval must work for a single query and in batch over a whole dataset of questions read from JSON. On the reference datasets, your system must reach at least **80 % recall@5 on docs questions** and **50 % on code questions** (the metric is defined in the [Evaluation](#evaluation) chapter).

A single-query search returns ranked source locations, each with its file path and character span:

```
$ uv run python -m src search "How to configure the OpenAI server?" --k 5
data/raw/vllm-0.10.1/examples/online_serving/openai_transcription_client.py [0:1352]
data/raw/vllm-0.10.1/examples/online_serving/prompt_embed_inference_with_openai_client.py [0:932]
data/raw/vllm-0.10.1/docs/deployment/frameworks/dstack.md [1936:3170]
data/raw/vllm-0.10.1/docs/models/supported_models.md [1699:3396]
data/raw/vllm-0.10.1/examples/online_serving/openai_chat_completion_client_with_tools.py [0:2000]
```

### Answer generation

With the right snippets retrieved, the system generates a natural-language answer using `Qwen/Qwen3-0.6B`. Pass the retrieved context to the model within its token budget, and produce structured JSON following the provided pydantic models.

A satisfactory answer is:

- **Coherent** and understandable,
- **Grounded** in the retrieved sources, with no major hallucination,
- **On point:** it answers the question actually asked.

> **Note:** The mandatory `Qwen/Qwen3-0.6B` model has known reasoning limits. These criteria describe the target an answer should aim for, not a strict pass/fail bar: grading prioritizes retrieval quality, grounding and prompt strategy over perfect final phrasing. A valid answer is coherent and mostly grounded in the retrieved documents, even if partially incomplete due to base-model limitations.

### Data Models

Implement the following `pydantic` models; they validate the data exchanged between the stages.

The **`MinimalSource`** model represents a single source of information:

```python
class MinimalSource(BaseModel):
    file_path: str
    first_character_index: int
    last_character_index: int
```

The **`UnansweredQuestion`** and **`AnsweredQuestion`** models represent an unanswered question and an answered question:

```python
class UnansweredQuestion(BaseModel):
    question_id: str = Field(default_factory=lambda: str(uuid.uuid4()))
    question: str


class AnsweredQuestion(UnansweredQuestion):
    sources: List[MinimalSource]
    answer: str
```

The **`RagDataset`** model represents a dataset of RAG questions:

```python
class RagDataset(BaseModel):
    rag_questions: List[AnsweredQuestion | UnansweredQuestion]
```

The **`MinimalSearchResults`** and **`MinimalAnswer`** models represent the search results and an answer:

```python
class MinimalSearchResults(BaseModel):
    question_id: str
    question: str
    retrieved_sources: List[MinimalSource]


class MinimalAnswer(MinimalSearchResults):
    answer: str
```

The **`StudentSearchResults`** and **`StudentSearchResultsAndAnswer`** models represent search results and search results with answers:

```python
class StudentSearchResults(BaseModel):
    search_results: List[MinimalSearchResults]
    k: int


class StudentSearchResultsAndAnswer(BaseModel):
    search_results: List[MinimalAnswer]
    k: int
```

The provided models are a foundation. You can expand them by adding new models or extra fields (for example in the search results model) if your implementation requires it.

### Output

Each command writes a JSON file that conforms to the provided pydantic models:

- **For search operations:** use the `StudentSearchResults` model with:
  - `search_results`: list of `MinimalSearchResults` containing `question_id`, `question` and `retrieved_sources`
  - `k`: number of results requested
- **For answer generation:** use the `StudentSearchResultsAndAnswer` model with:
  - `search_results`: list of `MinimalAnswer` containing `question_id`, `question`, `retrieved_sources`, and `answer`
  - `k`: number of results requested
- **Source information:** each `MinimalSource` contains:
  - `file_path`: path to the source file, relative to your project root and written exactly as in the ingested corpus (e.g. `data/raw/vllm-0.10.1/...`); it is compared verbatim to the reference
  - `first_character_index`: starting character position
  - `last_character_index`: ending character position

#### Output format

The output must respect the minimal basis of the provided models but can be enhanced as follows.

**Example: `StudentSearchResults` output**

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How to configure OpenAI server?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100
        },
        {
          "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
          "first_character_index": 267,
          "last_character_index": 400
        }
      ]
    }
  ],
  "k": 10
}
```

For answers, the output should follow the `StudentSearchResultsAndAnswer` model.

**Example: `StudentSearchResultsAndAnswer` output**

```json
{
  "search_results": [
    {
      "question_id": "q1",
      "question": "How to configure OpenAI server?",
      "retrieved_sources": [
        {
          "file_path": "data/raw/vllm-0.10.1/docs/serving/openai_compatible_server.md",
          "first_character_index": 9867,
          "last_character_index": 10100
        },
        {
          "file_path": "data/raw/vllm-0.10.1/vllm/entrypoints/openai/api_server.py",
          "first_character_index": 267,
          "last_character_index": 400
        }
      ],
      "answer": "To configure the OpenAI compatible server in vLLM..."
    }
  ],
  "k": 10
}
```

### Command-Line Interface

Provide a CLI built with **Python Fire**. Every command is invoked as `uv run python -m src <command> [options]`. The following commands are required (options shown are the minimum; you may add your own):

| Command | Description |
|---------|-------------|
| `index --max_chunk_size <int>` | Ingest `data/raw/` and build the index under `data/processed/`. |
| `search <query> --k <int>` | Return the top-`k` sources for a single query. |
| `search_dataset --dataset_path <path> --k <int> --save_directory <dir>` | Run search over a whole dataset and write a `StudentSearchResults` JSON file. |
| `answer <query> --k <int>` | Answer a single query using the retrieved context. |
| `answer_dataset --student_search_results_path <path> --save_directory <dir>` | Generate answers for a dataset, producing a `StudentSearchResultsAndAnswer` JSON file. |
| `evaluate --student_search_results_path <path> --dataset_path <path>` | Report your own recall@k against a ground-truth dataset, for your own testing. |

All input and output paths must be configurable CLI arguments and **never hard-coded**. Include progress bars (`tqdm`) for long-running operations, and handle degenerate inputs (empty query, nonsensical query, `k=0`, missing files, malformed JSON) gracefully. The CLI is tested with such edge cases and **must never crash with an unhandled traceback**.

> **Warning:** The `evaluate` command is for your own iteration. The official recall@k used during the defense is computed by the provided moulinette executable, not by your code. Your solution must never import or call the moulinette.

### End-to-end walkthrough

This section shows the complete workflow, from raw documents to evaluation and answer generation, exactly as the reference exam scripts run it.

#### Expected project layout for evaluation

During the defense, the retrieval pipeline is run end-to-end as a single automated flow (`index` → `search_dataset` → `moulinette evaluate_student_search_results`) using reference exam scripts. The individual commands remain available for debugging, but to make the automated run reproducible without manual restructuring, your repository must respect the following layout and conventions:

| Path | Purpose |
|------|---------|
| `src/` | Your Python module, runnable as `uv run python -m src <command>`. |
| `pyproject.toml` and `uv.lock` | At the repository root, so `uv sync` works from the root. |
| `README.md` | At the repository root. |
| `data/raw/` | The indexed sources (the provided vLLM repository). |
| `data/processed/` | The index produced by the `index` command. |
| `data/datasets/UnansweredQuestions/` and `data/datasets/AnsweredQuestions/` | The question datasets. |
| `data/output/search_results/<DatasetScope>/` | The output of `search_dataset`, scoped by dataset. |
| `data/output/search_results_and_answer/<DatasetScope>/` | The output of `answer_dataset`, scoped by dataset. |

All input and output paths (`--dataset_path`, `--save_directory`, `--student_search_results_path`) must be configurable CLI arguments and must never be hard-coded: the evaluator points them at the datasets and at dedicated output directories. **If the structure or the command interface is not respected, the reference scripts cannot run and the corresponding checks are considered failed by design.**

#### Running the full pipeline

The pipeline is driven by four commands, in order: index the corpus, search a whole dataset, score the results with the moulinette, then generate answers. The `search` and `answer` single-query commands shown earlier behave the same way on one question at a time.

**1. Index the corpus once:**

```bash
uv run python -m src index --max_chunk_size 2000
```

```
Ingestion complete! Indices saved under data/processed/
```

**2. Search a dataset.** Always scope `--save_directory` by dataset (`UnansweredQuestions` or `AnsweredQuestions`): the public datasets share file names, so writing every run into the same folder would overwrite previous results.

```bash
uv run python -m src search_dataset \
    --dataset_path data/datasets/UnansweredQuestions/dataset_docs_public.json \
    --k 10 \
    --save_directory data/output/search_results/UnansweredQuestions
```

```
Saved student_search_results to data/output/search_results/UnansweredQuestions/dataset_docs_public.json
```

**3. Score with the moulinette** (rename `moulinette-ubuntu` / `-fedora` to `moulinette` first). The student results come first, the ground-truth `AnsweredQuestions` dataset second:

```bash
./moulinette evaluate_student_search_results \
    data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    data/datasets/AnsweredQuestions/dataset_docs_public.json \
    --k 10 --max_context_length 2000
```

```
Student data is valid: True
Evaluation Results
========================================
Recall@1: 0.450 Recall@3: 0.590 Recall@5: 0.650 Recall@10: 0.720
```

> **Note:** These figures illustrate the output format only. They are not a reference score. The thresholds you must reach are defined in the [Evaluation](#evaluation) chapter.

**4. Generate answers from the search results:**

```bash
uv run python -m src answer_dataset \
    --student_search_results_path data/output/search_results/UnansweredQuestions/dataset_docs_public.json \
    --save_directory data/output/search_results_and_answer/UnansweredQuestions
```

```
Loaded 100 questions ... Processed 100 of 100 questions
Saved student_search_results_and_answer to .../UnansweredQuestions/dataset_docs_public.json
```

---

## Evaluation

Retrieval quality is measured with a **recall@k** metric.

### Recall@k calculation

For each question, recall@k is the share of its correct sources that you retrieve in your top-k results. A correct source counts as found when one of your results is **in the same file** and **overlaps its character range**.

The overlap bar is low (an **IoU of 0.05**), so you do not need to match the reference span exactly: retrieving a chunk that covers the right region of the right file is enough. A result in a different file never counts, which is why `file_path` must be exact.

The moulinette validates your output, then reports recall at several values of k. **Docs must reach 80 % recall@5, code 50 %.**

Example moulinette output:

```
$ ./moulinette evaluate_student_search_results \
    search_results/dataset_docs_public.json \
    datasets/AnsweredQuestions/dataset_docs_public.json --k 10
Student data is valid: True
Total number of questions: 100
Total number of questions with student sources: 100

Evaluation Results
========================================
Questions evaluated: 100
Recall@1: 0.620 (62.0%)
Recall@3: 0.770 (77.0%)
Recall@5: 0.880 (88.0%)   <- docs threshold is 80%
Recall@10: 0.890 (89.0%)
```

### Performance requirements

Your system must respect some minimal performances, listed below:

| Metric | Requirement |
|--------|-------------|
| **Indexing time** | At most **5 minutes** for the whole corpus. |
| **Retrieval throughput** | At most **90 seconds** for **200 questions**. |
| **Recall@5** | At least **80 %** on docs questions and **50 %** on code questions. |

---

## Bonus Part

> **Warning:** The bonus is graded only once the entire mandatory part is validated: every mandatory requirement met and the system robust even when a reviewer pushes it in odd directions. Until the mandatory part validates in full, the bonus is not evaluated at all.

There are **five bonuses, each worth one point**. They all run on a CPU-only campus machine, and each one extends the mandatory system in a direction you will meet again later. A bonus counts only if it is **implemented and working**, not merely described in the `README.md`, and you may be asked to demonstrate it.

1. **Semantic embeddings:** add a vector index built with a lightweight CPU model (such as `all-MiniLM-L6-v2`) next to your lexical index.
2. **Hybrid retrieval:** combine the lexical and semantic rankings into a single result list.
3. **Incremental indexing:** when a file changes, re-index only that file instead of rebuilding the whole index.
4. **Caching:** cache the index and query results to speed up cold start and repeated queries.
5. **Local HTTP API:** expose querying the index and answering questions over a small local HTTP API, so the system can be driven by something other than the CLI.

---

## Submission

Submit your assignment in your Git repository as usual. Only the work inside your repository will be evaluated during the defense. Don't hesitate to double-check the names of your files to ensure they are correct.

Your repository must contain:

- `src/` directory with your implementation
- `pyproject.toml` and `uv.lock` for dependency management
- a `Makefile` exposing the rules from the [Common Instructions](#makefile) (`install`, `run`, `debug`, `clean`, `lint`)
- `README.md` with comprehensive documentation
- Any additional configuration files needed to run your solution

> **Warning:** Do not include large data files, model weights, or generated outputs in your repository. The evaluator will generate these during the evaluation process. The deep-learning stack and the model weights can total several gigabytes, so build and run your project from a location with enough free disk space.
