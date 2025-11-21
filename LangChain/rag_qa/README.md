# Single-User RAG QA Pipeline

A CLI-based RAG (Retrieval-Augmented Generation) system for question-answering over PDF documents.

## Architecture

```
Documents → Load → Split → Embed → FAISS Store
                                    ↓
Query → Retrieve → Build Prompt → LLM → Parse → Answer + Citations
```

## Getting Started

### Prerequisites

1. **OpenAI API Key**: Get your API key from [OpenAI](https://platform.openai.com/api-keys) and set it:
   ```bash
   export OPENAI_API_KEY="your-api-key-here"
   ```
   Or create a `.env` file:
   ```
   OPENAI_API_KEY=your-api-key-here
   ```

2. **PDF Documents**: Place your PDF files in the `documents/` directory (or specify a custom directory). **Only PDF files are supported.**

### Install

```bash
pip install -r requirements.txt
```

## Quickstart

### Install
```bash
pip install -r requirements.txt
```

### Ingest Documents
```bash
python src/cli.py ingest --document-dir documents
```

### Chat
```bash
python src/cli.py chat "Your question here"
```

## Configuration

| Setting | Default | Description |
|---------|---------|-------------|
| `model` | `gpt-4o-mini` | OpenAI model name |
| `temperature` | `0.0` | Model temperature (0-1) |
| `chunk_size` | `400` | Characters per chunk |
| `chunk_overlap` | `20` | Overlap between chunks |
| `top_k` | `3` | Number of chunks to retrieve |
| `vector_store_path` | `./vector_store` | FAISS store location |
| `document_dir` | `documents` | PDF documents directory |

Override via CLI flags: `--model`, `--temperature`, `--chunk-size`, `--chunk-overlap`, `--top-k`, `--vector-store-path`, `--api-key`

## Logging Sample

python src/cli.py chat "What implications does the government restart have?"

```
Latency: 7.719367742538452 seconds
Prompt tokens: 934
Completion tokens: 316
Total tokens: 1250

Answer:
The restart of the government will allow government workers to return to their jobs and reverse recent mass layoffs, as furloughed workers were missing paychecks during the shutdown [1]. Additionally, funding could be available within 24 hours for most states [2], although it may take several days for everything to return to normal in areas like airports [3].

Citations:
[1] Government Shutdown 2025 Update: Trump signs bill, ending record 43 day disruption | Page 2 | page 2
    "Government workers will get back to work, and recent mass layoffs are to be reversed."
[2] Government Shutdown 2025 Update: Trump signs bill, ending record 43 day disruption | Page 6 | page 6
    "Funds could be available within 24 hours of the government reopening for most states."
[3] The longest government shutdown in U.S. history is over. Here's what you need to know | Page 3 | page 3
    "It will likely be several days or more until everything will get back to normal."
```

## Testing

```bash
pytest tests/
```

## Dry Run

Preview the prompt without calling the model:

```bash
python src/cli.py chat "Your question" --dry-run
```