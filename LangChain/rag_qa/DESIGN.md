# Design Decisions

## Chunk Rationale

**Default Settings:**
- `chunk_size: 400` - Balances context preservation with retrieval precision. Smaller chunks (200-300) may lose context; larger chunks (600+) reduce retrieval accuracy.
- `chunk_overlap: 20` - Prevents information loss at chunk boundaries. 5% overlap (20/400) is sufficient for most documents.
- `top_k: 3` - Retrieves 3 most relevant chunks per query. Fewer chunks (1-2) may miss context; more chunks (5+) increase token usage and latency.

**Trade-offs:**
- Larger chunks → better context, worse retrieval precision
- Smaller chunks → better precision, potential context fragmentation
- More overlap → less information loss, more redundant embeddings

## Memory

This chat is completely stateless. Meant to be used for Q&A purposes. No memory is implemented.

## No-Answer Strategy

When no relevant chunks are retrieved:
1. Return fallback message: `"I couldn't find relevant info in the provided docs."`
2. Return empty citations array: `[]`
3. No LLM call is made (saves tokens/cost)

This prevents hallucination when the knowledge base lacks relevant information.

## Model Swapping

**Via CLI:**
```bash
python src/cli.py chat "question" --model gpt-4o
```

**Supported Models:**
- Any OpenAI chat model (gpt-4o, gpt-4o-mini, gpt-4-turbo, etc.)
- Model must support structured JSON output for citations

**Note:** Ensure your API key has access to the chosen model.

