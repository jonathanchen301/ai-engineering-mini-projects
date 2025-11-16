# Splitter Parameters

### Default
- Default: 400 characters
    - Smaller chunks improve precision
    - Reduces irrelevant information in retrieved context
    - Better for fact-based Q&A
    - May lose broader context!
- Overlap: 40 characters
    - Make big if want to preserve information between chunks
    - Default around 10% of chunk size
- RecursiveCharacterTextSplitter
    - State-of-art
    - Preserves metadata automatically
    - Splits at natural boundaries (paragraphs, sentences, words)
    - Look at smaller chunks (words), then zoom out and grab larger chunks.

# Retrieval

### Default
- Method: Dot product top-k similarity search
    - Default. Good for small numbers of documents
    - Another one that is useful is multi-query search
- Top-K Choice: 3 - 5 is reasonable for small knowledge bases
    - Use similarity threshold when you want to exclude chunks that aren't similar enough bove a certain threshold