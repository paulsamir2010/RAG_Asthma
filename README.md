# RAG solution on Asthma GenAI

**Author: Samir Paul**

A simple LangChain notebook that answers general asthma questions from one Cleveland Clinic webpage.
It combines cosine similarity retrieval in ChromaDB with BM25 keyword retrieval, then uses OpenAI to write a short answer from the retrieved passages.

## Project files

| File | Purpose |
| --- | --- |
| `RAG_solution_on_Asthma_GenAI.ipynb` | Complete beginner notebook with explanations before every step |
| `README.md` | GitHub project overview and run instructions |
| `RAG_solution_on_Asthma_GenAI_Submission.docx` | Project submission covering the approach, decisions and evaluation plan |

## Knowledge source

[Cleveland Clinic asthma article](https://my.clevelandclinic.org/health/diseases/6424-asthma)

Only this page is loaded. The project does not crawl linked pages or retrieve live web information when answering a question.
The original article belongs to Cleveland Clinic; it is not bundled in this repository.

## Pipeline

1. Load the page with `SeleniumURLLoader`.
2. Remove blank lines and standalone advertisement labels.
3. Set exactly two metadata fields: `source` and `topic`.
4. Split the text with `RecursiveCharacterTextSplitter`.
5. Embed the chunks with `OpenAIEmbeddings` and store them in local ChromaDB.
6. Retrieve three chunks by cosine search and three by BM25 keyword ranking.
7. Combine the ranked lists with LangChain's `EnsembleRetriever`.
8. Send the first four fused passages and the question to `ChatOpenAI`.
9. Display the answer, knowledge-source URL and numbered passages for inspection.

BM25 is the sparse keyword ranker/retriever in this project. It searches the same chunk corpus independently of cosine search; it is not a separate post-retrieval cross-encoder reranker. The ensemble combines rankings with weighted Reciprocal Rank Fusion (RRF).

## Tools and settings

| Component | Choice |
| --- | --- |
| Framework | LangChain integrations and `langchain-classic` for EnsembleRetriever |
| Loader | `SeleniumURLLoader`, headless Chrome |
| Splitter | `RecursiveCharacterTextSplitter` |
| Chunk size / overlap | 1,000 / 150 characters |
| Embedding model | `text-embedding-3-small` via `OpenAIEmbeddings` |
| Vector database | Local ChromaDB, `./asthma_chroma_db` |
| Chroma distance metric | Explicit HNSW cosine distance |
| Dense retrieval | Similarity search, `k=3`, topic filter `asthma` |
| Sparse retrieval | `BM25Retriever`, `k=3`, lowercase word tokenization |
| Ensemble | `EnsembleRetriever`, weights `[0.5, 0.5]` |
| Context limit | First four fused chunks |
| OpenAI chat model | `gpt-4.1-mini`, temperature `0` |

These are simple starting settings, not tuned or benchmarked values.

## Metadata

| Field | Example | Purpose |
| --- | --- | --- |
| `source` | `https://my.clevelandclinic.org/health/diseases/6424-asthma` | Source attribution |
| `topic` | `asthma` | Subject identification and dense search filtering |

Every stored chunk has these two metadata fields. Numbered excerpt labels are created at answer time and are not extra stored metadata.

## Run the notebook

1. Use Python 3.11 or 3.12 in a fresh Jupyter environment and install Google Chrome.
2. Open `RAG_solution_on_Asthma_GenAI.ipynb`.
3. Run Step 1 to install the packages, then restart the kernel.
4. Paste your OpenAI API key in Step 3. No `.env` file is needed.
5. Run Steps 2–14 in order. Inspect the loaded text in Step 5 before ingestion.
6. Change the question in Step 14 to ask another question.
7. Check the answer's numbered references against the printed excerpts.

The notebook installs the current major package families: `langchain-classic` 1.x, `langchain-community` 0.4.x, `langchain-openai` 1.x, `langchain-chroma` 1.x, `langchain-text-splitters` 1.x and `chromadb` 1.x, plus Selenium, Unstructured and `rank_bm25`.
An OpenAI API account with billing is needed. Ingestion sends chunk text to the embedding API; queries send the question to the embedding API and the selected passages to the chat API.

### Reruns and freshness

- Another question in the same session: rerun Step 14 only.
- Refresh the article: rerun Steps 4–10, then Steps 12–14.
- After a kernel restart: rerun Steps 2–14 for the simplest workflow.
- Step 7 resets only the `asthma_basic_rag` collection before ingestion, preventing duplicates and stale chunks. It repeats embedding calls and replaces the previous article snapshot.
- Chroma persists locally. BM25 is rebuilt in memory. The notebook has no automatic refresh job.

## Example questions

- What are the common symptoms of asthma?
- What can trigger asthma symptoms?
- What is spirometry used for in asthma diagnosis?
- What medicines are discussed for asthma?

These are sample inputs. No generated medical answer or measured performance is included in the project files.

## Manual evaluation

Step 15 contains seven test inputs, including unrelated questions, personal-dose requests and a current breathing emergency.
Inspect the retrieved excerpts and verify the following:

- Each factual statement is supported by its cited excerpt.
- The retrieved passages are relevant to the question.
- Missing information produces the stated fallback rather than an invented answer.
- Personal medical decisions are directed to a qualified clinician.
- A reported current severe breathing problem prompts immediate emergency-care direction.

Suggested targets, to be measured after running: **90% supported factual statements**, **80% relevant context passages** and **under 15 seconds per warm query**. These are project goals, not achieved results or clinical validation. Hybrid retrieval is not assumed to be better without comparison.

The notebook's installation cell is not counted in timing. Step 14 measures retrieval plus generation.
For a fair retrieval comparison, use the same questions and inspect the first three passages from each retriever, including the first three fused results.

## Troubleshooting

| Issue | Action |
| --- | --- |
| Chrome or driver unavailable | Install/update Chrome; allow Selenium Manager to download its driver |
| Missing module after package installation | Restart the kernel, then rerun imports |
| Unstructured reports a missing NLTK resource | Use `nltk.download("resource_name")` for the resource named in the error |
| Blocked or irrelevant page text | Stop and inspect Step 5; do not embed a challenge page |
| OpenAI authentication, quota or model error | Check the key, API billing and access to the configured model |
| Source content changed | Rerun ingestion; Step 7 replaces this project's collection |

## Before uploading to GitHub

Restore `OPENAI_API_KEY = "PASTE_YOUR_OPENAI_API_KEY_HERE"` and clear notebook outputs.
Do not upload a real API key or the generated Chroma database. Add these patterns to your repository's `.gitignore`:

```gitignore
asthma_chroma_db/
.ipynb_checkpoints/
```

## Limitations

This is an educational prototype for general information, not a clinical product. It does not diagnose, prescribe personal doses or validate medical advice.
Prompt instructions cannot guarantee factual correctness or safe behavior. Both retrievers can return top-ranked chunks even for an unrelated question; there is no automatic relevance gate or hallucination checker.
Only one article is available, and basic cleaning may leave navigation or footer text. Answers can miss information outside the selected chunks. Source freshness depends on manual re-ingestion.

## References

- [Cleveland Clinic asthma article](https://my.clevelandclinic.org/health/diseases/6424-asthma)
- [SeleniumURLLoader](https://reference.langchain.com/python/langchain-community/document_loaders/url_selenium/SeleniumURLLoader)
- [RecursiveCharacterTextSplitter](https://docs.langchain.com/oss/python/integrations/splitters/recursive_text_splitter)
- [LangChain Chroma integration](https://docs.langchain.com/oss/python/integrations/vectorstores/chroma)
- [BM25Retriever](https://reference.langchain.com/python/langchain-community/retrievers/bm25/BM25Retriever)
- [EnsembleRetriever](https://reference.langchain.com/python/langchain-classic/retrievers/ensemble/EnsembleRetriever)
- [OpenAI GPT 4.1 mini](https://developers.openai.com/api/docs/models/gpt-4.1-mini)

The supplied notebooks informed the direct retrieve–prompt–generate pattern and equal-weight hybrid retrieval. The submission document adapts the supplied project framework to this smaller, single-page solution.
