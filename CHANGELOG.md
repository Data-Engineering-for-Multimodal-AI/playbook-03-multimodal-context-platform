# Changelog

## 2026.10 (2026-10-22): 2026 maintenance update

The printed version is tagged `first-edition`.

### Added
- **Sample documents.** The 22 documents the README uploads were never committed, so the first pipeline step failed as printed. `data/` now holds 22 synthetic documents for the same five customers and file names: driver's license photos, Forms W-2 and 1040 (tax year 2024), utility bills and bank statements. They come from `scripts/generate_sample_data.py`, and their figures agree across documents. Every name and number is invented; see `data/README.md`.
- **Local mode.** `--local` on every script runs the pipeline on a laptop CPU with no cloud account: pypdf, the text on each license card, `all-MiniLM-L6-v2` and a NumPy index.
- `scripts/teardown_vector_search.py` and a Clean up section: the deployed Vector Search index is billed per hour, and the README never said to remove it.
- `pyproject.toml`, `uv.lock` and a pinned `requirements.txt` (Python 3.12+); `.env.example`; `.gitignore`.
- `tests/test_smoke.py` (`make smoke`) and GitHub Actions on Python 3.12 and 3.13, including the printed `pip install` path.

### Fixed
- `--filter` crashed: `find_neighbors` takes a list of `Namespace`, not a dict.
- `src/vector_search.py` read `neighbor.metadata`, which doesn't exist, and passed upserts in the wrong format.
- `src/document_processor.py` called `aiplatform.TextEmbeddingModel` (doesn't exist), sent a Document AI request field that doesn't exist, and used `us-central1` as the Document AI location.
- `setup_vector_search.py` exited 0 when setup failed. `process_documents.py` embedded failed documents as their error message; it now reports them and exits 1.
- Document categories came from file names, and `'id' in name` matched any name containing "id". They now come from the sample-data folder.
- README: the real clone URL and folder, Python 3.12+, and the `gsutil mkdir` lines, which fail (`gsutil mkdir` means `gsutil mb`).

### Changed
- **Embeddings** go through the `google-genai` SDK with the same model, `text-embedding-005`. The `vertexai.language_models` API was scheduled for removal on June 24, 2026. Google lists `text-embedding-005` for retirement on April 1, 2027.
- **Utility bills** go to the Form Parser: Google discontinued the Utility Parser on June 30, 2026. Four processors instead of five.
- The embeddings file holds only the fields Vector Search reads (`id`, `embedding`, `restricts`). The labels the search scripts print are in `vector-search/document-metadata.json`. First-edition files still work.
- Processor IDs, project, bucket, location and endpoint come from environment variables instead of edits to the source. The printed flags still work.
- The README follows the standard playbook layout; its expected output comes from a local run of this version.

### Removed
- `vertexai` (a duplicate of `google-cloud-aiplatform`), `google-cloud-vision`, `pandas` and `python-dotenv` from the requirements: unused.
