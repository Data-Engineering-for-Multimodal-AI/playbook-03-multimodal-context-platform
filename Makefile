.PHONY: smoke local data clean

# Sample data, the local pipeline and the cloud calls (mocked). CPU only, no cloud account.
smoke:
	uv run --frozen pytest -q tests/test_smoke.py

# Run the five pipeline steps locally (README "Run it locally")
local:
	uv run --frozen python scripts/process_documents.py --local
	uv run --frozen python scripts/create_embeddings.py --local
	uv run --frozen python scripts/setup_vector_search.py --local
	uv run --frozen python scripts/search_documents.py --local
	uv run --frozen python scripts/test_person_search.py --local

# Regenerate the synthetic documents in data/ (identical output)
data:
	uv run --frozen python scripts/generate_sample_data.py

clean:
	rm -rf processed-documents processed_docs vector-search embeddings.json
