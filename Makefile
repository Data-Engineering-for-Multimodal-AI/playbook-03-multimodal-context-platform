.PHONY: setup run smoke local data clean

# Uniform targets for The Playbooks. The book prints no commands for this playbook. The README's
# commands (pip install -r requirements.txt, scripts/*.py --local, and the cloud mode with
# --project and --bucket) still work as printed.

# Install the locked dependencies into .venv.
setup:
	uv sync --frozen

# Run the five pipeline steps in local mode on the sample documents in data/ and print the search
# results. CPU only, no cloud account. Same steps as `make local`.
run: setup
	@$(MAKE) --no-print-directory local

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
