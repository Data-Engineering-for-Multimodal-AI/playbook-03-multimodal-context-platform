.PHONY: setup run smoke local search check-data load try-hybrid try-chunks try-index-size data clean

# A bare `make` runs the smoke test, as it did before the uniform targets were added.
.DEFAULT_GOAL := smoke

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

# Search the local index (after make run or make load). FILTER and RESULTS are optional.
#   make search QUERY="Laura Russo utility bill"
#   make search QUERY="James tax documents" FILTER=Form1040 RESULTS=3
search:
	@test -n "$(QUERY)" || { echo 'Usage: make search QUERY="a request"  [FILTER=Form1040] [RESULTS=5]'; exit 2; }
	uv run --frozen python scripts/search_documents.py --local --query "$(QUERY)" \
		--results $(or $(RESULTS),5) $(if $(FILTER),--filter $(FILTER))

# Your own documents, laid out like data/. check-data checks every file and writes nothing;
# load replaces the local index with your documents.
#   make check-data DIR=my_documents
#   make load DIR=my_documents
check-data: setup
	@test -n "$(DIR)" || { echo "Usage: make check-data DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/load_documents.py --dir "$(DIR)" --check-only

load: setup
	@test -n "$(DIR)" || { echo "Usage: make load DIR=path/to/folder"; exit 2; }
	uv run --frozen python scripts/load_documents.py --dir "$(DIR)"

# Try this: three measured experiments (see README, "Try this"). Local, no cloud account.
try-hybrid: setup
	uv run --frozen python experiments/hybrid_search.py

try-chunks: setup
	uv run --frozen python experiments/chunking.py

try-index-size: setup
	uv run --frozen python experiments/index_size.py $(SIZES)

# Regenerate the synthetic documents in data/ (identical output)
data:
	uv run --frozen python scripts/generate_sample_data.py

# Delete the pipeline output (processed documents and the local index). data/ is kept.
clean:
	rm -rf processed-documents processed_docs vector-search embeddings.json
