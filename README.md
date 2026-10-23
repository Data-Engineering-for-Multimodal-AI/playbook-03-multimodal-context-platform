# Multimodal Document Retrieval Pipeline

**Chapter 3 playbook** from *Data Engineering for Multimodal AI* (O'Reilly, 2026). Page numbers to be confirmed.
Book: https://oreil.ly/data-engineering-multimodal

> Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## What you'll build

A semantic search system for KYC (Know Your Customer) documents: driver's licenses, Forms W-2 and 1040, utility bills and bank statements. The pipeline extracts the text from each document, turns it into an embedding, indexes the embeddings, and answers natural-language queries such as "Form 1040 tax return" or "Mitchell Allen's tax information".

```
data/ (JPEG, PDF) ──▶ 1. extract text ──▶ 2. embed ──▶ 3. index ──▶ 4. search
                         Document AI       text-embedding-005   Vertex AI Vector Search   (cloud, as printed)
                         pypdf + card text all-MiniLM-L6-v2     NumPy matrix              (--local)
```

Every step runs in one of two modes:

| Step | Google Cloud (as printed) | Local (`--local`, laptop CPU, no account) |
|---|---|---|
| Extract text | Document AI processors (OCR and entities) | pypdf for PDFs; the card text in `data/extracted_text/` for the license photos |
| Embed | Vertex AI `text-embedding-005` (768 dimensions) | `all-MiniLM-L6-v2` (384 dimensions) |
| Index and search | Vertex AI Vector Search (tree-AH index on a deployed endpoint) | Cosine similarity over a NumPy matrix |

Both modes read the same 22 sample documents for five fictional customers and write files to the same relative paths (`processed-documents/`, `vector-search/`). The local mode uses a different embedding model, so its scores differ from a cloud run.

> **Vertex AI has a new name.** In April 2026 Google renamed Vertex AI "Gemini Enterprise Agent Platform", and the console uses the new name. The API (`aiplatform.googleapis.com`), the Python packages and this code are unchanged, so this README keeps the book's name.

## Quick start

You need Python 3.12 or newer. Local mode needs about 1.5 GB of disk space for the environment and no cloud account. Cloud mode needs a Google Cloud project with billing.

### Step 1: Clone the repository

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-03-multimodal-context-platform.git
cd playbook-03-multimodal-context-platform
```

### Step 2: Create a virtual environment

```bash
python -m venv venv
source venv/bin/activate        # macOS / Linux
# venv\Scripts\activate         # Windows
```

### Step 3: Install dependencies (as printed)

```bash
pip install -r requirements.txt
```

The versions are pinned, and the file covers both modes. On Linux, PyTorch comes from its CPU-only index, which `requirements.txt` adds.

Using [uv](https://docs.astral.sh/uv/) instead: `uv sync`, then prefix the Python commands below with `uv run` (or run `make local` for all five local steps).

### Run it locally (no cloud account)

```bash
python scripts/process_documents.py --local     # extract text from the 22 documents in data/
python scripts/create_embeddings.py --local     # embed them (downloads all-MiniLM-L6-v2, about 90 MB, once)
python scripts/setup_vector_search.py --local   # build the local index
python scripts/search_documents.py --local      # the four example queries
python scripts/test_person_search.py --local    # the five person queries
```

Single queries and the category filter work the same way:

```bash
python scripts/search_documents.py --local --query "Find Laura's driver's license" --results 5
python scripts/search_documents.py --local --query "James tax documents" --filter Form1040 --results 3
```

Categories for `--filter`: `ID`, `W2`, `Form1040`, `UtilityBill`, `BankStatement`.

### Run it on Google Cloud (as printed in the book)

These steps create billed resources. A deployed Vector Search index is billed for every hour it stays deployed, so finish with [Clean up](#clean-up).

#### 1. Set up Google Cloud

```bash
# Authenticate with Google Cloud
gcloud auth login
gcloud auth application-default login   # credentials for the Python client libraries

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable documentai.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable vision.googleapis.com   # as printed; the scripts don't use the Vision API
```

#### 2. Set up Document AI processors

In the [Google Cloud Console](https://console.cloud.google.com/), go to Document AI > Processors and create these processors in the `us` (or `eu`) location:

- US Driver License Parser
- W2 Parser
- Form Parser (for Forms 1040 and, since the Utility Parser was discontinued, utility bills)
- Bank Statement Parser

Note each processor's ID.

#### 3. Create a Cloud Storage bucket

```bash
# Create a GCS bucket to store your documents and processed results
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME/
```

Google has deprecated `gsutil` in favor of `gcloud storage`. The printed `gsutil` lines still work; the current command is:

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1
```

The book also runs `gsutil mkdir gs://YOUR_BUCKET_NAME/data/` and similar lines for the folders. Skip them: `gsutil mkdir` is another name for `gsutil mb` (make bucket), so they fail with "The mb command requires a URL that specifies a bucket". Cloud Storage creates the folders as files are copied into them.

#### 4. Configure

Copy `.env.example` to `.env`, fill in your project, bucket and the four processor IDs, and load it:

```bash
cp .env.example .env            # then edit .env
set -a; source .env; set +a
```

You no longer edit IDs into the source files. The printed `--project` and `--bucket` flags still work and take precedence.

#### 5. Upload the sample data

```bash
gsutil cp -r data/DrivingLicenses/* gs://YOUR_BUCKET_NAME/data/DrivingLicenses/
gsutil cp -r data/W2Sample/* gs://YOUR_BUCKET_NAME/data/W2Sample/
gsutil cp -r data/Form1040Sample/* gs://YOUR_BUCKET_NAME/data/Form1040Sample/
gsutil cp -r data/Utility/* gs://YOUR_BUCKET_NAME/data/Utility/
gsutil cp data/Realistic_Bank_Statements.pdf gs://YOUR_BUCKET_NAME/data/
```

Or with `gcloud storage`:

```bash
gcloud storage cp -r data/DrivingLicenses/* gs://YOUR_BUCKET_NAME/data/DrivingLicenses/
gcloud storage cp -r data/W2Sample/* gs://YOUR_BUCKET_NAME/data/W2Sample/
gcloud storage cp -r data/Form1040Sample/* gs://YOUR_BUCKET_NAME/data/Form1040Sample/
gcloud storage cp -r data/Utility/* gs://YOUR_BUCKET_NAME/data/Utility/
gcloud storage cp data/Realistic_Bank_Statements.pdf gs://YOUR_BUCKET_NAME/data/
```

#### 6. Run the pipeline

```bash
python scripts/process_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --sample-data data --output processed-documents
python scripts/create_embeddings.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME
python scripts/setup_vector_search.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME
```

The embeddings use `text-embedding-005`, which Google retires on April 1, 2027. After that date, set `EMBEDDING_MODEL` in `.env` to its replacement (for example `gemini-embedding-001`) and run the pipeline again from `create_embeddings.py`. `EMBEDDING_DIMENSIONS` (768) sets the size of the vectors and of the index, so you don't need to change any code. A different model gives different scores, and this repository was tested only with `text-embedding-005`.

`setup_vector_search.py` takes 30–45 minutes and prints the endpoint ID at the end (`projects/.../locations/us-central1/indexEndpoints/...`). Use it for the searches:

```bash
python scripts/search_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID"
python scripts/test_person_search.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID"
python scripts/search_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID" --query "James tax documents" --filter "Form1040" --results 3
```

#### Clean up

```bash
python scripts/teardown_vector_search.py --project YOUR_PROJECT_ID          # lists the index and endpoint
python scripts/teardown_vector_search.py --project YOUR_PROJECT_ID --yes    # undeploys and deletes them
gsutil -m rm -r gs://YOUR_BUCKET_NAME                                       # deletes the bucket and everything in it
# or: gcloud storage rm --recursive gs://YOUR_BUCKET_NAME/
```

Document AI is billed per page processed. Delete the processors in the console when you no longer need them.

#### Cost

The book estimates under $10 for 20–30 documents. That holds only if you clean up promptly: Document AI and the embeddings are billed per page and per request (cents here), but the deployed Vector Search index is billed per node-hour for as long as it stays deployed. Check the current [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) before you start.

### Checks

```bash
make smoke    # or: uv run pytest -q
```

`make smoke` (about 1 minute on a laptop CPU) checks the sample data, runs the five steps in local mode, and checks the cloud calls with mocks: the google-genai embedding call, the category filter, processor routing, exit codes and teardown. GitHub Actions runs it on Python 3.12 and 3.13, plus the printed `pip install -r requirements.txt` path.

## Tested on

| | |
|---|---|
| Date | 2026-10-22 |
| OS | macOS 26.5.1 (arm64); GitHub Actions `ubuntu-latest` |
| Python | 3.12.15 and 3.13.16 |
| Key packages | sentence-transformers 6.2.0, torch 2.14.1, pypdf 6.21.2, google-genai 2.33.0, google-cloud-aiplatform 2.6.0, google-cloud-documentai 3.16.0 |
| Install time | `pip install -r requirements.txt`: 64 s (environment 1.3 GB) |
| Local pipeline | About 60 s for the five steps from a fresh clone, including the one-time model download (46 s of it) |

The local mode was run end to end from a fresh clone on 3.12, and the results on 3.13 were identical. **The Google Cloud mode was not run for this update:** it needs a billed project and deploys a billed endpoint. Its calls were checked against the current SDKs and with mocked clients in the tests.

## Differences from the printed book

| What | Printed book | Now | Why |
|---|---|---|---|
| Sample documents | `gsutil cp data/DrivingLicenses/*` and the others failed: the 22 documents were never committed | 22 synthetic documents in `data/`, with the same customer names and file names, and a generator | The originals and their source were lost. See [Data and licenses](#data-and-licenses) |
| Running without Google Cloud | Not possible | `--local` on every script | Readers without a billed project can run every step |
| Clone URL and folder | `pb3-mm-docretrievalpipeline` | `playbook-03-multimodal-context-platform` | The repository's actual name |
| Python | 3.8+ | 3.12+ | Current dependencies |
| Embeddings | `vertexai.language_models.TextEmbeddingModel` | `google-genai` (`client.models.embed_content`), same model `text-embedding-005` | Google deprecated that API in June 2025 and scheduled its removal for June 24, 2026 |
| Embedding model | Fixed in the code | `EMBEDDING_MODEL` and `EMBEDDING_DIMENSIONS` in `.env`; default `text-embedding-005`, 768 | Google retires `text-embedding-005` on April 1, 2027. The setting lets you switch models without editing code |
| Product name | Vertex AI | The console says "Gemini Enterprise Agent Platform" | Google renamed it in April 2026. The APIs and code didn't change |
| `gsutil` commands | `gsutil mb`, `gsutil cp`, `gsutil rm` | Kept, with the `gcloud storage` equivalent alongside | Google has deprecated `gsutil` |
| Vector Search | Index and index endpoint | Unchanged | Google still supports this Vector Search. Its newer service, Agent Retrieval (formerly Vector Search 2.0), uses collections in place of indexes; moving to it would rewrite this chapter's cloud steps |
| `--filter` | Crashed (a dict passed where a list of `Namespace` is expected) | Works | Bug |
| Utility bills | Utility Parser | Form Parser | Google discontinued the Utility Parser on June 30, 2026 |
| Processor IDs, project, bucket | Edited into `src/document_processor.py` and `scripts/process_documents.py` | Environment variables (`.env.example`); printed flags still work | No edits to code; no IDs in commits |
| `gsutil mkdir gs://.../data/` lines | Printed | Skipped | `gsutil mkdir` means `gsutil mb` (make bucket) and fails on a folder path; Cloud Storage creates folders on upload |
| Embedding file | Each record also held a `metadata` field | Records hold `id`, `embedding`, `restricts`; metadata goes to `vector-search/document-metadata.json` | Vector Search's input format has no `metadata` field. The search scripts still read first-edition files |
| Failures | `setup_vector_search.py` exited 0 after failing; failed documents were embedded as their error message | Exit 1; failed documents are reported and skipped | Bugs |
| Clean up | Not covered | `scripts/teardown_vector_search.py` and a Clean up section | The deployed index is billed per hour |
| `src/document_processor.py` | `aiplatform.TextEmbeddingModel` (doesn't exist) and a Document AI request field that doesn't exist | Fixed; same embeddings helper as the scripts | Bugs. The scripts don't use this class |
| Example output | Scores from a cloud run with the original documents | Output from a local run with the new documents, below | The original documents and a cloud run aren't available |

Details are in [CHANGELOG.md](CHANGELOG.md).

## Data and licenses

| Path | Contents | Source | License |
|---|---|---|---|
| `data/` | 22 synthetic KYC documents (5 license photos, 5 W-2s, 5 Forms 1040, 6 utility bills, 1 bank statement file) and the text on each license card | Generated for this repository by `scripts/generate_sample_data.py` | MIT, like the code |

Every person, company, address and number in them is invented. The SSNs are in the block the Social Security Administration reserves for advertising, and every page is marked SPECIMEN. [data/README.md](data/README.md) lists every file and explains how the data was made. The first edition's documents were never committed, and their source isn't recorded; these replace them.

Never put real KYC documents in a public repository.

## Expected output

From my run of the local mode on 2026-10-22: macOS 26.5.1 (arm64), Python 3.12.15, fresh clone, `pip install -r requirements.txt`. Python 3.13.16 gave identical results.

`python scripts/process_documents.py --local` (excerpt):

```
Listing documents in data
Found 22 documents
Processing document 1/22: data/DrivingLicenses/AHDL.jpeg
Saved result to: processed-documents/AHDL_jpeg_processed.json
...
Processing document 22/22: data/W2Sample/TaylorCoxW2.pdf
Saved result to: processed-documents/TaylorCoxW2_pdf_processed.json
Processing complete. Summary saved to: processed-documents/processing_summary.json
```

`python scripts/create_embeddings.py --local` and `python scripts/setup_vector_search.py --local`:

```
Found 22 processed documents
Creating embeddings with sentence-transformers/all-MiniLM-L6-v2...
Saved 22 embeddings (384 dimensions) to vector-search/document-embeddings/embeddings.json
...
Local index: 22 vectors, 384 dimensions, saved to vector-search/local-index.npz
Vector search setup completed successfully.
```

`python scripts/search_documents.py --local`:

```
Query: 'Form 1040 tax return'
Found 3 similar documents:
1. processed-documents/AprilHensleyForm1040_pdf_processed.json (Form1040) - Score: 0.5541
2. processed-documents/TaylorCoxForm1040_pdf_processed.json (Form1040) - Score: 0.5362
3. processed-documents/JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.5041

Query: 'W2 income statement'
1. processed-documents/JamesWilliamsW2_pdf_processed.json (W2) - Score: 0.4571
2. processed-documents/AprilHensleyW2_pdf_processed.json (W2) - Score: 0.4515
3. processed-documents/LauraRussoW2_pdf_processed.json (W2) - Score: 0.4316

Query: 'Driver license ID'
1. processed-documents/AHDL_jpeg_processed.json (ID) - Score: 0.5108
2. processed-documents/JWDL_jpeg_processed.json (ID) - Score: 0.4982
3. processed-documents/LRDL_jpeg_processed.json (ID) - Score: 0.4663

Query: 'Utility bill for monthly payment'
1. processed-documents/TaylorCoxWaterUtility_pdf_processed.json (UtilityBill) - Score: 0.5149
2. processed-documents/LauraRussoUtility_pdf_processed.json (UtilityBill) - Score: 0.5094
3. processed-documents/JamesWilliamsUtility_pdf_processed.json (UtilityBill) - Score: 0.4719
```

(Snippet lines omitted.) For every query, the top three results all have the right document type.

`python scripts/test_person_search.py --local` (top 3 of 10 per query; snippets omitted):

```
=== Testing Query: 'Laura's KYC documents' ===
1. JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.2808
2. JWDL_jpeg_processed.json (ID) - Score: 0.2647
3. LauraRussoForm1040_pdf_processed.json (Form1040) - Score: 0.2611

=== Testing Query: 'April Hensley's identification documents' ===
1. AprilHensleyForm1040_pdf_processed.json (Form1040) - Score: 0.3467
2. LRDL_jpeg_processed.json (ID) - Score: 0.3118
3. TCDL_jpeg_processed.json (ID) - Score: 0.2752

=== Testing Query: 'James Williams' personal information' ===
1. JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.2934
2. JamesWilliamsW2_pdf_processed.json (W2) - Score: 0.2536
3. LauraRussoForm1040_pdf_processed.json (Form1040) - Score: 0.2165

=== Testing Query: 'Taylor Cox's financial records' ===
1. TaylorCoxW2_pdf_processed.json (W2) - Score: 0.3814
2. TaylorCoxForm1040_pdf_processed.json (Form1040) - Score: 0.3739
3. AprilHensleyForm1040_pdf_processed.json (Form1040) - Score: 0.3198

=== Testing Query: 'Mitchell Allen's tax information' ===
1. MitchellAllenW2_pdf_processed.json (W2) - Score: 0.4341
2. MitchellAllenForm1040_pdf_processed.json (Form1040) - Score: 0.4171
3. JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.4110
```

The person queries are where an embedding search shows its limits. Three of the five rank the right person's documents first. For "Laura's KYC documents", James Williams's Form 1040 and license come out ahead of Laura Russo's. For "April Hensley's identification documents", another customer's license is second. A sentence embedding captures what kind of document something is better than whose it is, especially in a small model that reads only the first 256 tokens. To find documents by name or ID number, use the entities Document AI extracts (the `entities` field in each processed file) or a keyword index, and keep vector search for the "what kind of document" part. The combined bank statement file doesn't reach the top 10 for any person: its text starts with the first customer's statement.

`python scripts/search_documents.py --local --query "James tax documents" --filter Form1040 --results 3`:

```
Search results for: 'James tax documents' (Filter: Form1040)
Found 3 similar documents:
1. processed-documents/JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.4881
2. processed-documents/AprilHensleyForm1040_pdf_processed.json (Form1040) - Score: 0.4205
3. processed-documents/MitchellAllenForm1040_pdf_processed.json (Form1040) - Score: 0.4106
```

## Project structure

```
├── data/                          # 22 synthetic sample documents (see data/README.md)
│   └── extracted_text/            # text on each license card (local mode only)
├── scripts/
│   ├── process_documents.py       # 1. extract text (Document AI, or --local)
│   ├── create_embeddings.py       # 2. embed (text-embedding-005, or --local)
│   ├── setup_vector_search.py     # 3. create and deploy the index (or --local)
│   ├── search_documents.py        # 4. example and single searches, --filter
│   ├── test_person_search.py      # 5. person-specific searches
│   ├── teardown_vector_search.py  # delete the cloud index and endpoint
│   └── generate_sample_data.py    # recreate data/ (dev dependencies)
├── src/
│   ├── kyc.py                     # shared helpers: categories, text extraction, embeddings, local index
│   ├── document_processor.py      # the cloud steps as one class (not used by the scripts)
│   └── vector_search.py           # Vector Search index, endpoint and queries
├── tests/test_smoke.py            # make smoke
├── .env.example                   # cloud settings
├── pyproject.toml, uv.lock        # dependencies (uv)
└── requirements.txt               # pinned, exported from uv.lock
```

## Troubleshooting

- **`--endpoint is required`**: cloud searches need the endpoint ID that `setup_vector_search.py` printed (or `VECTOR_SEARCH_ENDPOINT` in `.env`). For local searches, add `--local`.
- **`Set DOCAI_..._PROCESSOR_ID`**: fill in the processor IDs in `.env` and load it (`set -a; source .env; set +a`).
- **`DefaultCredentialsError`**: run `gcloud auth application-default login`.
- **Document AI processing errors**: the processors must be in the location `DOCAI_LOCATION` names (`us` or `eu`).
- **Vector Search errors**: the index has `EMBEDDING_DIMENSIONS` (768) dimensions. Embeddings from local mode (384) can't go into a cloud index. If you change `EMBEDDING_MODEL` or `EMBEDDING_DIMENSIONS`, rerun `create_embeddings.py` and create a new index.
- **`local-index.npz not found`**: run the local steps in order; each reads the previous step's output from the current folder.

## License

MIT; see [LICENSE](LICENSE). The sample documents in `data/` are MIT-licensed too.
