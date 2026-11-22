# Playbook 03: Find the right documents from a plain-language request
**How to build a document retrieval pipeline for KYC documents.** Chapter 3 of [*Data Engineering for Multimodal AI*](https://oreil.ly/data-engineering-multimodal) (O'Reilly, 2026), pages 90–97.

A compliance reviewer asks for "tax documents for Laura" and needs the right files back, not every file with "tax" in it. In this repo, we explore this with Revault, a fictional investment platform, and its identity and tax documents. Follow along to build a retrieval pipeline that answers plain-language requests with the matching documents.

**The problem.** An agent works best with only the documents that matter for a request, but the documents are a mix of photographed IDs and PDF tax forms, utility bills and bank statements, with no shared structure.

**What you'll work with.** You'll turn every document into a common representation with embeddings, then retrieve only the context a request needs, using a managed AI platform (Vertex AI).

> Following the book? It describes the pipeline without printing commands; the first-edition README's steps, including the Google Cloud mode, are in [Run it the way the book does](#run-it-the-way-the-book-does) and still work as printed. Want the code exactly as printed? Use the `first-edition` tag. This branch is the 2026 maintenance update; see [Differences from the printed book](#differences-from-the-printed-book).

## Run it

You need [uv](https://docs.astral.sh/uv/) and `make`. This runs the local mode: no cloud account, no API keys, no Docker. Or open the repo in a GitHub Codespace, where both are ready. On Windows without `make`, use a Codespace or the steps below. The Google Cloud mode is under [Run it the way the book does](#run-it-the-way-the-book-does).

**1. Get the code.** About 3 MB.

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/playbook-03-multimodal-context-platform.git
cd playbook-03-multimodal-context-platform
```

**2. Build the index and run the example searches.** Installs the locked dependencies (`uv sync`) and runs the five scripts with `--local`: `process_documents.py` extracts the text of the 22 documents, `create_embeddings.py` embeds them (downloading `all-MiniLM-L6-v2`, about 90 MB, once), `setup_vector_search.py` builds the index, and `search_documents.py` and `test_person_search.py` run the example queries. On a Mac the first run took 1 min 35 s, most of it creating the 1.1 GB Python environment; a second run took 48 s.

```bash
make run
```

You should see 22 documents processed and embedded, then the searches:

```
Saved 22 embeddings (384 dimensions) to vector-search/document-embeddings/embeddings.json
Local index: 22 vectors, 384 dimensions, saved to vector-search/local-index.npz
...
Query: 'Form 1040 tax return'
Found 3 similar documents:
1. processed-documents/AprilHensleyForm1040_pdf_processed.json (Form1040) - Score: 0.5541
...
=== Testing Query: 'Laura's KYC documents' ===
1. JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.2808
```

Document-type queries find the right type every time. Person queries often don't: James Williams's Form 1040 comes first for Laura. [Expected output](#expected-output) has the full results.

**3. Ask your own question.** Runs `search_documents.py --local --query`:

```bash
make search QUERY="Laura Russo utility bill" RESULTS=3
```

```
1. processed-documents/TaylorCoxWaterUtility_pdf_processed.json (UtilityBill) - Score: 0.3403
2. processed-documents/LauraRussoUtility_pdf_processed.json (UtilityBill) - Score: 0.3369
3. processed-documents/AprilHensleyUtility_pdf_processed.json (UtilityBill) - Score: 0.3269
```

All three are utility bills, and Laura's is second: the embedding matches the kind of document better than the name. Narrow a search to one category with `FILTER=` (`ID`, `W2`, `Form1040`, `UtilityBill`, `BankStatement`):

```bash
make search QUERY="James tax documents" FILTER=Form1040 RESULTS=3
```

```
1. processed-documents/JamesWilliamsForm1040_pdf_processed.json (Form1040) - Score: 0.4881
```

**4. Clean up.**

```bash
make clean
```

This deletes the pipeline output: `processed-documents/` and `vector-search/` (the local index). The documents in `data/` stay, and so does the downloaded model (in `~/.cache/huggingface`, 87 MB).

## How much time do you have?

| Time | What to do |
|---|---|
| 15 minutes | [Run it](#run-it), steps 1 to 3. On a Mac, step 2 took 1 min 35 s the first time and 48 s after that. |
| 1 hour | The three [Try this](#try-this) experiments (under 30 s each on a Mac), then your own queries with `make search`. |
| 1 day | [Use your own data](#use-your-own-data) with your own documents, and work through [Before production](#before-production). The cloud mode takes longer: index setup alone takes 30 to 45 minutes. |

## Try this

Each experiment changes one thing and measures the effect. They read `data/` directly, use the local model, and write nothing. No cloud account. The numbers below are from a Mac on 2026-11-21.

**1. Find a person's documents by meaning, by name, or both?** Runs the five person queries from `test_person_search.py` three ways and counts the documents in each top 3 that belong to the person asked about: by vector score alone (as the pipeline searches), by documents that contain the name in the query, and by vector score plus a bonus for the name. The bank statement file holds a page for every customer, so it counts for all five.

```bash
make try-hybrid
```

```
Query                                      vector only name match hybrid   (right person in top 3)
Laura's KYC documents                              1/3        3/3    3/3
April Hensley's identification documents           1/3        3/3    3/3
James Williams' personal information               2/3        3/3    3/3
Taylor Cox's financial records                     2/3        3/3    3/3
Mitchell Allen's tax information                   2/3        3/3    3/3
All five queries                                  8/15      15/15  15/15
```

The embedding alone finds 8 of the 15; adding a check for the name finds all 15. Names are exact strings, and a small sentence embedding barely tells two names apart. Try a query that mixes a person and a type, such as "Laura's driver's license", and change `WEIGHT` in `experiments/hybrid_search.py`.

**2. How much of each document does the embedding read?** `all-MiniLM-L6-v2` reads at most 256 tokens of a text, and `create_embeddings.py` sends each document as one text. This counts the tokens, then embeds each document whole (as the pipeline does) and in chunks, where a document scores as its best chunk.

```bash
make try-chunks
```

```
22 documents, 103 to 2337 tokens each; the model reads 256.
17 documents are longer; it ignores 62% of all tokens.

Embedding                    vectors person queries type queries
whole document                    22           8/15        12/12
chunks of 256 tokens              62           9/15         9/12
chunks of 128 tokens             110           8/15        12/12
chunks of 64 tokens              211           8/15        12/12
```

The model ignores 62% of the text, but reading all of it in chunks doesn't find people any better (8 or 9 of 15), and 256-token chunks lose three of the type matches. Chunking also makes up to ten times as many vectors to store. For person queries, [Try this 1](#try-this)'s name check matters more than how much text is embedded. Change `CHUNK_SIZES` and run it again.

**3. When does an exact search get too slow?** The local index (`LocalIndex` in `src/kyc.py`) scores the query against every vector, sorts all the scores and builds a Python list of every hit. This fills it with random vectors of the same size (384) and times 50 top-10 queries, then the same search with `argpartition`, which finds the top 10 without sorting everything. Both return the same results.

```bash
make try-index-size
```

```
  vectors   memory  LocalIndex (sort all) top 10 only
       22    0.0 MB                0.02 ms     0.01 ms
   10,000   15.4 MB                5.49 ms     0.46 ms
  100,000  153.6 MB               75.08 ms     9.78 ms
  300,000  460.8 MB              209.27 ms    17.57 ms
```

For 22 documents, any search is instant. At 300,000 vectors each query takes 209 ms, and keeping only the top 10 cuts that to 18 ms. Past that, you need an approximate index like the cloud mode's Vector Search. Pass other sizes with `make try-index-size SIZES="1000 1000000"` (a million vectors need about 1.5 GB of memory).

## Use your own data

Put your documents in a folder with the same layout as `data/`:

```
my_documents/
  DrivingLicenses/  jane_doe_dl.jpeg        photos or scans: .jpeg .jpg .png
  W2Sample/         JaneDoeW2.pdf           PDFs with a text layer
  Form1040Sample/   JaneDoeForm1040.pdf
  Utility/          JaneDoeUtility.pdf
  bank_statement.pdf                        outside those folders, the category comes from the file name
  extracted_text/DrivingLicenses/jane_doe_dl.txt   the text on each image (local mode reads it in place of OCR)
```

The folder name sets the category that `FILTER=` selects. Never put real KYC documents in a public repository.

Check it first. This reads every file and writes nothing:

```bash
make check-data DIR=my_documents
```

It reports every file the local pipeline would fail on, skip or mix up. On a test folder with mistakes in it:

```
10 documents in my_documents: 1 Form1040, 3 ID, 4 UtilityBill, 2 W2

7 problem(s):
  notes.docx: not a document type the pipeline reads (.pdf, .jpeg, .jpg, .png); it would be skipped
  DrivingLicenses/JDDL.jpeg: Git LFS pointer, not the file (run git lfs pull)
  DrivingLicenses/MADL.jpeg: no extracted_text/DrivingLicenses/MADL.txt; local mode reads the text on an image from there
  Utility/broken.pdf: content is not a PDF file
  Utility/scanned_bill.pdf: no text layer (a scan?); local mode can't read it. Add its text, or use cloud mode, where Document AI does OCR
  W2Sample/scan.pdf: content is not a PDF file
  Utility/LauraRussoW2.pdf, W2Sample/LauraRussoW2.pdf: would all be saved as processed-documents/LauraRussoW2_pdf_processed.json, and the last one would replace the others; rename one

Fix or remove these files and run again. Nothing was written.
```

It also catches empty files, password-protected PDFs and card text files that aren't UTF-8, and notes documents with no known category or more than 8,000 characters (the part `create_embeddings.py` keeps). When everything passes, index your documents and search them:

```bash
make load DIR=my_documents
make search QUERY="Laura's driver's license" RESULTS=3
```

```
Indexed 4 documents. Search them with: make search QUERY="..."
...
1. processed-documents/LRDL_jpeg_processed.json (ID) - Score: 0.3943
2. processed-documents/laura_tax_return_form1040_pdf_processed.json (Form1040) - Score: 0.3155
3. processed-documents/LauraRussoW2_pdf_processed.json (W2) - Score: 0.2870
```

`make load` deletes the previous run's output (`processed-documents/` and `vector-search/`), so your documents aren't mixed with the samples, and runs steps 1 to 3 with `--local` on your folder. It took 10 s for four documents on a Mac. `make run` puts the samples back.

## What you end up with

A local search index over your own documents that answers plain-language requests, a checker that tells you which files it can't read, and three measurements of where embedding search falls short.

Skills you'll practice:
- Turning PDFs and images into one text representation
- Embedding documents and searching them by cosine similarity
- Filtering results by metadata (document category)
- Combining vector search with exact matching on names
- Validating documents before they reach the pipeline
- Measuring retrieval quality and search time before changing anything

## Before production

This playbook runs on 22 synthetic documents. Before you run something like it for real:

1. **Find people by entity, not by embedding.** The embedding alone found 8 of 15 of the right person's documents; adding a name match found 15 of 15 ([Try this 1](#try-this)). The cloud mode already extracts entities with Document AI (the `entities` field); index them and filter on them. *Chapter 3, "Selective Context Retrieval" (page 87); Chapter 6, "Hybrid Storage and Retrieval" (page 189).*
2. **Split documents where they really split.** The local model ignores 62% of the text, and `Realistic_Bank_Statements.pdf` holds five customers' statements in one file, so its vector comes from the first page, Laura Russo's. Split multi-person files into one document per person before embedding, and decide how long documents are chunked ([Try this 2](#try-this)). *Chapter 3, "Stage 1: Document Parsing and Representation" (page 92).*
3. **Use an approximate index at scale.** The local index sorts every score for every query: 209 ms per query at 300,000 vectors on a Mac ([Try this 3](#try-this)). The cloud mode's tree-AH index is approximate; measure its recall against an exact search on your data. *Chapter 3, "Stage 3: Vector Indexing" (page 94); Chapter 6, "Vector Databases" (page 180).*
4. **Give every document a stable ID.** `processed_name()` builds the ID from the file name alone, so two files with the same name in different folders overwrite each other's processed file and embedding. Use the full path or a content hash. *Chapter 11, "Lineage and Provenance Tracking" (page 416).*
5. **Count what was skipped.** `process_documents.py` exits 1 when a document fails, but a scanned PDF with no text layer doesn't fail in local mode: it is saved with empty text, and `create_embeddings.py` skips it with a warning and builds the index without it. Fail the run, or report the count, when documents go missing. *Chapter 4, "Validation, Transformation, Observability, and Annotation Across Modalities" (page 119).*
6. **Record which model made the vectors.** The embeddings file stores the vectors and category but not the model. Local vectors (384 dimensions) and cloud vectors (768) can't be mixed, and `text-embedding-005` retires on April 1, 2027. Store the model name with the index and re-embed on a change. *Chapter 8, "Data Versioning and Experiment Tracking for Multimodal AI" (page 276).*
7. **Tear down what bills by the hour.** A deployed Vector Search index is billed per node-hour while it is deployed, whether or not anyone searches. `scripts/teardown_vector_search.py --yes` removes it; see [What it costs](#what-it-costs). *Chapter 3, "Choosing Between Archetypes: A Cost and Complexity Framework" (page 82); Chapter 9, "Cost Optimization for Cloud-Based Multimodal AI Infrastructure" (page 342).*
8. **Protect the documents and the snippets.** KYC files carry names, addresses and tax IDs. The metadata file keeps the first 200 characters of every document, and the search output prints 100 of them. Restrict access to the bucket and index, and keep personal data out of logs and snippets. *Chapter 11, "Protecting Data Across the Pipeline" (page 421); Chapter 11, "Compliance Across Modalities" (page 417).*

## What it costs

| Where you run it | Cost |
|---|---|
| Your machine (local mode) | $0. No cloud account, no API keys. About 1.2 GB of disk: 1.1 GB for the Python environment and 87 MB for the model. |
| A GitHub Codespace (2 cores, 8 GB, the default size; the repo asks for nothing larger) | Estimated $0.18 per hour at GitHub's list price (November 2026). A free personal account includes 120 core-hours a month, about 60 hours on this machine. In CI, `make run` took 125 s in a freshly built devcontainer, including the model download. |
| Google Cloud (cloud mode), the 22 sample documents | Estimated $4 for a run that keeps the index deployed for one hour, then about $0.75 for every further hour it stays deployed. Not measured: the cloud mode wasn't run for this update. Breakdown below. |

The cloud estimate is Google's list price (us-central1, November 2026) times the workload the local run measured: 22 documents, 33 pages, 47,612 characters embedded.

| Service | Workload | List price | Estimated |
|---|---|---|---|
| Document AI, US Driver License Parser | 5 documents | $0.10 per document | $0.50 |
| Document AI, W2 Parser | 5 documents | $0.30 per document | $1.50 |
| Document AI, Form Parser | 16 pages (5 Forms 1040 of 2 pages, 6 utility bills) | $30 per 1,000 pages | $0.48 |
| Document AI, Bank Statement Parser | 1 document | $0.75 per document | $0.75 |
| Embeddings, `text-embedding-005` | 47,612 characters | $0.000025 per 1,000 characters | under $0.01 |
| Vector Search, index build | 22 vectors × 768 dimensions × 4 bytes | $3.00 per GiB | under $0.01 |
| Vector Search, deployed index | 1 node, e2-standard-16 (assumed: the default for the default medium shard size) | $0.75 per node-hour | $0.75 per hour |

Sources: [Document AI pricing](https://cloud.google.com/document-ai/pricing), [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) (Vector Search), [Generative AI on Vertex AI pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) (embeddings). Cloud Storage for a few MB is a fraction of a cent. Check the pages before you start; the first-edition README's "under $10" holds only if you tear the index down the same day.

## How it works

The pipeline extracts the text from each KYC (Know Your Customer) document, turns it into an embedding, indexes the embeddings, and answers queries such as "Form 1040 tax return" or "Mitchell Allen's tax information".

```
data/ (JPEG, PDF) ──▶ 1. extract text ──▶ 2. embed ──▶ 3. index ──▶ 4. search
                         Document AI       text-embedding-005   Vertex AI Vector Search   (cloud, first edition)
                         pypdf + card text all-MiniLM-L6-v2     NumPy matrix              (--local)
```

Every step runs in one of two modes:

| Step | Google Cloud (first edition) | Local (`--local`, laptop CPU, no account) |
|---|---|---|
| Extract text | Document AI processors (OCR and entities) | pypdf for PDFs; the card text in `data/extracted_text/` for the license photos |
| Embed | Vertex AI `text-embedding-005` (768 dimensions) | `all-MiniLM-L6-v2` (384 dimensions) |
| Index and search | Vertex AI Vector Search (tree-AH index on a deployed endpoint) | Cosine similarity over a NumPy matrix |

Both modes read the same 22 sample documents for five fictional customers and write files to the same relative paths (`processed-documents/`, `vector-search/`). The local mode uses a different embedding model, so its scores differ from a cloud run.

> **Vertex AI has a new name.** In April 2026 Google renamed Vertex AI "Gemini Enterprise Agent Platform", and the console uses the new name. The API (`aiplatform.googleapis.com`), the Python packages and this code are unchanged, so this README keeps the book's name.

## Run it the way the book does

The book describes this pipeline in four stages (pages 90–97) without printing commands. These are the first-edition README's steps, as printed apart from the fixes in [Differences from the printed book](#differences-from-the-printed-book). The `make` targets in [Run it](#run-it) run the same local-mode scripts.

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

### Step 3: Install dependencies

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

### Run it on Google Cloud (as in the first edition)

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
gcloud services enable vision.googleapis.com   # as in the first edition; the scripts don't use the Vision API
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

Google has deprecated `gsutil` in favor of `gcloud storage`. The first edition's `gsutil` lines still work; the current command is:

```bash
gcloud storage buckets create gs://YOUR_BUCKET_NAME --location=us-central1
```

The first-edition README also ran `gsutil mkdir gs://YOUR_BUCKET_NAME/data/` and similar lines for the folders. Skip them: `gsutil mkdir` is another name for `gsutil mb` (make bucket), so they fail with "The mb command requires a URL that specifies a bucket". Cloud Storage creates the folders as files are copied into them.

#### 4. Configure

Copy `.env.example` to `.env`, fill in your project, bucket and the four processor IDs, and load it:

```bash
cp .env.example .env            # then edit .env
set -a; source .env; set +a
```

You no longer edit IDs into the source files. The first edition's `--project` and `--bucket` flags still work and take precedence.

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

The first-edition README estimated under $10 for 20–30 documents. That holds only if you clean up promptly: Document AI and the embeddings are billed per page and per request (cents here), but the deployed Vector Search index is billed per node-hour for as long as it stays deployed. Check the current [Vertex AI pricing](https://cloud.google.com/vertex-ai/pricing) before you start.

### Checks

```bash
make smoke    # or: uv run pytest -q
```

`make smoke` (about 1 minute on a laptop CPU) checks the sample data, runs the five steps in local mode, and checks the cloud calls with mocks: the google-genai embedding call, the category filter, processor routing, exit codes and teardown. GitHub Actions runs it on Python 3.12 and 3.13, plus the printed `pip install -r requirements.txt` path.

`uv run --frozen pytest -q tests/test_own_data.py` checks the loader behind `make check-data` and `make load` ([Use your own data](#use-your-own-data)).

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

The book (pages 90–97) describes the pipeline without printing commands. The "First edition" column is the code and README at the `first-edition` tag.

| What | First edition | Now | Why |
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
| Processor IDs, project, bucket | Edited into `src/document_processor.py` and `scripts/process_documents.py` | Environment variables (`.env.example`); the first edition's flags still work | No edits to code; no IDs in commits |
| `gsutil mkdir gs://.../data/` lines | In the README | Skipped | `gsutil mkdir` means `gsutil mb` (make bucket) and fails on a folder path; Cloud Storage creates folders on upload |
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

From my run of the local mode on 2026-10-22: macOS 26.5.1 (arm64), Python 3.12.15, fresh clone, `pip install -r requirements.txt`. Python 3.13.16 gave identical results. This is the output of this version; it may differ from the output printed in the book, and that's okay.

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
├── scripts/load_documents.py      # make check-data / make load: your own documents
├── experiments/                   # make try-hybrid, try-chunks, try-index-size
├── tests/test_smoke.py            # make smoke
├── tests/test_own_data.py         # checks for scripts/load_documents.py
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

If this playbook helped you, a star helps other readers find it.

## License

MIT; see [LICENSE](LICENSE). The sample documents in `data/` are MIT-licensed too.
