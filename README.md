# Multimodal Document Retrieval Pipeline

A semantic search system for KYC (Know Your Customer) documents using Google Cloud's Document AI, Vertex AI, and Vector Search.

## Overview

This project implements a complete pipeline for processing, embedding, and semantically searching KYC documents such as:
- ID cards/Driver's licenses
- Tax forms (Form 1040)
- W2 statements
- Utility bills
- Bank statements

The pipeline uses:
1. **Google Document AI** to extract text and entities from documents
2. **Vertex AI Embeddings** to convert document text into vector embeddings
3. **Vertex AI Vector Search** to build a searchable index of documents
4. A simple search interface to find documents using natural language queries

## Prerequisites

- Google Cloud account with billing enabled
- Python 3.8+
- The following Google Cloud APIs enabled:
  - Document AI API
  - Vision API
  - Vertex AI API
  - Cloud Storage API
- Document AI processors set up for different document types
- Service account with appropriate permissions
- Google Cloud CLI (gcloud) installed and configured

## Setup

### 1. Clone the Repository

```bash
git clone https://github.com/Data-Engineering-for-Multimodal-AI/pb3-mm-docretrievalpipeline.git
cd pb3-mm-docretrievalpipeline
```

### 2. Install Dependencies

```bash
pip install -r requirements.txt
```

### 3. Set Up Google Cloud

```bash
# Authenticate with Google Cloud
gcloud auth login

# Set your project ID
gcloud config set project YOUR_PROJECT_ID

# Enable required APIs
gcloud services enable documentai.googleapis.com
gcloud services enable aiplatform.googleapis.com
gcloud services enable storage.googleapis.com
gcloud services enable vision.googleapis.com
```

### 4. Set Up Document AI Processors

You'll need to create Document AI processors for each document type:

1. Go to the [Google Cloud Console](https://console.cloud.google.com/)
2. Navigate to Document AI > Processors
3. Create the following processors:
   - Driving License Parser
   - W2 Parser
   - Form Parser (for Form 1040)
   - Utility Bill Parser
   - Bank Statement Parser
4. Note the processor IDs for each processor

### 5. Create a Cloud Storage Bucket

```bash
# Create a GCS bucket to store your documents and processed results
gsutil mb -l us-central1 gs://YOUR_BUCKET_NAME/

# Create folders for sample data and processed documents
gsutil mkdir gs://YOUR_BUCKET_NAME/data/
gsutil mkdir gs://YOUR_BUCKET_NAME/processed-documents/
gsutil mkdir gs://YOUR_BUCKET_NAME/vector-search/
```

### 6. Update Configuration

Edit the processor IDs and project-specific information in:
- `src/document_processor.py`
- `scripts/process_documents.py`

## Pipeline Steps

### 1. Upload Sample Data

Upload sample KYC documents to your bucket:

```bash
# Create subfolders for different document types
gsutil mkdir gs://YOUR_BUCKET_NAME/data/DrivingLicenses/
gsutil mkdir gs://YOUR_BUCKET_NAME/data/W2Sample/
gsutil mkdir gs://YOUR_BUCKET_NAME/data/Form1040Sample/
gsutil mkdir gs://YOUR_BUCKET_NAME/data/Utility/

# Upload sample documents to their respective folders
gsutil cp -r data/DrivingLicenses/* gs://YOUR_BUCKET_NAME/data/DrivingLicenses/
gsutil cp -r data/W2Sample/* gs://YOUR_BUCKET_NAME/data/W2Sample/
gsutil cp -r data/Form1040Sample/* gs://YOUR_BUCKET_NAME/data/Form1040Sample/
gsutil cp -r data/Utility/* gs://YOUR_BUCKET_NAME/data/Utility/
gsutil cp data/Realistic_Bank_Statements.pdf gs://YOUR_BUCKET_NAME/data/
```

### 2. Process Documents with Document AI

This step uses Document AI processors to extract text and structured information from your documents:

```bash
python scripts/process_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --sample-data data --output processed-documents
```

Expected output:
```
Listing documents in gs://YOUR_BUCKET_NAME/data
Found 22 documents
Processing document 1/22: data/DrivingLicenses/AHDL.jpeg
Using processor ID: YOUR_DRIVING_LICENSE_PROCESSOR_ID
...
Processing complete. Summary saved to: gs://YOUR_BUCKET_NAME/processed-documents/processing_summary.json
```

### 3. Generate Embeddings

This step creates vector embeddings for each processed document using Vertex AI's text-embedding-005 model:

```bash
python scripts/create_embeddings.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME
```

Expected output:
```
Downloading processed documents...
100%|██████████| 22/22 [00:03<00:00, 6.32it/s]
Found 22 processed documents
Creating embeddings...
100%|██████████| 22/22 [00:42<00:00, 0.52it/s]
Uploaded 22 embeddings to vector-search/document-embeddings/embeddings.json
```

### 4. Set Up Vector Search

This step creates and deploys a Vector Search index:

```bash
python scripts/setup_vector_search.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME
```

Expected output:
```
Setting up vector search for project: YOUR_PROJECT_ID
Creating vector search index...
NOTE: This process will take 30-45 minutes to complete
Waiting for index creation... (this may take 20-30 minutes)
Creating index endpoint...
Waiting for endpoint creation... (this may take 5-10 minutes)
Deploying index to endpoint with ID: kyc_documents_index_deployed
Waiting for index deployment... (this may take 10-15 minutes)
Vector search setup completed successfully!
Index name: kyc-documents-index
Index ID: projects/YOUR_PROJECT_ID/locations/us-central1/indexes/1234567890123456789
Endpoint name: kyc-documents-index-endpoint
Endpoint ID: projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/9876543210987654321
Deployed index ID: kyc_documents_index_deployed
```

Make note of the Endpoint ID for the next step.

### 5. Search Documents

Test general document searches:

```bash
python scripts/search_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID"
```

Expected output:
```
Testing document search with Vector Search:

Query: 'Form 1040 tax return'
Found 3 similar documents:
1. processed-documents/MitchellAllenForm1040_pdf_processed.json (Form1040) - Score: 0.7357
   Snippet: "US Individual Income Tax Return 1040..."

2. processed-documents/TaylorCoxForm1040_pdf_processed.json (Form1040) - Score: 0.7356
   Snippet: "US Individual Income Tax Return 1040..."

3. processed-documents/AprilHensleyForm1040_pdf_processed.json (Form1040) - Score: 0.7349
   Snippet: "US Individual Income Tax Return 1040..."
```

### 6. Test Person-Specific Searches

Test person-specific searches to demonstrate the semantic understanding of the system:

```bash
python scripts/test_person_search.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID"
```

Expected output:
```
=== Testing Query: 'Laura's KYC documents' ===
Found 10 matching documents:
1. LRDL_jpeg_processed.json (ID) - Score: 0.5082
   Snippet: "ILLINOIS DRIVER'S LICENSE..."

2. LauraRussoForm1040_pdf_processed.json (Form1040) - Score: 0.5056
   Snippet: "US Individual Income Tax Return 1040..."

=== Testing Query: 'Mitchell Allen's tax information' ===
Found 10 matching documents:
1. MitchellAllenForm1040_pdf_processed.json (Form1040) - Score: 0.6167
   Snippet: "US Individual Income Tax Return 1040..."
```

## Project Structure

- `src/` - Core classes and modules
  - `document_processor.py` - Document processing with Document AI
  - `vector_search.py` - Vector search functionality
- `scripts/` - Pipeline execution scripts
  - `process_documents.py` - Process documents with Document AI
  - `create_embeddings.py` - Generate embeddings for documents
  - `setup_vector_search.py` - Set up vector search index
  - `search_documents.py` - Search for documents
  - `test_person_search.py` - Test person-specific queries
- `data/` - Sample document data
- `requirements.txt` - Project dependencies

## Advanced Usage

### Single Document Search Query

To search for a specific document:

```bash
python scripts/search_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID" --query "Find Laura's driver's license" --results 5
```

### Filter by Document Category

To filter results by document category:

```bash
python scripts/search_documents.py --project YOUR_PROJECT_ID --bucket YOUR_BUCKET_NAME --endpoint "projects/YOUR_PROJECT_ID/locations/us-central1/indexEndpoints/YOUR_ENDPOINT_ID" --query "James tax documents" --filter "Form1040" --results 3
```

## Troubleshooting

- **Document AI Processing Errors**: Check that your processor IDs are correct and the processors are trained for the specific document types.
- **Vector Search Errors**: Ensure the embedding dimension (768 for text-embedding-005) matches the configuration in vector_search.py.
- **Access Denied Errors**: Verify that your service account has the necessary permissions for all GCP services.

## Cost Considerations

This implementation uses several Google Cloud services that incur costs:
- Document AI processing (per page)
- Vertex AI embedding generation (per request)
- Vector Search creation and querying (per index and per query)
- Cloud Storage (storage and operations)

For a sample dataset of 20-30 documents, costs should be minimal (<$10) for setup and testing.

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

The MIT License is a permissive license that allows for reuse with few restrictions. You are free to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the software, provided that the original copyright notice and permission notice appear in all copies.

### What You Can Do

- Use the code commercially
- Modify the code
- Distribute the code
- Place warranty
- Use the code privately or publicly

### Limitations

- The license provides no warranty
- The authors/maintainers are not liable for any damages or issues arising from the use of this software

This open-source license is designed to be simple and business-friendly while requiring that the license text itself and copyright notice be preserved. 
