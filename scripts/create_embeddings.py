import os
import sys
import json
import glob
import argparse
from tqdm import tqdm

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kyc import METADATA_FILE, Embedder, document_category


def download_processed_documents(bucket_name, processed_docs_prefix, project_id=None):
    """Download processed document files from GCS."""
    from google.cloud import storage
    print("Downloading processed documents...")
    local_dir = "processed_docs"
    os.makedirs(local_dir, exist_ok=True)

    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)

    blobs = list(bucket.list_blobs(prefix=processed_docs_prefix))
    for blob in tqdm(blobs):
        if blob.name.endswith('_processed.json'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)

    return local_dir


def extract_document_content(doc_data):
    """Extract the text content from a processed document."""
    if doc_data.get('text'):
        return doc_data['text']
    # Try to find any long text field
    for value in doc_data.values():
        if isinstance(value, str) and len(value) > 100:
            return value
    return ""


def get_document_category(doc_data, file_path):
    """Document category from the original document's folder, or the file name."""
    return document_category(doc_data.get('document_uri') or os.path.basename(file_path))


def create_embeddings(project_id, location, bucket_name, processed_docs_prefix, vector_search_prefix,
                      local=False):
    """Create embeddings for all processed documents.

    Writes two files:
      <vector-search>/embeddings.json   JSON Lines in the Vector Search input format
                                        (id, embedding, restricts), the only files
                                        allowed in the index's contents folder
      <parent>/document-metadata.json   path, category and snippet for each id, which the
                                        search scripts use to label results
    """
    if local:
        local_dir = processed_docs_prefix
    else:
        local_dir = download_processed_documents(bucket_name, processed_docs_prefix, project_id)

    json_files = sorted(f for f in glob.glob(os.path.join(local_dir, "*_processed.json")))
    print(f"Found {len(json_files)} processed documents")

    embedder = Embedder(local, project_id, location)
    print(f"Creating embeddings with {embedder.model_name}...")

    records, metadata = [], {}
    for file_path in tqdm(json_files):
        doc_id = os.path.basename(file_path)  # Use filename as document ID for consistency
        with open(file_path) as fh:
            doc_data = json.load(fh)

        content = extract_document_content(doc_data)
        if not content:
            print(f"Warning: Could not extract content from {file_path}")
            continue

        # Truncate content if too long (text-embedding-005 accepts up to 2,048 tokens)
        content = content[:8000]

        try:
            embedding = embedder.embed([content])[0]
        except Exception as e:
            print(f"Error creating embedding for {file_path}: {str(e)}")
            continue

        category = get_document_category(doc_data, file_path)
        records.append({
            "id": doc_id,
            "embedding": embedding,
            "restricts": [{"namespace": "category", "allow": [category]}],
        })
        metadata[doc_id] = {
            "document_path": f"{processed_docs_prefix}/{doc_id}",
            "text_snippet": content[:200] + "...",
            "category": category,
        }

    if not records:
        raise SystemExit("No embeddings were created.")

    parent = os.path.dirname(vector_search_prefix.rstrip("/"))
    metadata_path = f"{parent}/{METADATA_FILE}" if parent else METADATA_FILE
    embeddings_lines = "".join(json.dumps(r) + "\n" for r in records)

    if local:
        os.makedirs(vector_search_prefix, exist_ok=True)
        embedding_path = os.path.join(vector_search_prefix, "embeddings.json")
        with open(embedding_path, "w") as fh:
            fh.write(embeddings_lines)
        with open(metadata_path, "w") as fh:
            json.dump(metadata, fh, indent=2)
        print(f"Saved {len(records)} embeddings ({len(records[0]['embedding'])} dimensions) to {embedding_path}")
        return embedding_path

    from google.cloud import storage
    bucket = storage.Client(project=project_id).bucket(bucket_name)
    embedding_blob = bucket.blob(f"{vector_search_prefix}/embeddings.json")
    embedding_blob.upload_from_string(embeddings_lines)
    bucket.blob(metadata_path).upload_from_string(json.dumps(metadata, indent=2))
    print(f"Uploaded {len(records)} embeddings to {embedding_blob.name}")
    return f"gs://{bucket_name}/{embedding_blob.name}"


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings from processed documents")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"), help="GCP region")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "YOUR_BUCKET_NAME"), help="GCS bucket name")
    parser.add_argument("--processed-docs", default="processed-documents", help="Prefix for processed documents")
    parser.add_argument("--vector-search", default="vector-search/document-embeddings", help="Path for vector search embeddings")
    parser.add_argument("--local", action="store_true",
                        help="Embed on this machine with a local model instead of Vertex AI")

    args = parser.parse_args()

    embedding_path = create_embeddings(
        args.project,
        args.location,
        args.bucket,
        args.processed_docs,
        args.vector_search,
        args.local,
    )

    print(f"Embeddings created and stored at: {embedding_path}")
    print(f"Now you can run the setup_vector_search.py script{' --local' if args.local else ''} to create the index.")
