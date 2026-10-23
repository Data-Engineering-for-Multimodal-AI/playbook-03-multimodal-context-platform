import os
import sys
import json
import argparse
from typing import List, Dict

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kyc import (DOCUMENT_EXTENSIONS, EXTRACTED_TEXT_DIR, extract_local_text,
                     list_local_documents, mime_type, processed_name)

# Document AI processor IDs, read from the environment (see .env.example).
# Google discontinued the Utility Parser on June 30, 2026; utility bills now go to the Form Parser.
PROCESSOR_ENV = {
    "driving_license": "DOCAI_DRIVER_LICENSE_PROCESSOR_ID",
    "w2": "DOCAI_W2_PROCESSOR_ID",
    "form": "DOCAI_FORM_PROCESSOR_ID",
    "bank_statement": "DOCAI_BANK_STATEMENT_PROCESSOR_ID",
}


def list_sample_documents(bucket_name: str, prefix: str = "data/") -> List[Dict]:
    """List all documents in the data folder and its subfolders."""
    from google.cloud import storage
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)

    # Use recursive listing of all files
    blobs = list(bucket.list_blobs(prefix=prefix))

    documents = []
    for blob in blobs:
        # Skip directories, hidden files and the local-mode text files
        if blob.name.endswith('/') or os.path.basename(blob.name).startswith('.'):
            continue
        if f"/{EXTRACTED_TEXT_DIR}/" in f"/{blob.name}":
            continue

        # Check if it's a document file
        if blob.name.lower().endswith(DOCUMENT_EXTENSIONS):
            documents.append({
                'id': os.path.basename(blob.name),
                'name': blob.name,
                'uri': f"gs://{bucket_name}/{blob.name}"
            })

    return sorted(documents, key=lambda d: d['name'])


def get_processor_type(document_uri: str) -> str:
    """Get the appropriate processor type based on document type/path."""
    document_path = document_uri.lower()
    if 'drivinglicenses' in document_path and document_path.endswith(('.jpg', '.jpeg')):
        return "driving_license"  # US Driver License Parser
    elif 'w2sample' in document_path and document_path.endswith('.pdf'):
        return "w2"  # W2 Parser
    elif 'bank_statements' in document_path or 'bankstatements' in document_path:
        return "bank_statement"  # Bank Statement Parser
    else:
        # Form Parser for Form 1040, utility bills and anything else
        return "form"


def get_processor_id(document_uri: str) -> str:
    """Get the processor ID for a document from the environment."""
    env = PROCESSOR_ENV[get_processor_type(document_uri)]
    processor_id = os.environ.get(env, "")
    if not processor_id:
        raise SystemExit(f"Set {env} to your Document AI processor ID (see .env.example).")
    return processor_id


def process_document(project_id: str, location: str, processor_id: str, file_uri: str) -> Dict:
    """Process a document using Document AI processor."""
    from google.cloud import documentai_v1 as documentai
    from google.cloud import storage
    try:
        # Initialize Document AI client for the processor's location (us or eu)
        client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{location}-documentai.googleapis.com"})

        # Format the processor name
        processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"

        # Read the file content from GCS
        gcs_client = storage.Client()
        bucket_name = file_uri.replace('gs://', '').split('/')[0]
        blob_name = file_uri.replace(f'gs://{bucket_name}/', '')
        content = gcs_client.bucket(bucket_name).blob(blob_name).download_as_bytes()

        # Configure the process request
        document = documentai.RawDocument(content=content, mime_type=mime_type(file_uri))
        request = documentai.ProcessRequest(name=processor_name, raw_document=document)

        # Process the document
        result = client.process_document(request=request)
        document = result.document

        # Extract the text and structured data
        text = document.text

        # Build the entity list
        entities = []
        for entity in document.entities:
            for segment in entity.text_anchor.text_segments:
                entities.append({
                    "type": entity.type_,
                    "text": text[segment.start_index:segment.end_index],
                    "confidence": entity.confidence
                })

        return {
            "text": text,
            "entities": entities,
            "pages": len(document.pages),
            "mime_type": mime_type(file_uri),
            "document_uri": file_uri
        }

    except Exception as e:
        print(f"Error processing document {file_uri}: {str(e)}")
        return {"document_uri": file_uri, "error": str(e)}


def process_local_document(path: str, sample_dir: str) -> Dict:
    """Local mode: the same output fields, with text from pypdf or extracted_text/."""
    try:
        text, pages = extract_local_text(path, sample_dir)
    except Exception as e:
        print(f"Error processing document {path}: {str(e)}")
        return {"document_uri": path, "error": str(e)}
    return {
        "text": text,
        "entities": [],   # entity extraction needs Document AI (cloud mode)
        "pages": pages,
        "mime_type": mime_type(path),
        "document_uri": path
    }


def save_processed_document(result: Dict, output_bucket: str, output_prefix: str):
    """Save the processed document result to GCS."""
    from google.cloud import storage
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket)
        output_file = f"{output_prefix}/{processed_name(result['document_uri'])}"
        bucket.blob(output_file).upload_from_string(json.dumps(result, indent=2))
        return output_file
    except Exception as e:
        print(f"Error saving processed result: {str(e)}")
        return None


def save_local_processed_document(result: Dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    output_file = os.path.join(output_dir, processed_name(result['document_uri']))
    with open(output_file, "w") as fh:
        json.dump(result, fh, indent=2)
    return output_file


def main(project_id: str, bucket_name: str, sample_data_prefix: str, output_prefix: str,
         docai_location: str = "us", local: bool = False):
    """Main function to process all documents. Returns the number of failed documents."""
    if local:
        print(f"Listing documents in {sample_data_prefix}")
        documents = [{'name': p, 'uri': p} for p in list_local_documents(sample_data_prefix)]
    else:
        print(f"Listing documents in gs://{bucket_name}/{sample_data_prefix}")
        documents = list_sample_documents(bucket_name, sample_data_prefix)
    print(f"Found {len(documents)} documents")

    if not documents:
        print("No documents found. Please check your bucket path.")
        return 1

    results, failed = [], 0
    for i, doc in enumerate(documents):
        print(f"Processing document {i+1}/{len(documents)}: {doc['name']}")
        if local:
            result = process_local_document(doc['uri'], sample_data_prefix)
        else:
            processor_id = get_processor_id(doc['uri'])
            print(f"Using processor ID: {processor_id}")
            result = process_document(project_id, docai_location, processor_id, doc['uri'])

        # Don't save failed documents: their error text would be embedded as content
        if "error" in result:
            failed += 1
            output_file = None
        elif local:
            output_file = save_local_processed_document(result, output_prefix)
            print(f"Saved result to: {output_file}")
        else:
            output_file = save_processed_document(result, bucket_name, output_prefix)
            if output_file:
                print(f"Saved result to: gs://{bucket_name}/{output_file}")
            else:
                failed += 1

        results.append({
            "document": doc['name'],
            "text_length": len(result.get("text", "")),
            "entities_found": len(result.get("entities", [])),
            "output_file": output_file,
            "error": result.get("error")
        })

    summary = {"processed_documents": len(results) - failed, "failed_documents": failed,
               "documents": results}
    if local:
        summary_path = save_summary_local(summary, output_prefix)
    else:
        from google.cloud import storage
        bucket = storage.Client().bucket(bucket_name)
        bucket.blob(f"{output_prefix}/processing_summary.json").upload_from_string(json.dumps(summary, indent=2))
        summary_path = f"gs://{bucket_name}/{output_prefix}/processing_summary.json"

    print(f"Processing complete. Summary saved to: {summary_path}")
    if failed:
        print(f"{failed} of {len(documents)} documents failed; see the errors above.")
    return failed


def save_summary_local(summary: Dict, output_dir: str) -> str:
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, "processing_summary.json")
    with open(path, "w") as fh:
        json.dump(summary, fh, indent=2)
    return path


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents with Document AI")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "YOUR_BUCKET_NAME"), help="GCS bucket name")
    parser.add_argument("--sample-data", default="data", help="Prefix for sample data (a local folder with --local)")
    parser.add_argument("--output", default="processed-documents", help="Output path for processed documents")
    parser.add_argument("--docai-location", default=os.environ.get("DOCAI_LOCATION", "us"),
                        help="Document AI location of your processors: us or eu")
    parser.add_argument("--local", action="store_true",
                        help="Run on this machine without Google Cloud (pypdf; text files for images)")

    args = parser.parse_args()

    failed = main(args.project, args.bucket, args.sample_data, args.output, args.docai_location, args.local)
    sys.exit(1 if failed else 0)
