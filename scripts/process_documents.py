from google.cloud import documentai_v1 as documentai
from google.cloud import storage
import os
import json
import argparse
from typing import List, Dict, Any
from tqdm import tqdm

def list_sample_documents(bucket_name: str, prefix: str = "data/") -> List[Dict]:
    """List all documents in the data folder and its subfolders."""
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    # Use recursive listing of all files
    blobs = list(bucket.list_blobs(prefix=prefix))
    
    documents = []
    for blob in blobs:
        # Skip directories and hidden files
        if blob.name.endswith('/') or os.path.basename(blob.name).startswith('.'):
            continue
            
        # Check if it's a document file
        if blob.name.lower().endswith(('.pdf', '.jpeg', '.jpg', '.png')):
            documents.append({
                'id': os.path.basename(blob.name),
                'name': blob.name,
                'uri': f"gs://{bucket_name}/{blob.name}"
            })
    
    return documents

def get_processor_id(document_uri: str) -> str:
    """Get the appropriate processor ID based on document type/path."""
    # Determine document type based on path and file extension
    document_path = document_uri.lower()
    
    # NOTE: You need to replace these with your own processor IDs from Document AI
    if 'drivinglicenses' in document_path and document_path.endswith(('.jpg', '.jpeg')):
        return "YOUR_DRIVING_LICENSE_PROCESSOR_ID"  # Driving License Parser
    elif 'w2sample' in document_path and document_path.endswith('.pdf'):
        return "YOUR_W2_PROCESSOR_ID"  # W2 Parser
    elif 'form1040sample' in document_path and document_path.endswith('.pdf'):
        return "YOUR_FORM_PROCESSOR_ID"  # Form Parser
    elif 'utility' in document_path and document_path.endswith('.pdf'):
        return "YOUR_UTILITY_PROCESSOR_ID"  # Utility Parser
    elif 'bank_statements' in document_path or 'bankstatements' in document_path or 'realistic_bank_statements' in document_path:
        return "YOUR_BANK_STATEMENT_PROCESSOR_ID"  # Bank Statement Parser
    else:
        # Default to Form Parser for other documents
        return "YOUR_FORM_PROCESSOR_ID"

def process_document(project_id: str, location: str, processor_id: str, file_uri: str) -> Dict:
    """Process a document using Document AI processor."""
    try:
        # Initialize Document AI client
        client = documentai.DocumentProcessorServiceClient()
        
        # Format the processor name
        processor_name = f"projects/{project_id}/locations/{location}/processors/{processor_id}"
        
        # Read the file content from GCS
        gcs_client = storage.Client()
        bucket_name = file_uri.replace('gs://', '').split('/')[0]
        blob_name = '/'.join(file_uri.replace(f'gs://{bucket_name}/', '').split('/'))
        bucket = gcs_client.bucket(bucket_name)
        blob = bucket.blob(blob_name)
        content = blob.download_as_bytes()
        
        # Determine the MIME type based on file extension
        mime_type = "application/pdf"
        if file_uri.lower().endswith(('.jpg', '.jpeg')):
            mime_type = "image/jpeg"
        elif file_uri.lower().endswith('.png'):
            mime_type = "image/png"
        
        # Configure the process request
        document = documentai.RawDocument(content=content, mime_type=mime_type)
        request = documentai.ProcessRequest(name=processor_name, raw_document=document)
        
        # Process the document
        result = client.process_document(request=request)
        document = result.document
        
        # Extract the text and structured data
        text = document.text
        
        # Build the entity list
        entities = []
        for entity in document.entities:
            # Get the text segments
            text_segments = entity.text_anchor.text_segments
            for segment in text_segments:
                # Extract the text for this segment
                start_index = segment.start_index
                end_index = segment.end_index
                segment_text = text[start_index:end_index]
                
                # Add to entities list
                entities.append({
                    "type": entity.type_,
                    "text": segment_text,
                    "confidence": entity.confidence
                })
        
        # Return the extracted text and additional information
        return {
            "text": text,
            "entities": entities,
            "pages": len(document.pages),
            "mime_type": mime_type,
            "document_uri": file_uri
        }
        
    except Exception as e:
        print(f"Error processing document {file_uri}: {str(e)}")
        return {
            "text": f"Error processing document: {str(e)}",
            "entities": [],
            "document_uri": file_uri,
            "error": str(e)
        }

def save_processed_document(result: Dict, output_bucket: str, output_prefix: str):
    """Save the processed document result to GCS."""
    try:
        storage_client = storage.Client()
        bucket = storage_client.bucket(output_bucket)
        
        # Create a unique filename
        document_id = os.path.basename(result["document_uri"]).replace(".", "_")
        output_file = f"{output_prefix}/{document_id}_processed.json"
        
        # Save to GCS
        blob = bucket.blob(output_file)
        blob.upload_from_string(json.dumps(result, indent=2))
        
        return output_file
    except Exception as e:
        print(f"Error saving processed result: {str(e)}")
        return None

def main(project_id: str, bucket_name: str, sample_data_prefix: str, output_prefix: str):
    """Main function to process all documents."""
    try:
        # List sample documents
        print(f"Listing documents in gs://{bucket_name}/{sample_data_prefix}")
        documents = list_sample_documents(bucket_name, sample_data_prefix)
        print(f"Found {len(documents)} documents")
        
        if not documents:
            print("No documents found. Please check your bucket path.")
            return
        
        # Process each document
        results = []
        for i, doc in enumerate(tqdm(documents)):
            print(f"Processing document {i+1}/{len(documents)}: {doc['name']}")
            
            # Get appropriate processor
            processor_id = get_processor_id(doc['uri'])
            print(f"Using processor ID: {processor_id}")
            
            # Process document
            result = process_document(project_id, "us", processor_id, doc['uri'])
            
            # Save result
            output_file = save_processed_document(result, bucket_name, output_prefix)
            if output_file:
                print(f"Saved result to: gs://{bucket_name}/{output_file}")
                
            results.append({
                "document": doc['name'],
                "text_length": len(result.get("text", "")),
                "entities_found": len(result.get("entities", [])),
                "output_file": output_file
            })
        
        # Save summary
        summary = {
            "processed_documents": len(results),
            "documents": results
        }
        
        storage_client = storage.Client()
        bucket = storage_client.bucket(bucket_name)
        summary_blob = bucket.blob(f"{output_prefix}/processing_summary.json")
        summary_blob.upload_from_string(json.dumps(summary, indent=2))
        
        print(f"Processing complete. Summary saved to: gs://{bucket_name}/{output_prefix}/processing_summary.json")
        
    except Exception as e:
        print(f"Error in main process: {str(e)}")
        raise

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Process documents with Document AI")
    parser.add_argument("--project", default="YOUR_PROJECT_ID", help="GCP project ID")
    parser.add_argument("--bucket", default="YOUR_BUCKET_NAME", help="GCS bucket name")
    parser.add_argument("--sample-data", default="data", help="Prefix for sample data")
    parser.add_argument("--output", default="processed-documents", help="Output path for processed documents")
    
    args = parser.parse_args()
    
    main(args.project, args.bucket, args.sample_data, args.output) 