import os
import json
import uuid
from google.cloud import storage
from google.cloud import aiplatform
import glob
from tqdm import tqdm
import argparse

def download_processed_documents(bucket_name, processed_docs_prefix):
    """Download processed document files from GCS."""
    print("Downloading processed documents...")
    local_dir = "processed_docs"
    os.makedirs(local_dir, exist_ok=True)
    
    # Initialize Storage client
    storage_client = storage.Client()
    bucket = storage_client.bucket(bucket_name)
    
    blobs = list(bucket.list_blobs(prefix=processed_docs_prefix))
    for blob in tqdm(blobs):
        if blob.name.endswith('.json'):
            local_path = os.path.join(local_dir, os.path.basename(blob.name))
            blob.download_to_filename(local_path)
    
    return local_dir

def extract_document_content(file_path):
    """Extract the text content from a processed document."""
    with open(file_path, 'r') as f:
        doc_data = json.load(f)
    
    # Extract text from the document
    if 'text' in doc_data:
        return doc_data['text']
    elif 'pages' in doc_data:
        text = []
        for page in doc_data['pages']:
            if 'text' in page:
                text.append(page['text'])
        return "\n".join(text)
    else:
        # Try to find any text field
        for key, value in doc_data.items():
            if isinstance(value, str) and len(value) > 100:
                return value
        
        return json.dumps(doc_data)  # Last resort: use the entire JSON

def get_document_category(file_path):
    """Determine document category from filename."""
    filename = os.path.basename(file_path)
    if 'Form1040' in filename:
        return 'Form1040'
    elif 'w2' in filename.lower():
        return 'W2'
    elif 'license' in filename.lower() or 'id' in filename.lower() or 'DL' in filename:
        return 'ID'
    elif 'utility' in filename.lower() or 'bill' in filename.lower():
        return 'UtilityBill'
    elif 'bank' in filename.lower() or 'statement' in filename.lower():
        return 'BankStatement'
    else:
        return 'Other'

def get_embedding(text, project_id, location):
    """Get embedding from Vertex AI API using text-embedding-005."""
    from vertexai.language_models import TextEmbeddingModel
    
    # Initialize Vertex AI
    import vertexai
    vertexai.init(project=project_id, location=location)
    
    # Get the embedding model
    model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
    # Get embeddings
    embeddings = model.get_embeddings([text])
    
    # Return the embedding values
    return embeddings[0].values

def create_embeddings(project_id, location, bucket_name, processed_docs_prefix, vector_search_prefix):
    """Create embeddings for all processed documents."""
    print("Creating embeddings...")
    
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Initialize Storage client
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    
    local_dir = download_processed_documents(bucket_name, processed_docs_prefix)
    
    # Create vector search directory in GCS if it doesn't exist
    vs_blob = bucket.blob(f"{vector_search_prefix}/")
    if not vs_blob.exists():
        vs_blob.upload_from_string('')
    
    # Process each document
    documents = []
    embeddings = []
    
    # Get all JSON files in the processed docs directory
    json_files = glob.glob(os.path.join(local_dir, "*.json"))
    print(f"Found {len(json_files)} processed documents")
    
    for file_path in tqdm(json_files):
        print(f"Processing {file_path}...")
        doc_id = os.path.basename(file_path)  # Use filename as document ID for consistency
        
        # Extract document content
        content = extract_document_content(file_path)
        if not content:
            print(f"Warning: Could not extract content from {file_path}")
            continue
        
        # Truncate content if too long (text-embedding-005 has a context limit)
        if len(content) > 8000:
            content = content[:8000]
        
        # Get embedding
        try:
            embedding = get_embedding(content, project_id, location)
            
            # Get document category
            category = get_document_category(file_path)
            
            # Create document metadata
            doc_metadata = {
                "document_path": f"{processed_docs_prefix}/{os.path.basename(file_path)}",
                "text_snippet": content[:200] + "...",
                "category": category
            }
            
            # Create embedding record in the format Vector Search expects
            embedding_record = {
                "id": doc_id,
                "embedding": embedding,
                "restricts": [
                    {
                        "namespace": "category",
                        "allow": [category]
                    }
                ],
                "metadata": doc_metadata
            }
            
            embeddings.append(embedding_record)
            
            # Store document info
            doc_info = {
                "id": doc_id,
                "path": f"{processed_docs_prefix}/{os.path.basename(file_path)}",
                "category": category
            }
            documents.append(doc_info)
            
        except Exception as e:
            print(f"Error creating embedding for {file_path}: {str(e)}")
    
    # Save all embeddings to a single JSON Lines file
    embedding_file = "embeddings.json"
    with open(embedding_file, 'w') as f:
        for emb in embeddings:
            f.write(json.dumps(emb) + '\n')
    
    # Upload to GCS
    embedding_blob = bucket.blob(f"{vector_search_prefix}/embeddings.json")
    embedding_blob.upload_from_filename(embedding_file)
    
    print(f"Uploaded {len(embeddings)} embeddings to {embedding_blob.name}")
    return embedding_blob.name

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Create embeddings from processed documents")
    parser.add_argument("--project", default="YOUR_PROJECT_ID", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--bucket", default="YOUR_BUCKET_NAME", help="GCS bucket name")
    parser.add_argument("--processed-docs", default="processed-documents", help="Prefix for processed documents")
    parser.add_argument("--vector-search", default="vector-search/document-embeddings", help="Path for vector search embeddings")
    
    args = parser.parse_args()
    
    embedding_path = create_embeddings(
        args.project, 
        args.location, 
        args.bucket, 
        args.processed_docs, 
        args.vector_search
    )
    
    print(f"Embeddings created and stored at: gs://{args.bucket}/{embedding_path}")
    print("Now you can run the setup_vector_search.py script to create the index.") 