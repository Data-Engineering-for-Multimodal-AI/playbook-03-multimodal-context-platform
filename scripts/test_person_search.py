import os
import json
import argparse
from google.cloud import storage
from google.cloud import aiplatform
import vertexai
from vertexai.language_models import TextEmbeddingModel

def get_embedding(text, project_id, location):
    """Get embedding from Vertex AI API using text-embedding-005."""
    # Initialize Vertex AI
    vertexai.init(project=project_id, location=location)
    
    # Get the embedding model
    embedding_model = TextEmbeddingModel.from_pretrained("text-embedding-005")
    
    # Get embeddings
    embeddings = embedding_model.get_embeddings([text])
    
    # Return the embedding values
    return embeddings[0].values

def load_document_mappings(bucket_name, vector_search_prefix, project_id):
    """Load document mappings from embeddings file in GCS."""
    storage_client = storage.Client(project=project_id)
    bucket = storage_client.bucket(bucket_name)
    blob = bucket.blob(f"{vector_search_prefix}/embeddings.json")
    
    content = blob.download_as_string().decode("utf-8")
    lines = content.strip().split("\n")
    
    mappings = {}
    for line in lines:
        embedding_data = json.loads(line)
        doc_id = embedding_data["id"]
        metadata = embedding_data["metadata"]
        mappings[doc_id] = metadata
    
    print(f"Loaded mappings for {len(mappings)} documents")
    return mappings

def search_similar_documents(query_text, endpoint_name, deployed_index_id, 
                           project_id, location, num_results=10):
    """Search for similar documents to the query text."""
    # Get embedding for the query text
    query_embedding = get_embedding(query_text, project_id, location)
    
    # Get the endpoint
    endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_name)
    print(f"Using endpoint: {endpoint.display_name}")
    
    # Find similar documents
    response = endpoint.find_neighbors(
        deployed_index_id=deployed_index_id,
        queries=[query_embedding],
        num_neighbors=num_results
    )
    
    return response[0]  # Return the neighbors for the first query

def main(project_id, location, bucket_name, vector_search_prefix, endpoint_name, deployed_index_id):
    """Main function to test person-specific searches."""
    # Initialize Vertex AI
    aiplatform.init(project=project_id, location=location)
    
    # Load document mappings
    doc_mappings = load_document_mappings(bucket_name, vector_search_prefix, project_id)
    
    # Test person-specific KYC queries
    person_queries = [
        "Laura's KYC documents",
        "April Hensley's identification documents",
        "James Williams' personal information",
        "Taylor Cox's financial records",
        "Mitchell Allen's tax information"
    ]
    
    for query in person_queries:
        print(f"\n\n=== Testing Query: '{query}' ===")
        try:
            neighbors = search_similar_documents(
                query, 
                endpoint_name, 
                deployed_index_id,
                project_id, 
                location
            )
            
            print(f"Found {len(neighbors)} matching documents:")
            for i, neighbor in enumerate(neighbors):
                doc_id = neighbor.id
                distance = neighbor.distance
                doc_info = doc_mappings.get(doc_id, {"document_path": "Unknown", "category": "Unknown"})
                
                doc_path = doc_info.get('document_path', 'Unknown')
                filename = doc_path.split('/')[-1] if '/' in doc_path else doc_path
                
                # Extract document category
                category = doc_info.get('category', 'Unknown')
                
                print(f"{i+1}. {filename} ({category}) - Score: {distance:.4f}")
                print(f"   Snippet: {doc_info.get('text_snippet', '')[:150]}...")
                
        except Exception as e:
            print(f"Error: {str(e)}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test person-specific document searches")
    parser.add_argument("--project", default="YOUR_PROJECT_ID", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--bucket", default="YOUR_BUCKET_NAME", help="GCS bucket name")
    parser.add_argument("--vector-search", default="vector-search/document-embeddings", help="Path for vector search embeddings")
    parser.add_argument("--endpoint", default=None, help="Vector search endpoint resource name (required)")
    parser.add_argument("--deployed-index", default="kyc_documents_index_deployed", help="Deployed index ID")
    
    args = parser.parse_args()
    
    if args.endpoint is None:
        print("Error: --endpoint is required. Please provide the full resource name of your MatchingEngineIndexEndpoint.")
        print("Example: projects/123456789/locations/us-central1/indexEndpoints/987654321")
        exit(1)
    
    main(
        args.project, 
        args.location, 
        args.bucket, 
        args.vector_search, 
        args.endpoint, 
        args.deployed_index
    ) 