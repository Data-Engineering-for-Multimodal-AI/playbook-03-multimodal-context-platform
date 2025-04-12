import argparse
import os
import sys
import time

# Add the src directory to the path so we can import the VectorSearch class
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.vector_search import VectorSearch

def main(project_id, location, bucket_name, index_name):
    """Set up vector search index and deploy it."""
    print(f"Setting up vector search for project: {project_id}")
    print(f"Using bucket: {bucket_name}")
    print(f"Index name: {index_name}")
    
    # Initialize vector search
    vector_search = VectorSearch(project_id, location, index_name)
    
    # Create and deploy the index
    print("Creating and deploying vector search index...")
    print("NOTE: This process will take 30-45 minutes to complete")
    
    try:
        vector_search.create_index(bucket_name)
        print("Vector search index created and deployed successfully!")
        return True
    except Exception as e:
        print(f"Error setting up vector search: {str(e)}")
        return False

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up vector search index")
    parser.add_argument("--project", default="YOUR_PROJECT_ID", help="GCP project ID")
    parser.add_argument("--location", default="us-central1", help="GCP region")
    parser.add_argument("--bucket", default="YOUR_BUCKET_NAME", help="GCS bucket name")
    parser.add_argument("--index-name", default="kyc-documents-index", help="Vector search index name")
    
    args = parser.parse_args()
    
    success = main(args.project, args.location, args.bucket, args.index_name)
    
    if success:
        print("Vector search setup completed successfully.")
        print("You can now use the search_documents.py script to search for documents.")
    else:
        print("Vector search setup encountered errors. Please check the logs above.") 