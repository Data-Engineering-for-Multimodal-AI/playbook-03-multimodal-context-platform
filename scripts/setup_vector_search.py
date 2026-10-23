import argparse
import os
import sys

# Add the repository root to the path so we can import from src/
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kyc import LOCAL_INDEX_FILE, build_local_index


def main(project_id, location, bucket_name, index_name, dimensions=768):
    """Set up vector search index and deploy it."""
    from src.vector_search import VectorSearch

    print(f"Setting up vector search for project: {project_id}")
    print(f"Using bucket: {bucket_name}")
    print(f"Index name: {index_name}")

    # Create and deploy the index
    print("Creating and deploying vector search index...")
    print("NOTE: This process will take 30-45 minutes to complete")
    print("The deployed index is billed for every hour it stays deployed. "
          "Run scripts/teardown_vector_search.py when you're done.")

    try:
        vector_search = VectorSearch(project_id, location, index_name)
        vector_search.create_index(bucket_name, dimensions)
        print("Vector search index created and deployed successfully!")
        return True
    except Exception as e:
        print(f"Error setting up vector search: {str(e)}")
        return False


def main_local(vector_search_prefix):
    """Local mode: load the embeddings into a NumPy matrix that the search scripts query."""
    embeddings_path = os.path.join(vector_search_prefix, "embeddings.json")
    if not os.path.exists(embeddings_path):
        print(f"Error: {embeddings_path} not found. Run scripts/create_embeddings.py --local first.")
        return False
    index_path = os.path.join(os.path.dirname(vector_search_prefix.rstrip("/")), LOCAL_INDEX_FILE)
    rows, dims = build_local_index(embeddings_path, index_path)
    print(f"Local index: {rows} vectors, {dims} dimensions, saved to {index_path}")
    return True


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Set up vector search index")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"), help="GCP region")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "YOUR_BUCKET_NAME"), help="GCS bucket name")
    parser.add_argument("--index-name", default="kyc-documents-index", help="Vector search index name")
    parser.add_argument("--vector-search", default="vector-search/document-embeddings",
                        help="Folder with embeddings.json (--local only; the cloud index reads it from the bucket)")
    parser.add_argument("--local", action="store_true", help="Build a local index instead of Vertex AI Vector Search")

    args = parser.parse_args()

    if args.local:
        success = main_local(args.vector_search)
    else:
        success = main(args.project, args.location, args.bucket, args.index_name)

    if success:
        print("Vector search setup completed successfully.")
        print("You can now use the search_documents.py script to search for documents.")
    else:
        print("Vector search setup encountered errors. Please check the logs above.")
        sys.exit(1)
