import os
import sys
import argparse

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from search_documents import Searcher, load_document_mappings, print_results

# Test person-specific KYC queries
PERSON_QUERIES = [
    "Laura's KYC documents",
    "April Hensley's identification documents",
    "James Williams' personal information",
    "Taylor Cox's financial records",
    "Mitchell Allen's tax information"
]


def main(searcher, doc_mappings, num_results=10):
    """Main function to test person-specific searches."""
    for query in PERSON_QUERIES:
        print(f"\n\n=== Testing Query: '{query}' ===")
        try:
            neighbors = searcher.search(query, num_results=num_results)
            print_results(neighbors, doc_mappings, snippet_chars=150, filename_only=True)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Test person-specific document searches")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"), help="GCP region")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "YOUR_BUCKET_NAME"), help="GCS bucket name")
    parser.add_argument("--vector-search", default="vector-search/document-embeddings", help="Path for vector search embeddings")
    parser.add_argument("--endpoint", default=os.environ.get("VECTOR_SEARCH_ENDPOINT"),
                        help="Vector search endpoint resource name (required without --local)")
    parser.add_argument("--deployed-index", default="kyc_documents_index_deployed", help="Deployed index ID")
    parser.add_argument("--results", type=int, default=10, help="Number of results per query")
    parser.add_argument("--local", action="store_true", help="Search the local index instead of Vertex AI")

    args = parser.parse_args()

    if args.endpoint is None and not args.local:
        print("Error: --endpoint is required. Please provide the full resource name of your MatchingEngineIndexEndpoint.")
        print("Example: projects/123456789/locations/us-central1/indexEndpoints/987654321")
        print("Or add --local to search the local index.")
        sys.exit(1)

    doc_mappings = load_document_mappings(args.bucket, args.vector_search, args.project, args.local)
    searcher = Searcher(args.project, args.location, args.vector_search, args.endpoint,
                        args.deployed_index, args.local)
    main(searcher, doc_mappings, args.results)
