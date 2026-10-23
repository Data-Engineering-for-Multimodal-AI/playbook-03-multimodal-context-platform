import os
import sys
import json
import argparse

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kyc import (LOCAL_INDEX_FILE, METADATA_FILE, Embedder, LocalIndex, load_metadata_file,
                     metadata_from_embeddings_lines)


class Searcher:
    """Embeds queries and finds the nearest documents, on Vertex AI Vector Search or locally."""

    def __init__(self, project_id, location, vector_search_prefix, endpoint_name=None,
                 deployed_index_id=None, local=False):
        self.local = local
        self.embedder = Embedder(local, project_id, location)
        if local:
            parent = os.path.dirname(vector_search_prefix.rstrip("/"))
            self.index = LocalIndex(os.path.join(parent, LOCAL_INDEX_FILE))
        else:
            from google.cloud import aiplatform
            aiplatform.init(project=project_id, location=location)
            self.endpoint = aiplatform.MatchingEngineIndexEndpoint(endpoint_name)
            self.deployed_index_id = deployed_index_id
            print(f"Using endpoint: {self.endpoint.display_name}")

    def search(self, query_text, num_results=5, filter_category=None):
        """Return [(document id, score)] for the documents most similar to the query."""
        query_embedding = self.embedder.embed([query_text])[0]
        if self.local:
            return self.index.find_neighbors(query_embedding, num_results, filter_category)

        from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
        restricts = [Namespace("category", [filter_category], [])] if filter_category else None
        response = self.endpoint.find_neighbors(
            deployed_index_id=self.deployed_index_id,
            queries=[query_embedding],
            num_neighbors=num_results,
            filter=restricts,
        )
        # Score: dot product of normalized embeddings (the index's default distance)
        return [(n.id, n.distance) for n in response[0]]


def load_document_mappings(bucket_name, vector_search_prefix, project_id, local=False):
    """Load the path, category and snippet of each document id."""
    parent = os.path.dirname(vector_search_prefix.rstrip("/"))
    metadata_path = f"{parent}/{METADATA_FILE}" if parent else METADATA_FILE
    if local:
        mappings = load_metadata_file(metadata_path)
    else:
        from google.cloud import storage
        bucket = storage.Client(project=project_id).bucket(bucket_name)
        blob = bucket.blob(metadata_path)
        if blob.exists():
            mappings = json.loads(blob.download_as_text())
        else:
            # Embeddings created by the first-edition code kept the metadata in each record
            lines = bucket.blob(f"{vector_search_prefix}/embeddings.json").download_as_text().splitlines()
            mappings = metadata_from_embeddings_lines(lines)

    print(f"Loaded mappings for {len(mappings)} documents")
    return mappings


def print_results(neighbors, doc_mappings, snippet_chars=100, filename_only=False):
    print(f"Found {len(neighbors)} similar documents:")
    for i, (doc_id, score) in enumerate(neighbors):
        doc_info = doc_mappings.get(doc_id, {"document_path": "Unknown", "category": "Unknown"})
        path = doc_info.get('document_path', 'Unknown')
        if filename_only:
            path = path.split('/')[-1]
        print(f"{i+1}. {path} ({doc_info.get('category', 'Unknown')}) - Score: {score:.4f}")
        snippet = " ".join(doc_info.get('text_snippet', '').split())
        print(f"   Snippet: {snippet[:snippet_chars]}...")


def main(searcher, doc_mappings):
    """Run the example queries."""
    queries = [
        {"text": "Form 1040 tax return", "filter": None},
        {"text": "W2 income statement", "filter": None},
        {"text": "Driver license ID", "filter": None},
        {"text": "Utility bill for monthly payment", "filter": None}
    ]

    print("\nTesting document search with Vector Search:")
    for query in queries:
        query_text = query["text"]
        filter_category = query["filter"]
        print(f"\nQuery: '{query_text}'{f' (Filter: {filter_category})' if filter_category else ''}")
        try:
            neighbors = searcher.search(query_text, num_results=3, filter_category=filter_category)
            print_results(neighbors, doc_mappings)
        except Exception as e:
            print(f"Error: {str(e)}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Search for documents using vector search")
    parser.add_argument("--project", default=os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID"), help="GCP project ID")
    parser.add_argument("--location", default=os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1"), help="GCP region")
    parser.add_argument("--bucket", default=os.environ.get("GCS_BUCKET", "YOUR_BUCKET_NAME"), help="GCS bucket name")
    parser.add_argument("--vector-search", default="vector-search/document-embeddings", help="Path for vector search embeddings")
    parser.add_argument("--endpoint", default=os.environ.get("VECTOR_SEARCH_ENDPOINT"),
                        help="Vector search endpoint resource name (required without --local)")
    parser.add_argument("--deployed-index", default="kyc_documents_index_deployed", help="Deployed index ID")
    parser.add_argument("--query", default=None, help="Search query (if not specified, will run test queries)")
    parser.add_argument("--filter", default=None,
                        help="Filter by document category: ID, W2, Form1040, UtilityBill or BankStatement")
    parser.add_argument("--results", type=int, default=5, help="Number of results to return")
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

    if args.query:
        # Run a single search
        neighbors = searcher.search(args.query, num_results=args.results, filter_category=args.filter)
        print(f"\nSearch results for: '{args.query}'{f' (Filter: {args.filter})' if args.filter else ''}")
        print_results(neighbors, doc_mappings)
    else:
        # Run test queries
        main(searcher, doc_mappings)
