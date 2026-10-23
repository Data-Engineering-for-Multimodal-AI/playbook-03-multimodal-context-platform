from google.cloud import aiplatform
from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
from typing import List, Dict, Optional

class VectorSearch:
    def __init__(self, project_id: str, location: str, index_name: str = "kyc-documents-index"):
        self.project_id = project_id
        self.location = location
        self.index_name = index_name
        self.index_endpoint = None
        self.index = None
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
    
    def create_index(self, bucket_name: str, dimensions: int = 768):
        """Create a vector search index following the setup process."""
        try:
            # Step 1: Create the index
            print("Creating vector search index...")
            self.index = aiplatform.MatchingEngineIndex.create_tree_ah_index(
                display_name=self.index_name,
                contents_delta_uri=f"gs://{bucket_name}/vector-search/document-embeddings",
                dimensions=dimensions,  # EMBEDDING_DIMENSIONS, 768 by default
                approximate_neighbors_count=150,
            )
            
            # Wait for index creation to complete
            print(f"Waiting for index creation... (this may take 20-30 minutes)")
            self.index.wait()
            
            # Step 2: Create index endpoint
            print("Creating index endpoint...")
            self.index_endpoint = aiplatform.MatchingEngineIndexEndpoint.create(
                display_name=f"{self.index_name}-endpoint",
                public_endpoint_enabled=True
            )
            
            # Wait for endpoint creation to complete
            print(f"Waiting for endpoint creation... (this may take 5-10 minutes)")
            self.index_endpoint.wait()
            
            # Step 3: Deploy the index
            deployed_index_id = f"{self.index_name.replace('-', '_')}_deployed"
            print(f"Deploying index to endpoint with ID: {deployed_index_id}")
            
            self.index_endpoint.deploy_index(
                index=self.index, 
                deployed_index_id=deployed_index_id
            )
            
            # Wait for deployment to complete
            print(f"Waiting for index deployment... (this may take 10-15 minutes)")
            self.index_endpoint.wait()
            
            print("Vector search setup completed successfully!")
            print(f"Index name: {self.index.display_name}")
            print(f"Index ID: {self.index.name}")
            print(f"Endpoint name: {self.index_endpoint.display_name}")
            print(f"Endpoint ID: {self.index_endpoint.name}")
            print(f"Deployed index ID: {deployed_index_id}")
            
        except Exception as e:
            print(f"Error in vector search setup: {str(e)}")
            raise
    
    def upsert_embeddings(self, datapoints: List[Dict]):
        """Upsert document embeddings into the index."""
        if not self.index or not self.index_endpoint:
            raise ValueError("Index not created. Call create_index() first.")
        
        print(f"Upserting {len(datapoints)} datapoints...")
        
        # Prepare datapoints for upsertion (upsert needs an index created with
        # index_update_method="STREAM_UPDATE"; create_index() makes a batch index)
        from google.cloud.aiplatform_v1.types import IndexDatapoint
        formatted_datapoints = [
            IndexDatapoint(
                datapoint_id=datapoint["id"],
                feature_vector=datapoint["embedding"],
                restricts=[IndexDatapoint.Restriction(namespace="category",
                                                      allow_list=[datapoint["category"]])],
            )
            for datapoint in datapoints
        ]

        # Upsert the datapoints
        self.index.upsert_datapoints(datapoints=formatted_datapoints)
        print("Datapoints upserted successfully!")
    
    def search(self, query_embedding: List[float], num_neighbors: int = 5,
               filter_category: Optional[str] = None) -> List[Dict]:
        """Search for similar documents with optional filters."""
        if not self.index_endpoint:
            raise ValueError("Index endpoint not created. Call create_index() first.")
        
        deployed_index_id = f"{self.index_name.replace('-', '_')}_deployed"
        
        # Prepare the search request
        search_request = {
            "deployed_index_id": deployed_index_id,
            "queries": [query_embedding],
            "num_neighbors": num_neighbors
        }
        
        # Restrict to one document category (find_neighbors takes a list of Namespace)
        if filter_category:
            search_request["filter"] = [Namespace("category", [filter_category], [])]
        
        # Perform the search
        response = self.index_endpoint.find_neighbors(**search_request)
        
        # Process and return results
        results = []
        for neighbor in response[0]:
            results.append({
                "id": neighbor.id,
                "distance": neighbor.distance,
            })
        
        return results

def main():
    # Example usage
    project_id = "YOUR_PROJECT_ID"  # Replace with your GCP project ID
    location = "us-central1"        # Replace with your preferred location
    bucket_name = "YOUR_BUCKET_NAME" # Replace with your GCS bucket name
    
    # Initialize VectorSearch
    vector_search = VectorSearch(project_id, location)
    
    # Create index (uncomment to run)
    # vector_search.create_index(bucket_name)
    
    print("Vector Search initialized successfully!")

if __name__ == "__main__":
    main() 