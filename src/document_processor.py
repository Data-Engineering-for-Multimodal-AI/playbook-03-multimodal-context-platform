from google.cloud import storage
from google.cloud import documentai_v1 as documentai
from google.cloud import vision
from google.cloud import aiplatform
import os
import json
from typing import List, Dict, Optional
import pandas as pd
from datetime import datetime

class DocumentProcessor:
    def __init__(self, project_id: str, location: str, bucket_name: str):
        self.project_id = project_id
        self.location = location
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)
        
        # Initialize Document AI client
        self.documentai_client = documentai.DocumentProcessorServiceClient()
        
        # Initialize Vision AI client
        self.vision_client = vision.ImageAnnotatorClient()
        
        # Initialize Vertex AI
        aiplatform.init(project=project_id, location=location)
        
        # Processor configurations
        # NOTE: You must replace these processor IDs with your own Document AI processors
        self.processors = {
            'driving_license': {
                'name': 'driving-license-parser',
                'id': 'YOUR_DRIVING_LICENSE_PROCESSOR_ID',
                'endpoint': f'https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/YOUR_DRIVING_LICENSE_PROCESSOR_ID:process'
            },
            'w2': {
                'name': 'w2-parser',
                'id': 'YOUR_W2_PROCESSOR_ID',
                'endpoint': f'https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/YOUR_W2_PROCESSOR_ID:process'
            },
            'form': {
                'name': 'form-parser',
                'id': 'YOUR_FORM_PROCESSOR_ID',
                'endpoint': f'https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/YOUR_FORM_PROCESSOR_ID:process'
            },
            'utility': {
                'name': 'utility-parser',
                'id': 'YOUR_UTILITY_PROCESSOR_ID',
                'endpoint': f'https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/YOUR_UTILITY_PROCESSOR_ID:process'
            },
            'bank_statement': {
                'name': 'bank-statement-parser',
                'id': 'YOUR_BANK_STATEMENT_PROCESSOR_ID',
                'endpoint': f'https://{location}-documentai.googleapis.com/v1/projects/{project_id}/locations/{location}/processors/YOUR_BANK_STATEMENT_PROCESSOR_ID:process'
            }
        }
    
    def upload_to_gcs(self, local_file_path: str, destination_blob_name: str) -> str:
        """Upload a file to Google Cloud Storage."""
        blob = self.bucket.blob(destination_blob_name)
        blob.upload_from_filename(local_file_path)
        return f"gs://{self.bucket_name}/{destination_blob_name}"
    
    def determine_processor_type(self, file_path: str) -> Optional[str]:
        """Determine the appropriate processor type based on file name and extension."""
        file_name = file_path.lower()
        if 'driving' in file_name or 'license' in file_name:
            return 'driving_license'
        elif 'w2' in file_name:
            return 'w2'
        elif 'utility' in file_name or 'bill' in file_name:
            return 'utility'
        elif 'bank' in file_name or 'statement' in file_name:
            return 'bank_statement'
        else:
            return 'form'
    
    def process_document(self, gcs_uri: str, processor_type: Optional[str] = None) -> Dict:
        """Process a document using the appropriate Document AI processor."""
        if not processor_type:
            processor_type = self.determine_processor_type(gcs_uri)
        
        processor_config = self.processors.get(processor_type)
        if not processor_config:
            raise ValueError(f"Unknown processor type: {processor_type}")
        
        # Configure the processor request
        request = documentai.ProcessRequest(
            name=f"projects/{self.project_id}/locations/{self.location}/processors/{processor_config['id']}",
            document=documentai.Document(
                gcs_uri=gcs_uri,
                mime_type="application/pdf" if gcs_uri.endswith('.pdf') else "image/jpeg"
            )
        )
        
        # Process the document
        result = self.documentai_client.process_document(request=request)
        document = result.document
        
        # Extract text and entities
        extracted_data = {
            'text': document.text,
            'entities': {entity.type_: entity.mention_text for entity in document.entities},
            'document_type': processor_type,
            'processing_timestamp': datetime.now().isoformat(),
            'original_uri': gcs_uri
        }
        
        return extracted_data
    
    def create_vector_embeddings(self, text: str) -> List[float]:
        """Create vector embeddings using Vertex AI's text-embedding-005 model."""
        model = aiplatform.TextEmbeddingModel.from_pretrained("text-embedding-005")
        embeddings = model.get_embeddings([text])[0].values
        return embeddings
    
    def prepare_document_for_indexing(self, document_id: str, extracted_data: Dict) -> Dict:
        """Prepare document data for vector search indexing."""
        # Create embeddings from the document text
        embeddings = self.create_vector_embeddings(extracted_data['text'])
        
        # Prepare metadata
        metadata = {
            'document_id': document_id,
            'document_type': extracted_data['document_type'],
            'processing_timestamp': extracted_data['processing_timestamp'],
            'original_uri': extracted_data['original_uri'],
            'entities': extracted_data['entities']
        }
        
        return {
            'id': document_id,
            'embedding': embeddings,
            'metadata': metadata
        }

def main():
    # Configuration
    project_id = "YOUR_PROJECT_ID"  # Replace with your GCP project ID
    location = "us-central1"        # Replace with your preferred location
    bucket_name = "YOUR_BUCKET_NAME" # Replace with your GCS bucket name
    
    # Initialize processor
    processor = DocumentProcessor(project_id, location, bucket_name)
    
    # Add your processing logic here
    print("Document Processor initialized successfully!")
    
if __name__ == "__main__":
    main() 