from google.cloud import storage
from google.cloud import documentai_v1 as documentai
import os
import sys
from typing import List, Dict, Optional
from datetime import datetime

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from src.kyc import Embedder, mime_type


class DocumentProcessor:
    """Library version of the cloud pipeline: Document AI extraction and Vertex AI embeddings.

    The scripts in scripts/ don't use this class; it shows the same steps as one object.
    Processor IDs are read from the environment (see .env.example).
    """

    def __init__(self, project_id: str, location: str, bucket_name: str, docai_location: str = "us"):
        self.project_id = project_id
        self.location = location              # Vertex AI region, e.g. us-central1
        self.docai_location = docai_location  # Document AI location: us or eu
        self.bucket_name = bucket_name
        self.storage_client = storage.Client()
        self.bucket = self.storage_client.bucket(bucket_name)

        # Initialize Document AI client
        self.documentai_client = documentai.DocumentProcessorServiceClient(
            client_options={"api_endpoint": f"{docai_location}-documentai.googleapis.com"})

        self._embedder = None

        # Processor configurations. Google discontinued the Utility Parser on
        # June 30, 2026; utility bills use the Form Parser.
        self.processors = {
            'driving_license': {'name': 'driving-license-parser',
                                'id': os.environ.get('DOCAI_DRIVER_LICENSE_PROCESSOR_ID', '')},
            'w2': {'name': 'w2-parser', 'id': os.environ.get('DOCAI_W2_PROCESSOR_ID', '')},
            'form': {'name': 'form-parser', 'id': os.environ.get('DOCAI_FORM_PROCESSOR_ID', '')},
            'bank_statement': {'name': 'bank-statement-parser',
                               'id': os.environ.get('DOCAI_BANK_STATEMENT_PROCESSOR_ID', '')},
        }
        self.processors['utility'] = self.processors['form']

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
        if not processor_config['id']:
            raise ValueError(f"No processor ID set for {processor_config['name']} (see .env.example)")

        # Configure the processor request (a document in Cloud Storage)
        request = documentai.ProcessRequest(
            name=f"projects/{self.project_id}/locations/{self.docai_location}/processors/{processor_config['id']}",
            gcs_document=documentai.GcsDocument(gcs_uri=gcs_uri, mime_type=mime_type(gcs_uri)),
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
        """Create vector embeddings with the Vertex AI model in EMBEDDING_MODEL (text-embedding-005 by default)."""
        if self._embedder is None:
            self._embedder = Embedder(local=False, project=self.project_id, location=self.location)
        return self._embedder.embed([text])[0]

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
    project_id = os.environ.get("GOOGLE_CLOUD_PROJECT", "YOUR_PROJECT_ID")
    location = os.environ.get("GOOGLE_CLOUD_LOCATION", "us-central1")
    bucket_name = os.environ.get("GCS_BUCKET", "YOUR_BUCKET_NAME")

    # Initialize processor
    processor = DocumentProcessor(project_id, location, bucket_name,
                                  os.environ.get("DOCAI_LOCATION", "us"))

    # Add your processing logic here
    print("Document Processor initialized successfully!")


if __name__ == "__main__":
    main()
