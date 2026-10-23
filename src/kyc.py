"""Shared helpers for the KYC pipeline scripts, in both modes.

Cloud mode (as printed in the book): Google Document AI, Vertex AI embeddings
(text-embedding-005 through the google-genai SDK) and Vertex AI Vector Search.
Google renamed Vertex AI "Gemini Enterprise Agent Platform" in April 2026; the
APIs and Python packages kept their names.

Local mode (--local): the same steps on a laptop CPU with no cloud account.
pypdf reads the PDFs, the text printed on each license card comes from
data/extracted_text/ in place of OCR, sentence-transformers embeds the text,
and a NumPy matrix stands in for the Vector Search index. Files go to the same
relative paths the cloud mode uses in the bucket.
"""

import json
import os

import numpy as np

# Cloud embedding model. Google retires text-embedding-005 on April 1, 2027; set
# EMBEDDING_MODEL to its replacement then. The index keeps EMBEDDING_DIMENSIONS.
CLOUD_EMBEDDING_MODEL = os.environ.get("EMBEDDING_MODEL", "text-embedding-005")
CLOUD_EMBEDDING_DIMENSIONS = int(os.environ.get("EMBEDDING_DIMENSIONS", "768"))
LOCAL_EMBEDDING_MODEL = os.environ.get("LOCAL_EMBEDDING_MODEL",
                                       "sentence-transformers/all-MiniLM-L6-v2")  # 384 dimensions

DOCUMENT_EXTENSIONS = (".pdf", ".jpeg", ".jpg", ".png")
EXTRACTED_TEXT_DIR = "extracted_text"
METADATA_FILE = "document-metadata.json"        # next to (not inside) the embeddings folder
LOCAL_INDEX_FILE = "local-index.npz"

# Sample-data folder -> document category (the Vector Search restrict and --filter value)
FOLDER_CATEGORIES = {
    "drivinglicenses": "ID",
    "w2sample": "W2",
    "form1040sample": "Form1040",
    "utility": "UtilityBill",
}


def document_category(path):
    """Category of a document, from its sample-data folder or, failing that, its file name."""
    parts = [p.lower() for p in path.replace("\\", "/").split("/")]
    for part in parts[:-1]:
        if part in FOLDER_CATEGORIES:
            return FOLDER_CATEGORIES[part]
    name = os.path.basename(path)
    lower = name.lower()
    if "form1040" in lower:
        return "Form1040"
    if "w2" in lower:
        return "W2"
    if "license" in lower or name.split(".")[0].endswith("DL"):
        return "ID"
    if "utility" in lower or "bill" in lower:
        return "UtilityBill"
    if "bank" in lower or "statement" in lower:
        return "BankStatement"
    return "Other"


def mime_type(path):
    lower = path.lower()
    if lower.endswith((".jpg", ".jpeg")):
        return "image/jpeg"
    if lower.endswith(".png"):
        return "image/png"
    return "application/pdf"


def processed_name(document_path):
    """File name for a processed document, e.g. LRDL.jpeg -> LRDL_jpeg_processed.json."""
    return os.path.basename(document_path).replace(".", "_") + "_processed.json"


# ---------------------------------------------------------------- local files

def list_local_documents(sample_dir):
    """All documents under sample_dir (sorted), skipping the extracted_text folder."""
    found = []
    for root, dirs, files in os.walk(sample_dir):
        dirs[:] = sorted(d for d in dirs if d != EXTRACTED_TEXT_DIR and not d.startswith("."))
        for f in sorted(files):
            if f.lower().endswith(DOCUMENT_EXTENSIONS) and not f.startswith("."):
                found.append(os.path.join(root, f))
    return sorted(found)


def extract_local_text(path, sample_dir):
    """Text of a local document: pypdf for PDFs, the extracted_text file for images."""
    if path.lower().endswith(".pdf"):
        from pypdf import PdfReader
        reader = PdfReader(path)
        return "\n".join(page.extract_text() or "" for page in reader.pages), len(reader.pages)
    rel = os.path.relpath(path, sample_dir)
    txt = os.path.join(sample_dir, EXTRACTED_TEXT_DIR, os.path.splitext(rel)[0] + ".txt")
    if not os.path.exists(txt):
        raise FileNotFoundError(
            f"Local mode reads image text from {txt}, which doesn't exist. "
            "Add the text there, or run in cloud mode, where Document AI does the OCR.")
    with open(txt) as fh:
        return fh.read(), 1


# ----------------------------------------------------------------- embeddings

class Embedder:
    """Embeds texts with the local model or with the Vertex AI model in EMBEDDING_MODEL."""

    def __init__(self, local, project=None, location=None):
        self.local = local
        if local:
            # Keep the output readable: no download/progress bars or Hub notices
            os.environ.setdefault("HF_HUB_DISABLE_PROGRESS_BARS", "1")
            os.environ.setdefault("TRANSFORMERS_VERBOSITY", "error")
            os.environ.setdefault("HF_HUB_VERBOSITY", "error")
            import logging
            logging.getLogger("torch.distributed.elastic.multiprocessing.redirects").setLevel(logging.ERROR)
            from sentence_transformers import SentenceTransformer
            from transformers.utils import logging as hf_logging
            hf_logging.disable_progress_bar()
            self.model_name = LOCAL_EMBEDDING_MODEL
            self._model = SentenceTransformer(LOCAL_EMBEDDING_MODEL, device="cpu")
        else:
            from google import genai
            self.model_name = CLOUD_EMBEDDING_MODEL
            self.dimensions = CLOUD_EMBEDDING_DIMENSIONS
            self._client = genai.Client(vertexai=True, project=project, location=location)

    def embed(self, texts):
        if self.local:
            vectors = self._model.encode(list(texts), normalize_embeddings=True,
                                         show_progress_bar=False)
            return [v.tolist() for v in vectors]
        from google.genai import types
        config = types.EmbedContentConfig(output_dimensionality=self.dimensions)
        # text-embedding-* models take up to 5 texts per request (under the token
        # limit); send one text per request to the Gemini embedding models.
        batch = 5 if self.model_name.startswith("text-") else 1
        out = []
        for i in range(0, len(texts), batch):
            chunk = list(texts[i:i + batch])
            response = self._client.models.embed_content(model=self.model_name, contents=chunk,
                                                         config=config)
            if len(response.embeddings) != len(chunk):
                raise RuntimeError(f"{self.model_name} returned {len(response.embeddings)} "
                                   f"embeddings for {len(chunk)} texts")
            for e in response.embeddings:
                # Unit length: the Vector Search index ranks by dot product
                v = np.asarray(e.values, dtype=np.float64)
                out.append((v / np.linalg.norm(v)).tolist())
        return out


# ------------------------------------------------------------- local index

def build_local_index(embeddings_path, index_path):
    """Read the JSON Lines embeddings file and save a normalized matrix for search."""
    ids, vectors, categories = [], [], []
    with open(embeddings_path) as fh:
        for line in fh:
            if line.strip():
                rec = json.loads(line)
                ids.append(rec["id"])
                vectors.append(rec["embedding"])
                cats = [r["allow"] for r in rec.get("restricts", []) if r["namespace"] == "category"]
                categories.append(cats[0][0] if cats else "")
    matrix = np.asarray(vectors, dtype=np.float32)
    matrix /= np.linalg.norm(matrix, axis=1, keepdims=True)
    np.savez(index_path, ids=np.array(ids), matrix=matrix, categories=np.array(categories))
    return matrix.shape


class LocalIndex:
    """Cosine-similarity search over the saved matrix (what Vector Search does in the cloud)."""

    def __init__(self, index_path):
        if not os.path.exists(index_path):
            raise FileNotFoundError(f"{index_path} not found. Run scripts/setup_vector_search.py --local first.")
        data = np.load(index_path)
        self.ids = [str(i) for i in data["ids"]]
        self.matrix = data["matrix"]
        self.categories = [str(c) for c in data["categories"]]

    def find_neighbors(self, query_vector, num_neighbors=5, filter_category=None):
        q = np.asarray(query_vector, dtype=np.float32)
        q /= np.linalg.norm(q)
        scores = self.matrix @ q
        order = np.argsort(-scores, kind="stable")
        hits = [(self.ids[i], float(scores[i])) for i in order
                if filter_category is None or self.categories[i] == filter_category]
        return hits[:num_neighbors]


def load_metadata_file(path):
    with open(path) as fh:
        return json.load(fh)


def metadata_from_embeddings_lines(lines):
    """Older embeddings files (first edition) carried a 'metadata' field in each record."""
    out = {}
    for line in lines:
        if line.strip():
            rec = json.loads(line)
            if "metadata" in rec:
                out[rec["id"]] = rec["metadata"]
    return out
