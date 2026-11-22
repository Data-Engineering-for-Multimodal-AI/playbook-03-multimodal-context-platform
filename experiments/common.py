"""Shared helpers for the Try this experiments: the sample documents, who owns
each one, and the queries. Reads data/ directly with the same text extraction
and local embedding model as the pipeline's --local mode; writes nothing."""

import os
import sys

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, ROOT)
sys.path.insert(0, os.path.join(ROOT, "scripts"))

from src import kyc  # noqa: E402
from test_person_search import PERSON_QUERIES  # noqa: E402  (the five queries the pipeline runs)

PEOPLE = {"Laura Russo": "LR", "Mitchell Allen": "MA", "April Hensley": "AH",
          "Taylor Cox": "TC", "James Williams": "JW"}
# The person each query asks about, in PERSON_QUERIES order
QUERY_PERSON = ["Laura Russo", "April Hensley", "James Williams", "Taylor Cox", "Mitchell Allen"]
# The four document-type queries search_documents.py runs, and the category each one wants
TYPE_QUERIES = [("Form 1040 tax return", "Form1040"), ("W2 income statement", "W2"),
                ("Driver license ID", "ID"), ("Utility bill for monthly payment", "UtilityBill")]
BANK_STATEMENTS = "Realistic_Bank_Statements.pdf"


def owners(path):
    """People a sample document belongs to (data/README.md). The bank statement
    file holds one page per customer, so it belongs to all five."""
    name = os.path.basename(path)
    if name == BANK_STATEMENTS:
        return set(PEOPLE)
    compact = name.lower()
    found = {p for p, initials in PEOPLE.items()
             if compact.startswith(p.replace(" ", "").lower()) or compact == f"{initials.lower()}dl.jpeg"}
    return found


def documents():
    """[(file name, category, text, owners)] for the 22 sample documents."""
    out = []
    for path in kyc.list_local_documents(DATA):
        text, _ = kyc.extract_local_text(path, DATA)
        out.append((os.path.basename(path), kyc.document_category(path), text, owners(path)))
    return out


def model():
    """The local embedding model, quietly (the same one create_embeddings.py --local uses)."""
    return kyc.Embedder(local=True)._model
