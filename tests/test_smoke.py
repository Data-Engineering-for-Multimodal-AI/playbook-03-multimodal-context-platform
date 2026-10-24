"""Smoke tests: sample data, the full local pipeline, and the cloud calls (mocked).

Run with `make smoke` (or `uv run pytest -q`). CPU only, no cloud account.
The local pipeline test downloads all-MiniLM-L6-v2 (about 90 MB) on first run.
"""

import hashlib
import importlib.util
import json
import os
import subprocess
import sys
from types import SimpleNamespace
from unittest import mock

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
SCRIPTS = os.path.join(ROOT, "scripts")
sys.path.insert(0, ROOT)
sys.path.insert(0, SCRIPTS)

from src import kyc  # noqa: E402


def load_script(name):
    spec = importlib.util.spec_from_file_location(name, os.path.join(SCRIPTS, f"{name}.py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def run(args, cwd, env=None):
    full_env = dict(os.environ, **(env or {}))
    return subprocess.run([sys.executable, *args], cwd=cwd, env=full_env,
                          capture_output=True, text=True, timeout=600)


def sha(path):
    with open(path, "rb") as fh:
        return hashlib.sha256(fh.read()).hexdigest()


# ------------------------------------------------------------------ sample data

def test_sample_data_complete_and_documented():
    docs = kyc.list_local_documents(DATA)
    assert len(docs) == 22
    with open(os.path.join(DATA, "README.md")) as fh:
        readme = fh.read()
    for path in docs:
        assert os.path.basename(path) in readme, f"{path} not listed in data/README.md"
        if path.endswith(".jpeg"):
            kyc.extract_local_text(path, DATA)  # raises if the card text is missing


def test_generator_reproduces_committed_data(tmp_path):
    gen = pytest.importorskip("generate_sample_data", reason="needs the dev group (Pillow, reportlab)")
    gen.main(str(tmp_path))
    for path in kyc.list_local_documents(DATA):
        rel = os.path.relpath(path, DATA)
        if path.endswith(".pdf"):
            assert sha(path) == sha(tmp_path / rel), f"{rel} differs from the generator's output"
        else:
            from PIL import Image
            assert Image.open(path).size == Image.open(tmp_path / rel).size
    for name in os.listdir(os.path.join(DATA, "extracted_text", "DrivingLicenses")):
        rel = os.path.join("extracted_text", "DrivingLicenses", name)
        assert sha(os.path.join(DATA, rel)) == sha(tmp_path / rel)


def test_documents_agree_with_each_other():
    gen = pytest.importorskip("generate_sample_data", reason="needs the dev group (Pillow, reportlab)")
    bank, _ = kyc.extract_local_text(os.path.join(DATA, "Realistic_Bank_Statements.pdf"), DATA)
    for p in gen.PEOPLE:
        per, annual = gen.payroll(p)
        w2, _ = kyc.extract_local_text(os.path.join(DATA, "W2Sample", f"{p.file_stem}W2.pdf"), DATA)
        f1040, _ = kyc.extract_local_text(
            os.path.join(DATA, "Form1040Sample", f"{p.file_stem}Form1040.pdf"), DATA)
        assert gen.money(annual["box1"]) in w2
        assert f"{round(annual['box1']):,}" in f1040          # line 1a = W-2 box 1
        assert gen.money(per["net"]) in bank                  # biweekly net pay deposited
        for u in p.utilities:
            _, total = gen.utility_amounts(u)
            assert gen.money(total) in bank                   # utility bill paid
        assert p.ssn.startswith("987-65-432")                 # SSA advertising block


# --------------------------------------------------------------- local pipeline

@pytest.fixture(scope="module")
def local_run(tmp_path_factory):
    work = tmp_path_factory.mktemp("pipeline")
    steps = [
        ["scripts/process_documents.py", "--local", "--sample-data", DATA],
        ["scripts/create_embeddings.py", "--local"],
        ["scripts/setup_vector_search.py", "--local"],
    ]
    for step in steps:
        result = run([os.path.join(ROOT, step[0]), *step[1:]], cwd=work)
        assert result.returncode == 0, result.stdout + result.stderr
    return work


def test_local_pipeline_outputs(local_run):
    processed = sorted(p for p in os.listdir(local_run / "processed-documents") if p.endswith("_processed.json"))
    assert len(processed) == 22
    with open(local_run / "vector-search" / "document-embeddings" / "embeddings.json") as fh:
        records = [json.loads(line) for line in fh]
    assert len(records) == 22
    for rec in records:
        # Vector Search's JSON input format: no extra fields such as "metadata"
        assert set(rec) == {"id", "embedding", "restricts"}
        assert len(rec["embedding"]) == 384
    with open(local_run / "vector-search" / "document-metadata.json") as fh:
        meta = json.load(fh)
    cats = {m["category"] for m in meta.values()}
    assert cats == {"ID", "W2", "Form1040", "UtilityBill", "BankStatement"}
    assert meta["LRDL_jpeg_processed.json"]["category"] == "ID"


@pytest.mark.parametrize("query,category", [
    ("Form 1040 tax return", "Form1040"),
    ("W2 income statement", "W2"),
    ("Driver license ID", "ID"),
    ("Utility bill for monthly payment", "UtilityBill"),
])
def test_local_search_finds_the_right_category(local_run, query, category):
    result = run([os.path.join(SCRIPTS, "search_documents.py"), "--local", "--query", query,
                  "--results", "3"], cwd=local_run)
    assert result.returncode == 0, result.stdout + result.stderr
    hits = [line for line in result.stdout.splitlines() if " - Score: " in line]
    assert len(hits) == 3
    assert all(f"({category})" in h for h in hits), result.stdout


def test_local_filter(local_run):
    result = run([os.path.join(SCRIPTS, "search_documents.py"), "--local", "--query", "James tax documents",
                  "--filter", "Form1040", "--results", "3"], cwd=local_run)
    assert result.returncode == 0, result.stdout + result.stderr
    hits = [line for line in result.stdout.splitlines() if " - Score: " in line]
    assert len(hits) == 3 and all("(Form1040)" in h for h in hits)
    assert "JamesWilliamsForm1040" in hits[0]


def test_person_search_runs(local_run):
    result = run([os.path.join(SCRIPTS, "test_person_search.py"), "--local"], cwd=local_run)
    assert result.returncode == 0, result.stdout + result.stderr
    assert result.stdout.count("=== Testing Query") == 5
    assert result.stdout.count("Found 10 similar documents") == 5


# ------------------------------------------------------- cloud path (mocked)

def fake_genai_client(dims=2):
    client = mock.MagicMock()
    client.models.embed_content.side_effect = lambda model, contents, config: SimpleNamespace(
        embeddings=[SimpleNamespace(values=[3.0, 4.0][:dims]) for _ in contents])
    return client


def test_cloud_embeddings_use_google_genai():
    from google.genai import types
    fake_client = fake_genai_client()
    with mock.patch("google.genai.Client", return_value=fake_client) as client_cls:
        vectors = kyc.Embedder(local=False, project="p", location="us-central1").embed(["hello"])
    client_cls.assert_called_once_with(vertexai=True, project="p", location="us-central1")
    fake_client.models.embed_content.assert_called_once_with(
        model="text-embedding-005", contents=["hello"],
        config=types.EmbedContentConfig(output_dimensionality=768))
    assert vectors == [[0.6, 0.8]]   # unit length for the dot-product index


def test_cloud_embedding_model_is_a_setting(monkeypatch):
    monkeypatch.setattr(kyc, "CLOUD_EMBEDDING_MODEL", "gemini-embedding-001")
    fake_client = fake_genai_client()
    with mock.patch("google.genai.Client", return_value=fake_client):
        embedder = kyc.Embedder(local=False, project="p", location="us-central1")
        vectors = embedder.embed(["a", "b", "c"])
    calls = fake_client.models.embed_content.call_args_list
    assert [c.kwargs["model"] for c in calls] == ["gemini-embedding-001"] * 3   # one text per request
    assert len(vectors) == 3
    fake_client.models.embed_content.side_effect = lambda **kw: SimpleNamespace(embeddings=[])
    with pytest.raises(RuntimeError, match="returned 0 embeddings for 1 texts"):
        embedder.embed(["a"])


def test_cloud_filter_is_a_list_of_namespaces():
    from google.cloud.aiplatform.matching_engine.matching_engine_index_endpoint import Namespace
    search_documents = load_script("search_documents")
    endpoint = mock.MagicMock()
    endpoint.find_neighbors.return_value = [[SimpleNamespace(id="a", distance=0.9)]]
    with mock.patch("google.cloud.aiplatform.init"), \
            mock.patch("google.cloud.aiplatform.MatchingEngineIndexEndpoint", return_value=endpoint), \
            mock.patch.object(search_documents, "Embedder") as embedder:
        embedder.return_value.embed.return_value = [[0.0] * 768]
        searcher = search_documents.Searcher("p", "us-central1", "vector-search/document-embeddings",
                                             "projects/p/locations/us-central1/indexEndpoints/1",
                                             "kyc_documents_index_deployed")
        hits = searcher.search("James tax documents", num_results=3, filter_category="Form1040")
    kwargs = endpoint.find_neighbors.call_args.kwargs
    assert kwargs["filter"] == [Namespace("category", ["Form1040"], [])]
    assert hits == [("a", 0.9)]


def test_processor_routing_and_missing_ids(monkeypatch):
    process_documents = load_script("process_documents")
    route = process_documents.get_processor_type
    assert route("gs://b/data/DrivingLicenses/LRDL.jpeg") == "driving_license"
    assert route("gs://b/data/W2Sample/LauraRussoW2.pdf") == "w2"
    assert route("gs://b/data/Utility/Taylorcoxutility.pdf") == "form"   # Utility Parser discontinued
    assert route("gs://b/data/Realistic_Bank_Statements.pdf") == "bank_statement"
    monkeypatch.delenv("DOCAI_W2_PROCESSOR_ID", raising=False)
    with pytest.raises(SystemExit, match="DOCAI_W2_PROCESSOR_ID"):
        process_documents.get_processor_id("gs://b/data/W2Sample/LauraRussoW2.pdf")


def test_cloud_setup_failure_exits_nonzero(tmp_path):
    env = {"GOOGLE_APPLICATION_CREDENTIALS": str(tmp_path / "missing.json"),
           "GCE_METADATA_HOST": "127.0.0.1:1", "NO_GCE_CHECK": "True"}
    result = run([os.path.join(SCRIPTS, "setup_vector_search.py")], cwd=tmp_path, env=env)
    assert result.returncode == 1, result.stdout + result.stderr
    assert "encountered errors" in result.stdout


def test_search_without_endpoint_explains(tmp_path):
    env = {k: v for k, v in os.environ.items() if k != "VECTOR_SEARCH_ENDPOINT"}
    result = subprocess.run([sys.executable, os.path.join(SCRIPTS, "search_documents.py")],
                            cwd=tmp_path, env=env, capture_output=True, text=True, timeout=120)
    assert result.returncode == 1
    assert "--endpoint is required" in result.stdout


def test_teardown_dry_run_deletes_nothing():
    teardown = load_script("teardown_vector_search")
    endpoint = mock.MagicMock(resource_name="projects/p/locations/l/indexEndpoints/1",
                              deployed_indexes=[SimpleNamespace(id="kyc_documents_index_deployed")])
    index = mock.MagicMock(resource_name="projects/p/locations/l/indexes/2")
    with mock.patch("google.cloud.aiplatform.init"), \
            mock.patch.object(teardown, "find_resources", return_value=([endpoint], [index])):
        teardown.main("p", "us-central1", "kyc-documents-index", yes=False)
        endpoint.undeploy_all.assert_not_called()
        index.delete.assert_not_called()
        teardown.main("p", "us-central1", "kyc-documents-index", yes=True)
    endpoint.undeploy_all.assert_called_once()
    endpoint.delete.assert_called_once()
    index.delete.assert_called_once()
