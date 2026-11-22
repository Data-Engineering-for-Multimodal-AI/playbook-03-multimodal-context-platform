"""Check a folder of your own documents and run the local pipeline on it.

    make check-data DIR=my_documents   # check every file, write nothing
    make load DIR=my_documents         # check it, then run steps 1-3 (--local) on it
    make search QUERY="..."            # then search it

The folder uses the same layout as data/:

    my_documents/
      DrivingLicenses/  jane_doe_dl.jpeg      photos or scans (.jpeg .jpg .png)
      W2Sample/         JaneDoeW2.pdf         PDFs with a text layer
      Form1040Sample/   JaneDoeForm1040.pdf
      Utility/          JaneDoeUtility.pdf
      any_other.pdf                           category from the file name, or Other
      extracted_text/DrivingLicenses/jane_doe_dl.txt   the text on each image (local mode)

The category (the value for --filter) comes from the folder name (DrivingLicenses,
W2Sample, Form1040Sample, Utility) or, outside those, from the file name.

The checks cover what makes the local pipeline fail, skip a document or mix
two up: file types it doesn't read, empty files, Git LFS pointers, files whose
content doesn't match their extension, PDFs that won't open or have no text
layer (scans), images without their text file in extracted_text/, text files
that aren't UTF-8, and two documents that would be saved under the same
processed file name. If any check fails, nothing is written.

`make load` deletes the previous run's output (processed-documents/ and
vector-search/, the same folders make clean deletes) so your documents aren't
mixed with the samples, then runs process_documents.py, create_embeddings.py and
setup_vector_search.py with --local. `make run` puts the samples back.
"""

import argparse
import os
import shutil
import subprocess
import sys
from collections import defaultdict

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT)
from src.kyc import (DOCUMENT_EXTENSIONS, EXTRACTED_TEXT_DIR, document_category,  # noqa: E402
                     processed_name)

MAGIC = {".pdf": (b"%PDF",), ".jpeg": (b"\xff\xd8\xff",), ".jpg": (b"\xff\xd8\xff",),
         ".png": (b"\x89PNG\r\n\x1a\n",)}
EMBED_CHARS = 8000  # create_embeddings.py keeps the first 8,000 characters


def head(path, n=100):
    with open(path, "rb") as fh:
        return fh.read(n)


def check_document(path, folder):
    """Return (problems, notes, text length) for one document."""
    start = head(path)
    if not start:
        return ["empty file"], [], 0
    if start.startswith(b"version https://git-lfs"):
        return ["Git LFS pointer, not the file (run git lfs pull)"], [], 0
    ext = os.path.splitext(path)[1].lower()
    if not start.startswith(MAGIC[ext]):
        return [f"content is not a {ext[1:].upper()} file"], [], 0
    if ext == ".pdf":
        from pypdf import PdfReader
        try:
            reader = PdfReader(path)
            if reader.is_encrypted:
                return ["PDF is password-protected"], [], 0
            text = "\n".join(page.extract_text() or "" for page in reader.pages)
        except Exception as e:
            return [f"can't read the PDF ({e.__class__.__name__})"], [], 0
        if not text.strip():
            return ["no text layer (a scan?); local mode can't read it. Add its text, or use "
                    "cloud mode, where Document AI does OCR"], [], 0
        return [], [], len(text)
    rel = os.path.relpath(path, folder)
    txt = os.path.join(folder, EXTRACTED_TEXT_DIR, os.path.splitext(rel)[0] + ".txt")
    txt_rel = os.path.relpath(txt, folder)
    if not os.path.exists(txt):
        return [f"no {txt_rel}; local mode reads the text on an image from there"], [], 0
    try:
        with open(txt, encoding="utf-8") as fh:
            text = fh.read()
    except UnicodeDecodeError:
        return [f"{txt_rel} is not UTF-8 text"], [], 0
    if not text.strip():
        return [f"{txt_rel} is empty"], [], 0
    return [], [], len(text)


def scan(folder):
    """Check every file. Returns ([(path, category)], [problems], [notes])."""
    problems, notes, docs = [], [], []
    names = defaultdict(list)
    for root, dirs, files in os.walk(folder):
        dirs[:] = sorted(d for d in dirs if not d.startswith("."))
        if os.path.relpath(root, folder).split(os.sep)[0] == EXTRACTED_TEXT_DIR:
            continue
        for f in sorted(files):
            if f.startswith(".") or f.upper().startswith("README"):
                continue
            path = os.path.join(root, f)
            rel = os.path.relpath(path, folder)
            if not f.lower().endswith(DOCUMENT_EXTENSIONS):
                problems.append(f"{rel}: not a document type the pipeline reads "
                                f"({', '.join(DOCUMENT_EXTENSIONS)}); it would be skipped")
                continue
            found, extra, length = check_document(path, folder)
            problems += [f"{rel}: {p}" for p in found]
            notes += [f"{rel}: {n}" for n in extra]
            category = document_category(rel)
            if not found:
                if category == "Other":
                    notes.append(f"{rel}: category Other (not a known folder or file name); "
                                 f"--filter can't select it")
                if length > EMBED_CHARS:
                    notes.append(f"{rel}: {length:,} characters; the embedding uses the first {EMBED_CHARS:,}")
            names[processed_name(path)].append(rel)
            docs.append((path, category))
    for name, rels in names.items():
        if len(rels) > 1:
            problems.append(f"{', '.join(rels)}: would all be saved as processed-documents/{name}, "
                            f"and the last one would replace the others; rename one")
    return docs, problems, notes


def run_step(args):
    print(f"\n$ python {' '.join(args)}", flush=True)
    result = subprocess.run([sys.executable, *args], cwd=ROOT)
    if result.returncode:
        sys.exit(f"{args[0]} failed (exit {result.returncode}).")


def main():
    p = argparse.ArgumentParser(description="Check a folder of documents and index it locally.")
    p.add_argument("--dir", required=True, help="your folder, laid out like data/")
    p.add_argument("--check-only", action="store_true", help="check the files and write nothing")
    args = p.parse_args()
    folder = os.path.abspath(args.dir)
    if not os.path.isdir(folder):
        sys.exit(f"No such folder: {args.dir}")

    docs, problems, notes = scan(folder)
    counts = defaultdict(int)
    for _, category in docs:
        counts[category] += 1
    summary = ", ".join(f"{n} {c}" for c, n in sorted(counts.items()))
    print(f"{len(docs)} documents in {args.dir}" + (f": {summary}" if summary else ""))
    for line in notes:
        print(f"  note: {line}")
    if problems:
        print(f"\n{len(problems)} problem(s):")
        for line in problems:
            print(f"  {line}")
        sys.exit("\nFix or remove these files and run again. Nothing was written.")
    if not docs:
        sys.exit("No documents found. Nothing was written.")
    print("All files passed the checks.", flush=True)
    if args.check_only:
        return

    for out in ("processed-documents", "vector-search"):
        shutil.rmtree(os.path.join(ROOT, out), ignore_errors=True)
    rel = os.path.relpath(folder, ROOT)
    run_step(["scripts/process_documents.py", "--local", "--sample-data",
              folder if rel.startswith("..") else rel])
    run_step(["scripts/create_embeddings.py", "--local"])
    run_step(["scripts/setup_vector_search.py", "--local"])
    print(f"\nIndexed {len(docs)} documents. Search them with: make search QUERY=\"...\"")


if __name__ == "__main__":
    main()
