"""Checks for the "Use your own data" loader (scripts/load_documents.py). Writes only to tmp_path."""

import os
import shutil
import subprocess
import sys

import pytest

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
DATA = os.path.join(ROOT, "data")
sys.path.insert(0, os.path.join(ROOT, "scripts"))
from load_documents import scan  # noqa: E402


def good_folder(tmp_path):
    for sub in ("W2Sample", "DrivingLicenses", "extracted_text/DrivingLicenses"):
        (tmp_path / sub).mkdir(parents=True)
    shutil.copy(os.path.join(DATA, "W2Sample", "LauraRussoW2.pdf"), tmp_path / "W2Sample")
    shutil.copy(os.path.join(DATA, "DrivingLicenses", "LRDL.jpeg"), tmp_path / "DrivingLicenses")
    shutil.copy(os.path.join(DATA, "extracted_text", "DrivingLicenses", "LRDL.txt"),
                tmp_path / "extracted_text" / "DrivingLicenses")
    return tmp_path


def rel(problems):
    return [p.split(": ", 1)[0] for p in problems]


def test_sample_data_passes():
    docs, problems, _ = scan(DATA)
    assert problems == []
    assert len(docs) == 22


def test_good_folder(tmp_path):
    folder = good_folder(tmp_path)
    (folder / ".DS_Store").write_bytes(b"\0")
    docs, problems, notes = scan(str(folder))
    assert problems == [] and notes == []
    assert sorted(c for _, c in docs) == ["ID", "W2"]


@pytest.mark.parametrize("name,content,expected", [
    ("notes.docx", b"PK", "not a document type the pipeline reads"),
    ("W2Sample/empty.pdf", b"", "empty file"),
    ("W2Sample/lfs.pdf", b"version https://git-lfs.github.com/spec/v1\n", "Git LFS pointer"),
    ("W2Sample/fake.pdf", b"\xff\xd8\xff\xe0 a jpeg", "content is not a PDF file"),
    ("W2Sample/cut.pdf", b"%PDF-1.4 truncated", "can't read the PDF"),
    ("DrivingLicenses/fake.png", b"%PDF-1.4", "content is not a PNG file"),
    ("DrivingLicenses/nocard.jpeg", b"\xff\xd8\xff\xe0", "no extracted_text/DrivingLicenses/nocard.txt"),
])
def test_bad_files_are_reported(tmp_path, name, content, expected):
    folder = good_folder(tmp_path)
    (folder / name).write_bytes(content)
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and expected in problems[0], problems
    assert rel(problems) == [name]


def test_pdf_without_text_layer(tmp_path):
    from pypdf import PdfWriter
    folder = good_folder(tmp_path)
    writer = PdfWriter()
    writer.add_blank_page(612, 792)
    writer.write(str(folder / "W2Sample" / "scan.pdf"))
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and "no text layer" in problems[0]


def test_card_text_not_utf8(tmp_path):
    folder = good_folder(tmp_path)
    (folder / "extracted_text" / "DrivingLicenses" / "LRDL.txt").write_bytes("CAFÉ".encode("latin-1"))
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and "is not UTF-8 text" in problems[0]


def test_same_processed_name(tmp_path):
    folder = good_folder(tmp_path)
    (folder / "Utility").mkdir()
    shutil.copy(folder / "W2Sample" / "LauraRussoW2.pdf", folder / "Utility" / "LauraRussoW2.pdf")
    _, problems, _ = scan(str(folder))
    assert len(problems) == 1 and "LauraRussoW2_pdf_processed.json" in problems[0]


def test_unknown_category_is_a_note(tmp_path):
    folder = good_folder(tmp_path)
    shutil.copy(folder / "W2Sample" / "LauraRussoW2.pdf", folder / "paystub.pdf")
    docs, problems, notes = scan(str(folder))
    assert problems == []
    assert any("paystub.pdf: category Other" in n for n in notes)


def test_check_only_writes_nothing(tmp_path):
    folder = good_folder(tmp_path / "docs")
    work = tmp_path / "work"
    work.mkdir()
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "load_documents.py"),
                             "--dir", str(folder), "--check-only"],
                            cwd=work, capture_output=True, text=True)
    assert result.returncode == 0, result.stdout + result.stderr
    assert "All files passed the checks." in result.stdout
    assert list(work.iterdir()) == []


def test_problems_stop_before_writing(tmp_path):
    folder = good_folder(tmp_path / "docs")
    (folder / "W2Sample" / "empty.pdf").write_bytes(b"")
    before = sorted(os.listdir(ROOT))
    result = subprocess.run([sys.executable, os.path.join(ROOT, "scripts", "load_documents.py"),
                             "--dir", str(folder)], capture_output=True, text=True)
    assert result.returncode == 1
    assert "Nothing was written." in result.stderr
    assert sorted(os.listdir(ROOT)) == before
