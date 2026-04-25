from __future__ import annotations

import json
from pathlib import Path

from forge.retrieval import Document, TfidfRetriever, from_directory, from_jsonl


def _docs() -> list[Document]:
    return [
        Document(id="parser", text="def parse_json(text):\n    import json\n    return json.loads(text)"),
        Document(id="server", text="def start_server(port):\n    import http.server\n    server = http.server.HTTPServer(('', port), None)\n    server.serve_forever()"),
        Document(id="auth",   text="def hash_password(p):\n    import hashlib\n    return hashlib.sha256(p.encode()).hexdigest()"),
        Document(id="email",  text="def send_email(to, body):\n    import smtplib\n    smtplib.SMTP('localhost').sendmail('a', [to], body)"),
    ]


def test_tfidf_top1_matches_obvious_query():
    r = TfidfRetriever().index(_docs())
    top = r.search("parse JSON from a string", k=1)
    assert top and top[0].id == "parser"

    top = r.search("hash a password securely", k=1)
    assert top and top[0].id == "auth"


def test_tfidf_returns_k_in_descending_order():
    r = TfidfRetriever().index(_docs())
    top = r.search("server http port", k=4)
    assert len(top) >= 1
    assert top[0].id == "server"


def test_tfidf_save_and_load_roundtrip(tmp_path: Path):
    r = TfidfRetriever().index(_docs())
    path = tmp_path / "idx.json"
    r.save(path)

    r2 = TfidfRetriever.load(path)
    top = r2.search("parse json", k=1)
    assert top and top[0].id == "parser"


def test_corpus_from_directory_filters_by_suffix(tmp_path: Path):
    (tmp_path / "a.py").write_text("print(1)")
    (tmp_path / "b.txt").write_text("note")
    (tmp_path / "sub").mkdir()
    (tmp_path / "sub" / "c.py").write_text("print(2)")

    docs = from_directory(tmp_path, suffixes={".py"})
    ids = sorted(d.id for d in docs)
    assert ids == ["a.py", "sub/c.py"]


def test_corpus_from_directory_skips_oversized(tmp_path: Path):
    (tmp_path / "small.py").write_text("x = 1")
    (tmp_path / "big.py").write_text("x = 1\n" * 100_000)
    docs = from_directory(tmp_path, suffixes={".py"}, max_bytes_per_file=1000)
    assert {d.id for d in docs} == {"small.py"}


def test_corpus_from_jsonl(tmp_path: Path):
    p = tmp_path / "c.jsonl"
    p.write_text(json.dumps({"id": "1", "text": "hello", "tag": "x"}) + "\n")
    docs = from_jsonl(p)
    assert docs[0].id == "1"
    assert docs[0].text == "hello"
    assert docs[0].metadata == {"tag": "x"}
