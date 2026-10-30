import numpy as np
import pytest
from rag_lab.core import Chunk, Index, citations_valid, prompt_for, source_recall_at_k, word_chunks


def test_chunks_overlap_and_reject_invalid():
    assert word_chunks("a b c d e", size=3, overlap=1) == ["a b c", "c d e", "e"]
    with pytest.raises(ValueError):
        word_chunks("abc", size=3, overlap=3)


def test_cosine_search_and_persistence(tmp_path):
    idx = Index([Chunk("a.pdf", 1, "refunds"), Chunk("b.md", None, "returns")], np.array([[1, 0], [0, 1]]), "fake")
    assert idx.search(np.array([0.9, 0.1]), 1)[0][2].source == "a.pdf"
    saved = tmp_path / "index.npz"
    idx.save(saved)
    restored = Index.load(saved)
    assert restored.model == "fake"
    assert restored.chunks == idx.chunks


def test_retrieval_metric_and_citations():
    idx = Index([Chunk("a", None, "refunds"), Chunk("b", None, "returns")], np.eye(2), "fake")
    assert source_recall_at_k(idx, [{"question":"refunds","relevant_sources":["a"]}], lambda _: np.array([1.,0.]), 1) == 1
    assert citations_valid("Refunds take 5 days [1]", 1)
    assert not citations_valid("Refunds take 5 days [9]", 1)
    assert not citations_valid("Refunds take 5 days", 1)
    assert citations_valid("INSUFFICIENT_EVIDENCE", 0)
    assert "page 1" in prompt_for("when?", [(0,1.0,Chunk("a.pdf",1,"five days"))])
