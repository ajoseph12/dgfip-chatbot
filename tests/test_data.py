"""Tests — loading, cleaning, chunking, splitting, and pipeline integrity."""

import math

import pytest

from dgfip_chatbot.data.build import build_chunks
from dgfip_chatbot.data.chunk import chunk_text, count_words
from dgfip_chatbot.data.clean import canonicalize, lexical_normalize
from dgfip_chatbot.data.loaders import load_fiches, load_questions
from dgfip_chatbot.data.split import stratified_split

# --- loaders ---


def test_load_fiches_count_and_ids():
    fiches = load_fiches()
    assert len(fiches) == 113
    assert {f.fiche_id for f in fiches} == set(range(113))


def test_load_questions_count_and_ids():
    df = load_questions()
    assert len(df) == 1427
    assert df["fiche_id"].between(0, 112).all()


# --- cleaning ---


def test_canonicalize_folds_nbsp_keeps_newlines_and_is_idempotent():
    raw = "Titre\nLigne avec   espaces\net une autre"
    clean = canonicalize(raw)
    assert " " not in clean  # nbsp folded by NFKC
    assert "  " not in clean  # space runs collapsed
    assert "\n" in clean  # line structure preserved
    assert canonicalize(clean) == clean  # idempotent


def test_canonicalize_strip_title_optional():
    raw = "Obligations\nLe corps du texte."
    assert canonicalize(raw, strip_title=True, title="OBLIGATIONS").startswith("Le corps")
    assert canonicalize(raw, strip_title=False, title="OBLIGATIONS").startswith("Obligations")


def test_lexical_normalize_lowercases_folds_drops_stopwords():
    out = lexical_normalize("Déclarer LES revenus à l'État")
    assert out == out.lower()
    assert "é" not in out and "à" not in out  # accents folded
    assert "les" not in out.split()  # stopword removed
    assert "declarer" in out.split() and "revenus" in out.split()
    assert lexical_normalize("Déclarer") == lexical_normalize("Déclarer")  # deterministic


# --- chunking ---


def test_chunk_whole_fiche_when_cap_inf():
    text = "ligne 1\nligne 2\nligne 3"
    assert chunk_text(text, cap=math.inf) == [text]


def test_chunk_respects_cap_without_overlap():
    text = "\n".join(f"mot{i} encore{i}" for i in range(40))  # 80 words over many lines
    chunks = chunk_text(text, cap=10, overlap=0)
    assert len(chunks) > 1
    assert all(count_words(c) <= 10 for c in chunks)


def test_chunk_overlap_shares_content():
    text = "\n".join(f"ligne{i}" for i in range(20))
    chunks = chunk_text(text, cap=5, overlap=2)
    # consecutive chunks should share at least one line
    first_lines = set(chunks[1].split("\n"))
    assert first_lines & set(chunks[0].split("\n"))


# --- pipeline integrity ---


@pytest.fixture(scope="module")
def chunks():
    return build_chunks()


def test_chunks_cover_all_fiches_and_are_nonempty(chunks):
    assert chunks["fiche_id"].nunique() == 113
    assert chunks["fiche_id"].between(0, 112).all()
    assert (chunks["text"].str.len() > 0).all()
    assert (chunks["text_lexical"].str.len() > 0).all()
    assert chunks["chunk_id"].is_unique


def test_build_chunks_is_deterministic(chunks):
    assert chunks.equals(build_chunks())


# --- split ---


def test_split_is_stratified_disjoint_and_covers_every_fiche():
    df = load_questions()
    dev, test = stratified_split(df, test_size=0.30, seed=42)
    assert set(dev["question_id"]).isdisjoint(set(test["question_id"]))
    assert len(dev) + len(test) == len(df)
    assert set(dev["fiche_id"]) == set(test["fiche_id"]) == set(df["fiche_id"])
    assert 0.25 <= len(test) / len(df) <= 0.35  # ~30%
