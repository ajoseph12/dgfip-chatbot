"""Typed contracts for the Phase 1 records."""

from pydantic import BaseModel


class Fiche(BaseModel):
    """A knowledge-base fiche (one row of info_particulier_impot.csv)."""

    fiche_id: int
    category: str  # renamed niveau_0 (impots.gouv URL, as-is)
    subcategory: str  # renamed niveau_1 (impots.gouv URL, as-is)
    url: str
    titre: str
    texte: str


class Question(BaseModel):
    """An eval question (one row of questions_fiches_fip.csv)."""

    question_id: int
    question: str
    fiche_id: int  # renamed num_texte (the ground-truth target fiche)


class Chunk(BaseModel):
    """A retrievable passage of a fiche, with both text views."""

    chunk_id: str
    fiche_id: int
    chunk_index: int
    text: str  # natural text -> embeddings + display
    text_lexical: str  # normalized text -> BM25
    titre: str
    url: str
    category: str
    subcategory: str
    n_chars: int
    n_tokens: int
