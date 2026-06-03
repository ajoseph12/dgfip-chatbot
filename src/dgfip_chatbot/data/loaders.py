"""Load + validate the raw CSVs, renaming the id and category columns."""

import pandas as pd

from dgfip_chatbot.config import settings
from dgfip_chatbot.data.schema import Fiche, Question

# The leading id column in each CSV has a blank header, which pandas reads as "Unnamed: 0".
_FICHE_RENAME = {
    "Unnamed: 0": "fiche_id",
    "niveau_0": "category",  # kept as-is (the impots.gouv URL), just renamed
    "niveau_1": "subcategory",
    "URL": "url",
    "Titre": "titre",
    "Texte": "texte",
}
_QUESTION_RENAME = {"Unnamed: 0": "question_id", "num_texte": "fiche_id"}


def load_fiches() -> list[Fiche]:
    """Read the 113 fiches; validate count and that ids form 0..112."""
    df = pd.read_csv(settings.kb_path).rename(columns=_FICHE_RENAME)
    # Fail fast if the data isn't the shape we expect (guards against silent drift).
    if len(df) != 113:
        raise ValueError(f"expected 113 fiches, got {len(df)}")
    if set(df["fiche_id"]) != set(range(113)):
        raise ValueError("fiche ids are not the contiguous range 0..112")
    # Replace any missing text fields with "" so model construction never sees NaN.
    df = df.fillna({"category": "", "subcategory": "", "url": "", "titre": "", "texte": ""})
    # Build typed Fiche objects. Casting here keeps pydantic happy (CSV gives numpy ints).
    return [
        Fiche(
            fiche_id=int(r.fiche_id),
            category=str(r.category),
            subcategory=str(r.subcategory),
            url=str(r.url),
            titre=str(r.titre),
            texte=str(r.texte),
        )
        for r in df.itertuples(index=False)
    ]


def load_questions() -> pd.DataFrame:
    """Read the 1,427 eval questions; validate count and target fiche range."""
    df = pd.read_csv(settings.questions_path).rename(columns=_QUESTION_RENAME)
    if len(df) != 1427:
        raise ValueError(f"expected 1427 questions, got {len(df)}")
    # Every question must point at a real fiche id (the join key into the KB).
    if not df["fiche_id"].between(0, 112).all():
        raise ValueError("some questions reference a fiche_id outside 0..112")
    # Cheap per-row type check (raises if a value can't be coerced to the schema).
    for r in df.itertuples(index=False):
        Question(question_id=int(r.question_id), question=str(r.question), fiche_id=int(r.fiche_id))
    return df[["question_id", "question", "fiche_id"]]
