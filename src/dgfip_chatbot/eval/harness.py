"""Run a retriever over a question set and aggregate metrics (overall / per-theme / macro)."""

import pandas as pd

from dgfip_chatbot.config import settings
from dgfip_chatbot.eval.metrics import hit_at_k, reciprocal_rank
from dgfip_chatbot.retrieval.base import Retriever

_FULL = 1_000_000  # retrieve the full fiche ranking so MRR / hit@k are well-defined


def evaluate(retriever: Retriever, questions: pd.DataFrame, ks: list[int] | None = None) -> dict:
    """Evaluate `retriever` on `questions` (needs columns `question`, `fiche_id`, `cat`).

    Returns ``{"overall": Series, "per_theme": DataFrame, "macro": Series}``.
    """
    ks = ks or settings.eval_k
    rows = []
    for q in questions.itertuples(index=False):
        ranked = [r.fiche_id for r in retriever.retrieve(q.question, k=_FULL)]
        row = {"cat": q.cat, "MRR": reciprocal_rank(ranked, q.fiche_id)}
        for k in ks:
            row[f"hit@{k}"] = hit_at_k(ranked, q.fiche_id, k)
        rows.append(row)

    df = pd.DataFrame(rows)
    metric_cols = ["MRR"] + [f"hit@{k}" for k in ks]
    overall = df[metric_cols].mean()  # micro: every question weighted equally
    per_theme = df.groupby("cat")[metric_cols].mean()  # one row per category
    macro = per_theme.mean()  # themes weighted equally
    return {"overall": overall, "per_theme": per_theme, "macro": macro}
