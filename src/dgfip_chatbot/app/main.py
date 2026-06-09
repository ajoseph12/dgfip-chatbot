"""Streamlit demo — a retrieval *chat* over the DGFiP fiches pratiques.

Retrieval-only: the assistant "answers" by routing the question to the most relevant
**official fiche** (the chosen hybrid score-fusion config) and citing it — a "FAQ déguisée".
Run with ``make app`` (needs the ml + app dependency groups).
"""

from __future__ import annotations

import streamlit as st

from dgfip_chatbot.config import settings
from dgfip_chatbot.retrieval.base import RetrievalResult
from dgfip_chatbot.retrieval.bm25 import BM25Retriever
from dgfip_chatbot.retrieval.dense import DenseRetriever
from dgfip_chatbot.retrieval.hybrid import HybridRetriever

# A few starter questions (one per theme) surfaced as sidebar buttons.
EXAMPLES = [
    "Comment corriger une erreur sur ma déclaration de revenus ?",
    "Dans quel délai puis-je déposer une réclamation ?",
    "Comment déclarer les revenus d'une location meublée ?",
]
_SNIPPET_CHARS = 280


@st.cache_resource(show_spinner="Chargement de l'index…")
def get_retriever() -> HybridRetriever:
    """Load the chosen retriever once and reuse it across reruns (model + index are heavy)."""
    return HybridRetriever(dense=DenseRetriever(), bm25=BM25Retriever())


def _snippet(text: str) -> str:
    """Collapse whitespace and truncate a chunk to a short preview."""
    s = " ".join(text.split())
    return s[:_SNIPPET_CHARS].rsplit(" ", 1)[0] + "…" if len(s) > _SNIPPET_CHARS else s


def format_answer(results: list[RetrievalResult]) -> str:
    """Render the retrieved fiches as a chat answer (markdown): best match + alternatives.

    Always offering alternatives is deliberate — the error analysis showed the remaining
    errors are near-duplicate *sibling* fiches, so a shortlist is the honest UX.
    """
    if not results:
        return "Je n'ai trouvé aucune fiche pertinente. Pouvez-vous reformuler la question ?"
    top = results[0]
    md = [
        "Voici la fiche la plus pertinente :",
        "",
        f"### 📄 [{top.titre}]({top.url})",
        f"> {_snippet(top.snippet)}",
        "",
        f"_Score de pertinence : {top.score:.2f}_",
    ]
    if len(results) > 1:
        md.append("\n**Autres fiches qui pourraient aider :**")
        md += [f"- [{r.titre}]({r.url})  ·  {r.score:.2f}" for r in results[1:]]
    md.append("\n_Fiches officielles d'impots.gouv.fr — démo de recherche, pas un conseil fiscal._")
    return "\n".join(md)


def main() -> None:
    st.set_page_config(page_title="DGFiP — trouver la bonne fiche", page_icon="📄")
    st.title("📄 Trouver la bonne fiche")
    st.caption(
        "Posez une question fiscale ; je vous oriente vers la *fiche pratique* officielle la "
        "plus pertinente (impots.gouv.fr). Aucune réponse n'est rédigée — uniquement la recherche."
    )

    with st.sidebar:
        st.header("À propos")
        st.markdown(
            "Recherche parmi les **113 fiches pratiques** des particuliers, sur les 5 thèmes "
            "(déclarer, payer, gérer son patrimoine, signaler un changement, faire un recours).\n\n"
            "Moteur : **hybride** — embeddings *e5-base* + *BM25*, fusion de scores 0,5 / 0,5."
        )
        st.markdown("**Exemples de questions :**")
        for ex in EXAMPLES:
            if st.button(ex, use_container_width=True):
                st.session_state.pending = ex
        if st.button("🗑️ Effacer la conversation", use_container_width=True):
            st.session_state.messages = []
            st.rerun()

    # Load the retriever (cached). Friendly message if the index hasn't been built yet.
    try:
        retriever = get_retriever()
    except FileNotFoundError:
        st.error("Index introuvable — lancez `make data && make index`, puis rechargez la page.")
        st.stop()

    if "messages" not in st.session_state:
        st.session_state.messages = []

    # Replay the conversation so far.
    for m in st.session_state.messages:
        with st.chat_message(m["role"]):
            st.markdown(m["content"])

    # A question can come from the chat box or from an example button.
    query = st.chat_input("Posez votre question… (ex. : comment corriger ma déclaration ?)")
    if not query and st.session_state.get("pending"):
        query = st.session_state.pop("pending")

    if query:
        st.session_state.messages.append({"role": "user", "content": query})
        with st.chat_message("user"):
            st.markdown(query)
        with st.chat_message("assistant"):
            with st.spinner("Recherche de la fiche…"):
                results = retriever.retrieve(query, k=settings.top_k)
            answer = format_answer(results)
            st.markdown(answer)
        st.session_state.messages.append({"role": "assistant", "content": answer})


if __name__ == "__main__":
    main()
