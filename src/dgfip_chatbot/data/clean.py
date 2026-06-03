"""Text cleaning, in two tiers:

- ``canonicalize`` — fix encoding artifacts but keep text *natural* (for embeddings + display).
- ``lexical_normalize`` — lowercase / accent-fold / drop stopwords (for the BM25 baseline only).
"""

import re
import unicodedata

# Common French stopwords. Folded to no-accent lowercase at import (see below), so they
# match the output of ``lexical_normalize``'s own folding.
_RAW_FRENCH_STOPWORDS = {
    "au",
    "aux",
    "avec",
    "ce",
    "ces",
    "dans",
    "de",
    "des",
    "du",
    "elle",
    "en",
    "et",
    "eux",
    "il",
    "ils",
    "je",
    "la",
    "le",
    "les",
    "leur",
    "leurs",
    "lui",
    "ma",
    "mais",
    "me",
    "même",
    "mes",
    "moi",
    "mon",
    "ne",
    "nos",
    "notre",
    "nous",
    "on",
    "ou",
    "où",
    "par",
    "pas",
    "pour",
    "qu",
    "que",
    "qui",
    "sa",
    "se",
    "ses",
    "son",
    "sur",
    "ta",
    "te",
    "tes",
    "toi",
    "ton",
    "tu",
    "un",
    "une",
    "vos",
    "votre",
    "vous",
    "c",
    "d",
    "j",
    "l",
    "m",
    "n",
    "s",
    "t",
    "y",
    "à",
    "été",
    "être",
    "avoir",
    "ai",
    "as",
    "avait",
    "ont",
    "est",
    "sont",
    "sera",
    "cette",
    "cet",
    "ceux",
    "celui",
    "celle",
    "dont",
    "donc",
    "alors",
    "aussi",
    "comme",
    "plus",
    "moins",
    "très",
    "peu",
    "si",
    "ni",
    "car",
    "entre",
    "sans",
    "sous",
    "vers",
    "chez",
    "quel",
    "quelle",
    "quels",
    "quelles",
    "cela",
    "ça",
    "ceci",
    "tout",
    "tous",
    "toute",
    "toutes",
    "autre",
    "autres",
    "aucun",
    "aucune",
}

_SPACES_NO_NEWLINE = re.compile(r"[^\S\n]+")
_MULTI_NEWLINE = re.compile(r"\n{3,}")
_WORD = re.compile(r"[a-z0-9]+")


def _strip_accents(text: str) -> str:
    """Fold accents/ligatures to ASCII (é→e, ç→c, œ→oe), preserving the base letters."""
    # Ligatures have no accent to strip, so map them to letter pairs explicitly first.
    text = text.replace("œ", "oe").replace("Œ", "OE").replace("æ", "ae").replace("Æ", "AE")
    # NFKD splits an accented char into "base letter + combining accent mark"...
    decomposed = unicodedata.normalize("NFKD", text)
    # ...then we drop those combining marks, leaving just the base ASCII letters.
    return "".join(c for c in decomposed if not unicodedata.combining(c))


# Pre-fold the stopword set exactly how lexical_normalize folds the text, so they match.
FRENCH_STOPWORDS = {_strip_accents(w).lower() for w in _RAW_FRENCH_STOPWORDS}


def canonicalize(text: str, *, strip_title: bool = False, title: str | None = None) -> str:
    """Clean to natural text: NFKC (folds nbsp→space, composes accents), collapse runs of
    spaces, keep the single-``\\n`` structure. Case/accents/stopwords are preserved."""
    if not text:
        return ""
    # NFKC composes accents AND folds compatibility chars — this is what turns the
    # non-breaking spaces (\xa0, present in 106/113 fiches) into ordinary spaces.
    text = unicodedata.normalize("NFKC", text)
    # Normalize any Windows/Mac line endings to plain "\n".
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # Collapse runs of spaces/tabs to a single space — but NOT newlines (\n is structure).
    text = _SPACES_NO_NEWLINE.sub(" ", text)
    # Trim each line individually, keeping the line breaks between them.
    text = "\n".join(line.strip() for line in text.split("\n"))
    # Cap any 3+ blank-line runs at one blank line, then trim the whole string.
    text = _MULTI_NEWLINE.sub("\n\n", text).strip()
    # Optionally drop a leading line that merely repeats the title (true for all 113 fiches).
    if strip_title and title:
        lines = text.split("\n")
        if lines and _strip_accents(lines[0]).lower() == _strip_accents(title).lower():
            text = "\n".join(lines[1:]).strip()
    return text


def lexical_normalize(text: str, *, remove_stopwords: bool = True) -> str:
    """Normalize for lexical/BM25 retrieval: lowercase, accent-fold, tokenize, drop
    French stopwords. Returns a space-joined token string."""
    # Fold accents (é→e) and lowercase so surface variants collapse to one token form.
    folded = _strip_accents(text).lower()
    # Keep only alphanumeric runs (drops punctuation); after folding these are ASCII.
    tokens = _WORD.findall(folded)
    # Drop high-frequency French function words that just add noise to bag-of-words matching.
    if remove_stopwords:
        tokens = [t for t in tokens if t not in FRENCH_STOPWORDS]
    return " ".join(tokens)
