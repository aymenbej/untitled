"""Détection heuristique du titre, du type et de l'année d'une œuvre."""

from __future__ import annotations

import re

from .lexique import normaliser

TYPES = ("film", "série", "documentaire", "animation")

_RE_GUILLEMETS = re.compile(r"[«“\"]\s*([^«»“”\"\n]{2,80}?)\s*[»”\"]")
_RE_HASHTAG = re.compile(r"#([A-Za-zÀ-ÿ0-9][\wÀ-ÿ]{2,60})")
_RE_APRES_TYPE = re.compile(
    r"(?i:film|s[ée]rie|mini-s[ée]rie|docu(?:mentaire)?|anim[ée]|manga|saga)\s*(?:[:\-–—]\s*|\s+)"
    r"((?:[A-Z0-9À-Ý][\w'’\-À-ÿ]*)(?:\s+(?:[A-Z0-9À-Ý&][\w'’\-À-ÿ]*|de|du|des|of|the|and|la|le|les|et|à|a|d['’][\wÀ-ÿ]+|in|on|:))*)"
)
_RE_ANNEE = re.compile(r"\((19[2-9]\d|20[0-4]\d)\)|\b(19[5-9]\d|20[0-4]\d)\b")

_RE_PLATEFORME = re.compile(
    r"\s+(sur|de|on|via)\s+(Netflix|Prime( Video)?|Amazon|Disney\+?|Canal\+?|Apple( TV\+?)?|HBO|Max|"
    r"Paramount\+?|OCS|Arte|M6\+?|TF1\+?|France\.tv|Crunchyroll|ADN|YouTube).*$",
    re.I,
)

_MOTS_PARASITES = {
    "je", "j'ai", "salut", "bonjour", "hello", "coucou", "hey", "petite", "petit",
    "nouvelle", "nouveau", "reco", "recommandation", "avis", "on", "a", "regardé", "regarde",
}


def _decouper_hashtag(tag: str) -> str:
    if "_" in tag:
        return tag.replace("_", " ")
    return re.sub(r"(?<=[a-zà-ÿ])(?=[A-ZÀ-Ý0-9])", " ", tag)


def deviner_type(texte: str) -> str:
    t = normaliser(texte)
    if re.search(r"\b(documentaire|docu|docuserie|docu-serie)\b", t):
        return "documentaire"
    if re.search(r"\b(anime|animation|manga|dessin anime|pixar|ghibli)\b", t):
        return "animation"
    if re.search(r"\b(serie|mini-serie|saison|saisons|episode|episodes|s\d{1,2}e\d{1,2}|s\d{1,2}\b)", t):
        return "série"
    return "film"


def deviner_annee(texte: str) -> int | None:
    m = _RE_ANNEE.search(texte)
    if not m:
        return None
    return int(m.group(1) or m.group(2))


def deviner_titre(texte: str) -> str | None:
    """Essaie d'extraire le titre ; None si rien de convaincant."""
    m = _RE_GUILLEMETS.search(texte)
    if m:
        return m.group(1).strip(" .,:;!")

    m = _RE_APRES_TYPE.search(texte)
    if m:
        titre = m.group(1)
        titre = _RE_PLATEFORME.sub("", titre)
        # Retire les mots de liaison restés en fin (« Dark de » -> « Dark »)
        titre = re.sub(r"(\s+(de|du|des|of|the|and|la|le|les|et|à|a|in|on|:))+$", "", titre, flags=re.I)
        titre = titre.strip(" .,:;!")
        if titre and normaliser(titre.split()[0]) not in _MOTS_PARASITES:
            return titre

    m = _RE_HASHTAG.search(texte)
    if m:
        return _decouper_hashtag(m.group(1))

    premiere = texte.strip().split("\n", 1)[0].strip(" .,:;!")
    premiere = re.sub(r"\s*\((19|20)\d{2}\)\s*$", "", premiere)
    if 1 <= len(premiere) <= 50 and len(premiere.split()) <= 7 and not re.search(r"[!?]", premiere):
        if normaliser(premiere.split()[0]) not in _MOTS_PARASITES:
            return premiere
    return None


def cle_titre(titre: str) -> str:
    """Clé de regroupement : insensible à la casse, aux accents et à la ponctuation."""
    t = normaliser(titre)
    t = re.sub(r"^(the|le|la|les|l')\s*", "", t)
    return re.sub(r"[^a-z0-9]+", "", t)
