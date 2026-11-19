"""Nettoyage et anonymisation du texte collé depuis Facebook.

Objectif : ne garder que le texte des avis. Les noms des auteurs, les
boutons de l'interface (« J'aime », « Répondre »...), les horodatages et
les mentions sont supprimés avant tout stockage ou analyse.
"""

from __future__ import annotations

import re

# Lignes produites par l'interface Facebook lors d'un copier-coller.
_LIGNES_INTERFACE = {
    "j'aime", "j’aime", "aimer", "répondre", "repondre", "partager", "modifié",
    "modifie", "auteur", "auteure", "voir la traduction", "voir plus",
    "afficher plus", "voir moins", "masquer", "commenter", "commentaire",
    "commentaires", "top fan", "fan assidu", "fan assidue", "top contributeur",
    "top contributrice", "contributeur", "contributrice", "admin",
    "administrateur", "administratrice", "modérateur", "modératrice",
    "nouveau membre", "nouvelle membre", "membre", "à l’instant", "à l'instant",
    "hier", "les plus pertinents", "plus pertinents", "tous les commentaires",
    "plus récents", "gif", "photo", "vidéo", "· ", "·", "suivre", "s’abonner",
    "s'abonner", "like", "reply", "share", "edited", "author", "see translation",
    "most relevant", "all comments", "newest", "top contributor", "rising contributor",
    "contributeur en pleine ascension", "contributrice en pleine ascension",
}

_REGEX_INTERFACE = [
    # Horodatages relatifs : « 2 sem », « 3 j », « 5 h », « 12 min », « 1 an »
    re.compile(r"^\d+\s*(s|sem|semaines?|j|jours?|h|heures?|min|minutes?|mois|an|ans|a|w|d|y|m)\.?$"),
    # Dates : « 12 mars », « 12 mars 2024 », « 12 mars, 14:30 »
    re.compile(
        r"^\d{1,2}\s+(janv|févr|fevr|mars|avr|mai|juin|juil|août|aout|sept|oct|nov|déc|dec)"
        r"[a-zéû]*\.?(\s+\d{4})?(,?\s*(à\s*)?\d{1,2}[:h]\d{2})?$"
    ),
    # Compteurs de réactions / réponses : « 12 », « 3 réponses », « Afficher les 4 réponses »
    re.compile(r"^\d+([.,]\d+)?\s*(k|m)?$"),
    re.compile(r"^(afficher|voir)\s+(les\s+|la\s+)?(\d+\s+)?(autres?\s+)?(réponses?|commentaires?|reponses?)"),
    re.compile(r"^\d+\s+(réponses?|reponses?|commentaires?|partages?|j['’]aime|réactions?)$"),
    re.compile(r"^(a|ont)\s+répondu"),
    re.compile(r"^écrire\s+(un\s+commentaire|une\s+réponse)"),
    re.compile(r"^(voir|afficher)\s+(plus|les)\s+(de\s+)?commentaires"),
]

_RE_URL = re.compile(r"https?://\S+|www\.\S+")
_RE_MENTION = re.compile(r"@[\w.\-À-ÿ]+(\s+[A-ZÀ-Ý][\w\-À-ÿ]+)?")
_RE_ESPACES = re.compile(r"[ \t]+")

# Mots qui commencent souvent un avis et ne sont donc pas des noms.
_MOTS_NON_NOMS = {
    "Trop", "Super", "Top", "Nul", "Bof", "Génial", "Excellent", "Vraiment", "Merci",
    "Oui", "Non", "Pas", "Très", "Le", "La", "Les", "Un", "Une", "Je", "J'ai", "Moi",
    "Ce", "Cette", "Il", "Elle", "On", "Grave", "Perso", "Franchement", "Magnifique",
    "Incroyable", "Excellente", "Sublime", "Bien", "Mais", "Et", "Ah", "Oh", "Bah",
}


def est_ligne_interface(ligne: str) -> bool:
    l = ligne.strip().strip("·•").strip().lower()
    if not l:
        return True
    if l in _LIGNES_INTERFACE:
        return True
    return any(r.match(l) for r in _REGEX_INTERFACE)


def ressemble_a_un_nom(ligne: str) -> bool:
    """Heuristique : 1 à 4 mots capitalisés, sans ponctuation de phrase."""
    l = ligne.strip()
    if not l or len(l) > 40 or re.search(r"[.!?,;:/0-9()«»\"]", l):
        return False
    mots = l.split()
    if not 1 <= len(mots) <= 4:
        return False
    if mots[0] in _MOTS_NON_NOMS:
        return False
    return all(m[0].isupper() or m.lower() in {"de", "du", "des", "le", "la", "el", "ben", "bin", "van", "von"} for m in mots)


def nettoyer_texte(texte: str) -> str:
    """Supprime URL, mentions et espaces superflus d'un avis."""
    texte = _RE_URL.sub("", texte)
    texte = _RE_MENTION.sub("", texte)
    lignes = [_RE_ESPACES.sub(" ", l).strip() for l in texte.splitlines()]
    return "\n".join(l for l in lignes if l).strip()


def decouper_commentaires(texte: str, format_facebook: bool = True) -> list[str]:
    """Découpe un bloc de commentaires collé en une liste d'avis anonymes.

    - ``format_facebook=True`` : texte copié tel quel depuis Facebook. Les
      lignes d'interface servent de séparateurs et la première ligne de chaque
      bloc (le nom de l'auteur) est supprimée.
    - ``format_facebook=False`` : un commentaire par paragraphe (séparés par
      une ligne vide), aucun nom attendu.
    """
    texte = texte.replace("\r\n", "\n").replace("\r", "\n")
    blocs: list[list[str]] = []
    courant: list[str] = []

    for ligne in texte.split("\n"):
        separateur = (not ligne.strip()) if not format_facebook else est_ligne_interface(ligne)
        if separateur:
            if courant:
                blocs.append(courant)
                courant = []
        else:
            courant.append(ligne.strip())
    if courant:
        blocs.append(courant)

    commentaires: list[str] = []
    for bloc in blocs:
        if format_facebook:
            if len(bloc) >= 2 and ressemble_a_un_nom(bloc[0]):
                bloc = bloc[1:]
            elif len(bloc) == 1 and ressemble_a_un_nom(bloc[0]):
                continue  # nom seul (commentaire GIF/photo) : rien à garder
        propre = nettoyer_texte("\n".join(bloc))
        if propre:
            commentaires.append(propre)
    return commentaires


def nettoyer_publication(texte: str, format_facebook: bool = True) -> str:
    """Nettoie le texte d'une publication.

    Dans un copier-coller Facebook, l'en-tête d'une publication est « Nom »
    suivi d'une ligne de date (« 2 j », « 12 mars »...). On ne retire la
    première ligne que dans ce cas, pour ne pas supprimer un titre de film
    placé en tête du texte.
    """
    lignes = [l.strip() for l in texte.replace("\r", "").split("\n") if l.strip()]
    if format_facebook:
        if len(lignes) >= 2 and ressemble_a_un_nom(lignes[0]) and est_ligne_interface(lignes[1].strip(" ·")):
            lignes = lignes[1:]
        lignes = [l for l in lignes if not est_ligne_interface(l.strip(" ·"))]
    return nettoyer_texte("\n".join(lignes))


# Lignes qui marquent la fin d'une publication et le début des commentaires.
_RE_PIED_PUBLICATION = re.compile(
    r"^(j['’]aime|commenter|partager|les plus pertinents|plus pertinents|tous les commentaires|"
    r"plus récents|\d+\s+commentaires?|\d+\s+partages?|toutes les réactions\s*:?.*|"
    r"like|comment|share|most relevant|all comments|\d+\s+comments?)$",
    re.I,
)


def separer_publication_commentaires(texte: str) -> tuple[str, str]:
    """Sépare un collage unique (publication + commentaires) en deux parties.

    La publication s'arrête à la première ligne de pied Facebook (« J'aime »,
    « Commenter », « 12 commentaires », « Les plus pertinents »...).
    """
    lignes = texte.replace("\r\n", "\n").split("\n")
    for i, ligne in enumerate(lignes):
        if _RE_PIED_PUBLICATION.match(ligne.strip().strip("·•").strip()):
            return "\n".join(lignes[:i]), "\n".join(lignes[i:])
    return texte, ""
