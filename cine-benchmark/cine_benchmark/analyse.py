"""Analyse d'une publication et de ses commentaires.

Deux moteurs, même résultat :
- ``lexique`` (par défaut, hors ligne, gratuit) ;
- ``claude`` (si ANTHROPIC_API_KEY est défini) : extraction du titre et
  notation plus fine (ironie, nuances). En cas d'échec de l'appel, on
  retombe automatiquement sur le lexique.
"""

from __future__ import annotations

import json
import logging
import os
from dataclasses import dataclass, field

from . import lexique, titres

log = logging.getLogger(__name__)

MODELE_CLAUDE = os.environ.get("CINE_MODELE", "claude-opus-5-5")


@dataclass
class AvisNote:
    texte: str
    note: float | None


@dataclass
class Analyse:
    titre: str | None
    type: str
    annee: int | None
    publication: AvisNote
    commentaires: list[AvisNote] = field(default_factory=list)
    moteur: str = "lexique"


def analyser_lexique(publication: str, commentaires: list[str]) -> Analyse:
    return Analyse(
        titre=titres.deviner_titre(publication),
        type=titres.deviner_type(publication),
        annee=titres.deviner_annee(publication),
        publication=AvisNote(publication, lexique.noter(publication)),
        commentaires=[AvisNote(c, lexique.noter(c)) for c in commentaires],
        moteur="lexique",
    )


# --------------------------------------------------------------------------- Claude

_CONSIGNE = """Tu analyses des avis publiés dans un groupe francophone où les membres partagent \
les films et séries qu'ils ont regardés. On te donne une publication et ses commentaires, déjà anonymisés.

1. Identifie l'œuvre dont parle la publication : son titre officiel (titre français s'il est \
couramment utilisé, sinon titre original), son type (film, série, documentaire ou animation) \
et son année si tu la connais avec certitude.
2. Note de 0 à 10 l'opinion de l'auteur de la publication sur l'œuvre, puis celle de chaque commentaire.
   - 0 = détesté, 5 = mitigé / moyen, 10 = chef-d'œuvre absolu.
   - Tiens compte de l'ironie, des nuances (« bien mais la fin est ratée ») et des notes explicites (« 7/10 »).
   - Mets null si le commentaire n'exprime aucun avis sur l'œuvre (identification d'un ami, question, \
« je note », hors sujet, avis sur une autre œuvre).
Réponds uniquement avec le JSON demandé, sans aucun nom de personne."""

_SCHEMA = {
    "type": "object",
    "properties": {
        "titre": {"anyOf": [{"type": "string"}, {"type": "null"}]},
        "type": {"type": "string", "enum": list(titres.TYPES)},
        "annee": {"anyOf": [{"type": "integer"}, {"type": "null"}]},
        "note_publication": {"anyOf": [{"type": "number"}, {"type": "null"}]},
        "notes_commentaires": {
            "type": "array",
            "items": {
                "type": "object",
                "properties": {
                    "numero": {"type": "integer"},
                    "note": {"anyOf": [{"type": "number"}, {"type": "null"}]},
                },
                "required": ["numero", "note"],
                "additionalProperties": False,
            },
        },
    },
    "required": ["titre", "type", "annee", "note_publication", "notes_commentaires"],
    "additionalProperties": False,
}


def _borner(note: float | None) -> float | None:
    if note is None:
        return None
    return round(min(10.0, max(0.0, float(note))), 1)


def claude_disponible() -> bool:
    if not (os.environ.get("ANTHROPIC_API_KEY") or os.environ.get("ANTHROPIC_AUTH_TOKEN")):
        return False
    try:
        import anthropic  # noqa: F401
    except ImportError:
        return False
    return True


def analyser_claude(publication: str, commentaires: list[str], client=None) -> Analyse:
    import anthropic

    client = client or anthropic.Anthropic()
    contenu = "PUBLICATION :\n" + publication + "\n\nCOMMENTAIRES :\n" + (
        "\n".join(f"[{i}] {c}" for i, c in enumerate(commentaires, 1)) or "(aucun)"
    )
    reponse = client.beta.messages.create(
        model=MODELE_CLAUDE,
        max_tokens=16000,
        system=_CONSIGNE,
        messages=[{"role": "user", "content": contenu}],
        output_config={
            "effort": "low",
            "format": {"type": "json_schema", "schema": _SCHEMA},
        },
        # Si le modèle décline la requête, l'API la rejoue sur un modèle de repli.
        betas=["server-side-fallback-2026-07-01"],
        fallbacks="default",
    )
    if reponse.stop_reason in ("refusal", "max_tokens"):
        raise RuntimeError(f"Réponse inutilisable (stop_reason={reponse.stop_reason})")
    texte = next(b.text for b in reponse.content if b.type == "text")
    data = json.loads(texte)

    notes = {n["numero"]: _borner(n["note"]) for n in data["notes_commentaires"]}
    type_ = data["type"] if data["type"] in titres.TYPES else titres.deviner_type(publication)
    return Analyse(
        titre=(data["titre"] or "").strip() or titres.deviner_titre(publication),
        type=type_,
        annee=data["annee"],
        publication=AvisNote(publication, _borner(data["note_publication"])),
        commentaires=[AvisNote(c, notes.get(i)) for i, c in enumerate(commentaires, 1)],
        moteur="claude",
    )


def analyser(publication: str, commentaires: list[str], moteur: str = "auto") -> Analyse:
    """Analyse avec le moteur demandé (« auto », « claude » ou « lexique »)."""
    if moteur == "claude" or (moteur == "auto" and claude_disponible()):
        try:
            return analyser_claude(publication, commentaires)
        except Exception as e:  # réseau, quota, refus... : on ne bloque pas l'import
            log.warning("Analyse Claude impossible (%s), repli sur le lexique.", e)
    return analyser_lexique(publication, commentaires)
