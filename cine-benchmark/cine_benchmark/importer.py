"""Import en lot depuis un fichier JSON ou texte.

JSON : liste d'objets
    [{"publication": "...", "commentaires": ["...", "..."], "titre": "facultatif"}]

Texte : collages Facebook séparés par une ligne « ===== ». La publication et
ses commentaires sont séparés automatiquement ; on peut aussi forcer la
séparation avec une ligne « --- ».

Usage : cine-import fichier.json [--moteur lexique|claude|auto] [--brut]
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass
from pathlib import Path

from . import db
from .analyse import analyser
from .nettoyage import decouper_commentaires, nettoyer_publication, separer_publication_commentaires
from .titres import TYPES


@dataclass
class Entree:
    publication: str
    commentaires: str | list[str]
    titre: str | None = None
    type: str | None = None


def lire_fichier(chemin: Path) -> list[Entree]:
    contenu = chemin.read_text(encoding="utf-8")
    if chemin.suffix.lower() == ".json":
        return [
            Entree(d["publication"], d.get("commentaires", []), d.get("titre"), d.get("type"))
            for d in json.loads(contenu)
        ]
    entrees = []
    for bloc in re.split(r"^\s*={3,}\s*$", contenu, flags=re.M):
        if not bloc.strip():
            continue
        parties = re.split(r"^\s*-{3,}\s*$", bloc, maxsplit=1, flags=re.M)
        if len(parties) == 2:
            entrees.append(Entree(parties[0], parties[1]))
        else:
            entrees.append(Entree(*separer_publication_commentaires(bloc)))
    return entrees


def importer_entree(con, e: Entree, moteur: str = "auto", format_facebook: bool = True) -> int:
    publication = nettoyer_publication(e.publication, format_facebook)
    if isinstance(e.commentaires, list):
        commentaires = [c for brut in e.commentaires for c in decouper_commentaires(brut, False)]
    else:
        commentaires = decouper_commentaires(e.commentaires, format_facebook)
    if not publication:
        raise ValueError("Publication vide après nettoyage.")
    analyse = analyser(publication, commentaires, moteur)
    if e.titre:
        analyse.titre = e.titre.strip()
    if e.type in TYPES:
        analyse.type = e.type
    return db.enregistrer(con, analyse)


def main(argv: list[str] | None = None) -> int:
    p = argparse.ArgumentParser(description="Importe des avis (JSON ou texte) dans la base.")
    p.add_argument("fichiers", nargs="+", type=Path)
    p.add_argument("--moteur", choices=["auto", "lexique", "claude"], default="auto")
    p.add_argument("--brut", action="store_true",
                   help="Commentaires déjà propres (un par paragraphe), pas un copier-coller Facebook.")
    p.add_argument("--db", type=Path, default=db.CHEMIN_DEFAUT)
    args = p.parse_args(argv)

    con = db.connecter(args.db)
    ok = doublons = erreurs = 0
    for f in args.fichiers:
        for e in lire_fichier(f):
            try:
                importer_entree(con, e, args.moteur, not args.brut)
                ok += 1
            except db.DoublonError:
                doublons += 1
            except Exception as ex:
                erreurs += 1
                print(f"Erreur : {ex}", file=sys.stderr)
    print(f"{ok} publication(s) importée(s), {doublons} doublon(s), {erreurs} erreur(s).")
    return 1 if erreurs and not ok else 0


if __name__ == "__main__":
    sys.exit(main())
