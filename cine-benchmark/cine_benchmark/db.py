"""Stockage SQLite : œuvres, publications et commentaires (anonymes)."""

from __future__ import annotations

import hashlib
import os
import sqlite3
from dataclasses import dataclass
from pathlib import Path

from .analyse import Analyse
from .titres import cle_titre

CHEMIN_DEFAUT = Path(os.environ.get("CINE_DB", Path(__file__).resolve().parent.parent / "data" / "cine.db"))

_SCHEMA = """
CREATE TABLE IF NOT EXISTS oeuvres (
    id INTEGER PRIMARY KEY,
    titre TEXT NOT NULL,
    cle TEXT NOT NULL UNIQUE,
    type TEXT NOT NULL DEFAULT 'film',
    annee INTEGER
);
CREATE TABLE IF NOT EXISTS publications (
    id INTEGER PRIMARY KEY,
    oeuvre_id INTEGER NOT NULL REFERENCES oeuvres(id) ON DELETE CASCADE,
    texte TEXT NOT NULL,
    note REAL,
    moteur TEXT NOT NULL,
    empreinte TEXT NOT NULL UNIQUE,
    cree_le TEXT NOT NULL DEFAULT (datetime('now'))
);
CREATE TABLE IF NOT EXISTS commentaires (
    id INTEGER PRIMARY KEY,
    publication_id INTEGER NOT NULL REFERENCES publications(id) ON DELETE CASCADE,
    texte TEXT NOT NULL,
    note REAL
);
CREATE INDEX IF NOT EXISTS idx_pub_oeuvre ON publications(oeuvre_id);
CREATE INDEX IF NOT EXISTS idx_com_pub ON commentaires(publication_id);
"""

# Poids du « prior » de la moyenne bayésienne : une œuvre doit accumuler
# plusieurs avis avant de pouvoir dépasser la moyenne générale.
POIDS_PRIOR = 3


def connecter(chemin: Path | str = CHEMIN_DEFAUT) -> sqlite3.Connection:
    if str(chemin) != ":memory:":
        Path(chemin).parent.mkdir(parents=True, exist_ok=True)
    con = sqlite3.connect(chemin, check_same_thread=False)
    con.row_factory = sqlite3.Row
    con.execute("PRAGMA foreign_keys = ON")
    con.executescript(_SCHEMA)
    return con


def _empreinte(texte: str, commentaires: list[str]) -> str:
    h = hashlib.sha256(texte.strip().lower().encode())
    for c in commentaires:
        h.update(b"\x00" + c.strip().lower().encode())
    return h.hexdigest()


class DoublonError(Exception):
    """La même publication (texte + commentaires) a déjà été importée."""


def _oeuvre_id(con: sqlite3.Connection, titre: str, type_: str, annee: int | None) -> int:
    cle = cle_titre(titre) or titre.lower()
    row = con.execute("SELECT id, annee FROM oeuvres WHERE cle = ?", (cle,)).fetchone()
    if row:
        if annee and not row["annee"]:
            con.execute("UPDATE oeuvres SET annee = ? WHERE id = ?", (annee, row["id"]))
        return row["id"]
    cur = con.execute(
        "INSERT INTO oeuvres (titre, cle, type, annee) VALUES (?, ?, ?, ?)", (titre, cle, type_, annee)
    )
    return cur.lastrowid


def enregistrer(con: sqlite3.Connection, analyse: Analyse) -> int:
    """Enregistre une analyse ; renvoie l'id de l'œuvre."""
    textes = [c.texte for c in analyse.commentaires]
    empreinte = _empreinte(analyse.publication.texte, textes)
    if con.execute("SELECT 1 FROM publications WHERE empreinte = ?", (empreinte,)).fetchone():
        raise DoublonError("Publication déjà importée.")
    titre = (analyse.titre or "Titre à identifier").strip()
    with con:
        oid = _oeuvre_id(con, titre, analyse.type, analyse.annee)
        pid = con.execute(
            "INSERT INTO publications (oeuvre_id, texte, note, moteur, empreinte) VALUES (?, ?, ?, ?, ?)",
            (oid, analyse.publication.texte, analyse.publication.note, analyse.moteur, empreinte),
        ).lastrowid
        con.executemany(
            "INSERT INTO commentaires (publication_id, texte, note) VALUES (?, ?, ?)",
            [(pid, c.texte, c.note) for c in analyse.commentaires],
        )
    return oid


@dataclass
class LigneClassement:
    id: int
    titre: str
    type: str
    annee: int | None
    nb_avis: int
    nb_publications: int
    moyenne: float | None
    score: float | None  # moyenne bayésienne, sert au classement
    repartition: tuple[int, int, int]  # avis négatifs (<4), mitigés, positifs (>=7)


def classement(
    con: sqlite3.Connection,
    type_: str | None = None,
    recherche: str | None = None,
    min_avis: int = 1,
    tri: str = "score",
) -> list[LigneClassement]:
    notes = con.execute(
        """
        SELECT o.id AS oid, p.note AS note FROM oeuvres o JOIN publications p ON p.oeuvre_id = o.id
        UNION ALL
        SELECT o.id, c.note FROM oeuvres o JOIN publications p ON p.oeuvre_id = o.id
            JOIN commentaires c ON c.publication_id = p.id
        """
    ).fetchall()
    par_oeuvre: dict[int, list[float]] = {}
    for r in notes:
        if r["note"] is not None:
            par_oeuvre.setdefault(r["oid"], []).append(r["note"])
    toutes = [n for ns in par_oeuvre.values() for n in ns]
    moyenne_globale = sum(toutes) / len(toutes) if toutes else 5.0

    sql = """SELECT o.*, (SELECT COUNT(*) FROM publications p WHERE p.oeuvre_id = o.id) AS nb_pub
             FROM oeuvres o WHERE 1=1"""
    params: list = []
    if type_:
        sql += " AND o.type = ?"
        params.append(type_)
    lignes = []
    for o in con.execute(sql, params).fetchall():
        if recherche and cle_titre(recherche) not in o["cle"] and recherche.lower() not in o["titre"].lower():
            continue
        ns = par_oeuvre.get(o["id"], [])
        if len(ns) < min_avis:
            continue
        moyenne = sum(ns) / len(ns) if ns else None
        score = (POIDS_PRIOR * moyenne_globale + sum(ns)) / (POIDS_PRIOR + len(ns)) if ns else None
        lignes.append(LigneClassement(
            id=o["id"], titre=o["titre"], type=o["type"], annee=o["annee"], nb_avis=len(ns),
            nb_publications=o["nb_pub"],
            moyenne=round(moyenne, 1) if moyenne is not None else None,
            score=round(score, 2) if score is not None else None,
            repartition=(sum(n < 4 for n in ns), sum(4 <= n < 7 for n in ns), sum(n >= 7 for n in ns)),
        ))
    cles = {
        "score": lambda l: (l.score or 0, l.nb_avis),
        "moyenne": lambda l: (l.moyenne or 0, l.nb_avis),
        "avis": lambda l: (l.nb_avis, l.score or 0),
        "titre": None,
    }
    if tri == "titre":
        lignes.sort(key=lambda l: l.titre.lower())
    else:
        lignes.sort(key=cles.get(tri, cles["score"]), reverse=True)
    return lignes


def oeuvre(con: sqlite3.Connection, oid: int) -> dict | None:
    o = con.execute("SELECT * FROM oeuvres WHERE id = ?", (oid,)).fetchone()
    if not o:
        return None
    pubs = []
    for p in con.execute("SELECT * FROM publications WHERE oeuvre_id = ? ORDER BY id DESC", (oid,)):
        coms = con.execute(
            "SELECT * FROM commentaires WHERE publication_id = ? ORDER BY note IS NULL, note DESC", (p["id"],)
        ).fetchall()
        pubs.append({**dict(p), "commentaires": [dict(c) for c in coms]})
    return {**dict(o), "publications": pubs}


def modifier_oeuvre(con: sqlite3.Connection, oid: int, titre: str, type_: str, annee: int | None) -> int:
    """Renomme une œuvre. Si le nouveau titre existe déjà, fusionne les deux ; renvoie l'id final."""
    cle = cle_titre(titre) or titre.lower()
    with con:
        autre = con.execute("SELECT id FROM oeuvres WHERE cle = ? AND id != ?", (cle, oid)).fetchone()
        if autre:
            con.execute("UPDATE publications SET oeuvre_id = ? WHERE oeuvre_id = ?", (autre["id"], oid))
            con.execute("DELETE FROM oeuvres WHERE id = ?", (oid,))
            con.execute("UPDATE oeuvres SET type = ?, annee = COALESCE(?, annee) WHERE id = ?",
                        (type_, annee, autre["id"]))
            return autre["id"]
        con.execute("UPDATE oeuvres SET titre = ?, cle = ?, type = ?, annee = ? WHERE id = ?",
                    (titre, cle, type_, annee, oid))
        return oid


def supprimer_publication(con: sqlite3.Connection, pid: int) -> int | None:
    """Supprime une publication (et l'œuvre si elle n'a plus d'avis) ; renvoie l'id de l'œuvre restante."""
    row = con.execute("SELECT oeuvre_id FROM publications WHERE id = ?", (pid,)).fetchone()
    if not row:
        return None
    with con:
        con.execute("DELETE FROM publications WHERE id = ?", (pid,))
        reste = con.execute("SELECT 1 FROM publications WHERE oeuvre_id = ?", (row["oeuvre_id"],)).fetchone()
        if not reste:
            con.execute("DELETE FROM oeuvres WHERE id = ?", (row["oeuvre_id"],))
            return None
    return row["oeuvre_id"]


def statistiques(con: sqlite3.Connection) -> dict:
    r = con.execute(
        """SELECT (SELECT COUNT(*) FROM oeuvres) AS oeuvres,
                  (SELECT COUNT(*) FROM publications) AS publications,
                  (SELECT COUNT(*) FROM commentaires) AS commentaires,
                  (SELECT COUNT(*) FROM commentaires WHERE note IS NOT NULL)
                + (SELECT COUNT(*) FROM publications WHERE note IS NOT NULL) AS avis_notes"""
    ).fetchone()
    return dict(r)
