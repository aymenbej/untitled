"""Application web : classement, fiches et import des avis.

Lancement : uvicorn cine_benchmark.app:app --reload
"""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI, File, Form, HTTPException, Request, UploadFile
from fastapi.responses import HTMLResponse, JSONResponse, RedirectResponse
from fastapi.templating import Jinja2Templates

from . import db
from .analyse import claude_disponible
from .importer import Entree, importer_entree, lire_fichier
from .nettoyage import separer_publication_commentaires
from .titres import TYPES

ICI = Path(__file__).resolve().parent
templates = Jinja2Templates(directory=ICI / "templates")


def _couleur(note):
    if note is None:
        return "neutre"
    return "bas" if note < 4 else "moyen" if note < 7 else "haut"


templates.env.filters["couleur"] = _couleur
templates.env.filters["note"] = lambda n: "—" if n is None else f"{n:.1f}".replace(".", ",")


def creer_app(chemin_db=db.CHEMIN_DEFAUT) -> FastAPI:
    app = FastAPI(title="Cine Benchmark")
    con = db.connecter(chemin_db)
    app.state.con = con

    def page(request: Request, nom: str, **ctx):
        return templates.TemplateResponse(
            request, nom, {"types": TYPES, "stats": db.statistiques(con), "claude": claude_disponible(), **ctx}
        )

    @app.get("/", response_class=HTMLResponse)
    def accueil(request: Request, type: str = "", q: str = "", tri: str = "score", min_avis: int = 1):
        lignes = db.classement(con, type or None, q or None, max(1, min_avis), tri)
        return page(request, "classement.html", lignes=lignes,
                    filtres={"type": type, "q": q, "tri": tri, "min_avis": min_avis})

    @app.get("/oeuvre/{oid}", response_class=HTMLResponse)
    def fiche(request: Request, oid: int):
        o = db.oeuvre(con, oid)
        if not o:
            raise HTTPException(404, "Œuvre introuvable")
        notes = [p["note"] for p in o["publications"]] + [
            c["note"] for p in o["publications"] for c in p["commentaires"]
        ]
        notes = [n for n in notes if n is not None]
        histo = [sum(1 for n in notes if i <= n < i + 1 or (i == 9 and n == 10)) for i in range(10)]
        ligne = next((l for l in db.classement(con) if l.id == oid), None)
        return page(request, "oeuvre.html", o=o, ligne=ligne, histo=histo, max_histo=max(histo or [1]) or 1)

    @app.post("/oeuvre/{oid}/modifier")
    def modifier(oid: int, titre: str = Form(...), type: str = Form(...), annee: str = Form("")):
        if type not in TYPES:
            raise HTTPException(400, "Type inconnu")
        nouvel_id = db.modifier_oeuvre(con, oid, titre.strip(), type, int(annee) if annee.strip().isdigit() else None)
        return RedirectResponse(f"/oeuvre/{nouvel_id}", status_code=303)

    @app.post("/publication/{pid}/supprimer")
    def supprimer(pid: int):
        oid = db.supprimer_publication(con, pid)
        return RedirectResponse(f"/oeuvre/{oid}" if oid else "/", status_code=303)

    @app.get("/importer", response_class=HTMLResponse)
    def formulaire(request: Request):
        return page(request, "importer.html", message=None, erreur=None, valeurs={})

    @app.post("/importer", response_class=HTMLResponse)
    def importer(
        request: Request,
        collage: str = Form(""),
        publication: str = Form(""),
        commentaires: str = Form(""),
        titre: str = Form(""),
        type: str = Form(""),
        moteur: str = Form("auto"),
        format: str = Form("facebook"),
    ):
        valeurs = {"titre": titre, "type": type, "moteur": moteur, "format": format}
        if collage.strip():
            publication, commentaires = separer_publication_commentaires(collage)
        try:
            oid = importer_entree(
                con, Entree(publication, commentaires, titre or None, type or None), moteur, format == "facebook"
            )
        except db.DoublonError:
            return page(request, "importer.html", message=None, erreur="Cette publication a déjà été importée.",
                        valeurs=valeurs)
        except ValueError as e:
            return page(request, "importer.html", message=None, erreur=str(e), valeurs=valeurs)
        return RedirectResponse(f"/oeuvre/{oid}?importe=1", status_code=303)

    @app.post("/importer/fichier", response_class=HTMLResponse)
    async def importer_fichier(request: Request, fichier: UploadFile = File(...), moteur: str = Form("auto")):
        import tempfile

        suffixe = Path(fichier.filename or "import.txt").suffix or ".txt"
        with tempfile.NamedTemporaryFile(suffix=suffixe, delete=False) as tmp:
            tmp.write(await fichier.read())
        ok = doublons = 0
        erreurs: list[str] = []
        try:
            for e in lire_fichier(Path(tmp.name)):
                try:
                    importer_entree(con, e, moteur)
                    ok += 1
                except db.DoublonError:
                    doublons += 1
                except Exception as ex:
                    erreurs.append(str(ex))
        except (json.JSONDecodeError, KeyError, UnicodeDecodeError) as ex:
            return page(request, "importer.html", message=None, erreur=f"Fichier illisible : {ex}", valeurs={})
        finally:
            Path(tmp.name).unlink(missing_ok=True)
        msg = f"{ok} publication(s) importée(s), {doublons} doublon(s) ignoré(s)."
        if erreurs:
            msg += f" {len(erreurs)} erreur(s) : " + " ; ".join(erreurs[:3])
        return page(request, "importer.html", message=msg, erreur=None, valeurs={})

    @app.get("/api/classement")
    def api_classement(type: str = "", q: str = "", tri: str = "score", min_avis: int = 1):
        return JSONResponse([l.__dict__ for l in db.classement(con, type or None, q or None, min_avis, tri)])

    @app.get("/api/oeuvre/{oid}")
    def api_oeuvre(oid: int):
        o = db.oeuvre(con, oid)
        if not o:
            raise HTTPException(404, "Œuvre introuvable")
        return o

    return app


app = creer_app()
