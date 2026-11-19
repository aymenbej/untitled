import json
from pathlib import Path
from types import SimpleNamespace

import pytest
from fastapi.testclient import TestClient

from cine_benchmark import analyse, db, lexique
from cine_benchmark.app import creer_app
from cine_benchmark.importer import Entree, importer_entree, lire_fichier
from cine_benchmark.nettoyage import (
    decouper_commentaires,
    nettoyer_publication,
    separer_publication_commentaires,
)
from cine_benchmark.titres import cle_titre, deviner_annee, deviner_titre, deviner_type

EXEMPLE = Path(__file__).resolve().parent.parent / "exemples" / "collage_facebook.txt"
NOMS = ["Karim", "Benali", "Sophie", "Martin", "Lucas", "Petit", "Nadia", "Yasmine", "Haddad",
        "Thomas", "Bernard", "Inès", "Laurent", "Julie", "Moreau", "Mehdi", "Claire", "Antoine"]


# ---------------------------------------------------------------- anonymisation

def test_aucun_nom_ne_survit_au_nettoyage():
    for bloc in EXEMPLE.read_text(encoding="utf-8").split("====="):
        pub, coms = separer_publication_commentaires(bloc)
        textes = [nettoyer_publication(pub)] + decouper_commentaires(coms)
        tout = "\n".join(textes)
        for nom in NOMS:
            assert nom not in tout, nom
        assert "J’aime" not in tout and "Répondre" not in tout


def test_decoupage_commentaires_facebook():
    coms = decouper_commentaires("Jean Dupont\nSuper film\nJ’aime\nRépondre\n2 sem\nMarie Curie\nBof\nJ’aime\n3 j")
    assert coms == ["Super film", "Bof"]


def test_mentions_et_liens_supprimes():
    assert decouper_commentaires("@Paul regarde https://x.com/a ça", format_facebook=False) == ["regarde ça"]


def test_titre_en_tete_de_publication_conserve():
    # « The Bear » ressemble à un nom mais n'est pas suivi d'une date : c'est le titre.
    assert nettoyer_publication("The Bear\nUne pépite").startswith("The Bear")
    assert nettoyer_publication("Jean Dupont\n2 j\nThe Bear, une pépite") == "The Bear, une pépite"


# ---------------------------------------------------------------- notation

@pytest.mark.parametrize("texte, mini, maxi", [
    ("Un chef-d'œuvre absolu, je recommande", 8.5, 10),
    ("Génial", 7, 9.5),
    ("C'est pas mal", 5, 7.5),
    ("Bof, sans plus", 2, 5),
    ("Nul, une perte de temps", 0, 2.5),
    ("Pas génial du tout", 1, 5),
    ("Pas mal mais la fin est décevante", 2, 5),
])
def test_notes_lexique(texte, mini, maxi):
    note = lexique.noter(texte)
    assert note is not None and mini <= note <= maxi, (texte, note)


def test_note_explicite_prioritaire():
    assert lexique.noter("Génial ! 7/10") == 7.0
    assert lexique.noter("4/5") == 8.0
    assert lexique.noter("★★★") == 6.0


def test_avis_sans_opinion_non_note():
    assert lexique.noter("regarde ça") is None
    assert lexique.noter("C'est sur quelle plateforme ?") is None


# ---------------------------------------------------------------- titres

def test_detection_titre_type_annee():
    assert deviner_titre("Je viens de finir la série « Dark »") == "Dark"
    assert deviner_titre("Film : Le Comte de Monte-Cristo (2024). Magnifique") == "Le Comte de Monte-Cristo"
    assert deviner_titre("#BreakingBad à voir") == "Breaking Bad"
    assert deviner_type("3 saisons d'une traite") == "série"
    assert deviner_type("Un documentaire passionnant") == "documentaire"
    assert deviner_annee("Dune (2021)") == 2021
    assert cle_titre("The Bear") == cle_titre("the bear!") == cle_titre("Bear")


# ---------------------------------------------------------------- base et classement

@pytest.fixture
def con():
    return db.connecter(":memory:")


def test_import_regroupe_et_dedoublonne(con):
    e1 = Entree("Série « Dark », géniale", ["Excellent", "Bof"])
    oid = importer_entree(con, e1, "lexique", format_facebook=False)
    oid2 = importer_entree(con, Entree("« dark » encore une pépite", ["Top"]), "lexique", format_facebook=False)
    assert oid == oid2
    with pytest.raises(db.DoublonError):
        importer_entree(con, e1, "lexique", format_facebook=False)
    (ligne,) = db.classement(con)
    assert ligne.nb_publications == 2 and ligne.nb_avis == 5


def test_moyenne_bayesienne_favorise_les_titres_tres_commentes(con):
    importer_entree(con, Entree("« Unique » 10/10", []), "lexique", False)
    importer_entree(con, Entree("« Populaire » 9/10", ["9/10"] * 30), "lexique", False)
    importer_entree(con, Entree("« Navet » 2/10", ["1/10"] * 10), "lexique", False)
    titres = [l.titre for l in db.classement(con)]
    assert titres[0] == "Populaire" and titres[-1] == "Navet"


def test_renommer_fusionne(con):
    a = importer_entree(con, Entree("« Dark » génial", []), "lexique", False)
    b = importer_entree(con, Entree("« Dark Netflix » génial aussi", []), "lexique", False)
    assert a != b
    assert db.modifier_oeuvre(con, b, "Dark", "série", None) == a
    assert len(db.oeuvre(con, a)["publications"]) == 2


def test_lire_fichier_txt_et_json(tmp_path):
    entrees = lire_fichier(EXEMPLE)
    assert len(entrees) == 2  # sans ligne « --- », la séparation publication/commentaires est automatique
    assert "Dark" in entrees[0].publication and "Meilleure série" in entrees[0].commentaires
    f = tmp_path / "a.json"
    f.write_text(json.dumps([{"publication": "« X » top", "commentaires": ["bien"], "titre": "X"}]))
    (e,) = lire_fichier(f)
    assert e.titre == "X" and e.commentaires == ["bien"]


# ---------------------------------------------------------------- Claude (simulé, sans réseau)

class _FauxClient:
    def __init__(self, donnees, stop_reason="end_turn"):
        bloc = SimpleNamespace(type="text", text=json.dumps(donnees))
        self.reponse = SimpleNamespace(content=[bloc], stop_reason=stop_reason)
        self.appel = None
        self.beta = SimpleNamespace(messages=SimpleNamespace(create=self._create))

    def _create(self, **kwargs):
        self.appel = kwargs
        return self.reponse


def test_analyse_claude_simulee():
    client = _FauxClient({
        "titre": "Dark", "type": "série", "annee": 2017, "note_publication": 9.5,
        "notes_commentaires": [{"numero": 1, "note": 12}, {"numero": 2, "note": None}],
    })
    a = analyse.analyser_claude("La série Dark !", ["Génial", "@ami"], client=client)
    assert (a.titre, a.type, a.annee, a.moteur) == ("Dark", "série", 2017, "claude")
    assert a.publication.note == 9.5
    assert [c.note for c in a.commentaires] == [10.0, None]  # note bornée à 10
    assert client.appel["output_config"]["format"]["type"] == "json_schema"


def test_repli_sur_lexique_si_claude_echoue(monkeypatch):
    monkeypatch.setattr(analyse, "claude_disponible", lambda: True)
    monkeypatch.setattr(analyse, "analyser_claude", lambda *a, **k: (_ for _ in ()).throw(RuntimeError("réseau")))
    assert analyse.analyser("« Dark » génial", [], "auto").moteur == "lexique"


# ---------------------------------------------------------------- application web

def test_parcours_web():
    client = TestClient(creer_app(":memory:"))
    assert "La base est vide" in client.get("/").text

    r = client.post("/importer", data={"collage": EXEMPLE.read_text(encoding="utf-8").split("=====")[0]},
                    follow_redirects=False)
    assert r.status_code == 303
    fiche = client.get(r.headers["location"]).text
    assert "Dark" in fiche and "Sophie" not in fiche and "Karim" not in fiche

    page = client.get("/?type=série").text
    assert "Dark" in page
    data = client.get("/api/classement").json()
    assert data[0]["titre"] == "Dark" and data[0]["nb_avis"] == 5

    r = client.post("/importer", data={"collage": EXEMPLE.read_text(encoding="utf-8").split("=====")[0]})
    assert "déjà été importée" in r.text
