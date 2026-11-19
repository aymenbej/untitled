"""Notation hors ligne d'un avis en français (0 à 10) par lexique.

Pas de dépendance ni de clé API : fonctionne partout. Moins fin qu'un
modèle de langage (ironie, avis nuancés), mais donne une note cohérente
sur les avis courts typiques des groupes Facebook.

Règles :
1. Une note explicite (« 8/10 », « 4/5 », « ★★★★ ») est prioritaire.
2. Sinon on additionne les poids des expressions et mots repérés, avec
   gestion des négations (« pas génial ») et des intensifs (« vraiment nul »).
3. Après « mais », la fin de la phrase pèse double.
4. Le total est ramené sur 0-10 par une tangente hyperbolique centrée sur 5.
5. Un avis sans aucun signal (« @Paul regarde ça ») n'est pas noté (None).
"""

from __future__ import annotations

import math
import re
import unicodedata


def _sans_accents(texte: str) -> str:
    texte = unicodedata.normalize("NFD", texte)
    return "".join(c for c in texte if unicodedata.category(c) != "Mn")


def normaliser(texte: str) -> str:
    texte = texte.lower().replace("’", "'").replace("œ", "oe")
    return _sans_accents(texte)


# Expressions (testées avant les mots isolés). Poids de -3 à +3.
EXPRESSIONS: dict[str, float] = {
    "chef d'oeuvre": 3, "chef-d'oeuvre": 3, "coup de coeur": 3, "a voir absolument": 3,
    "je recommande vivement": 3, "je recommande fortement": 3, "une des meilleures": 3,
    "un des meilleurs": 3, "l'une des meilleures": 3, "le meilleur": 2.5, "la meilleure": 2.5,
    "meilleure serie": 3, "meilleur film": 3, "j'ai adore": 2.5, "j'ai kiffe": 2.5,
    "je recommande": 2, "a voir": 1.5, "a regarder": 1, "j'ai aime": 2, "j'ai bien aime": 1.5,
    "je conseille": 2, "foncez": 2.5, "fonce": 2, "pas mal": 1, "pas mauvais": 0.8,
    "pas decu": 1.5, "pas decue": 1.5, "pas deçu": 1.5, "sans plus": -0.8, "pas terrible": -1.5,
    "pas ouf": -1.5, "pas top": -1.2, "pas fan": -1.2, "pas accroche": -1.5, "pas accrochee": -1.5,
    "j'ai pas accroche": -1.5, "pas aime": -1.8, "pas du tout": -0.5, "perte de temps": -2.5,
    "a eviter": -2.5, "a fuir": -3, "passez votre chemin": -2.5, "j'ai lache": -1.8,
    "j'ai abandonne": -2, "j'ai arrete": -1.5, "trop long": -1, "trop lent": -1.2,
    "tire en longueur": -1.2, "tire en longueurs": -1.2, "fin baclee": -1.5, "fin decevante": -1.5,
    "je me suis ennuye": -2, "je me suis ennuyee": -2, "on s'ennuie": -2, "on s'ennuit": -2,
    "sans interet": -2, "aucun interet": -2.5, "un navet": -3, "quel navet": -3,
    "tient en haleine": 2, "tenu en haleine": 2, "pas pu m'arreter": 2.5, "binge watche": 2,
    "d'une traite": 2, "en une nuit": 1.5, "en un week-end": 1.5, "bien joue": 1.5,
    "bien ficele": 2, "bien fait": 1.5, "bien realise": 1.5, "vaut le coup": 2,
    "vaut le detour": 2, "vaut pas le coup": -2, "vaut rien": -2.5, "mitige": -0.3,
    "plutot bien": 1, "assez bien": 0.8, "pas si mal": 0.8, "trop bien": 2.5,
    "trop fort": 2, "trop beau": 2.2, "trop nul": -2.8, "bof bof": -1.5,
    "regardable": 0, "se laisse regarder": 0.5, "sans pretention": 0.3, "on ne s'ennuie pas": 2,
    "on s'ennuie pas": 2, "pas vu passer": 1.8, "deception": -2, "pas a la hauteur": -1.8,
}

MOTS: dict[str, float] = {
    # Très positifs
    "chef-d'oeuvre": 3, "pepite": 3, "incontournable": 3, "exceptionnel": 3, "exceptionnelle": 3,
    "masterclass": 3, "parfait": 2.8, "parfaite": 2.8, "sublime": 2.8, "magistral": 3,
    "magistrale": 3, "bijou": 2.8, "phenomenal": 3, "phenomenale": 3, "culte": 2.5,
    "extraordinaire": 2.8, "grandiose": 2.8, "epoustouflant": 2.8, "epoustouflante": 2.8,
    "incroyable": 2.5, "genial": 2.5, "geniale": 2.5, "excellent": 2.5, "excellente": 2.5,
    "magnifique": 2.5, "fantastique": 2.5, "formidable": 2.3, "brillant": 2.3, "brillante": 2.3,
    "adore": 2.5, "adoree": 2.5, "adorer": 2.3, "addictif": 2.3, "addictive": 2.3,
    "captivant": 2.3, "captivante": 2.3, "bluffant": 2.3, "bluffante": 2.3, "pepite": 3,
    "ouf": 2, "dingue": 2, "top": 2, "canon": 1.8, "kiffe": 2.2, "kiff": 2.2, "love": 2,
    "wow": 2, "waouh": 2, "waw": 2, "bravo": 1.8, "merveilleux": 2.5, "merveilleuse": 2.5,
    "emouvant": 2, "emouvante": 2, "bouleversant": 2.3, "bouleversante": 2.3, "poignant": 2,
    "poignante": 2, "prenant": 2, "prenante": 2, "haletant": 2, "haletante": 2,
    "palpitant": 2, "palpitante": 2, "passionnant": 2.2, "passionnante": 2.2,
    "hilarant": 2, "hilarante": 2, "drole": 1.5, "touchant": 1.8, "touchante": 1.8,
    "superbe": 2.3, "beau": 1.3, "belle": 1.3, "super": 2, "terrible": 1, "efficace": 1.3,
    "original": 1.2, "originale": 1.2, "intelligent": 1.5, "intelligente": 1.5,
    "recommande": 2, "recommandee": 2, "conseille": 1.8, "conseillee": 1.8,
    "aime": 1.8, "aimee": 1.8, "bien": 1, "bon": 1.2, "bonne": 1.2, "sympa": 1.2,
    "cool": 1.2, "chouette": 1.3, "agreable": 1.2, "interessant": 1.3, "interessante": 1.3,
    "divertissant": 1.3, "divertissante": 1.3, "reussi": 1.8, "reussie": 1.8,
    "pas mal": 1, "correct": 0.4, "correcte": 0.4, "sympathique": 1.2, "plait": 1.3,
    "plu": 1.5, "valide": 1.5, "approuve": 1.5, "frissons": 1.5, "pleure": 1.2, "pleurer": 1,
    # Négatifs
    "navet": -3, "nul": -2.5, "nulle": -2.5, "nullissime": -3, "horrible": -2.8,
    "catastrophique": -3, "catastrophe": -2.5, "affreux": -2.5, "affreuse": -2.5, "pire": -2.5,
    "mauvais": -2, "mauvaise": -2, "decevant": -2, "decevante": -2, "decu": -2, "decue": -2,
    "ennuyeux": -2, "ennuyeuse": -2, "ennui": -1.8, "ennuye": -1.8, "chiant": -2.3,
    "chiante": -2.3, "soporifique": -2.5, "lent": -1, "lente": -1, "longuet": -1.2,
    "longuette": -1.2, "long": -0.6, "longue": -0.6, "longueurs": -1.2, "bof": -1.5,
    "moyen": -0.5, "moyenne": -0.5, "banal": -1.2, "banale": -1.2, "plat": -1.2, "plate": -1.2,
    "fade": -1.5, "rate": -1.8, "ratee": -1.8, "bacle": -1.8, "baclee": -1.8, "decevoir": -1.5,
    "ridicule": -2.2, "absurde": -1, "incoherent": -1.5, "incoherente": -1.5,
    "previsible": -1, "cliche": -1, "cliches": -1, "surcote": -1.5, "surcotee": -1.5,
    "survendu": -1.5, "survendue": -1.5, "insupportable": -2.3, "deteste": -2.5,
    "abandonne": -1.8, "lache": -1.5, "gache": -1.5, "gachee": -1.5, "gachis": -2,
    "inutile": -1.5, "vide": -1.2, "creux": -1.2, "lourd": -1, "lourde": -1,
    "mediocre": -2.2, "pathetique": -2.5, "nase": -2, "naze": -2, "pourri": -2.5,
    "pourrie": -2.5, "dommage": -0.8, "deception": -2, "flop": -2, "zero": -2,
}

NEGATIONS = {"pas", "jamais", "aucun", "aucune", "ni", "rien", "guere", "point", "sans"}
INTENSIFS = {
    "tres": 1.5, "vraiment": 1.4, "trop": 1.4, "tellement": 1.5, "hyper": 1.5, "super": 1.4,
    "grave": 1.3, "carrement": 1.4, "extremement": 1.7, "completement": 1.4,
    "totalement": 1.4, "absolument": 1.5, "franchement": 1.2, "vachement": 1.4,
    "archi": 1.6, "ultra": 1.6, "si": 1.3, "assez": 0.7, "plutot": 0.8, "un_peu": 0.6,
}

EMOJIS: dict[str, float] = {
    "😍": 2.5, "🥰": 2.3, "🤩": 2.5, "🔥": 2, "👍": 1.5, "👌": 1.5, "❤": 2, "♥": 2,
    "💯": 2.5, "👏": 1.8, "🙌": 1.8, "😭": 0.5, "🥹": 1.5, "💖": 2, "💕": 2, "😁": 1,
    "😊": 1, "🤣": 0.8, "😂": 0.6, "✨": 1, "🏆": 2,
    "👎": -2.5, "😴": -2, "🥱": -2, "🤮": -3, "🤢": -2.5, "😡": -2, "💩": -3, "😒": -1.5,
    "🙄": -1.5, "😞": -1.5, "😕": -1, "🫤": -1, "😐": -0.5, "🤦": -1.5, "😤": -1.2,
}

# Plus ECHELLE est grand, plus il faut de signaux pour approcher 0 ou 10.
ECHELLE = 4.0

_RE_NOTE = re.compile(r"(?<![\d/])(\d{1,3}(?:[.,]\d+)?)\s*/\s*(5|10|20|100)(?!\d)")
_RE_NOTE_SUR = re.compile(r"(?<!\d)(\d{1,2}(?:[.,]\d+)?)\s*sur\s*(5|10|20)(?!\d)")
_RE_ETOILES = re.compile(r"[★⭐🌟]")
_RE_MOT = re.compile(r"[a-z0-9_]+(?:['\-][a-z0-9_]+)*")

# Expressions triées de la plus longue à la plus courte pour éviter les recouvrements.
_EXPRESSIONS_NORM = sorted(
    ((normaliser(k), v) for k, v in EXPRESSIONS.items()), key=lambda kv: -len(kv[0])
)


def note_explicite(texte: str) -> float | None:
    """Renvoie une note sur 10 si l'avis en contient une (« 8/10 », « ★★★★ »)."""
    t = normaliser(texte)
    for regex in (_RE_NOTE, _RE_NOTE_SUR):
        m = regex.search(t)
        if m:
            valeur = float(m.group(1).replace(",", "."))
            echelle = float(m.group(2))
            if 0 <= valeur <= echelle:
                return round(valeur / echelle * 10, 1)
    etoiles = len(_RE_ETOILES.findall(texte))
    if 1 <= etoiles <= 5:
        return float(etoiles * 2)
    return None


def score_brut(texte: str) -> tuple[float, int]:
    """Somme pondérée des signaux et nombre de signaux trouvés.

    Après « mais » / « par contre » / « sauf que », c'est la fin de la phrase
    qui compte le plus : la partie qui précède est pondérée à moitié.
    """
    t = normaliser(texte)
    m = re.search(r"\b(mais|par contre|sauf que|en revanche|cependant|neanmoins)\b", t)
    if m and m.start() > 0:
        avant, signaux_avant = _score_segment(t[:m.start()])
        apres, signaux_apres = _score_segment(t[m.end():])
        total, signaux = 0.5 * avant + apres, signaux_avant + signaux_apres
    else:
        total, signaux = _score_segment(t)

    # Émojis et ponctuation expressive (sur le texte d'origine)
    for emoji, poids in EMOJIS.items():
        n = texte.count(emoji)
        if n:
            total += poids * min(n, 3) ** 0.5
            signaux += n
    if signaux and "!!" in texte:
        total *= 1.15
    return total, signaux


def _score_segment(t: str) -> tuple[float, int]:
    t = " " + t.replace("un peu", "un_peu") + " "
    total, signaux = 0.0, 0

    # 1. Expressions
    for expr, poids in _EXPRESSIONS_NORM:
        motif = re.compile(r"(?<![a-z])" + re.escape(expr) + r"(?![a-z])")
        for m in motif.finditer(t):
            avant = t[max(0, m.start() - 25):m.start()].split()[-2:]
            facteur = next((INTENSIFS[a] for a in reversed(avant) if a in INTENSIFS), 1.0)
            total += poids * facteur
            signaux += 1
        t = motif.sub(" _ ", t)

    # 2. Mots isolés, avec négation et intensifs sur les 3 mots précédents
    mots = _RE_MOT.findall(t)
    for i, mot in enumerate(mots):
        poids = MOTS.get(mot)
        if poids is None and mot.startswith(("j'", "l'", "d'", "c'", "n'", "qu'")):
            poids = MOTS.get(mot.split("'", 1)[1])
        if poids is None:
            continue
        if mot in INTENSIFS and i + 1 < len(mots) and (mots[i + 1] in MOTS):
            continue  # « super bien » : « super » sert d'intensif
        fenetre = mots[max(0, i - 3):i]
        facteur = 1.0
        for avant in fenetre:
            if avant in INTENSIFS:
                facteur *= INTENSIFS[avant]
        if any(a in NEGATIONS for a in fenetre):
            facteur *= -0.7 if poids > 0 else -0.4
        total += poids * facteur
        signaux += 1
    return total, signaux


def noter(texte: str) -> float | None:
    """Note de 0 à 10, ou None si l'avis n'exprime aucune opinion."""
    explicite = note_explicite(texte)
    if explicite is not None:
        return explicite
    total, signaux = score_brut(texte)
    if signaux == 0:
        return None
    return round(5 + 5 * math.tanh(total / ECHELLE), 1)
