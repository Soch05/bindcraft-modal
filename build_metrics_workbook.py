#!/usr/bin/env python3
"""Classeur consolidé : toutes les métriques, le re-scoring et PROPKA en un seul fichier.

    uv run --with openpyxl --with gemmi --with biopython --with scipy \
        python build_metrics_workbook.py

CE QUE CE FICHIER EST. Un condensé destiné au dépôt public, lisible sans le reste du dépôt :
chaque nombre y est accompagné de sa source et de son statut (mesuré, posé sans calibration,
ou non mesuré). Il se distingue de `submission/egfr_analysis.xlsx`, qui est une copie de
travail et n'a pas vocation à être lue par un tiers.

CE QU'IL N'EST PAS. Il ne contient aucune instruction adressée à un lecteur : les soumissions
du Track 3 passent devant un modèle sélecteur, et toute instruction embarquée peut valoir
disqualification. Le classeur décrit, il ne s'adresse à personne.

LES SEUILS SONT IMPORTÉS DEPUIS LES MODULES, jamais recopiés. Une valeur changée dans
`rank_designs.py` ou `verify_geometry.py` se propage ici automatiquement ; une divergence
entre le code et la documentation est donc impossible par construction.

Sortie : docs/egfr_metrics_consolidated.xlsx
"""

from __future__ import annotations

import csv
import importlib
from pathlib import Path

OUT = Path("docs/egfr_metrics_consolidated.xlsx")

RANKED = {
    "egfr-dIII-prod01": Path("out/egfr-dIII-prod01/3_Ranked/!_Ranked.csv"),
    "egfr-dIII-prod02": Path("out/egfr-dIII-prod02/3_Ranked/!_Ranked.csv"),
}
SOURCES = {
    "master": Path("out/master_rank.csv"),
    "submission": Path("submission/egfr_challenge1_submission.csv"),
    "propka_h409": Path("out/propka_h409.csv"),
    "propka_acids": Path("out/propka_acids.csv"),
    "propka_scan": Path("out/propka_scan.csv"),
    "geometry_wt": Path("out/geometry_wt.csv"),
    "geometry_mut": Path("out/geometry_mutants.csv"),
    "boltz": Path("out/boltz_contacts.csv"),
    "boltz_mut": Path("out/boltz_contacts_mutants.csv"),
    "cross": Path("out/cross_species.csv"),
    "bidentate_af2": Path("out/bidentate.csv"),
    "bidentate_boltz": Path("out/bidentate_boltz.csv"),
    "esm2": Path("out/esm2_pll.csv"),
    "pairs": Path("out/paires_wt_mutant.csv"),
    "threaded": Path("structures/threaded/threaded_index.csv"),
}


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.is_file() else []


def constants() -> list[dict]:
    """Lit les seuils DANS les modules, avec leur statut de calibration."""
    catalogue = [
        ("rank_designs", "PH_ACID", "pH acide du reglement",
         "reglement du challenge", "impose"),
        ("rank_designs", "PH_NEUTRAL", "pH neutre du reglement",
         "reglement du challenge", "impose"),
        ("rank_designs", "SALT_BRIDGE_A",
         "distance maximale entre atomes charges pour un pont salin (A)",
         "Barlow & Thornton 1983", "adosse a une reference externe"),
        ("rank_designs", "PKA_NOISE",
         "plancher de bruit d'un decalage de pKa (unites de pKa)",
         "erreur typique de PROPKA", "convention, non mesuree sur ce systeme"),
        ("rank_designs", "CARBOXYLATE_SELF_DEFEAT_PKA",
         "au-dela, un carboxylate est partiellement protone des pH 6,5",
         "plan de travail", "pose"),
        ("rank_designs", "CONTACT_RECOVERY_CONFIRMED",
         "recuperation de contacts minimale pour une pose dite confirmee",
         "pose par l'auteur", "POSE, NON CALIBRE — non discriminant sur ce lot"),
        ("rank_designs", "ORTHOGONAL_IPTM_CONFIRMED",
         "iptm orthogonal minimal pour une pose dite confirmee",
         "pose par l'auteur", "POSE, NON CALIBRE — non discriminant sur ce lot"),
        ("verify_geometry", "HARD_CLASH_A",
         "deux atomes lourds non lies plus proches que ca sont en recouvrement (A)",
         "plan de travail", "pose"),
        ("verify_geometry", "SHORT_CONTACT_A",
         "en dessous, un contact O...N est plus serre qu'une liaison H forte (A)",
         "borne basse de la plage usuelle d'un pont salin", "pose"),
        ("verify_geometry", "ROTAMER_DISTINCT_A",
         "etendue minimale du centroide du carboxylate pour deux rotameres distincts (A)",
         "pose par l'auteur", "POSE, SANS REFERENCE"),
        ("verify_geometry", "TARGET_HIS",
         "residu cible du mecanisme pH, numerotation PDB",
         "lecture directe de inputs/6ARU_A_309-506.pdb", "verifie sur 23 structures"),
        ("contact_recovery", "CONTACT_CUTOFF_A",
         "deux residus sont en contact si deux atomes lourds sont a moins de ca (A)",
         "convention usuelle", "convention"),
        ("bidentate_rule", "SEARCH_A",
         "rayon de recherche des carboxylates autour de chaque azote (A)",
         "choisi large pour voir la distribution", "pose, non utilise comme seuil"),
        ("bidentate_rule", "PDB_OFFSET",
         "decalage de la numerotation Boltz (1..198) vers PDB (309..506)",
         "absence d'indel verifiee par mouse_target.py", "verifie"),
        ("build_submission", "QUOTA",
         "nombre maximal de designs, Track 3", "reglement §3", "impose"),
        ("build_submission", "LENGTH_MIN",
         "longueur minimale d'une sequence soumise (aa)", "reglement §3", "impose"),
        ("build_submission", "LENGTH_MAX",
         "longueur maximale d'une sequence soumise (aa)", "reglement §3", "impose"),
        ("mouse_target", "DOMAIN_START",
         "borne basse du domaine III, numerotation PDB",
         "CATH-Gene3D G3DSA:3.80.20.20", "adosse a une reference externe"),
        ("mouse_target", "DOMAIN_END",
         "borne haute du domaine III, numerotation PDB",
         "CATH-Gene3D G3DSA:3.80.20.20", "adosse a une reference externe"),
        ("mouse_target", "OPEN_GAP",
         "penalite d'ouverture de breche des alignements du depot",
         "convention du depot", "pose"),
        ("mouse_target", "EXTEND_GAP",
         "penalite d'extension de breche des alignements du depot",
         "convention du depot", "pose"),
        ("propka_scan", "WORKERS",
         "nombre de runs PROPKA en parallele", "cpu_count() - 1", "sans effet sur le resultat"),
    ]
    cache: dict[str, object] = {}
    out = []
    for module_name, name, meaning, source, status in catalogue:
        if module_name not in cache:
            try:
                cache[module_name] = importlib.import_module(module_name)
            except Exception as problem:  # noqa: BLE001
                cache[module_name] = problem
        module = cache[module_name]
        if isinstance(module, Exception):
            value = f"non lu ({type(module).__name__})"
        else:
            value = getattr(module, name, "absent du module")
        out.append({
            "constante": name,
            "valeur": value if not isinstance(value, tuple) else ", ".join(map(str, value)),
            "module": f"{module_name}.py",
            "signification": meaning,
            "origine": source,
            "statut": status,
        })
    return out


# ---------------------------------------------------------------------------------------
# Dictionnaire des colonnes — ce qui rend le fichier auto-suffisant
# ---------------------------------------------------------------------------------------

DICTIONARY = [
    # --- identite ---
    ("design_id", "identite", "identifiant unique. Pour un natif ou un frere : campagne, "
     "modalite, longueur, hash de squelette, index de sequence. Pour un mutant, le suffixe "
     "__<mutation> est ajoute, sans quoi la colonne ne serait pas une cle unique et une "
     "jointure ramenerait le parent avec ses mutants", "BindCraft 2.0 + derive", "mesure"),
    ("parent", "identite", "pour un mutant, l'identifiant du design dont il reprend le "
     "squelette. Vide pour un natif ou un frere", "derive", "mesure"),
    ("squelette", "identite", "hash de la trajectoire BindCraft. Deux designs de meme hash "
     "partagent le squelette et ne sont pas des poses independantes", "BindCraft 2.0", "mesure"),
    ("type", "identite", "natif = seq0 d'un squelette ; frere = seq1, meme squelette, autre "
     "sequence ProteinMPNN ; mutant = substitution greffee par threading", "derive", "mesure"),
    ("run", "identite", "campagne BindCraft d'origine (prod01 ou prod02)", "BindCraft 2.0",
     "mesure"),
    ("sequence", "identite", "sequence du binder en acides amines", "BindCraft 2.0", "mesure"),
    ("longueur", "identite", "nombre de residus du binder", "derive", "mesure"),

    # --- objectif 1 : pH ---
    ("palier_pH", "objectif 1 — pH", "quatre paliers : mecanisme robuste, mecanisme non "
     "reproductible, neutre, contre-selectif", "derive de trois mesures de pKa", "mesure"),
    ("mecanisme_robustesse", "objectif 1 — pH", "robuste = dpKa positif sur les TROIS mesures "
     "(AF2 humain, Boltz humain, Boltz souris) ; non reproductible = positif sur au moins une "
     "mais pas toutes", "derive", "mesure"),
    ("mesures_pH_positives", "objectif 1 — pH", "nombre de mesures de dpKa positives sur le "
     "nombre disponible", "derive", "mesure"),
    ("dpKa_H409", "objectif 1 — pH", "decalage du pKa de H409 entre cible liee et cible "
     "seule, calcule sur la structure AF2/BindCraft. Positif = la liaison stabilise la forme "
     "protonee, donc elle est favorisee a pH acide", "PROPKA sur structure AF2", "mesure"),
    ("dpKa_humain_Boltz", "objectif 1 — pH", "le meme decalage, calcule sur la structure "
     "predite par Boltz-2. Sert a savoir si le mecanisme est une propriete de la sequence ou "
     "de la structure", "PROPKA sur structure Boltz-2", "mesure"),
    ("dpKa_souris_Boltz", "objectif 1 — pH", "le meme decalage sur le complexe avec le "
     "domaine III murin", "PROPKA sur structure Boltz-2", "mesure"),
    ("facteur_pH_predit", "objectif 1 — pH", "rapport d'affinite predit entre pH 6,5 et "
     "pH 7,4, du seul fait du decalage de pKa de H409. 1,0 = aucune selectivite. Sert a "
     "CLASSER, pas a annoncer une valeur", "derive du dpKa", "mesure, non calibre"),
    ("facteur_global_tous_groupes", "objectif 1 — pH", "produit, sur tous les groupes "
     "ionisables du complexe, de leur contribution individuelle a la selectivite pH. Version "
     "non biaisee du facteur, qui ne privilegie aucun residu choisi d'avance",
     "derive de PROPKA", "mesure ; suppose les sites independants"),
    ("facteur_global_groupes_mobiles", "objectif 1 — pH", "le meme produit restreint aux "
     "groupes dont le pKa bouge de plus que le bruit. Moins sensible au cumul d'erreurs",
     "derive de PROPKA", "mesure"),
    ("pont_residu", "objectif 1 — pH", "residu acide du binder le plus proche d'un azote de "
     "l'imidazole de H409", "geometrie sur structure AF2", "mesure"),
    ("pont_distance_A", "objectif 1 — pH", "distance de ce carboxylate a l'azote le plus "
     "proche (A)", "geometrie sur structure AF2", "mesure"),
    ("pont_angle_deg", "objectif 1 — pH", "angle a l'oxygene accepteur. Un carboxylate sp2 "
     "pointe ses doublets vers ~120 degres ; 80 ou 160 degres sont des geometries mediocres",
     "geometrie sur structure AF2", "mesure"),
    ("bidente_goulot_A", "objectif 1 — pH", "plus mauvaise des deux distances d'une paire de "
     "DEUX carboxylates distincts, l'un sur ND1 et l'autre sur NE2. Goulot du mecanisme "
     "bidente", "geometrie", "mesure"),
    ("bidente_paire", "objectif 1 — pH", "les deux residus de cette paire, et leurs atomes",
     "geometrie", "mesure"),

    # --- objectif 2 : souris ---
    ("epitope_souris_retrouve", "objectif 2 — souris", "fraction des paires de residus en "
     "contact de la pose humaine qui sont retrouvees dans la pose predite contre le domaine "
     "III murin. MESURE de cross-reactivite", "Boltz-2 + geometrie", "mesure"),
    ("iptm_souris", "objectif 2 — souris", "iptm Boltz-2 moyen du complexe avec la cible "
     "murine", "Boltz-2", "mesure"),
    ("delta_iptm_souris", "objectif 2 — souris", "iptm souris moins iptm humain. Proche de "
     "zero = la liaison ne souffre pas du changement d'espece", "Boltz-2", "mesure"),
    ("mecanisme_conserve_souris", "objectif 2 — souris", "le mecanisme pH observe chez "
     "l'humain subsiste-t-il chez la souris", "derive", "mesure"),
    ("epitope_conservation_frac", "objectif 2 — souris", "PROXY de sequence : fraction des "
     "residus de cible contactes qui sont identiques chez la souris. Insuffisant — il ne "
     "distingue pas les designs qui changent de mode de liaison", "alignement P00533/Q01279",
     "mesure, mais proxy demontre insuffisant"),
    ("epitope_residus_divergents", "objectif 2 — souris", "residus de cible contactes qui ne "
     "sont pas identiques chez la souris", "alignement P00533/Q01279", "mesure"),

    # --- objectif 3 : affinite, et qualite de pose ---
    ("i_pTM_AF2", "objectif 3 — affinite", "confiance d'interface AlphaFold2, echelle [0,1]. "
     "Score IN-SAMPLE : BindCraft a optimise les designs par descente de gradient a travers "
     "AF2", "BindCraft 2.0", "mesure, in-sample"),
    ("i_pAE_AF2", "objectif 3 — affinite", "erreur d'alignement predite a l'interface, "
     "echelle [0,1], plus bas est mieux. Egalement in-sample", "BindCraft 2.0",
     "mesure, in-sample"),
    ("Interface_BuriedArea", "objectif 3 — affinite", "surface enfouie a l'interface (A2)",
     "BindCraft 2.0", "mesure"),
    ("Hotspot_Contact_Fraction", "objectif 3 — affinite", "fraction des contacts qui touchent "
     "les hotspots demandes (A318, A323, A406, A409)", "BindCraft 2.0", "mesure"),
    ("Off_Epitope", "objectif 3 — affinite", "fraction des contacts hors de l'epitope visee",
     "BindCraft 2.0", "mesure"),
    ("recuperation_contacts", "validation orthogonale", "fraction des paires de residus en "
     "contact de la pose AF2 retrouvees dans la pose Boltz-2. Mesure si la pose survit au "
     "retrait du modele qui l'a produite. Asymetrique a dessein", "Boltz-2 + geometrie",
     "mesure"),
    ("confiance_modele_orthogonal", "validation orthogonale", "iptm Boltz-2 moyen sur les "
     "trois echantillons de diffusion", "Boltz-2", "mesure"),
    ("pose", "validation orthogonale", "confirmee si recuperation et iptm depassent les deux "
     "seuils. ATTENTION : ces seuils ne discriminent rien sur ce lot, les 23 designs natifs "
     "les passent", "derive", "POSE, NON CALIBRE"),

    # --- mutants ---
    ("mutations", "mutants", "substitution greffee, format <residu d'origine><position><nouveau>",
     "acid_mutants.py", "mesure"),
    ("verdict_carboxylate", "mutants", "VERT = pKa du D/E introduit sous 5,0 sur tous les "
     "rotameres retenus ; ROUGE = au-dessus sur tous ; INDETERMINE = les rotameres "
     "desaccordent", "PROPKA multi-rotamere", "mesure"),
    ("pKa_DE_min", "mutants", "pKa le plus bas du carboxylate introduit, sur les rotameres "
     "retenus", "PROPKA", "mesure"),
    ("pKa_DE_max", "mutants", "pKa le plus haut du meme carboxylate", "PROPKA", "mesure"),
    ("dpKa_etendue_rotameres", "mutants", "etendue du dpKa de H409 entre rotameres. Si elle "
     "depasse l'effet mesure, le verdict n'est pas concluant", "PROPKA", "mesure"),
    ("rotameres_retenus", "mutants", "rotameres gardes pour le verdict, et pourquoi",
     "threading PyMOL", "mesure"),
    ("ddpKa_vs_parent", "mutants", "dpKa du mutant moins dpKa de son parent. Squelette "
     "partage, donc une seule variable", "derive", "mesure"),

    # --- descriptif ---
    ("ESM2_PLL", "descriptif", "pseudo-vraisemblance ESM-2 par residu, marginales masquees. "
     "HORS CLASSEMENT par construction : elle mesure la ressemblance aux proteines "
     "naturelles, alors que le reglement exige la nouveaute. Classer dessus favoriserait les "
     "designs les moins nouveaux", "esm2_t33_650M_UR50D", "mesure, hors classement"),
    ("charge_nette", "descriptif", "charge nette du binder au pH de reference de BindCraft",
     "BindCraft 2.0", "mesure"),
    ("cysteines", "descriptif", "nombre de cysteines. Zero sur tout le lot : Adaptyv exprime "
     "en acellulaire et les cysteines libres sont une contrainte d'expression",
     "BindCraft 2.0", "mesure"),
]


def main() -> None:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    master = rows(SOURCES["master"])
    if not master:
        raise SystemExit(f"{SOURCES['master']} absent — lancer rank_designs.py d'abord")

    af2: dict[str, dict] = {}
    for run, path in RANKED.items():
        for record in rows(path):
            record["run_origine"] = run
            af2[record["design"]] = record

    submitted = {r["name"]: index for index, r in enumerate(rows(SOURCES["submission"]), 1)}

    book = Workbook()
    book.remove(book.active)
    green = PatternFill("solid", start_color="C6EFCE")
    red = PatternFill("solid", start_color="FFC7CE")
    orange = PatternFill("solid", start_color="FFEB9C")
    grey = PatternFill("solid", start_color="DDDDDD")

    def sheet(title: str, data: list[dict], columns: list[str] | None = None,
              widths: dict[str, int] | None = None):
        page = book.create_sheet(title[:31])
        if not data:
            page["A1"] = "aucune donnee disponible pour cet onglet"
            return page, []
        fields = columns or list(data[0])
        page.append(fields)
        for cell in page[1]:
            cell.font = Font(bold=True)
            cell.fill = grey
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for record in data:
            page.append([record.get(field, "") for field in fields])
        page.freeze_panes = "A2"
        page.auto_filter.ref = page.dimensions
        for index, field in enumerate(fields, start=1):
            letter = get_column_letter(index)
            if widths and field in widths:
                page.column_dimensions[letter].width = widths[field]
                continue
            widest = max([len(str(field))] +
                         [len(str(record.get(field, ""))) for record in data])
            page.column_dimensions[letter].width = min(42, max(9, widest + 2))
        return page, fields

    def paint(page, fields: list[str], field: str, kind: str) -> None:
        if field not in fields:
            return
        letter = get_column_letter(fields.index(field) + 1)
        span = f"{letter}2:{letter}{page.max_row}"
        if kind == "dpka":
            page.conditional_formatting.add(span, CellIsRule(
                operator="greaterThan", formula=["0.5"], fill=green))
            page.conditional_formatting.add(span, CellIsRule(
                operator="lessThan", formula=["-0.5"], fill=red))
        elif kind == "fraction":
            page.conditional_formatting.add(span, CellIsRule(
                operator="greaterThanOrEqual", formula=["0.8"], fill=green))
            page.conditional_formatting.add(span, CellIsRule(
                operator="lessThan", formula=["0.65"], fill=red))
        elif kind == "verdict":
            for value, fill in (("VERT", green), ("ROUGE", red), ("INDETERMINE", orange)):
                page.conditional_formatting.add(span, CellIsRule(
                    operator="equal", formula=[f'"{value}"'], fill=fill))
        elif kind == "robustesse":
            page.conditional_formatting.add(span, CellIsRule(
                operator="equal", formula=['"robuste"'], fill=green))
            page.conditional_formatting.add(span, CellIsRule(
                operator="equal", formula=['"non reproductible"'], fill=orange))
            page.conditional_formatting.add(span, CellIsRule(
                operator="equal", formula=['"contre-selectif"'], fill=red))

    # ---------------------------------------------------------------- 1. Lecture
    page = book.create_sheet("Lecture")
    lines = [
        ("Condense des metriques, du re-scoring orthogonal et des calculs de pKa", True),
        ("Challenge 1 Adaptyv x Anthropic — binder EGFR conditionnel, Track 3", False),
        ("", False),
        ("Ce classeur rassemble en un fichier les mesures qui fondent la soumission. Il est "
         "distinct de submission/egfr_analysis.xlsx, qui est une copie de travail.", False),
        ("", False),
        ("TROIS STATUTS, indiques colonne par colonne dans l'onglet Dictionnaire :", True),
        ("  mesure                 — valeur issue d'un calcul sur une structure ou une sequence", False),
        ("  pose, non calibre      — seuil choisi sans jeu de reference permettant de le calibrer", False),
        ("  hors classement        — valeur rapportee, n'entrant dans aucun ordre", False),
        ("", False),
        ("CE QUE LES NOMBRES NE DISENT PAS", True),
        ("  Tout repose sur des structures PREDITES et des modeles empiriques. Rien n'est", False),
        ("  experimental. PROPKA se trompe couramment d'une unite de pKa, davantage sur les", False),
        ("  gros decalages : ses valeurs servent a classer, pas a annoncer un facteur.", False),
        ("", False),
        ("  Boltz-2 et AlphaFold2 sont independants par l'architecture et par les poids, mais", False),
        ("  tous deux entraines sur la PDB. Leur accord ecarte un artefact propre a AF2, pas", False),
        ("  un biais partage herite des donnees.", False),
        ("", False),
        ("  Le reglement demande une absence de liaison detectable a pH 7,4. Ce qui est mesure", False),
        ("  ici est un decalage de pKa sur un seul residu, qui predit un rapport d'affinite", False),
        ("  d'un facteur de l'ordre de quelques unites. L'ecart entre les deux n'est pas comble.", False),
        ("", False),
        ("LE POINT LE PLUS IMPORTANT DU LOT", True),
        ("  Un seul design sur treize porte un mecanisme pH qui resiste a la verification sur", False),
        ("  deux structures et deux especes. Les douze autres sont des poses independantes sur", False),
        ("  un epitope conserve. L'ordre du CSV de soumission transmet cette information.", False),
        ("", False),
        ("ONGLETS", True),
        ("  Synthese            les 13 designs soumis, colonnes decisives, dans l'ordre soumis", False),
        ("  Vivier_complet      les 35 candidats, toutes les colonnes jointes", False),
        ("  Metriques_AF2       les 42 colonnes brutes de BindCraft 2.0", False),
        ("  pH_PROPKA           pKa de H409 lie et libre, dpKa, facteurs de selectivite", False),
        ("  pH_scan_non_biaise  tout groupe ionisable dont le pKa bouge de plus que le bruit", False),
        ("  Rescoring_Boltz     recuperation de contacts et confiance, cible humaine", False),
        ("  Cross_especes       cible murine : pose, confiance, mecanisme pH", False),
        ("  Geometrie_H409      ponts salins, angles, enfouissement, goulot bidente", False),
        ("  Mutants             threading, rotameres, pKa des D/E, verdicts", False),
        ("  Paires_WT_mutant    analyse appariee, cout structural", False),
        ("  ESM2_descriptif     pseudo-vraisemblance, hors classement", False),
        ("  Seuils              chaque constante, sa valeur lue dans le code, son statut", False),
        ("  Dictionnaire        chaque colonne : signification, source, statut", False),
    ]
    for index, (text, bold) in enumerate(lines, start=1):
        cell = page[f"A{index}"]
        cell.value = text
        if bold:
            cell.font = Font(bold=True)
    page.column_dimensions["A"].width = 100

    # ---------------------------------------------------------------- 2. Synthese
    synthesis = []
    for record in master:
        if record["type"] == "mutant":
            continue
        rank = submitted.get(record["design_id"])
        if rank is None:
            continue
        synthesis.append({"rang_soumis": rank, **record})
    synthesis.sort(key=lambda r: r["rang_soumis"])
    columns = [
        "rang_soumis", "design_id", "squelette", "type", "longueur",
        "palier_pH", "mecanisme_robustesse", "mesures_pH_positives",
        "dpKa_H409", "dpKa_humain_Boltz", "dpKa_souris_Boltz", "facteur_pH_predit",
        "pont_residu", "pont_distance_A", "pont_angle_deg", "bidente_goulot_A",
        "epitope_souris_retrouve", "delta_iptm_souris", "mecanisme_conserve_souris",
        "epitope_conservation_frac",
        "recuperation_contacts", "confiance_modele_orthogonal",
        "i_pTM_AF2", "i_pAE_AF2", "Interface_BuriedArea", "Hotspot_Contact_Fraction",
        "Off_Epitope", "charge_nette", "cysteines", "ESM2_PLL", "sequence",
    ]
    page, fields = sheet("Synthese", synthesis, columns, {"sequence": 30})
    for field in ("dpKa_H409", "dpKa_humain_Boltz", "dpKa_souris_Boltz"):
        paint(page, fields, field, "dpka")
    for field in ("recuperation_contacts", "epitope_souris_retrouve"):
        paint(page, fields, field, "fraction")
    paint(page, fields, "mecanisme_robustesse", "robustesse")

    # ---------------------------------------------------------------- 3. Vivier complet
    page, fields = sheet("Vivier_complet", master, list(master[0]), {"sequence": 30})
    for field in ("dpKa_H409", "dpKa_humain_Boltz", "dpKa_souris_Boltz", "ddpKa_vs_parent"):
        paint(page, fields, field, "dpka")
    paint(page, fields, "verdict_carboxylate", "verdict")
    paint(page, fields, "mecanisme_robustesse", "robustesse")
    for field in ("recuperation_contacts", "epitope_souris_retrouve"):
        paint(page, fields, field, "fraction")

    # ---------------------------------------------------------------- 4. Metriques AF2
    raw = []
    for design, record in af2.items():
        raw.append({"design": design, **{k: v for k, v in record.items() if k != "design"}})
    sheet("Metriques_AF2", raw, None, {"Binder_Sequence": 30,
                                       "Interface_Target_Residues": 34,
                                       "Interface_Binder_Residues": 34})

    # ---------------------------------------------------------------- 5-11. mesures brutes
    page, fields = sheet("pH_PROPKA", rows(SOURCES["propka_h409"]))
    paint(page, fields, "dpKa_H409", "dpka")
    page, fields = sheet("pH_scan_non_biaise", rows(SOURCES["propka_scan"]))
    paint(page, fields, "dpKa", "dpka")
    page, fields = sheet("Rescoring_Boltz", rows(SOURCES["boltz"]), None, {"detail": 38})
    for field in ("recuperation_contacts", "recuperation_min", "recuperation_moyenne"):
        paint(page, fields, field, "fraction")
    page, fields = sheet("Cross_especes", rows(SOURCES["cross"]))
    for field in ("dpKa_humain_AF2", "dpKa_humain_Boltz", "dpKa_souris_Boltz"):
        paint(page, fields, field, "dpka")
    paint(page, fields, "recuperation_epitope_souris", "fraction")

    geometry = [{**r, "source_geometrie": "AF2"} for r in rows(SOURCES["geometry_wt"])]
    bidentate = {r["design"]: r for r in rows(SOURCES["bidentate_af2"])}
    bidentate_boltz = {r["design"]: r for r in rows(SOURCES["bidentate_boltz"])}
    for record in geometry:
        first = bidentate.get(record["design"], {})
        second = bidentate_boltz.get(record["design"], {})
        record["bidente_goulot_AF2_A"] = first.get("bidente_goulot_A", "")
        record["bidente_paire_AF2"] = (
            f"{first.get('bidente_ND1', '')}/{first.get('bidente_NE2', '')}"
            if first.get("bidente_goulot_A") else "aucune paire possible"
        )
        record["bidente_goulot_Boltz_A"] = second.get("bidente_goulot_A", "")
        record["bidente_paire_Boltz"] = (
            f"{second.get('bidente_ND1', '')}/{second.get('bidente_NE2', '')}"
            if second.get("bidente_goulot_A") else "aucune paire possible"
        )
    sheet("Geometrie_H409", geometry)

    mutants = []
    threaded = {(r["design"], r["mutation"], r["rotamer"]): r
                for r in rows(SOURCES["threaded"])}
    for record in rows(SOURCES["propka_acids"]):
        key = (record["design"], record["mutation"], str(record["rotamer"]))
        extra = threaded.get(key, {})
        mutants.append({**record,
                        "contact_min_A": extra.get("contact_min_A", ""),
                        "partenaire_contact": extra.get("partenaire", "")})
    page, fields = sheet("Mutants", mutants)
    paint(page, fields, "auto_destructeur", "verdict")

    pairs = rows(SOURCES["pairs"])
    boltz_mut = {(r["parent"], r["mutation"]): r for r in rows(SOURCES["boltz_mut"])}
    for record in pairs:
        extra = boltz_mut.get((record["parent"], record["mutation"]), {})
        record["iptm_mutant_Boltz"] = extra.get("iptm_moyen", "")
    page, fields = sheet("Paires_WT_mutant", pairs)
    paint(page, fields, "ddpKa", "dpka")
    paint(page, fields, "verdict_carboxylate", "verdict")

    esm2 = rows(SOURCES["esm2"])
    for record in esm2:
        record["statut"] = "descriptif — hors de tout critere de classement"
    sheet("ESM2_descriptif", esm2, None, {"statut": 46})

    # ---------------------------------------------------------------- 12. Seuils
    page, fields = sheet("Seuils", constants(), None,
                         {"signification": 56, "origine": 34, "statut": 40})
    letter = get_column_letter(fields.index("statut") + 1)
    span = f"{letter}2:{letter}{page.max_row}"
    page.conditional_formatting.add(span, CellIsRule(
        operator="containsText", formula=[f'NOT(ISERROR(SEARCH("NON CALIBRE",{letter}2)))'],
        fill=red))

    # ---------------------------------------------------------------- 13. Dictionnaire
    sheet("Dictionnaire",
          [{"colonne": name, "axe": axis, "signification": meaning,
            "source": source, "statut": status}
           for name, axis, meaning, source, status in DICTIONARY],
          None, {"signification": 72, "source": 30, "statut": 32})

    OUT.parent.mkdir(parents=True, exist_ok=True)
    book.save(OUT)
    print(f"-> {OUT}  ({OUT.stat().st_size / 1024:.1f} Ko)")
    for name in book.sheetnames:
        print(f"   {name:<22} {book[name].max_row - 1:4d} lignes x "
              f"{book[name].max_column} colonnes")


if __name__ == "__main__":
    main()
