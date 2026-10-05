#!/usr/bin/env python3
"""Construit le CSV de soumission, le classeur de travail, et passe tous les contrôles.

    uv run --with openpyxl --with biopython --with pandas python build_submission.py

ALLOCATION : COUVERTURE MAXIMALE. Un seul design par squelette, le meilleur de chacun. Le
règlement ne tranche pas comment l'unicité est évaluée entre les designs d'un même
participant (voir docs/SUBMISSION_REPORT.md phase 0f) ; soumettre un WT et son mutant
ponctuel, à 98–99 % d'identité, serait donc un risque de rejet des deux plutôt qu'une
stratégie. Un design par squelette élimine ce risque et maximise le nombre de poses
indépendantes, qui est la ressource rare.

LE QUOTA EST UN PLAFOND. 20 designs maximum ne sont pas un objectif à atteindre, et le lot
n'est jamais complété par des frères pour arriver à un chiffre rond.

Sorties : submission/egfr_challenge1_submission.csv, submission/egfr_analysis.xlsx
"""

from __future__ import annotations

import csv
import re
import warnings
from pathlib import Path

warnings.filterwarnings("ignore")

MASTER = Path("out/master_rank.csv")
PAIRS = Path("out/paires_wt_mutant.csv")
PROPKA_SCAN = Path("out/propka_scan.csv")
PROPKA_ACIDS = Path("out/propka_acids.csv")
BOLTZ_CONTACTS = Path("out/boltz_contacts.csv")
GEOMETRY_WT = Path("out/geometry_wt.csv")
GEOMETRY_MUT = Path("out/geometry_mutants.csv")
STRUCTURES = Path("structures/wt")

OUT_DIR = Path("submission")
OUT_CSV = OUT_DIR / "egfr_challenge1_submission.csv"
OUT_XLSX = OUT_DIR / "egfr_analysis.xlsx"

# Règlement §3 : 20 max en Track 3, et le CSV est ordonné par classement.
QUOTA = 20
LENGTH_MIN, LENGTH_MAX = 10, 250
MOLECULE_CLASS = "protein"
STANDARD_AA = set("ACDEFGHIKLMNPQRSTVWY")

# Un mutant n'est éligible que si le threading a CONCLU en sa faveur : mécanisme robuste et
# carboxylate sain. Un mutant à mécanisme absent ou indéterminé n'apporte rien sur
# l'objectif n°1 et perd au passage le filtrage BindCraft et la validation orthogonale que
# son parent possède. Règle posée ici, appliquée dans `select`.
MUTANT_REQUIRES = {"mecanisme_robustesse": "robuste", "verdict_carboxylate": "VERT"}


def rows(path: Path) -> list[dict]:
    return list(csv.DictReader(path.open(newline=""))) if path.is_file() else []


def number(value, default: float = float("nan")) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


# ---------------------------------------------------------------------------------------
# Sélection
# ---------------------------------------------------------------------------------------


def select(candidates: list[dict]) -> tuple[list[dict], list[dict]]:
    """Un design par squelette, dans l'ordre du classement global déjà établi."""
    eligible, rejected = [], []
    for entry in candidates:
        if entry["type"] == "mutant":
            failures = [
                f"{key}={entry.get(key)!r} attendu {want!r}"
                for key, want in MUTANT_REQUIRES.items() if entry.get(key) != want
            ]
            if failures:
                rejected.append({**entry, "motif": "mutant non eligible : "
                                                   + " ; ".join(failures)})
                continue
        eligible.append(entry)

    chosen, seen = [], set()
    for entry in eligible:
        if entry["squelette"] in seen:
            rejected.append({**entry, "motif": "squelette deja represente par un design "
                                               "mieux classe"})
            continue
        seen.add(entry["squelette"])
        chosen.append(entry)
    return (chosen[:QUOTA], rejected)


def paired_allocation(candidates: list[dict]) -> dict:
    """Chiffre l'allocation « paires appariées » SANS la retenir, pour comparaison."""
    by_skeleton: dict[str, list[dict]] = {}
    for entry in candidates:
        by_skeleton.setdefault(entry["squelette"], []).append(entry)
    pairs = 0
    for members in by_skeleton.values():
        has_native = any(m["type"] != "mutant" for m in members)
        has_mutant = any(m["type"] == "mutant" for m in members)
        if has_native and has_mutant:
            pairs += 1
    return {
        "squelettes_avec_paire_possible": pairs,
        "designs_si_paires": min(QUOTA, 2 * pairs),
        "squelettes_couverts_si_paires": min(pairs, QUOTA // 2),
    }


# ---------------------------------------------------------------------------------------
# Contrôles
# ---------------------------------------------------------------------------------------


def identity(first: str, second: str) -> float:
    """Identité de séquence par alignement global, paramètres du dépôt (−11/−1)."""
    from Bio import Align

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.open_gap_score = -11
    aligner.extend_gap_score = -1
    aligner.match_score = 1
    aligner.mismatch_score = 0
    alignment = aligner.align(first, second)[0]
    matches = sum(
        1 for a, b in zip(alignment[0], alignment[1]) if a == b and a != "-"
    )
    return matches / min(len(first), len(second))


def binder_sequence_from_structure(design: str) -> str | None:
    """Relit la séquence du binder DANS la structure, pour confirmer la correspondance."""
    path = STRUCTURES / f"{design}.pdb"
    if not path.is_file():
        return None
    three_to_one = {
        "ALA": "A", "ARG": "R", "ASN": "N", "ASP": "D", "CYS": "C", "GLN": "Q",
        "GLU": "E", "GLY": "G", "HIS": "H", "ILE": "I", "LEU": "L", "LYS": "K",
        "MET": "M", "PHE": "F", "PRO": "P", "SER": "S", "THR": "T", "TRP": "W",
        "TYR": "Y", "VAL": "V",
    }
    seen, sequence = set(), []
    for line in path.read_text().splitlines():
        if not line.startswith("ATOM") or line[21] == "A":
            continue
        key = (line[21], int(line[22:26]))
        if key in seen:
            continue
        seen.add(key)
        sequence.append(three_to_one.get(line[17:20].strip(), "X"))
    return "".join(sequence) or None


def checks(chosen: list[dict]) -> list[dict]:
    results = []

    def record(name: str, ok: bool, detail: str) -> None:
        results.append({"controle": name, "resultat": "OK" if ok else "ECHEC",
                        "detail": detail})

    record("nombre de lignes <= quota", len(chosen) <= QUOTA,
           f"{len(chosen)} lignes pour un plafond de {QUOTA} — non force a l'egalite")

    sequences = [e["sequence"] for e in chosen]
    duplicates = len(sequences) - len(set(sequences))
    record("aucun doublon de sequence", duplicates == 0,
           f"{duplicates} doublon(s)")

    worst, worst_pair = 0.0, ("", "")
    for i in range(len(sequences)):
        for j in range(i + 1, len(sequences)):
            value = identity(sequences[i], sequences[j])
            if value > worst:
                worst, worst_pair = value, (chosen[i]["design_id"], chosen[j]["design_id"])
    record("identite de sequence maximale entre deux lignes", True,
           f"{worst:.1%} entre {worst_pair[0][-24:]} et {worst_pair[1][-24:]}")

    out_of_range = [
        e["design_id"] for e in chosen
        if not LENGTH_MIN <= len(e["sequence"]) <= LENGTH_MAX
    ]
    record("longueurs dans les bornes 10-250", not out_of_range,
           f"hors bornes : {out_of_range or 'aucun'} ; "
           f"observe {min(map(len, sequences))}-{max(map(len, sequences))} aa")

    with_cysteine = [e["design_id"] for e in chosen if "C" in e["sequence"]]
    record("aucune cysteine", not with_cysteine,
           f"{len(with_cysteine)} sequence(s) avec cysteine : "
           f"{with_cysteine or 'aucune'}")

    bad = []
    for entry in chosen:
        sequence = entry["sequence"]
        if set(sequence) - STANDARD_AA or re.search(r"\s", sequence):
            bad.append(entry["design_id"])
    record("aucun caractere non standard ni espace", not bad,
           f"{bad or 'aucun'}")

    mismatched = []
    for entry in chosen:
        from_structure = binder_sequence_from_structure(entry["design_id"])
        if from_structure is None:
            mismatched.append(f"{entry['design_id']}: structure absente")
        elif from_structure != entry["sequence"]:
            mismatched.append(f"{entry['design_id']}: structure != CSV")
    record("chaque sequence correspond a sa structure", not mismatched,
           f"{mismatched or 'les ' + str(len(chosen)) + ' concordent'}")

    return results


def reread(path: Path) -> dict:
    import pandas

    frame = pandas.read_csv(path, encoding="utf-8")
    return {
        "controle": "relecture pandas, encodage UTF-8",
        "resultat": "OK" if list(frame.columns) == ["name", "sequence", "molecule_class"]
        else "ECHEC",
        "detail": f"{len(frame)} lignes, colonnes {list(frame.columns)}",
    }


# ---------------------------------------------------------------------------------------
# Classeur
# ---------------------------------------------------------------------------------------


def workbook(chosen: list[dict], candidates: list[dict], control_rows: list[dict],
             allocation: dict, rejected: list[dict]) -> None:
    from openpyxl import Workbook
    from openpyxl.formatting.rule import CellIsRule
    from openpyxl.styles import Alignment, Font, PatternFill
    from openpyxl.utils import get_column_letter

    book = Workbook()
    book.remove(book.active)

    green = PatternFill("solid", start_color="C6EFCE")
    red = PatternFill("solid", start_color="FFC7CE")
    orange = PatternFill("solid", start_color="FFEB9C")
    header_fill = PatternFill("solid", start_color="DDDDDD")

    def sheet(title: str, data: list[dict], columns: list[str] | None = None):
        page = book.create_sheet(title[:31])
        if not data:
            page["A1"] = "aucune donnee"
            return page
        fields = columns or list(data[0])
        page.append(fields)
        for cell in page[1]:
            cell.font = Font(bold=True)
            cell.fill = header_fill
            cell.alignment = Alignment(vertical="top", wrap_text=True)
        for record in data:
            page.append([record.get(field, "") for field in fields])
        page.freeze_panes = "A2"
        page.auto_filter.ref = page.dimensions
        for index, field in enumerate(fields, start=1):
            widest = max(
                [len(str(field))] +
                [len(str(record.get(field, ""))) for record in data]
            )
            page.column_dimensions[get_column_letter(index)].width = min(48, max(9, widest + 2))
        return page

    def conditional(page, field: str, fields: list[str], kind: str) -> None:
        if field not in fields:
            return
        letter = get_column_letter(fields.index(field) + 1)
        span = f"{letter}2:{letter}{page.max_row}"
        if kind == "dpka":
            page.conditional_formatting.add(span, CellIsRule(
                operator="greaterThan", formula=["0.5"], fill=green))
            page.conditional_formatting.add(span, CellIsRule(
                operator="lessThan", formula=["-0.5"], fill=red))
        elif kind == "recuperation":
            page.conditional_formatting.add(span, CellIsRule(
                operator="greaterThanOrEqual", formula=["0.5"], fill=green))
            page.conditional_formatting.add(span, CellIsRule(
                operator="lessThan", formula=["0.5"], fill=red))
        elif kind == "verdict":
            for value, fill in (("VERT", green), ("ROUGE", red),
                                ("INDETERMINE", orange)):
                page.conditional_formatting.add(span, CellIsRule(
                    operator="equal", formula=[f'"{value}"'], fill=fill))

    # 1. Soumission
    submission_columns = [
        "rang_global", "design_id", "squelette", "type", "longueur", "charge_nette",
        "palier_pH", "mecanisme_robustesse", "mesures_pH_positives",
        "facteur_pH_predit", "dpKa_H409", "dpKa_humain_Boltz", "dpKa_souris_Boltz",
        "mecanisme_conserve_souris", "iptm_souris", "delta_iptm_souris",
        "epitope_souris_retrouve",
        "pont_residu", "pont_distance_A", "pont_angle_deg",
        "epitope_conservation_frac", "i_pTM_AF2", "i_pAE_AF2",
        "Interface_BuriedArea", "Hotspot_Contact_Fraction", "Off_Epitope",
        "pose", "recuperation_contacts", "confiance_modele_orthogonal",
        "bidente_goulot_A", "bidente_paire", "sequence",
    ]
    page = sheet("Soumission", chosen, submission_columns)
    conditional(page, "dpKa_H409", submission_columns, "dpka")
    conditional(page, "recuperation_contacts", submission_columns, "recuperation")

    # 2. Classement complet
    full_columns = list(candidates[0])
    page = sheet("Classement complet", candidates, full_columns)
    conditional(page, "dpKa_H409", full_columns, "dpka")
    conditional(page, "verdict_carboxylate", full_columns, "verdict")
    conditional(page, "recuperation_contacts", full_columns, "recuperation")

    # 3. Paires WT-mutant
    pair_rows = rows(PAIRS)
    if pair_rows:
        fields = list(pair_rows[0])
        page = sheet("Paires WT-mutant", pair_rows, fields)
        conditional(page, "ddpKa", fields, "dpka")
        conditional(page, "verdict_carboxylate", fields, "verdict")

    # 4. PROPKA detail
    acid_rows = rows(PROPKA_ACIDS)
    if acid_rows:
        fields = list(acid_rows[0])
        page = sheet("PROPKA pKa des DE", acid_rows, fields)
        letter = get_column_letter(fields.index("auto_destructeur") + 1)
        span = f"{letter}2:{letter}{page.max_row}"
        page.conditional_formatting.add(span, CellIsRule(
            operator="equal", formula=['"oui"'], fill=red))
        page.conditional_formatting.add(span, CellIsRule(
            operator="equal", formula=['"non"'], fill=green))
    scan_rows = rows(PROPKA_SCAN)
    if scan_rows:
        fields = list(scan_rows[0])
        page = sheet("PROPKA scan non biaise", scan_rows, fields)
        conditional(page, "dpKa", fields, "dpka")
    geometry = rows(GEOMETRY_WT)
    if geometry:
        sheet("Geometrie ponts WT", geometry, list(geometry[0]))
    mutant_geometry = rows(GEOMETRY_MUT)
    if mutant_geometry:
        sheet("Geometrie mutants", mutant_geometry, list(mutant_geometry[0]))

    # 5. Orthogonal detail
    contacts = rows(BOLTZ_CONTACTS)
    if contacts:
        fields = list(contacts[0])
        page = sheet("Orthogonal detail", contacts, fields)
        conditional(page, "recuperation_contacts", fields, "recuperation")
    else:
        page = book.create_sheet("Orthogonal detail")
        page["A1"] = "Re-scoring orthogonal : AUCUN RESULTAT"
        page["A1"].font = Font(bold=True)
        for index, line in enumerate([
            "",
            "Niveau A (Boltz-2 sur Modal) : build valide (torch 2.14.1+cu130, L40S,",
            "capability 8.9, CLI boltz ok, MSA 3725 sequences embarquee).",
            "",
            "Toutes les paires sont donc INCOMPLETES, et la colonne",
            "recuperation_contacts du classement porte NON MESUREE partout.",
            "",
            "Consequence sur le classement : aucun design ne peut atteindre le",
            "groupe 1 ni le groupe 3, qui exigent une pose confirmee. Les deux",
            "designs a mecanisme pH sont donc au groupe 2, le reste au groupe 4.",
        ], start=2):
            page[f"A{index}"] = line
        page.column_dimensions["A"].width = 95

    # 6. Methode et limites
    method = [
        {"rubrique": "Pipeline", "detail": "cible+hotspots -> BindCraft 2.0 (prod01, "
         "prod02) -> threading multi-rotamere PyMOL -> PROPKA (complexe / cible seule / "
         "binder seul) -> re-scoring orthogonal Boltz-2 -> classement lexicographique"},
        {"rubrique": "Generateur", "detail": "BindCraft 2.0, depot PacesaLab/BindCraft2, "
         "commit a8d0f2002df373842b86a3c20c5a060c5cfdf980, sur Modal L40S"},
        {"rubrique": "Threading", "detail": "PyMOL open-source headless, assistant de "
         "mutagenese, 3 rotameres par mutant, squelette du parent inchange, seuil de "
         "clash 2,2 A entre atomes lourds"},
        {"rubrique": "pKa", "detail": "propka3 via uv run --with propka, 141 runs. "
         "Numerotation verifiee sur les 23 structures : A409=HIS et empreinte des 6 His "
         "334/346/359/394/409/483 conforme partout"},
        {"rubrique": "MSA", "detail": "ColabFold MMseqs2 API, mode env, precalculee une "
         "fois (3725 sequences apres fusion en un alignement unique). MSA vide pour le "
         "binder, qui est de novo et sans homologue naturel"},
        {"rubrique": "Orthogonal", "detail": "Boltz-2 2.2.0, 3 echantillons de diffusion, "
         "3 etapes de recyclage. Module d'affinite NON utilise (calibre petites molecules)"},
        {"rubrique": "Classement", "detail": "lexicographique sur pH > cross-reactivite "
         "souris > affinite. Aucun score composite pondere. Critere pH DISCRETISE en trois "
         "paliers, parce que PROPKA se trompe d'environ une unite de pKa et qu'un ordre fin "
         "sur des ecarts de 0,4 unite serait du bruit"},
        {"rubrique": "ESM-2 — MESURE, hors classement", "detail": "esm2_t33_650M_UR50D, "
         "pseudo-vraisemblance par marginales masquees, normalisee par la longueur. "
         "PLL/residu de -2,72 a -1,93, mediane -2,21 sur 35 sequences. HORS CLASSEMENT PAR "
         "CONSTRUCTION : la metrique mesure la ressemblance aux proteines naturelles, alors "
         "que le reglement exige la nouveaute ; classer dessus favoriserait les designs les "
         "moins nouveaux. Lue comme detecteur d'anomalie, elle ne remonte RIEN : la plage "
         "est etroite (0,8 unite log) et aucune sequence ne sort du lot"},
        {"rubrique": "Pose orthogonale — MESUREE", "detail": "23/23 designs natifs. "
         "Recuperation de contacts 0,70 a 0,97 ; iptm Boltz 0,85 a 0,96. ATTENTION : le "
         "seuil de 'pose confirmee' (recup >= 0,50 et iptm >= 0,60) ne discrimine RIEN sur "
         "ce lot, tous le passent largement. Il n'apporte aucune information de classement"},
        {"rubrique": "Limite d'orthogonalite", "detail": "Boltz-2 et AF2 sont independants "
         "par l'architecture et les poids, mais tous deux entraines sur la PDB. Leur accord "
         "ecarte un artefact propre a AF2, pas un biais partage herite des donnees. Ce n'est "
         "pas une validation experimentale"},
        {"rubrique": "Regle bidentee (resultat de methode)", "detail": "les 2 designs a "
         "mecanisme pH ont DEUX carboxylates distincts, un par azote de l'imidazole de "
         "H409, goulot 3,34-3,35 A. Les 21 autres : 4,54-6,40 A ou aucune paire possible. "
         "Separation nette MAIS la comparaison informative est 2 contre 3 parmi les 5 "
         "designs capables de former la paire, ou un partage propre a ~10 % de chance sous "
         "l'hypothese nulle. Suggestive, non etablie, et N'ENTRE DANS AUCUN CRITERE de "
         "classement"},
        {"rubrique": "Cout structural des mutants — MESURE", "detail": "les 12 mutants "
         "ont ete predits par Boltz-2 apres le report d'echeance de 24 h. Reference : les "
         "contacts de la pose AF2 du PARENT, un mutant n'ayant pas de pose AF2. RESULTAT : "
         "aucune des 12 mutations ne casse l'interface, cout maximal 0,08 de recuperation "
         "de contacts. L'echec des mutants est donc CHIMIQUE, pas structural"},
        {"rubrique": "NON MESURE — cross-reactivite souris reelle", "detail": "aucune "
         "structure du domaine III murin n'a ete obtenue. L'objectif n°2 est approche par "
         "un proxy : fraction des residus de cible contactes qui sont identiques chez la "
         "souris, calculee sur l'alignement P00533/Q01279 de data/egfr_residues.csv"},
        {"rubrique": "NON MESURE — conformation etendue", "detail": "6ARU est replie. "
         "Toute SASA et toute geometrie calculees ici en heritent"},
        {"rubrique": "LIMITE — structures non relaxees", "detail": "les mutants sont des "
         "greffes de chaine laterale sur squelette rigide. Les pKa absolus en sont "
         "grossiers ; la comparaison appariee, qui partage le squelette, est en revanche "
         "la plus propre possible"},
        {"rubrique": "LIMITE — facteur de selectivite", "detail": "calcule depuis des pKa "
         "PROPKA sur des structures PREDITES. Sert a CLASSER. Aucun facteur absolu n'est "
         "revendique dans la soumission"},
        {"rubrique": "Allocation", "detail": f"couverture maximale, {len(chosen)} designs, "
         f"un par squelette. Alternative 'paires appariees' chiffree a "
         f"{allocation['designs_si_paires']} designs sur "
         f"{allocation['squelettes_avec_paire_possible']} squelettes, NON retenue faute de "
         f"savoir comment l'unicite est evaluee"},
    ]
    sheet("Methode et limites", method, ["rubrique", "detail"])

    control_sheet = sheet("Controles", control_rows, ["controle", "resultat", "detail"])
    letter = get_column_letter(2)
    span = f"{letter}2:{letter}{control_sheet.max_row}"
    control_sheet.conditional_formatting.add(span, CellIsRule(
        operator="equal", formula=['"OK"'], fill=green))
    control_sheet.conditional_formatting.add(span, CellIsRule(
        operator="equal", formula=['"ECHEC"'], fill=red))

    if rejected:
        sheet("Ecartes et motifs", rejected,
              ["rang_global", "design_id", "squelette", "type", "mutations",
               "mecanisme_pH", "verdict_carboxylate", "motif"])

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    book.save(OUT_XLSX)


def main() -> None:
    candidates = rows(MASTER)
    if not candidates:
        raise SystemExit(f"{MASTER} absent ou vide — lancer rank_designs.py d'abord")
    for entry in candidates:
        entry["sequence"] = entry["sequence"].strip().upper()

    chosen, rejected = select(candidates)
    allocation = paired_allocation(candidates)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(["name", "sequence", "molecule_class"])
        for entry in chosen:
            writer.writerow([entry["design_id"], entry["sequence"], MOLECULE_CLASS])

    control_rows = checks(chosen)
    control_rows.append(reread(OUT_CSV))
    workbook(chosen, candidates, control_rows, allocation, rejected)

    print(f"-> {OUT_CSV} : {len(chosen)} designs")
    print(f"-> {OUT_XLSX}")
    print()
    print("ALLOCATIONS")
    print(f"  couverture maximale (RETENUE) : {len(chosen)} designs, "
          f"{len({e['squelette'] for e in chosen})} squelettes couverts")
    print(f"  paires appariees (non retenue) : {allocation['designs_si_paires']} designs, "
          f"{allocation['squelettes_avec_paire_possible']} squelettes avec paire possible")
    print()
    print("CLASSEMENT SOUMIS")
    for position, entry in enumerate(chosen, start=1):
        print(f"  {position:2d}. {entry['design_id'][-26:]:<28} "
              f"G{entry['groupe']} {entry['palier_pH']:<16} "
              f"facteur {entry['facteur_pH_predit']!s:<7} "
              f"conserv {entry['epitope_conservation_frac']:<6} "
              f"{len(entry['sequence'])} aa")
    print()
    print("CONTROLES")
    for record in control_rows:
        print(f"  [{record['resultat']}] {record['controle']}")
        print(f"         {record['detail']}")


if __name__ == "__main__":
    main()
