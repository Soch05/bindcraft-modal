#!/usr/bin/env python3
"""Extrait le domaine III de l'EGFR MURIN (Q01279), par DEUX dérivations indépendantes.

    uv run --with biopython --with gemmi python mouse_target.py

POURQUOI CE FICHIER EXISTE. L'objectif n°2 du challenge — la même séquence doit reconnaître
P00533 et Q01279 — n'était jusqu'ici approché que par un proxy de séquence : la fraction des
résidus de cible contactés qui sont identiques chez la souris. Un proxy de séquence ne dit
rien de la conformation locale murine, et aucune structure du domaine III murin n'avait été
obtenue. C'est l'action 5 de CLAUDE.md, restée ouverte.

POURQUOI DEUX DÉRIVATIONS. Se tromper de région donnerait une cible murine plausible mais
fausse, et rien en aval ne le signalerait : Boltz-2 replierait la mauvaise séquence avec une
confiance élevée, et la « cross-réactivité mesurée » serait un artefact. On construit donc la
séquence de deux manières qui ne partagent aucune étape, et on refuse de continuer si elles
diffèrent :

  A. Par ALIGNEMENT de la séquence Q01279 complète (1210 aa, cache UniProt) sur la séquence
     du domaine III humain lue dans le PDB cible.
  B. Par la colonne `aa_mouse` de data/egfr_residues.csv, qui vient d'un alignement fait
     séparément par egfr_epitope_map.py, restreinte aux PDB 309–506.

Les deux doivent donner la même chaîne de 198 résidus. La dérivation B n'est valable que si
l'alignement y est sans indel dans cette fenêtre — ce qui est vérifié, pas supposé.

Sortie : inputs/mEGFR_dIII.fasta
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

HUMAN_PDB = Path("inputs/6ARU_A_309-506.pdb")
MOUSE_JSON = Path("data/Q01279.json")
RESIDUES = Path("data/egfr_residues.csv")
OUT_FASTA = Path("inputs/mEGFR_dIII.fasta")

DOMAIN_START, DOMAIN_END = 309, 506

# Paramètres d'alignement du dépôt, pour que l'offset trouvé ici soit comparable à celui
# qu'ont déjà produit les autres scripts.
OPEN_GAP, EXTEND_GAP = -11, -1


def human_domain() -> str:
    import gemmi

    structure = gemmi.read_structure(str(HUMAN_PDB))
    structure.setup_entities()
    return gemmi.one_letter_code([r.name for r in structure[0]["A"]]).upper()


def mouse_full() -> str:
    return json.loads(MOUSE_JSON.read_text())["sequence"]["value"].upper()


def derive_by_alignment(human: str, mouse: str) -> tuple[str, int, int]:
    """Dérivation A : aligne le domaine III humain sur la séquence murine complète."""
    from Bio import Align
    from Bio.Align import substitution_matrices

    aligner = Align.PairwiseAligner()
    aligner.mode = "global"
    aligner.substitution_matrix = substitution_matrices.load("BLOSUM62")
    aligner.open_gap_score = OPEN_GAP
    aligner.extend_gap_score = EXTEND_GAP
    # La requête est un fragment : pénaliser les brèches en bout de la cible interdirait
    # tout appariement local.
    aligner.target_end_gap_score = 0.0
    aligner.query_end_gap_score = 0.0

    alignment = aligner.align(mouse, human)[0]
    mouse_indices, human_indices = alignment.indices
    paired = [
        (int(m), int(h))
        for m, h in zip(mouse_indices, human_indices) if m >= 0 and h >= 0
    ]
    if not paired:
        raise SystemExit("aucun appariement entre le domaine III humain et Q01279")
    first, last = paired[0][0], paired[-1][0]
    insertions = (last - first + 1) - len(paired)
    deletions = len(human) - len(paired)
    if insertions or deletions:
        raise SystemExit(
            f"alignement avec indels dans la fenetre : {insertions} insertion(s) murine(s), "
            f"{deletions} deletion(s). La derivation colonne par colonne serait invalide et "
            f"le mapping de numerotation vers l'humain aussi. Traitement refuse."
        )
    return (mouse[first:last + 1], first + 1, last + 1)


def derive_by_column() -> str:
    """Dérivation B : la colonne `aa_mouse`, restreinte au domaine III."""
    rows = [
        r for r in csv.DictReader(RESIDUES.open(newline=""))
        if DOMAIN_START <= int(r["pdb_resnum"]) <= DOMAIN_END
    ]
    rows.sort(key=lambda r: int(r["pdb_resnum"]))
    numbers = [int(r["pdb_resnum"]) for r in rows]
    if numbers != list(range(DOMAIN_START, DOMAIN_END + 1)):
        raise SystemExit("la numerotation PDB du domaine III n'est pas contigue dans le CSV")
    letters = [r["aa_mouse"].strip().upper() for r in rows]
    if any(len(a) != 1 or not a.isalpha() for a in letters):
        raise SystemExit("colonne aa_mouse non exploitable (lacune ou valeur multiple)")
    return "".join(letters)


def main() -> None:
    human = human_domain()
    mouse = mouse_full()
    print(f"domaine III humain : {len(human)} residus, PDB {DOMAIN_START}-{DOMAIN_END}")
    print(f"Q01279 complet     : {len(mouse)} residus")
    print()

    by_alignment, start, end = derive_by_alignment(human, mouse)
    by_column = derive_by_column()

    print(f"derivation A, alignement   : {len(by_alignment)} residus, "
          f"UniProt murin {start}-{end}")
    print(f"derivation B, colonne CSV  : {len(by_column)} residus")

    if by_alignment != by_column:
        differences = [
            (index, a, b)
            for index, (a, b) in enumerate(zip(by_alignment, by_column)) if a != b
        ]
        print()
        print(f"DESACCORD sur {len(differences)} position(s) :")
        for index, a, b in differences[:12]:
            print(f"   position {index + DOMAIN_START} (PDB) : alignement {a}, colonne {b}")
        raise SystemExit(
            "les deux derivations divergent — la cible murine n'est pas etablie, "
            "traitement refuse"
        )

    print()
    print("LES DEUX DERIVATIONS CONCORDENT. Cible murine etablie.")

    identical = sum(1 for a, b in zip(human, by_column) if a == b)
    print(f"identite humain/souris sur le domaine III : "
          f"{identical}/{len(human)} = {identical / len(human):.1%}")

    his_human = [DOMAIN_START + i for i, a in enumerate(human) if a == "H"]
    his_mouse = [DOMAIN_START + i for i, a in enumerate(by_column) if a == "H"]
    print(f"His humaines (num. PDB) : {his_human}")
    print(f"His murines  (num. PDB) : {his_mouse}")
    target_index = 409 - DOMAIN_START
    print(f"position 409 : humain {human[target_index]}, souris {by_column[target_index]}")
    if by_column[target_index] != "H":
        raise SystemExit(
            "H409 n'est pas une histidine chez la souris : le mecanisme pH Route 2 ne peut "
            "pas y etre transpose, et tout le raisonnement de cross-reactivite tombe."
        )
    print("H409 est conservee chez la souris — le mecanisme pH est transposable.")

    divergent = [
        (DOMAIN_START + i, a, b)
        for i, (a, b) in enumerate(zip(human, by_column)) if a != b
    ]
    print(f"\n{len(divergent)} positions divergentes dans le domaine III :")
    print("   " + ", ".join(f"{a}{number}{b}" for number, a, b in divergent))

    OUT_FASTA.write_text(f">mEGFR_dIII_Q01279_{start}-{end}\n{by_column}\n")
    print(f"\n-> {OUT_FASTA}")


if __name__ == "__main__":
    main()
