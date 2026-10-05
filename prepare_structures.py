#!/usr/bin/env python3
"""Convertit les .cif de BindCraft en .pdb, et VÉRIFIE la numérotation avant la série.

    uv run --with gemmi python prepare_structures.py

POURQUOI CETTE ÉTAPE EXISTE : les .cif écrits par BindCraft 2.0 n'ont pas de colonne
`_atom_site.occupancy`, et le MMCIFParser de biopython la tient pour obligatoire — il lève un
KeyError. gemmi est tolérant. Comme PROPKA lit le PDB de toute façon, une conversion unique
sert les deux étapes au lieu d'être refaite deux fois.

LA VÉRIFICATION DE NUMÉROTATION EST LE POINT CRITIQUE. Le mécanisme pH visé repose sur H409 de
la cible, en numérotation PDB. Si le fichier était numéroté en UniProt (décalage +24), le
résidu 409 serait un autre acide aminé et TOUT le reste du pipeline mesurerait le mauvais
résidu sans qu'aucune valeur n'ait l'air anormale. On refuse donc de convertir un fichier dont
le résidu 409 de la chaîne cible n'est pas une histidine.

Sortie : structures/wt/<design>.pdb, et structures/target_only/<design>.pdb (cible seule,
extraite du MÊME fichier, pour le calcul PROPKA de référence en phase 3).
"""

from __future__ import annotations

import csv
from pathlib import Path

import gemmi

RANKED_RUNS = ("egfr-dIII-prod01", "egfr-dIII-prod02")
OUT_COMPLEX = Path("structures/wt")
OUT_TARGET = Path("structures/target_only")
INDEX = Path("out/structures_index.csv")

TARGET_CHAIN = "A"
TARGET_HIS = 409

# Les six His de la cible, lues dans inputs/6ARU_A_309-506.pdb le 3 octobre (CLAUDE.md §3).
# Si la numérotation glissait, cette empreinte ne serait pas retrouvée.
EXPECTED_HIS = {334, 346, 359, 394, 409, 483}


def check_numbering(structure: gemmi.Structure, label: str) -> dict:
    """Confirme que 409 est une His, et retrouve l'empreinte des six His de la cible."""
    model = structure[0]
    chain = model[TARGET_CHAIN]
    found = {r.seqid.num: r.name for r in chain}
    if TARGET_HIS not in found:
        raise SystemExit(f"{label} : pas de residu {TARGET_HIS} dans la chaine {TARGET_CHAIN}")
    if found[TARGET_HIS] != "HIS":
        raise SystemExit(
            f"{label} : residu {TARGET_CHAIN}{TARGET_HIS} = {found[TARGET_HIS]}, "
            f"attendu HIS. Numerotation suspecte, conversion refusee."
        )
    his = {n for n, name in found.items() if name == "HIS"}
    return {
        "residu_409": found[TARGET_HIS],
        "his_cible": ",".join(str(n) for n in sorted(his)),
        "empreinte_his_conforme": "oui" if his == EXPECTED_HIS else "NON",
        "n_residus_cible": len(found),
    }


def target_only(structure: gemmi.Structure) -> gemmi.Structure:
    """Cible seule, extraite du même fichier — pas d'un autre fichier, pour que la
    comparaison lié/libre de la phase 3 ne porte que sur la présence du binder."""
    copy = structure.clone()
    model = copy[0]
    for chain in [c.name for c in model]:
        if chain != TARGET_CHAIN:
            model.remove_chain(chain)
    return copy


def main() -> None:
    OUT_COMPLEX.mkdir(parents=True, exist_ok=True)
    OUT_TARGET.mkdir(parents=True, exist_ok=True)
    rows = []
    for run in RANKED_RUNS:
        for source in sorted(Path(f"out/{run}/3_Ranked").glob("*.cif")):
            design = source.stem
            structure = gemmi.read_structure(str(source))
            structure.setup_entities()
            info = check_numbering(structure, design)

            chains = [c.name for c in structure[0]]
            binder = [c for c in chains if c != TARGET_CHAIN]

            complex_path = OUT_COMPLEX / f"{design}.pdb"
            structure.write_pdb(str(complex_path))

            alone = target_only(structure)
            target_path = OUT_TARGET / f"{design}.pdb"
            alone.write_pdb(str(target_path))

            rows.append({
                "design": design, "run": run,
                "squelette": design.split("_")[-2],
                "chaines": ",".join(chains),
                "chaine_binder": ",".join(binder),
                "complexe_pdb": str(complex_path),
                "cible_seule_pdb": str(target_path),
                **info,
            })
            print(f"  {design[-26:]:<28} chaines {','.join(chains):<5} "
                  f"409={info['residu_409']}  His cible {info['his_cible']}  "
                  f"empreinte {info['empreinte_his_conforme']}")

    with INDEX.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    bad = [r for r in rows if r["empreinte_his_conforme"] != "oui"]
    print()
    print(f"-> {len(rows)} complexes convertis, index dans {INDEX}")
    print(f"   empreinte des 6 His non conforme : {len(bad)}")


if __name__ == "__main__":
    main()
