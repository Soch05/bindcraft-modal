#!/usr/bin/env python3
"""Mesure l'appariement His–acide aux interfaces des designs acceptés.

Action 6 du §8 de CLAUDE.md, jamais faite avant le 4 octobre. C'est l'objectif n°1 du
challenge — la sélectivité pH — et il reposait jusqu'ici sur une co-occurrence dans une
liste d'interface, ce qui ne dit rien de la géométrie.

    uv run his_acid_pairing.py out/egfr-dIII-prod01/3_Ranked/*.cif

Deux routes, cf. CLAUDE.md §6 :

  route 1  His du BINDER  contre  acide de la CIBLE (D323)
           -> gain de liaison à pH 6,5 : le pont salin n'existe que sous forme protonée
  route 2  acide du BINDER contre  His de la CIBLE (H409)
           -> même mécanisme, et H409 est `identical` chez la souris, donc la route sert
              aussi l'objectif n°2

Les `.cif` de BindCraft 2.0 n'ont pas de colonne `_atom_site.occupancy`, donc le parseur
mmCIF de Biopython échoue dessus. On lit la boucle `_atom_site` directement, en se repérant
sur l'ordre des colonnes déclaré dans le fichier plutôt que sur des positions fixes.
"""

from __future__ import annotations

import argparse
import math
import re
from dataclasses import dataclass
from pathlib import Path

# ----------------------------------------------------------------------------------------
# Seuils. Dans le code et non dans la prose, cf. CLAUDE.md §7.
# ----------------------------------------------------------------------------------------

# Convention de pont salin : atomes lourds des groupes chargés à moins de 4,0 Å. C'est le
# critère de Barlow & Thornton (1983), celui qu'emploie la littérature sur les ponts salins.
SALT_BRIDGE_A = 4.0

# Seuil large pour montrer les quasi-contacts : une paire à 5–6 Å n'est pas un pont salin
# mais peut en devenir un après relaxation des chaînes latérales. ⚠️ POSÉ, non calibré.
WITHIN_REACH_A = 6.0

# Atomes porteurs de la charge. L'imidazole de l'histidine se protone sur ND1 ou NE2 ;
# le carboxylate porte sa charge délocalisée sur ses deux oxygènes.
HIS_CHARGED_ATOMS = ("ND1", "NE2")
ACID_CHARGED_ATOMS = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}

# Les deux résidus de cible qui portent les mécanismes, en numérotation PDB — vérifiée par
# lecture directe de `inputs/6ARU_A_309-506.pdb` le 3 octobre : 323 = ASP, 409 = HIS.
TARGET_ACID_ANCHOR = 323
TARGET_HIS_ANCHOR = 409


@dataclass(frozen=True)
class Atom:
    chain: str
    resnum: int
    resname: str
    atom: str
    x: float
    y: float
    z: float

    def distance(self, other: Atom) -> float:
        return math.dist((self.x, self.y, self.z), (other.x, other.y, other.z))


def read_cif_atoms(path: Path) -> tuple[list[Atom], dict[str, str]]:
    """Lit la boucle `_atom_site` et les métadonnées `_bindcraft.*`.

    On repère les colonnes par leur nom déclaré, pas par un index en dur : l'ordre des
    colonnes d'une boucle mmCIF n'est pas normatif.
    """
    text = path.read_text()

    stamp = dict(re.findall(r"^_bindcraft\.(\S+)\s+'?([^'\n]+?)'?\s*$", text, re.M))

    columns: list[str] = []
    atoms: list[Atom] = []
    in_loop = False
    for line in text.splitlines():
        if line.startswith("_atom_site."):
            columns.append(line.strip().removeprefix("_atom_site."))
            in_loop = True
            continue
        if in_loop and line.startswith(("ATOM", "HETATM")):
            fields = line.split()
            if len(fields) < len(columns):
                continue
            row = dict(zip(columns, fields))
            try:
                atoms.append(
                    Atom(
                        chain=row["auth_asym_id"],
                        resnum=int(row["auth_seq_id"]),
                        resname=row["auth_comp_id"],
                        atom=row["auth_atom_id"],
                        x=float(row["Cartn_x"]),
                        y=float(row["Cartn_y"]),
                        z=float(row["Cartn_z"]),
                    )
                )
            except (KeyError, ValueError):
                continue
        elif in_loop and atoms and not line.startswith(("ATOM", "HETATM")):
            break
    return atoms, stamp


def charged_atoms(atoms: list[Atom], chain: str, resname: str) -> list[Atom]:
    wanted = HIS_CHARGED_ATOMS if resname == "HIS" else ACID_CHARGED_ATOMS.get(resname, ())
    return [a for a in atoms if a.chain == chain and a.resname == resname and a.atom in wanted]


def closest_pairs(
    source: list[Atom], target: list[Atom]
) -> list[tuple[float, Atom, Atom]]:
    """Distance minimale par couple de résidus, triée."""
    best: dict[tuple[int, int], tuple[float, Atom, Atom]] = {}
    for a in source:
        for b in target:
            d = a.distance(b)
            key = (a.resnum, b.resnum)
            if key not in best or d < best[key][0]:
                best[key] = (d, a, b)
    return sorted(best.values(), key=lambda item: item[0])


def verdict(distance: float) -> str:
    if distance <= SALT_BRIDGE_A:
        return f"PONT SALIN (<= {SALT_BRIDGE_A} A)"
    if distance <= WITHIN_REACH_A:
        return f"a portee (<= {WITHIN_REACH_A} A, pose)"
    return "trop loin"


def report_route(
    label: str,
    pairs: list[tuple[float, Atom, Atom]],
    anchor: int,
    anchor_label: str,
    show: int = 6,
) -> bool:
    print(f"  {label}")
    if not pairs:
        print("    aucun couple possible — un des deux partenaires est absent")
        return False
    on_anchor = [p for p in pairs if p[2].resnum == anchor]
    for distance, a, b in pairs[:show]:
        flag = "  <<< ANCRE" if b.resnum == anchor else ""
        print(
            f"    {a.resname}{a.resnum}:{a.atom:<4} -- {b.resname}{b.resnum}:{b.atom:<4} "
            f"{distance:5.2f} A   {verdict(distance)}{flag}"
        )
    if not on_anchor:
        print(f"    (aucun contact avec l'ancre {anchor_label})")
        return False
    best = on_anchor[0][0]
    print(f"    -> meilleure distance a l'ancre {anchor_label} : {best:.2f} A  {verdict(best)}")
    return best <= SALT_BRIDGE_A


def analyse(path: Path) -> dict[str, bool]:
    atoms, stamp = read_cif_atoms(path)
    binder = stamp.get("binder_chains", "B").split(",")[0]
    target = stamp.get("target_chains", "A").split(",")[0]

    print("=" * 78)
    print(f"{path.name}")
    print(f"  binder = chaine {binder}   cible = chaine {target}   atomes lus : {len(atoms)}")
    if stamp.get("bindcraft_revision", "").endswith("-dirty"):
        print(f"  ⚠️ revision amont marquee -dirty : {stamp['bindcraft_revision']}")
    print("=" * 78)

    binder_his = charged_atoms(atoms, binder, "HIS")
    binder_acid = charged_atoms(atoms, binder, "ASP") + charged_atoms(atoms, binder, "GLU")
    target_his = charged_atoms(atoms, target, "HIS")
    target_acid = charged_atoms(atoms, target, "ASP") + charged_atoms(atoms, target, "GLU")

    print(
        f"  groupes charges : binder {len(set(a.resnum for a in binder_his))} His / "
        f"{len(set(a.resnum for a in binder_acid))} acides | "
        f"cible {len(set(a.resnum for a in target_his))} His / "
        f"{len(set(a.resnum for a in target_acid))} acides"
    )
    print()

    route1 = report_route(
        f"ROUTE 1 — His du binder  ->  acide de la cible (ancre D{TARGET_ACID_ANCHOR})",
        closest_pairs(binder_his, target_acid),
        TARGET_ACID_ANCHOR,
        f"D{TARGET_ACID_ANCHOR}",
    )
    print()
    route2 = report_route(
        f"ROUTE 2 — acide du binder  ->  His de la cible (ancre H{TARGET_HIS_ANCHOR})",
        closest_pairs(binder_acid, target_his),
        TARGET_HIS_ANCHOR,
        f"H{TARGET_HIS_ANCHOR}",
    )
    print()
    return {"route1": route1, "route2": route2}


def main() -> None:
    parser = argparse.ArgumentParser(
        description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument("structures", type=Path, nargs="+", help="fichiers .cif de designs")
    args = parser.parse_args()

    results = {}
    for path in args.structures:
        if not path.is_file():
            print(f"(absent : {path})")
            continue
        results[path.name] = analyse(path)

    print("=" * 78)
    print("BILAN")
    print("=" * 78)
    print(f"  critere de pont salin : atomes charges a <= {SALT_BRIDGE_A} A (Barlow & Thornton)")
    print()
    for name, routes in results.items():
        marks = " ".join(
            f"{route}={'OUI' if ok else 'non'}" for route, ok in routes.items()
        )
        print(f"  {name[:58]:<58} {marks}")
    total = sum(any(r.values()) for r in results.values())
    print()
    print(f"  designs avec AU MOINS une route etablie : {total} / {len(results)}")
    if not total:
        print("  -> aucun mecanisme pH mesure. Ne PAS presenter ce lot comme pH-dependant.")


if __name__ == "__main__":
    main()
