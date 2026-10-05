#!/usr/bin/env python3
"""Vérifie la géométrie des mécanismes pH : ponts salins WT, portée des mutations, rotamères.

    uv run --with biopython python verify_geometry.py

Trois mesures, qui répondent à trois questions distinctes du plan.

PHASE 1g — LES PONTS SALINS WT SONT-ILS RÉELS ? Les six squelettes à mécanisme pH portent un
carboxylate de binder contre H409 de la cible. La distance seule ne suffit pas : sur une
structure PRÉDITE un contact trop court est un artefact d'AF2, pas une liaison forte. On
mesure donc distance, angle à l'oxygène accepteur, enfouissement, et on cherche des contacts
lourds anormalement courts dans le voisinage.

PHASE 1e — LES MUTATIONS SONT-ELLES À PORTÉE ? Un D/E greffé dont le carboxylate reste loin de
H409 n'est pas un design, c'est du bruit. On mesure la distance carboxylate→imidazole sur
CHAQUE rotamère.

PHASE 2b (complément) — LES ROTAMÈRES SONT-ILS DISTINCTS ? Le tri du threading classe sur le
contact le plus serré. Quand ce contact est porté par le CB — atome qui ne bouge PAS d'un
rotamère à l'autre, sa position étant fixée par le squelette — le tri est aveugle et trois
« rotamères » peuvent être le même. Un verdict multi-rotamère construit sur des rotamères
identiques serait une fausse confiance. On mesure donc chi1, chi2 et l'étendue du centroïde
du carboxylate.

Sortie : out/geometry_wt.csv, out/geometry_mutants.csv
"""

from __future__ import annotations

import csv
import itertools
import math
import warnings
from pathlib import Path

import numpy as np
from Bio.PDB import MMCIFParser, PDBParser, ShrakeRupley
from Bio.PDB.Structure import Structure

warnings.filterwarnings("ignore")

MUTANTS_CSV = Path("out/mutants_acide.csv")
THREADED_INDEX = Path("structures/threaded/threaded_index.csv")
OUT_WT = Path("out/geometry_wt.csv")
OUT_MUT = Path("out/geometry_mutants.csv")

# Critère de pont salin entre atomes chargés, Barlow & Thornton 1983. Même seuil que
# his_acid_pairing.py, pour que les deux scripts restent comparables.
SALT_BRIDGE_A = 4.0

# Deux atomes lourds non liés plus proches que ça sont en recouvrement physique.
HARD_CLASH_A = 2.2

# Un contact O···N de pont salin se situe normalement entre 2,6 et 3,2 A. En dessous de ce
# seuil la géométrie est plus serrée que ce qu'une liaison hydrogène forte justifie, et sur
# une structure prédite c'est un signe d'artefact plutôt que d'affinité.
SHORT_CONTACT_A = 2.60

# Au-delà de cette étendue du centroïde du carboxylate, deux rotamères explorent des
# positions réellement différentes. En dessous, ils sont le même rotamère à bruit près.
ROTAMER_DISTINCT_A = 0.30

TARGET_CHAIN = "A"
TARGET_HIS = 409
HIS_CHARGED = ("ND1", "NE2")
ACID_CHARGED = {"ASP": ("OD1", "OD2"), "GLU": ("OE1", "OE2")}
ACID_STEM = {"ASP": ("CG", ("OD1", "OD2")), "GLU": ("CD", ("OE1", "OE2"))}
BACKBONE = {"N", "CA", "C", "O", "OXT"}

# SASA de la chaîne latérale isolée, tripeptide Gly-X-Gly étendu, Tien et al. 2013.
SIDECHAIN_REF_A2 = {"ASP": 97.0, "GLU": 127.0}


def load(path: Path) -> Structure:
    parser = MMCIFParser(QUIET=True) if path.suffix == ".cif" else PDBParser(QUIET=True)
    return parser.get_structure(path.stem, str(path))


def binder_chain_id(structure: Structure) -> str:
    """La cible est la chaîne A ; le binder est l'autre. On ne devine pas, on regarde."""
    ids = [chain.id for chain in structure[0]]
    others = [i for i in ids if i != TARGET_CHAIN]
    return others[0] if others else ids[0]


def angle_deg(a: np.ndarray, b: np.ndarray, c: np.ndarray) -> float:
    """Angle au sommet b, en degrés."""
    u, v = a - b, c - b
    cosine = float(np.dot(u, v) / (np.linalg.norm(u) * np.linalg.norm(v)))
    return math.degrees(math.acos(max(-1.0, min(1.0, cosine))))


def dihedral_deg(p0: np.ndarray, p1: np.ndarray, p2: np.ndarray, p3: np.ndarray) -> float:
    b0, b1, b2 = p0 - p1, p2 - p1, p3 - p2
    b1 = b1 / np.linalg.norm(b1)
    v = b0 - np.dot(b0, b1) * b1
    w = b2 - np.dot(b2, b1) * b1
    return math.degrees(math.atan2(np.dot(np.cross(b1, v), w), np.dot(v, w)))


def his_charged_coords(structure: Structure) -> dict[str, np.ndarray]:
    chain = structure[0][TARGET_CHAIN]
    if TARGET_HIS not in [r.id[1] for r in chain]:
        return {}
    residue = chain[(" ", TARGET_HIS, " ")]
    if residue.get_resname() != "HIS":
        raise SystemExit(
            f"ATTENDU HIS au residu {TARGET_CHAIN}{TARGET_HIS}, trouve "
            f"{residue.get_resname()} — numerotation a verifier"
        )
    return {name: residue[name].coord for name in HIS_CHARGED if name in residue}


def closest_to_his(
    residue, his: dict[str, np.ndarray]
) -> tuple[float, str, str] | None:
    """Oxygène de carboxylate le plus proche d'un azote d'imidazole."""
    names = ACID_CHARGED.get(residue.get_resname())
    if not names or not his:
        return None
    pairs = [
        (float(np.linalg.norm(residue[o].coord - coord)), o, n)
        for o in names if o in residue
        for n, coord in his.items()
    ]
    return min(pairs) if pairs else None


def short_contacts(structure: Structure, residue, binder: str) -> tuple[int, float, str]:
    """Contacts lourds anormalement courts entre la chaîne latérale et le reste.

    Exclut les atomes du résidu lui-même : ses liaisons internes sont courtes par nature.
    """
    side = [a for a in residue if a.element != "H" and a.get_name() not in BACKBONE]
    others = [
        a for ch in structure[0] for r in ch for a in r
        if a.element != "H" and not (ch.id == binder and r.id[1] == residue.id[1])
    ]
    if not side or not others:
        return (0, 9e9, "")
    pairs = sorted(
        (float(np.linalg.norm(s.coord - o.coord)),
         f"{s.get_name()}--{o.get_parent().get_parent().id}"
         f"{o.get_parent().id[1]}:{o.get_name()}")
        for s in side for o in others
    )
    count = sum(1 for d, _ in pairs if d < HARD_CLASH_A)
    return (count, pairs[0][0], pairs[0][1])


def sidechain_sasa(structure: Structure, binder: str, resnum: int) -> tuple[float, float]:
    """SASA de la chaîne latérale dans le complexe, et fraction par rapport au résidu libre.

    Deux passes : complexe entier, puis binder seul. La différence dit ce que l'interface
    enfouit ; la fraction par rapport au tripeptide de référence dit l'enfouissement absolu.
    """
    sr = ShrakeRupley()

    def side_total(entity, chain_id: str) -> float:
        residue = entity[0][chain_id][(" ", resnum, " ")]
        return sum(
            a.sasa for a in residue
            if a.element != "H" and a.get_name() not in BACKBONE
        )

    sr.compute(structure[0], level="A")
    bound = side_total(structure, binder)

    import copy
    alone = copy.deepcopy(structure)
    for chain in list(alone[0]):
        if chain.id != binder:
            alone[0].detach_child(chain.id)
    sr.compute(alone[0], level="A")
    free = side_total(alone, binder)
    return (bound, free)


def wt_structure(design: str) -> Path | None:
    """Les PDB produits par prepare_structures.py, pas les .cif d'origine : ceux-ci n'ont pas
    de colonne `_atom_site.occupancy` et biopython la refuse."""
    path = Path(f"structures/wt/{design}.pdb")
    return path if path.is_file() else None


def do_wt() -> list[dict]:
    """Phase 1g : les ponts salins existants, sur les 23 WT."""
    rows = []
    print("=" * 78)
    print("PHASE 1g — PONTS SALINS WT CONTRE H409")
    print("=" * 78)
    for record in csv.DictReader(MUTANTS_CSV.open(newline="")):
        design = record["design"]
        path = wt_structure(design)
        if path is None:
            print(f"  {design} : structure introuvable")
            continue
        structure = load(path)
        binder = binder_chain_id(structure)
        his = his_charged_coords(structure)
        if not his:
            print(f"  {design} : pas de H409 dans la cible")
            continue

        best = None
        for residue in structure[0][binder]:
            found = closest_to_his(residue, his)
            if found and (best is None or found[0] < best[0][0]):
                best = (found, residue)
        if best is None:
            continue
        (distance, oxygen, nitrogen), residue = best

        stem_name, oxygens = ACID_STEM[residue.get_resname()]
        angle = angle_deg(residue[stem_name].coord, residue[oxygen].coord, his[nitrogen])
        clashes, nearest, partner = short_contacts(structure, residue, binder)
        bound, free = sidechain_sasa(structure, binder, residue.id[1])
        reference = SIDECHAIN_REF_A2[residue.get_resname()]

        flags = []
        if distance < SHORT_CONTACT_A:
            flags.append("CONTACT COURT")
        if clashes:
            flags.append(f"{clashes} CLASH")
        if distance > SALT_BRIDGE_A:
            flags.append("hors portee")

        rows.append({
            "design": design,
            "squelette": design.split("_")[-2],
            "pont": "oui" if distance <= SALT_BRIDGE_A else "non",
            "residu_acide": f"{residue.get_resname()}{residue.id[1]}",
            "atomes": f"{oxygen}-{nitrogen}",
            "distance_A": round(distance, 2),
            "angle_deg": round(angle, 1),
            "sasa_liee_A2": round(bound, 1),
            "sasa_libre_A2": round(free, 1),
            "enfouie_par_interface_A2": round(free - bound, 1),
            "frac_exposee_libre": round(free / reference, 3),
            "contacts_courts": clashes,
            "contact_le_plus_court_A": round(nearest, 2),
            "partenaire": partner,
            "alerte": " / ".join(flags) or "-",
        })
        marker = f"   <<< {' / '.join(flags)}" if flags else ""
        print(
            f"  {design[-24:]:<26} {residue.get_resname()}{residue.id[1]:<4} "
            f"{oxygen}-{nitrogen}  {distance:5.2f} A  angle {angle:5.1f} deg  "
            f"enfoui {free - bound:5.1f} A2  court {nearest:4.2f} A{marker}"
        )
    return rows


def do_mutants() -> list[dict]:
    """Phases 1e et 2b : portée vers H409, et distinction réelle des rotamères."""
    rows = []
    print()
    print("=" * 78)
    print("PHASES 1e / 2b — MUTANTS : PORTEE VERS H409 ET DISTINCTION DES ROTAMERES")
    print("=" * 78)
    index = list(csv.DictReader(THREADED_INDEX.open(newline="")))
    by_mutant: dict[tuple[str, str], list[dict]] = {}
    for record in index:
        by_mutant.setdefault((record["design"], record["mutation"]), []).append(record)

    for (design, mutation), records in by_mutant.items():
        resnum = int(mutation[1:-1])
        centroids, measured = {}, []
        for record in sorted(records, key=lambda r: int(r["rotamer"])):
            path = Path(record["path"])
            structure = load(path)
            binder = binder_chain_id(structure)
            his = his_charged_coords(structure)
            residue = structure[0][binder][(" ", resnum, " ")]
            found = closest_to_his(residue, his)
            if found is None:
                continue
            distance, oxygen, nitrogen = found
            stem_name, oxygens = ACID_STEM[residue.get_resname()]
            angle = angle_deg(residue[stem_name].coord, residue[oxygen].coord, his[nitrogen])
            chi1 = dihedral_deg(*(residue[n].coord for n in ("N", "CA", "CB", "CG")))
            second = ("CA", "CB", "CG", "OD1") if residue.get_resname() == "ASP" else \
                     ("CA", "CB", "CG", "CD")
            chi2 = dihedral_deg(*(residue[n].coord for n in second))
            centroid = np.mean([residue[o].coord for o in oxygens if o in residue], axis=0)
            centroids[record["rotamer"]] = centroid
            clashes, nearest, partner = short_contacts(structure, residue, binder)
            bound, free = sidechain_sasa(structure, binder, resnum)
            measured.append({
                "design": design, "squelette": design.split("_")[-2], "mutation": mutation,
                "rotamer": int(record["rotamer"]),
                "distance_H409_A": round(distance, 2),
                "atomes": f"{oxygen}-{nitrogen}",
                "angle_deg": round(angle, 1),
                "a_portee": "oui" if distance <= SALT_BRIDGE_A else "non",
                "chi1_deg": round(chi1, 1), "chi2_deg": round(chi2, 1),
                "contacts_courts": clashes,
                "contact_le_plus_court_A": round(nearest, 2),
                "partenaire": partner,
                "clash_threading": record["clash"],
                "enfouie_par_interface_A2": round(free - bound, 1),
            })

        spread = 0.0
        if len(centroids) > 1:
            spread = max(
                float(np.linalg.norm(a - b))
                for a, b in itertools.combinations(centroids.values(), 2)
            )
        distinct = "oui" if spread >= ROTAMER_DISTINCT_A else "NON"
        for row in measured:
            row["etendue_centroide_A"] = round(spread, 2)
            row["rotameres_distincts"] = distinct
        rows.extend(measured)

        reach = [r for r in measured if r["a_portee"] == "oui"]
        clean = [r for r in measured if r["contacts_courts"] == 0]
        print(f"  {design[-24:]:<26} {mutation:<6} etendue centroide {spread:4.2f} A  "
              f"distincts {distinct:<3}  a portee {len(reach)}/{len(measured)}  "
              f"sans clash {len(clean)}/{len(measured)}")
        for row in measured:
            print(f"      rot{row['rotamer']}  H409 {row['distance_H409_A']:5.2f} A "
                  f"({row['atomes']})  angle {row['angle_deg']:5.1f}  "
                  f"chi1 {row['chi1_deg']:7.1f} chi2 {row['chi2_deg']:7.1f}  "
                  f"court {row['contact_le_plus_court_A']:4.2f} A "
                  f"({row['contacts_courts']} < {HARD_CLASH_A})")
    return rows


def write(path: Path, rows: list[dict]) -> None:
    if not rows:
        print(f"  (rien a ecrire dans {path})")
        return
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print(f"-> {len(rows)} lignes dans {path}")


def main() -> None:
    wt = do_wt()
    mutants = do_mutants()
    print()
    write(OUT_WT, wt)
    write(OUT_MUT, mutants)


if __name__ == "__main__":
    main()
