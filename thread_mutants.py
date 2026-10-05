#!/usr/bin/env python3
"""Greffe les substitutions acides sur le squelette du parent, plusieurs rotamères par mutant.

    uv run python thread_mutants.py

POURQUOI LE THREADING ET PAS UN CO-REPLIEMENT : en greffant la mutation sur le `.cif` du
parent, le squelette reste **rigoureusement identique** entre parent et mutant. L'analyse
appariée n'a donc qu'une seule variable. Un co-repliement laisserait le squelette bouger et
confondrait l'effet de la substitution avec un déplacement global.

POURQUOI PLUSIEURS ROTAMÈRES : le pKa d'un carboxylate enfoui dépend presque entièrement de
la désolvatation et des termes coulombiens locaux, donc de la **position exacte** de la chaîne
latérale. Un seul rotamère non relaxé donnerait un pKa dont on ne saurait pas s'il décrit la
chimie ou l'artefact de placement. En sortant plusieurs rotamères, PROPKA peut dire si le
verdict est **stable** ou **indéterminé** — ce qui évite de rétrograder un mutant sur un
artefact.

LIMITE À ÉCRIRE DANS LE RAPPORT : squelette non relaxé, chaîne latérale posée par rotamère
idéal. Les pKa **absolus** en sont plus grossiers. Pour la **comparaison appariée** parent vs
mutant, c'est au contraire l'approche la plus propre, puisque le squelette est partagé.

Sortie : `structures/threaded/<design>__<mutation>__rot<N>_strain<S>.pdb`
PDB et non mmCIF, parce que PROPKA lit le PDB — la conversion est ainsi gratuite.
"""

from __future__ import annotations

import csv
import math
import re
import subprocess
from pathlib import Path

MUTANTS_CSV = Path("out/mutants_acide.csv")
OUT_DIR = Path("structures/threaded")

# Nombre de rotamères conservés par mutant, les moins contraints d'abord. Trois suffisent
# pour dire si le pKa est stable ; au-delà on multiplie les runs PROPKA sans gagner.
ROTAMERS_KEPT = 3

# Deux atomes lourds plus proches que ça sont en recouvrement. Seuil du prompt.
CLASH_A = 2.2

AA_THREE = {"D": "ASP", "E": "GLU"}


def parent_structure(design: str) -> Path | None:
    for run in ("egfr-dIII-prod01", "egfr-dIII-prod02"):
        candidate = Path(f"out/{run}/3_Ranked/{design}.cif")
        if candidate.is_file():
            return candidate
    return None


def binder_chain(path: Path) -> str:
    stamp = dict(re.findall(r"^_bindcraft\.(\S+)\s+'?([^'\n]+?)'?\s*$", path.read_text(), re.M))
    return stamp.get("binder_chains", "B").split(",")[0]


def thread(parent: Path, chain: str, resnum: int, new_aa: str, stem: str) -> list[Path]:
    """Applique la mutation et écrit TOUS les rotamères. Le tri est fait ensuite en Python.

    `pymol -cq fichier.py` et non `pymol -d` : l'option `-d` exécute ligne par ligne au
    toplevel, donc aucun bloc multi-ligne ne passe. Et on n'interroge pas la contrainte de
    PyMOL — `get_prompt()` ne la contient pas, elle n'apparaît que dans son flux de sortie.
    On classe sur notre propre mesure de contact, qui est celle qu'on rapporte.
    """
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    script = OUT_DIR / f"_{stem}.pml.py"
    script.write_text(f"""
from pymol import cmd

def fresh():
    cmd.delete('all')
    cmd.load({str(parent)!r}, 'parent')
    cmd.wizard('mutagenesis')
    cmd.get_wizard().set_mode({AA_THREE[new_aa]!r})
    cmd.get_wizard().do_select('/parent//{chain}/{resnum}')
    return cmd.count_states('mutation')

total = fresh()
print('TOTAL=%d' % total)
for state in range(1, total + 1):
    fresh()
    cmd.frame(state)
    cmd.get_wizard().apply()
    cmd.save('{OUT_DIR}/{stem}__state%d.pdb' % state, 'parent')
""")
    result = subprocess.run(
        ["pymol", "-cq", str(script)], capture_output=True, text=True, timeout=600
    )
    script.unlink(missing_ok=True)
    found = re.search(r"TOTAL=(\d+)", result.stdout)
    if not found:
        tail = (result.stderr or result.stdout).strip()[-250:]
        print(f"    ECHEC pymol : {tail}")
        return []
    return [p for p in (OUT_DIR / f"{stem}__state{n}.pdb" for n in range(1, int(found.group(1)) + 1)) if p.is_file()]


def heavy_atoms(path: Path) -> list[tuple[str, int, str, float, float, float]]:
    out = []
    for line in path.read_text().splitlines():
        if not line.startswith(("ATOM", "HETATM")):
            continue
        element = line[76:78].strip() or line[12:16].strip()[:1]
        if element == "H":
            continue
        out.append(
            (line[21], int(line[22:26]), line[12:16].strip(),
             float(line[30:38]), float(line[38:46]), float(line[46:54]))
        )
    return out


def worst_clash(path: Path, chain: str, resnum: int) -> tuple[float, str]:
    """Contact le plus serré entre la chaîne latérale greffée et le reste de la structure."""
    atoms = heavy_atoms(path)
    mutated = [a for a in atoms if a[0] == chain and a[1] == resnum
               and a[2] not in {"N", "CA", "C", "O"}]
    others = [a for a in atoms if not (a[0] == chain and a[1] == resnum)]
    if not mutated or not others:
        return (9e9, "")
    best = min(
        (math.dist(m[3:], o[3:]), f"{m[2]}--{o[0]}{o[1]}:{o[2]}")
        for m in mutated for o in others
    )
    return best


def main() -> None:
    rows = [r for r in csv.DictReader(MUTANTS_CSV.open(newline="")) if r["Binder_Sequence_mutee"]]
    print(f"{len(rows)} mutants a greffer, {ROTAMERS_KEPT} rotameres chacun")
    print()

    records = []
    for row in rows:
        design, mutation = row["design"], row["mutation"]
        found = re.fullmatch(r"([A-Z])(\d+)([DE])", mutation)
        if not found:
            print(f"  {design} : mutation illisible {mutation!r}, ignore")
            continue
        _, resnum, new_aa = found.groups()
        parent = parent_structure(design)
        if parent is None:
            print(f"  {design} : parent introuvable, ignore")
            continue
        chain = binder_chain(parent)
        stem = f"{design}__{mutation}"
        print(f"  {design[-28:]:<30} {mutation:<6} chaine {chain} res {resnum}")

        states = thread(parent, chain, int(resnum), new_aa, stem)
        if not states:
            continue
        scored = []
        for path in states:
            distance, partner = worst_clash(path, chain, int(resnum))
            scored.append((distance, path, partner))
        # Le contact le plus LARGE est le rotamere le moins encombre.
        scored.sort(key=lambda item: -item[0])
        for rank, (distance, path, partner) in enumerate(scored, start=1):
            if rank > ROTAMERS_KEPT:
                path.unlink(missing_ok=True)
                continue
            final = OUT_DIR / f"{stem}__rot{rank}.pdb"
            path.replace(final)
            clash = distance < CLASH_A
            print(f"      rot{rank}  contact min {distance:5.2f} A ({partner})"
                  f"{'   *** CLASH ***' if clash else ''}")
            records.append({
                "design": design, "mutation": mutation, "rotamer": rank,
                "contact_min_A": round(distance, 2), "partenaire": partner,
                "clash": "oui" if clash else "non", "path": str(final),
            })
        print()

    summary = OUT_DIR / "threaded_index.csv"
    with summary.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(records[0]) if records else ["design"])
        writer.writeheader()
        writer.writerows(records)
    clashing = [r for r in records if r["clash"] == "oui"]
    print(f"-> {len(records)} structures ecrites, index dans {summary}")
    print(f"   avec clash (< {CLASH_A} A) : {len(clashing)}")
    for r in clashing:
        print(f"     {r['design'][-26:]} {r['mutation']} rot{r['rotamer']} "
              f"{r['contact_min_A']} A sur {r['partenaire']}")


if __name__ == "__main__":
    main()
