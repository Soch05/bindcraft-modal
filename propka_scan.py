#!/usr/bin/env python3
"""PROPKA sur tout le vivier : pKa de H409 lié vs libre, pKa des D/E greffés, scan non biaisé.

    uv run --with propka --with gemmi python propka_scan.py

CE QUE PROPKA CALCULE, ET POURQUOI C'EST LE BON OUTIL ICI. Le pKa d'un groupe ionisable
dépend de son environnement : un carboxylate enfoui loin de l'eau paie un coût de
désolvatation qui FAIT MONTER son pKa, et une charge positive voisine le FAIT BAISSER.
PROPKA est un modèle empirique de ces termes. L'analogie ML : c'est une régression sur des
descripteurs géométriques, pas une simulation — rapide, et calibrée sur des pKa mesurés.

POURQUOI LIÉ **ET** LIBRE. Un pKa absolu ne dit rien sur la sélectivité pH. Ce qui la crée,
c'est le DÉCALAGE de pKa de H409 entre la cible seule et la cible liée au binder. Si lier le
binder fait monter le pKa de H409, alors H409 reste protonée plus haut en pH une fois liée :
la liaison stabilise la forme protonée, donc elle est favorisée à pH acide. C'est exactement
le mécanisme « Route 2 » recherché.

LE PIÈGE DU CARBOXYLATE AUTO-DESTRUCTEUR. Le D/E du binder doit être CHARGÉ à pH 6,5 pour
former le pont salin. S'il est enfoui contre une charge positive, son propre pKa monte ; au
dessus de ~5,0 il est partiellement protoné dès pH 6,5, perd sa charge là où on en a besoin,
et le mécanisme s'auto-détruit. On rapporte donc son pKa sur CHAQUE rotamère.

CALIBRATION — À NE PAS OUBLIER. PROPKA se trompe couramment d'une unité de pKa, davantage sur
les gros décalages. Les valeurs servent à CLASSER, pas à annoncer un facteur de sélectivité
absolu. Aucun facteur absolu ne doit entrer dans la soumission.

Sortie : out/propka_h409.csv, out/propka_acids.csv, out/propka_scan.csv
"""

from __future__ import annotations

import csv
import multiprocessing
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

import gemmi

STRUCT_INDEX = Path("out/structures_index.csv")
THREADED_INDEX = Path("structures/threaded/threaded_index.csv")
OUT_H409 = Path("out/propka_h409.csv")
OUT_ACIDS = Path("out/propka_acids.csv")
OUT_SCAN = Path("out/propka_scan.csv")
# Toutes les tables brutes, tous les groupes de toutes les structures. Sert à calculer le
# facteur de sélectivité GLOBAL, qui somme la contribution de chaque groupe ionisable au
# lieu de ne regarder que H409 — voir rank_designs.py.
OUT_ALL = Path("out/propka_all_groups.csv")

TARGET_CHAIN = "A"
TARGET_HIS = 409

# Les deux pH du règlement : « binds human EGFR at pH 6.5 and shows no detectable binding
# at pH 7.4 ». Ce sont eux, et pas d'autres, qui entrent dans le facteur de sélectivité.
PH_ACID = 6.5
PH_NEUTRAL = 7.4

# Au-dessus de ce pKa, un carboxylate est partiellement protoné dès pH 6,5 et le mécanisme
# s'auto-détruit. Seuil du plan de travail.
CARBOXYLATE_SELF_DEFEAT_PKA = 5.0

# Un décalage de pKa inférieur à ça est dans le bruit de PROPKA. Sert au scan non biaisé
# ET au verdict de stabilité entre rotamères.
PKA_SHIFT_NOISE = 0.5

WORKERS = max(1, (multiprocessing.cpu_count() or 2) - 1)

SUMMARY_LINE = re.compile(
    r"^\s*([A-Z][A-Z0-9]{1,3})\s+(-?\d+)\s+([A-Za-z0-9])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


def extract_chains(source: Path, keep: set[str], destination: Path) -> bool:
    """Écrit un PDB ne contenant que `keep`, depuis le MÊME fichier source.

    Extraire la cible du fichier du complexe et non d'un autre fichier garantit que la
    comparaison lié/libre ne porte que sur la présence du partenaire, pas sur une
    différence de conformation.
    """
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    model = structure[0]
    for name in [c.name for c in model]:
        if name not in keep:
            model.remove_chain(name)
    if not len(model):
        return False
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(destination))
    return True


def run_propka(path: Path) -> dict[tuple[str, int, str], tuple[float, float]] | None:
    """Lance propka3 dans un dossier jetable et renvoie {(resname, resnum, chain): (pKa, modele)}.

    Dossier jetable parce que propka3 écrit son .pka à côté de son entrée et pollue sinon
    l'arborescence des structures.
    """
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        local = work / path.name
        shutil.copy(path, local)
        completed = subprocess.run(
            ["propka3", local.name],
            cwd=work, capture_output=True, text=True, timeout=1200,
        )
        output = work / f"{local.stem}.pka"
        if not output.is_file():
            tail = (completed.stderr or completed.stdout).strip()[-200:]
            print(f"    ECHEC propka {path.name} : {tail}")
            return None
        text = output.read_text()

    groups: dict[tuple[str, int, str], tuple[float, float]] = {}
    inside = False
    for line in text.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            inside = True
            continue
        if inside:
            if line.strip().startswith("-") or "Free energy" in line:
                break
            found = SUMMARY_LINE.match(line)
            if found:
                name, number, chain, pka, model = found.groups()
                groups[(name, int(number), chain)] = (float(pka), float(model))
    return groups or None


def selectivity(pka_bound: float, pka_free: float) -> float:
    """Ka(6,5)/Ka(7,4) rapporté à l'état libre, formule du plan de travail.

    Interprétation : combien de fois la liaison est plus favorable à pH 6,5 qu'à pH 7,4,
    du seul fait du décalage de pKa de H409. 1,0 = aucune sélectivité.
    """
    def fraction(pka: float) -> float:
        return (1 + 10 ** (pka - PH_ACID)) / (1 + 10 ** (pka - PH_NEUTRAL))
    return fraction(pka_bound) / fraction(pka_free)


def job(task: tuple[str, str, str, str]) -> tuple[str, str, dict | None]:
    key, label, path, _ = task
    groups = run_propka(Path(path))
    print(f"    {label:<58} {'ok' if groups else 'ECHEC'} "
          f"({len(groups) if groups else 0} groupes)")
    return (key, label, groups)


def build_tasks(scratch: Path) -> list[tuple[str, str, str, str]]:
    """Trois jeux de structures : complexe, cible seule, binder seul.

    Le binder seul est nécessaire au scan non biaisé : sans lui, les groupes ionisables du
    binder n'ont aucun état « libre » auquel comparer leur pKa lié.
    """
    tasks = []
    index = {r["design"]: r for r in csv.DictReader(STRUCT_INDEX.open(newline=""))}

    for design, record in index.items():
        binder = record["chaine_binder"]
        tasks.append((f"WT|{design}|complexe", f"WT {design[-26:]} complexe",
                      record["complexe_pdb"], ""))
        tasks.append((f"WT|{design}|cible", f"WT {design[-26:]} cible seule",
                      record["cible_seule_pdb"], ""))
        alone = scratch / "binder_only" / f"{design}.pdb"
        if extract_chains(Path(record["complexe_pdb"]), {binder}, alone):
            tasks.append((f"WT|{design}|binder", f"WT {design[-26:]} binder seul",
                          str(alone), ""))

    for record in csv.DictReader(THREADED_INDEX.open(newline="")):
        design, mutation, rotamer = record["design"], record["mutation"], record["rotamer"]
        stem = f"{design}__{mutation}__rot{rotamer}"
        binder = index[design]["chaine_binder"]
        tasks.append((f"MUT|{stem}|complexe", f"MUT {mutation} rot{rotamer} "
                      f"{design[-20:]} complexe", record["path"], ""))
        alone = scratch / "binder_only" / f"{stem}.pdb"
        if extract_chains(Path(record["path"]), {binder}, alone):
            tasks.append((f"MUT|{stem}|binder", f"MUT {mutation} rot{rotamer} "
                          f"{design[-20:]} binder seul", str(alone), ""))
    return tasks


def main() -> None:
    scratch = Path(tempfile.mkdtemp(prefix="propka_"))
    tasks = build_tasks(scratch)
    print(f"{len(tasks)} runs PROPKA, {WORKERS} en parallele")
    print()

    with multiprocessing.Pool(WORKERS) as pool:
        done = pool.map(job, tasks)
    results = {key: groups for key, _, groups in done if groups}
    shutil.rmtree(scratch, ignore_errors=True)
    print()
    print(f"-> {len(results)}/{len(tasks)} runs exploitables")

    threaded = list(csv.DictReader(THREADED_INDEX.open(newline="")))
    struct = {r["design"]: r for r in csv.DictReader(STRUCT_INDEX.open(newline=""))}

    h409_rows, acid_rows, scan_rows = [], [], []

    def h409_of(groups: dict) -> float | None:
        hit = groups.get(("HIS", TARGET_HIS, TARGET_CHAIN))
        return hit[0] if hit else None

    # --- H409 : lié vs libre, pour les WT et pour chaque rotamère de mutant ---
    for design in struct:
        free = results.get(f"WT|{design}|cible")
        pka_free = h409_of(free) if free else None

        entries = [(f"WT|{design}|complexe", "WT", "", 0)]
        entries += [
            (f"MUT|{design}__{r['mutation']}__rot{r['rotamer']}|complexe",
             "mutant", r["mutation"], int(r["rotamer"]))
            for r in threaded if r["design"] == design
        ]
        for key, kind, mutation, rotamer in entries:
            groups = results.get(key)
            pka_bound = h409_of(groups) if groups else None
            if pka_bound is None or pka_free is None:
                continue
            h409_rows.append({
                "design": design, "squelette": design.split("_")[-2],
                "type": kind, "mutation": mutation, "rotamer": rotamer,
                "pKa_H409_lie": round(pka_bound, 2),
                "pKa_H409_libre": round(pka_free, 2),
                "dpKa_H409": round(pka_bound - pka_free, 2),
                "facteur_pH_predit": round(selectivity(pka_bound, pka_free), 3),
            })

    # --- pKa du D/E greffé, par rotamère : le piège auto-destructeur ---
    for record in threaded:
        design, mutation, rotamer = record["design"], record["mutation"], record["rotamer"]
        stem = f"{design}__{mutation}__rot{rotamer}"
        groups = results.get(f"MUT|{stem}|complexe")
        if not groups:
            continue
        resnum = int(mutation[1:-1])
        resname = "ASP" if mutation[-1] == "D" else "GLU"
        binder = struct[design]["chaine_binder"]
        hit = groups.get((resname, resnum, binder))
        if not hit:
            continue
        acid_rows.append({
            "design": design, "squelette": design.split("_")[-2],
            "mutation": mutation, "rotamer": int(rotamer),
            "residu": f"{resname}{resnum}",
            "pKa_DE": round(hit[0], 2),
            "pKa_modele": round(hit[1], 2),
            "decalage_vs_modele": round(hit[0] - hit[1], 2),
            "clash_threading": record["clash"],
            "auto_destructeur": "oui" if hit[0] > CARBOXYLATE_SELF_DEFEAT_PKA else "non",
        })

    # --- Scan non biaisé : tout groupe qui bouge de plus de PKA_SHIFT_NOISE ---
    for design in struct:
        binder = struct[design]["chaine_binder"]
        bound = results.get(f"WT|{design}|complexe")
        target = results.get(f"WT|{design}|cible")
        binder_alone = results.get(f"WT|{design}|binder")
        if not bound:
            continue
        for (name, number, chain), (pka, model) in bound.items():
            reference = target if chain == TARGET_CHAIN else binder_alone
            if not reference:
                continue
            free = reference.get((name, number, chain))
            if not free:
                continue
            shift = pka - free[0]
            if abs(shift) <= PKA_SHIFT_NOISE:
                continue
            scan_rows.append({
                "design": design, "squelette": design.split("_")[-2],
                "cote": "cible" if chain == TARGET_CHAIN else "binder",
                "groupe": f"{name}{number}{chain}",
                "pKa_lie": round(pka, 2), "pKa_libre": round(free[0], 2),
                "dpKa": round(shift, 2),
                "sens": "monte" if shift > 0 else "baisse",
            })

    # --- Tables brutes complètes : chaque groupe de chaque structure, sans filtre ---
    all_rows = []
    for key, groups in results.items():
        kind, stem, state = key.split("|")
        for (name, number, chain), (pka, model) in groups.items():
            all_rows.append({
                "cle": key, "type": kind, "structure": stem, "etat": state,
                "groupe": f"{name}{number}{chain}", "resname": name,
                "resnum": number, "chaine": chain,
                "pKa": round(pka, 2), "pKa_modele": round(model, 2),
            })

    for path, rows in ((OUT_H409, h409_rows), (OUT_ACIDS, acid_rows),
                       (OUT_SCAN, scan_rows), (OUT_ALL, all_rows)):
        if not rows:
            print(f"   (aucune ligne pour {path})")
            continue
        with path.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        print(f"   {len(rows):4d} lignes -> {path}")


if __name__ == "__main__":
    main()
