#!/usr/bin/env python3
"""Cross-réactivité souris MESURÉE : pose et mécanisme pH sur Q01279 contre P00533.

    modal volume get bindcraft 'boltz/mouse01' out/
    uv run --with biopython --with scipy --with propka --with gemmi python cross_species.py

CE QUE CE SCRIPT REMPLACE. L'objectif n°2 du challenge — la même séquence doit reconnaître
P00533 et Q01279 — n'était approché que par un proxy de séquence : la fraction des résidus de
cible contactés identiques chez la souris. Ce proxy ignore la conformation locale murine et ne
dit rien du mécanisme pH chez la souris.

DEUX CONTRÔLES QUI RENDENT LA COMPARAISON LISIBLE.

1. MÊME PRÉDICTEUR DES DEUX CÔTÉS. Le ΔpKa humain publié jusqu'ici vient des structures
   AF2/BindCraft. Le comparer à un ΔpKa murin issu de Boltz-2 mélangerait l'effet d'espèce et
   l'effet de prédicteur. On recalcule donc PROPKA sur les structures Boltz HUMAINES aussi, et
   la comparaison d'espèce se fait Boltz contre Boltz. Le ΔpKa humain sur AF2 reste rapporté à
   côté : sa concordance avec le ΔpKa humain sur Boltz est une vérification de robustesse du
   résultat principal au choix du prédicteur.

2. NUMÉROTATION REMISE EN PDB. Boltz renumérote chaque chaîne à partir de 1. Comme il n'y a
   AUCUN indel entre humain et souris dans la fenêtre 309–506 (vérifié par mouse_target.py),
   l'offset +308 s'applique aux deux espèces et « 409 » désigne le même résidu partout. On
   refuse de calculer si le résidu 409 obtenu n'est pas une histidine.

Sortie : out/cross_species.csv
"""

from __future__ import annotations

import csv
import json
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path

import gemmi

from contact_recovery import CONTACT_CUTOFF_A, contacts, load, residues, sequence

warnings.filterwarnings("ignore")

AF2_DIR = Path("structures/wt")
HUMAN_BOLTZ = Path("out/rescore01")
MOUSE_BOLTZ = Path("out/mouse01")
MOUSE_FASTA = Path("inputs/mEGFR_dIII.fasta")
PROPKA_H409 = Path("out/propka_h409.csv")
OUT = Path("out/cross_species.csv")

TARGET_CHAIN, BINDER_CHAIN = "A", "B"
TARGET_HIS = 409
DOMAIN_START = 309

# Décalage de la numérotation Boltz (1..198) vers la numérotation PDB (309..506).
PDB_OFFSET = DOMAIN_START - 1

PH_ACID, PH_NEUTRAL = 6.5, 7.4
PKA_NOISE = 0.5

SUMMARY_LINE = __import__("re").compile(
    r"^\s*([A-Z][A-Z0-9]{1,3})\s+(-?\d+)\s+([A-Za-z0-9])\s+(-?\d+\.\d+)\s+(-?\d+\.\d+)"
)


def mouse_sequence() -> str:
    lines = [l.strip() for l in MOUSE_FASTA.read_text().splitlines() if l.strip()]
    return "".join(l for l in lines if not l.startswith(">")).upper()


def renumber(source: Path, destination: Path) -> gemmi.Structure:
    """Remet la chaîne cible en numérotation PDB 309–506, et VÉRIFIE le résidu 409."""
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    model = structure[0]
    for residue in model[TARGET_CHAIN]:
        residue.seqid.num += PDB_OFFSET
    found = {r.seqid.num: r.name for r in model[TARGET_CHAIN]}
    if found.get(TARGET_HIS) != "HIS":
        raise ValueError(
            f"{source.name} : apres renumerotation, {TARGET_CHAIN}{TARGET_HIS} = "
            f"{found.get(TARGET_HIS)}, attendu HIS"
        )
    destination.parent.mkdir(parents=True, exist_ok=True)
    structure.write_pdb(str(destination))
    return structure


def target_only(source: Path, destination: Path) -> None:
    structure = gemmi.read_structure(str(source))
    structure.setup_entities()
    model = structure[0]
    for name in [c.name for c in model]:
        if name != TARGET_CHAIN:
            model.remove_chain(name)
    structure.write_pdb(str(destination))


def run_propka(path: Path) -> dict | None:
    with tempfile.TemporaryDirectory() as workdir:
        work = Path(workdir)
        local = work / path.name
        shutil.copy(path, local)
        subprocess.run(["propka3", local.name], cwd=work,
                       capture_output=True, text=True, timeout=1800)
        output = work / f"{local.stem}.pka"
        if not output.is_file():
            return None
        text = output.read_text()
    groups, inside = {}, False
    for line in text.splitlines():
        if "SUMMARY OF THIS PREDICTION" in line:
            inside = True
            continue
        if inside:
            if line.strip().startswith("-") or "Free energy" in line:
                break
            found = SUMMARY_LINE.match(line)
            if found:
                name, number, chain, pka, _ = found.groups()
                groups[(name, int(number), chain)] = float(pka)
    return groups or None


def selectivity(bound: float, free: float) -> float:
    def fraction(pka: float) -> float:
        return (1 + 10 ** (pka - PH_ACID)) / (1 + 10 ** (pka - PH_NEUTRAL))
    return fraction(bound) / fraction(free)


def best_sample(folder: Path) -> Path | None:
    """L'échantillon de plus haut iptm. Un seul suffit pour PROPKA, qui est coûteux."""
    best, best_value = None, -1.0
    for confidence in folder.glob("confidence_*.json"):
        value = json.loads(confidence.read_text()).get("iptm")
        if value is None:
            continue
        stem = confidence.name.replace("confidence_", "").replace(".json", "")
        structure = folder / f"{stem}.cif"
        if structure.is_file() and value > best_value:
            best, best_value = structure, value
    return best


def iptm_mean(folder: Path) -> float | str:
    values = [
        json.loads(p.read_text()).get("iptm") for p in folder.glob("confidence_*.json")
    ]
    values = [v for v in values if v is not None]
    return round(sum(values) / len(values), 3) if values else ""


def ph_mechanism(folder: Path, scratch: Path, label: str) -> dict:
    """ΔpKa de H409 sur la meilleure pose Boltz d'un dossier."""
    structure = best_sample(folder)
    if structure is None:
        return {}
    complexe = scratch / f"{label}_complexe.pdb"
    try:
        renumber(structure, complexe)
    except (ValueError, KeyError) as problem:
        print(f"    {problem}")
        return {}
    alone = scratch / f"{label}_cible.pdb"
    target_only(complexe, alone)

    bound = run_propka(complexe)
    free = run_propka(alone)
    if not bound or not free:
        return {}
    key = ("HIS", TARGET_HIS, TARGET_CHAIN)
    if key not in bound or key not in free:
        return {}
    return {
        "pKa_lie": round(bound[key], 2),
        "pKa_libre": round(free[key], 2),
        "dpKa": round(bound[key] - free[key], 2),
        "facteur": round(selectivity(bound[key], free[key]), 3),
    }


def main() -> None:
    if not MOUSE_BOLTZ.is_dir():
        raise SystemExit(
            f"{MOUSE_BOLTZ} absent. Rapatrier d'abord :\n"
            f"  modal volume get bindcraft 'boltz/mouse01' out/"
        )
    mouse = mouse_sequence()
    af2_shift = {
        r["design"]: float(r["dpKa_H409"])
        for r in csv.DictReader(PROPKA_H409.open(newline="")) if r["type"] == "WT"
    }

    scratch = Path(tempfile.mkdtemp(prefix="cross_"))
    rows = []
    designs = sorted(p.name for p in MOUSE_BOLTZ.iterdir() if p.is_dir())
    print(f"{len(designs)} designs, cible murine {len(mouse)} residus")
    print()

    for index, design in enumerate(designs, start=1):
        mouse_folder = MOUSE_BOLTZ / design
        human_folder = HUMAN_BOLTZ / design
        af2_path = AF2_DIR / f"{design}.pdb"
        if not (human_folder.is_dir() and af2_path.is_file()):
            print(f"  {design} : contrepartie humaine absente, ignore")
            continue

        print(f"[{index}/{len(designs)}] {design[-30:]}")

        # --- pose : les contacts murins retrouvent-ils l'epitope humain ? ---
        af2_model = load(af2_path)
        reference = contacts(af2_model, 0, 0)
        recoveries = []
        for structure_path in sorted(mouse_folder.glob("*.cif")):
            model = load(structure_path)
            # Aucun indel : l'offset PDB est le meme pour les deux especes, donc les
            # numeros de residus sont directement comparables.
            if len(residues(model, TARGET_CHAIN)) != len(residues(af2_model, TARGET_CHAIN)):
                continue
            if sequence(model, BINDER_CHAIN) != sequence(af2_model, BINDER_CHAIN):
                print("    binder different — compare refuse")
                continue
            predicted = contacts(model, PDB_OFFSET, 0)
            recoveries.append(len(reference & predicted) / len(reference))

        # --- mecanisme pH, Boltz des deux cotes ---
        human_ph = ph_mechanism(human_folder, scratch, f"{design}_h")
        mouse_ph = ph_mechanism(mouse_folder, scratch, f"{design}_m")

        row = {
            "design": design,
            "squelette": design.split("_")[-2],
            "iptm_humain": iptm_mean(human_folder),
            "iptm_souris": iptm_mean(mouse_folder),
            "recuperation_epitope_souris": round(max(recoveries), 3) if recoveries else "",
            "recuperation_souris_min": round(min(recoveries), 3) if recoveries else "",
            "dpKa_humain_AF2": af2_shift.get(design, ""),
            "dpKa_humain_Boltz": human_ph.get("dpKa", ""),
            "dpKa_souris_Boltz": mouse_ph.get("dpKa", ""),
            "facteur_humain_Boltz": human_ph.get("facteur", ""),
            "facteur_souris_Boltz": mouse_ph.get("facteur", ""),
            "pKa_H409_lie_souris": mouse_ph.get("pKa_lie", ""),
            "pKa_H409_libre_souris": mouse_ph.get("pKa_libre", ""),
        }
        if isinstance(row["iptm_humain"], float) and isinstance(row["iptm_souris"], float):
            row["delta_iptm"] = round(row["iptm_souris"] - row["iptm_humain"], 3)
        else:
            row["delta_iptm"] = ""
        if isinstance(row["dpKa_humain_Boltz"], float) and \
           isinstance(row["dpKa_souris_Boltz"], float):
            row["delta_dpKa_souris_vs_humain"] = round(
                row["dpKa_souris_Boltz"] - row["dpKa_humain_Boltz"], 2
            )
        else:
            row["delta_dpKa_souris_vs_humain"] = ""

        mechanism_mouse = (
            isinstance(mouse_ph.get("dpKa"), float) and mouse_ph["dpKa"] > PKA_NOISE
        )
        mechanism_human = (
            isinstance(human_ph.get("dpKa"), float) and human_ph["dpKa"] > PKA_NOISE
        )
        row["mecanisme_humain_Boltz"] = "oui" if mechanism_human else "non"
        row["mecanisme_souris_Boltz"] = "oui" if mechanism_mouse else "non"
        row["mecanisme_conserve"] = (
            "oui" if mechanism_human and mechanism_mouse
            else ("perdu chez la souris" if mechanism_human else "sans objet")
        )
        rows.append(row)
        print(f"    iptm  humain {row['iptm_humain']}  souris {row['iptm_souris']}  "
              f"(delta {row['delta_iptm']})")
        print(f"    epitope retrouve chez la souris : "
              f"{row['recuperation_epitope_souris']}")
        print(f"    dpKa H409  AF2-humain {row['dpKa_humain_AF2']}  "
              f"Boltz-humain {row['dpKa_humain_Boltz']}  "
              f"Boltz-souris {row['dpKa_souris_Boltz']}")

    shutil.rmtree(scratch, ignore_errors=True)
    if not rows:
        raise SystemExit("aucun resultat")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    print()
    print(f"-> {len(rows)} lignes dans {OUT}")

    conserved = [r for r in rows if r["mecanisme_conserve"] == "oui"]
    lost = [r for r in rows if r["mecanisme_conserve"] == "perdu chez la souris"]
    print(f"   mecanisme pH conserve chez la souris : {len(conserved)}")
    print(f"   mecanisme pH perdu chez la souris    : {len(lost)}")
    deltas = [r["delta_iptm"] for r in rows if isinstance(r["delta_iptm"], float)]
    if deltas:
        print(f"   delta iptm souris-humain : {min(deltas):+.3f} a {max(deltas):+.3f}, "
              f"moyen {sum(deltas) / len(deltas):+.3f}")
    recoveries = [
        r["recuperation_epitope_souris"] for r in rows
        if isinstance(r["recuperation_epitope_souris"], float)
    ]
    if recoveries:
        print(f"   epitope retrouve chez la souris : {min(recoveries):.3f} a "
              f"{max(recoveries):.3f}")


if __name__ == "__main__":
    main()
