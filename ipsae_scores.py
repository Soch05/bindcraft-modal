#!/usr/bin/env python3
"""ipSAE sur les prédictions Boltz-2, via l'implémentation de RÉFÉRENCE de ses auteurs.

    modal volume get bindcraft 'boltz/pae02' out/
    uv run --with numpy --with scipy python ipsae_scores.py

POURQUOI CETTE MÉTRIQUE. Le règlement du challenge nomme `ipSAE` parmi les métriques
encouragées. Elle corrige un défaut connu de l'ipTM sur les interfaces petites : l'ipTM est
normalisé sur la taille du complexe entier, donc un binder de 60 résidus contre une cible de
198 voit sa contribution d'interface diluée. L'ipSAE restreint le calcul aux paires de
résidus réellement à l'interface et renormalise dessus.

POURQUOI L'IMPLÉMENTATION DES AUTEURS ET PAS LA MIENNE. La définition exacte d'ipSAE — le
choix de d0, la restriction par cutoff de PAE et de distance, la distinction entre scores
asymétriques et symétrisés — ne se devine pas. La réécrire de mémoire produirait un nombre
plausible appelé « ipSAE » sans en être un. Le script de référence est donc téléchargé, son
empreinte SHA256 enregistrée dans la sortie, et exécuté tel quel.

CUTOFFS : 10 pour le PAE et 15 A pour la distance, qui sont les valeurs des exemples de
l'outil. Elles ne sont pas réglées sur ce jeu de données.

Sortie : out/ipsae.csv
"""

from __future__ import annotations

import csv
import hashlib
import subprocess
import sys
import urllib.request
from pathlib import Path

BOLTZ_DIR = Path("out/pae02")
SCRIPT = Path("out/_ipsae_reference.py")
OUT = Path("out/ipsae.csv")

SOURCE_URL = "https://raw.githubusercontent.com/DunbrackLab/IPSAE/main/ipsae.py"
PAE_CUTOFF = "10"
DIST_CUTOFF = "15"

TARGET_CHAIN, BINDER_CHAIN = "A", "B"

# En-tête exact du fichier .txt produit par l'outil, utilisé pour indexer les colonnes par
# nom au lieu de compter des positions.
HEADER_FIELDS = [
    "Chn1", "Chn2", "PAE", "Dist", "Type", "ipSAE", "ipSAE_d0chn", "ipSAE_d0dom",
    "ipTM_af", "ipTM_d0chn", "pDockQ", "pDockQ2", "LIS", "n0res", "n0chn", "n0dom",
    "d0res", "d0chn", "d0dom", "nres1", "nres2", "dist1", "dist2", "Model",
]

# Colonnes RETENUES. Les autres sont calculees par l'outil mais inexploitables depuis une
# sortie Boltz-2, et les rapporter serait presenter une colonne non renseignee comme une
# mesure :
#   ipTM_af = 0,000 partout     l'outil attend un JSON AF2/AF3 pour lire l'ipTM
#   pDockQ, pDockQ2 constants   0,0183 et 0,0073 sur les 6 designs ; la branche Boltz ne
#                               fournit pas ce dont ces scores ont besoin
# Verifie en lisant le .txt brut, pas suppose.
KEEP = ["ipSAE", "ipSAE_d0chn", "ipSAE_d0dom", "LIS", "d0res", "nres1", "nres2"]

# Colonnes ecartees, conservees ici pour que la raison soit dans le code.
DISCARDED = {
    "ipTM_af": "vaut 0,000 : l'outil ne lit pas l'ipTM depuis une sortie Boltz-2",
    "pDockQ": "constant sur tous les designs : colonne non renseignee par la branche Boltz",
    "pDockQ2": "idem",
    "n0res": "compte de normalisation de d0, egal a la longueur de chaine alignee — ce "
             "n'est PAS un nombre de residus d'interface",
}


def reference_script() -> tuple[Path, str]:
    """Télécharge le script de référence si absent, et renvoie son empreinte."""
    if not SCRIPT.is_file():
        SCRIPT.parent.mkdir(parents=True, exist_ok=True)
        print(f"telechargement de {SOURCE_URL}")
        with urllib.request.urlopen(SOURCE_URL, timeout=120) as response:
            SCRIPT.write_bytes(response.read())
    digest = hashlib.sha256(SCRIPT.read_bytes()).hexdigest()
    print(f"script de reference : {SCRIPT}  sha256 {digest[:16]}...  "
          f"{SCRIPT.stat().st_size} octets")
    return (SCRIPT, digest)


def parse_summary(path: Path) -> list[dict]:
    """Lit le .txt de l'outil et renvoie une ligne par (paire de chaines, type)."""
    rows = []
    for line in path.read_text().splitlines():
        parts = line.split()
        if len(parts) < len(HEADER_FIELDS) or parts[0] == "Chn1":
            continue
        if parts[4] not in {"asym", "max"}:
            continue
        rows.append(dict(zip(HEADER_FIELDS, parts)))
    return rows


def run_one(structure: Path, pae: Path) -> list[dict]:
    completed = subprocess.run(
        [sys.executable, str(SCRIPT.resolve()), str(pae.resolve()),
         str(structure.resolve()), PAE_CUTOFF, DIST_CUTOFF],
        capture_output=True, text=True, timeout=900,
    )
    summary = structure.with_name(
        f"{structure.stem}_{PAE_CUTOFF}_{DIST_CUTOFF}.txt"
    )
    if not summary.is_file():
        tail = (completed.stderr or completed.stdout).strip()[-400:]
        print(f"    ECHEC sur {structure.name} : {tail}")
        return []
    return parse_summary(summary)


def main() -> None:
    if not BOLTZ_DIR.is_dir():
        raise SystemExit(
            f"{BOLTZ_DIR} absent. Rapatrier d'abord :\n"
            f"  modal volume get bindcraft 'boltz/pae02' out/"
        )
    _, digest = reference_script()
    print()

    results = []
    for folder in sorted(p for p in BOLTZ_DIR.iterdir() if p.is_dir()):
        design = folder.name
        samples = []
        for structure in sorted(folder.glob("*.cif")):
            stem = structure.stem
            candidates = [
                folder / f"pae_{stem}.npz",
                folder / f"pae_{stem}.npy",
            ]
            pae = next((c for c in candidates if c.is_file()), None)
            if pae is None:
                found = list(folder.glob(f"*{stem}*.npz"))
                pae = found[0] if found else None
            if pae is None:
                continue
            for record in run_one(structure, pae):
                if {record["Chn1"], record["Chn2"]} != {TARGET_CHAIN, BINDER_CHAIN}:
                    continue
                samples.append({"echantillon": stem, **record})

        if not samples:
            print(f"  {design[-26:]:<28} aucune matrice PAE exploitable")
            continue

        symmetric = [s for s in samples if s["Type"] == "max"] or samples
        values = {}
        for field in KEEP:
            numbers = []
            for item in symmetric:
                try:
                    numbers.append(float(item[field]))
                except (KeyError, ValueError):
                    continue
            if numbers:
                values[f"{field}_moyen"] = round(sum(numbers) / len(numbers), 4)
                values[f"{field}_max"] = round(max(numbers), 4)

        # Les deux directions asymetriques, pour que le choix du « max » soit visible.
        directions = {}
        for item in samples:
            if item["Type"] != "asym":
                continue
            key = f"ipSAE_{item['Chn1']}_vers_{item['Chn2']}"
            directions.setdefault(key, []).append(float(item["ipSAE"]))
        for key, numbers in directions.items():
            values[f"{key}_moyen"] = round(sum(numbers) / len(numbers), 4)

        results.append({
            "design": design,
            "n_echantillons": len(symmetric),
            "pae_cutoff": PAE_CUTOFF,
            "dist_cutoff": DIST_CUTOFF,
            "script_sha256": digest,
            "colonnes_ecartees": "; ".join(f"{k} : {v}" for k, v in DISCARDED.items()),
            **values,
        })
        print(f"  {design[-26:]:<28} ipSAE(max) {values.get('ipSAE_moyen', '?'):<9} "
              f"A->B {values.get('ipSAE_A_vers_B_moyen', '?'):<9} "
              f"B->A {values.get('ipSAE_B_vers_A_moyen', '?'):<9} "
              f"LIS {values.get('LIS_moyen', '?')}")

    if not results:
        raise SystemExit("aucun resultat — verifier que --write_full_pae etait actif")
    with OUT.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    print()
    print(f"-> {len(results)} designs dans {OUT}")
    scores = [r["ipSAE_moyen"] for r in results if "ipSAE_moyen" in r]
    if scores:
        print(f"   ipSAE : {min(scores):.4f} a {max(scores):.4f}, "
              f"median {sorted(scores)[len(scores) // 2]:.4f}")


if __name__ == "__main__":
    main()
