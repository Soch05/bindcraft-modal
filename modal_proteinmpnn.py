#!/usr/bin/env python3
"""Auto-cohérence de séquence : ProteinMPNN redessine-t-il la séquence qui a été soumise ?

    modal run modal_proteinmpnn.py::self_consistency

CE QUE MESURE L'AUTO-COHÉRENCE. On repart du squelette du design, on demande à un modèle de
repliement inverse quelles séquences il choisirait pour ce squelette, et on compare à la
séquence réellement soumise. Une récupération élevée dit que la séquence est proche d'un
optimum du modèle pour ce squelette ; une récupération basse dit que la séquence est un choix
inhabituel, donc plus fragile.

⚠️ CIRCULARITÉ À ÉNONCER, PAS À CACHER. BindCraft 2.0 utilise **ProteinMPNN** pour produire
ses séquences. Mesurer l'auto-cohérence avec ProteinMPNN revient donc à demander au
générateur s'il est d'accord avec lui-même. La récupération sera haute par construction, et
ce n'est PAS une validation independante — c'est exactement le meme defaut que les i_pTM
d'AF2, qui sont in-sample parce que BindCraft optimise a travers AF2.

Ce que la mesure garde d'utile malgre ca : une récupération ANORMALEMENT BASSE par rapport au
reste du lot signale une séquence que le modèle qui l'a pourtant produite ne rechoisirait pas,
donc un design sur un squelette difficile. C'est un detecteur de valeur aberrante, lu en
relatif a l'interieur du lot.

La vraie auto-cohérence STRUCTURALE de ce projet est ailleurs : c'est la récupération de
contacts par Boltz-2, un modèle qui n'a jamais servi a concevoir ces designs.

LE BINDER EST REDESSINE DANS SON CONTEXTE. La cible est passee en chaine fixe et non retiree :
BindCraft a lui aussi concu la sequence en presence de la cible, et redessiner le binder seul
poserait une autre question.

CPU ET NON GPU. ProteinMPNN sur un complexe de ~260 residus prend quelques secondes. Payer
une L40S pour ca serait du gaspillage.

Sortie : out/self_consistency.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modal import App, Image

# Variante « vanilla » par defaut de ProteinMPNN, celle sur laquelle il est evalue.
MODEL_NAME = os.environ.get("MPNN_MODEL", "v_48_020")
# 0,1 est la temperature usuelle pour demander « quelle sequence ce modele choisirait »,
# par opposition a un echantillonnage diversifiant.
SAMPLING_TEMP = os.environ.get("MPNN_TEMP", "0.1")
SAMPLES = int(os.environ.get("MPNN_SAMPLES", 8))
SEED = 37

TARGET_CHAIN, BINDER_CHAIN = "A", "B"
REPO = "/opt/ProteinMPNN"

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git")
    .pip_install("torch", "numpy<3")
    .run_commands(
        f"git clone --depth 1 https://github.com/dauparas/ProteinMPNN {REPO}",
        # Le build echoue expres si les poids ne sont pas la : sans eux le run tournerait
        # et rendrait des sequences aleatoires sans rien signaler.
        f"test -f {REPO}/vanilla_model_weights/{MODEL_NAME}.pt",
    )
)

app = App("proteinmpnn-egfr")


@app.function(image=image, timeout=3600, cpu=4.0, memory=8192)
def redesign(jobs: list[dict]) -> str:
    import re
    import subprocess
    import tempfile

    results = []
    for index, task in enumerate(jobs, start=1):
        name, sequence, pdb_text = task["name"], task["sequence"], task["pdb"]
        work = Path(tempfile.mkdtemp())
        pdb = work / f"{name[:60]}.pdb"
        pdb.write_text(pdb_text)
        out_dir = work / "out"
        out_dir.mkdir()

        completed = subprocess.run(
            ["python", f"{REPO}/protein_mpnn_run.py",
             "--pdb_path", str(pdb),
             "--pdb_path_chains", BINDER_CHAIN,
             "--out_folder", str(out_dir),
             "--num_seq_per_target", str(SAMPLES),
             "--sampling_temp", SAMPLING_TEMP,
             "--model_name", MODEL_NAME,
             "--seed", str(SEED),
             "--batch_size", "1"],
            capture_output=True, text=True, timeout=1800,
        )
        fasta = list(out_dir.rglob("*.fa")) + list(out_dir.rglob("*.fasta"))
        if completed.returncode != 0 or not fasta:
            tail = (completed.stderr or completed.stdout)[-600:]
            print(f"[{index}/{len(jobs)}] {name[-30:]} ECHEC\n{tail}", flush=True)
            results.append({"design": name, "ok": False, "erreur": tail[-400:]})
            continue

        text = fasta[0].read_text()
        entries = []
        header = None
        for line in text.splitlines():
            if line.startswith(">"):
                header = line
                continue
            if header is None or not line.strip():
                continue
            # La premiere entree du fasta est la sequence d'ENTREE, pas un echantillon.
            score = re.search(r"score=([0-9.]+)", header)
            entries.append({
                "sequence": line.strip().split("/")[-1],
                "score": float(score.group(1)) if score else None,
                "entree": "sample" not in header and "T=" not in header,
            })
            header = None

        samples = [e for e in entries if not e["entree"]] or entries[1:]
        recoveries = []
        for item in samples:
            candidate = item["sequence"]
            if len(candidate) != len(sequence):
                continue
            identical = sum(1 for a, b in zip(candidate, sequence) if a == b)
            recoveries.append(identical / len(sequence))

        if not recoveries:
            print(f"[{index}/{len(jobs)}] {name[-30:]} aucune sequence comparable",
                  flush=True)
            results.append({"design": name, "ok": False,
                            "erreur": "longueurs incomparables"})
            continue

        scores = [i["score"] for i in samples if i["score"] is not None]
        record = {
            "design": name, "ok": True,
            "n_echantillons": len(recoveries),
            "recuperation_moyenne": round(sum(recoveries) / len(recoveries), 4),
            "recuperation_max": round(max(recoveries), 4),
            "recuperation_min": round(min(recoveries), 4),
            "score_mpnn_moyen": round(sum(scores) / len(scores), 4) if scores else "",
            "modele": MODEL_NAME, "temperature": SAMPLING_TEMP,
        }
        results.append(record)
        print(f"[{index}/{len(jobs)}] {name[-30:]:<32} "
              f"recuperation {record['recuperation_moyenne']:.3f} "
              f"(min {record['recuperation_min']:.3f}, "
              f"max {record['recuperation_max']:.3f})", flush=True)
    return json.dumps(results)


@app.local_entrypoint()
def self_consistency(designs: str | None = None) -> None:
    import csv

    master = list(csv.DictReader(Path("out/master_rank.csv").open(newline="")))
    wanted = set(designs.split(",")) if designs else None
    jobs = []
    for record in master:
        if record["type"] != "bindcraft":
            continue
        name = record["design_id"]
        if wanted and name not in wanted:
            continue
        pdb = Path("structures/wt") / f"{name}.pdb"
        if not pdb.is_file():
            continue
        jobs.append({"name": name, "sequence": record["sequence"].strip().upper(),
                     "pdb": pdb.read_text()})
    if not jobs:
        raise SystemExit("aucun design a traiter")

    print(f"{len(jobs)} designs | modele {MODEL_NAME} | temperature {SAMPLING_TEMP} | "
          f"{SAMPLES} echantillons chacun")
    print("RAPPEL : BindCraft utilise ProteinMPNN pour produire ses sequences. Cette")
    print("mesure est donc PARTIELLEMENT CIRCULAIRE et ne vaut qu'en relatif.")
    print()
    results = json.loads(redesign.remote(jobs))
    ok = [r for r in results if r.get("ok")]
    out = Path("out/self_consistency.csv")
    if ok:
        with out.open("w", newline="") as handle:
            writer = csv.DictWriter(handle, fieldnames=list(ok[0]))
            writer.writeheader()
            writer.writerows(ok)
    print()
    print(f"-> {len(ok)}/{len(results)} designs dans {out}")
    values = [r["recuperation_moyenne"] for r in ok]
    if values:
        print(f"   recuperation : {min(values):.3f} a {max(values):.3f}, "
              f"median {sorted(values)[len(values) // 2]:.3f}")
