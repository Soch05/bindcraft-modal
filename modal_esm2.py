#!/usr/bin/env python3
"""Pseudo-vraisemblance ESM-2 des séquences du vivier. COLONNE DESCRIPTIVE, HORS CLASSEMENT.

    modal run modal_esm2.py::score

⚠️ CETTE MÉTRIQUE N'ENTRE DANS AUCUN ORDRE, AUCUN GROUPE, AUCUN DÉPARTAGE. Ce n'est pas une
précaution de style, c'est la conséquence de ce qu'elle mesure.

La pseudo-vraisemblance ESM-2 quantifie à quel point une séquence ressemble aux protéines
NATURELLES sur lesquelles le modèle a été entraîné. Or nos binders sont de novo, et le
règlement du challenge EXIGE une « diversité de séquence et de structure suffisante par
rapport aux protéines connues ». Ne pas ressembler au naturel est donc le cahier des charges,
pas un défaut. Classer sur le PLL favoriserait mécaniquement les designs les PLUS proches du
naturel, c'est-à-dire les MOINS nouveaux — exactement ce que le règlement pénalise.

À quoi elle sert quand même : un PLL très bas par rapport au reste du lot peut signaler une
séquence pathologique (composition aberrante, répétitions, région désordonnée improbable).
C'est un détecteur d'anomalie, lu en relatif à l'intérieur du lot, et rien d'autre.

MÉTHODE : pseudo-vraisemblance par marginales masquées. Pour chaque position, on masque le
résidu et on lit la log-probabilité que le modèle attribue au résidu réellement présent. Le
PLL est la moyenne de ces log-probabilités — normalisée par la longueur, sans quoi les
séquences courtes paraîtraient systématiquement meilleures.

Sortie : out/esm2_pll.csv
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modal import App, Image, Volume

GPU = os.environ.get("GPU", "L40S")
USD_PER_HOUR = {"L40S": 1.95}
BUDGET_USD = float(os.environ.get("BUDGET_USD", 1.0))

# 650M paramètres : le point d'équilibre habituel d'ESM-2. Le 3B n'apporterait rien à un
# usage purement descriptif, et le 35M donnerait un PLL trop bruité pour servir même de
# détecteur d'anomalie.
MODEL = "facebook/esm2_t33_650M_UR50D"

# Positions masquées par lot. Une séquence de 94 résidus tient largement sur une L40S.
BATCH = 32

WEIGHTS = "/weights"


def budget_timeout_seconds() -> int:
    rate = USD_PER_HOUR.get(GPU)
    if rate is None:
        raise ValueError(
            f"tarif horaire inconnu pour GPU={GPU} : ajouter la valeur mesuree dans "
            f"USD_PER_HOUR avant de lancer"
        )
    return max(300, min(int(3600 * BUDGET_USD / rate), 86400))


image = (
    Image.debian_slim(python_version="3.12")
    .pip_install("torch", "transformers", "numpy<3")
    .env({"HF_HOME": WEIGHTS})
)

app = App("esm2-egfr")
weights_volume = Volume.from_name("esm2-weights", create_if_missing=True)


@app.function(
    image=image, gpu=GPU, volumes={WEIGHTS: weights_volume},
    timeout=budget_timeout_seconds(),
)
def pll(sequences: list[dict]) -> str:
    import torch
    from transformers import AutoModelForMaskedLM, AutoTokenizer

    if not torch.cuda.is_available():
        raise RuntimeError("pas de GPU CUDA")

    tokenizer = AutoTokenizer.from_pretrained(MODEL, cache_dir=WEIGHTS)
    model = AutoModelForMaskedLM.from_pretrained(MODEL, cache_dir=WEIGHTS).cuda().eval()
    weights_volume.commit()

    results = []
    for index, record in enumerate(sequences, start=1):
        sequence = record["sequence"]
        encoded = tokenizer(sequence, return_tensors="pt")
        ids = encoded["input_ids"].cuda()
        # Les positions de résidus, sans les jetons spéciaux de début et de fin.
        positions = list(range(1, ids.shape[1] - 1))

        total = 0.0
        with torch.no_grad():
            for start in range(0, len(positions), BATCH):
                chunk = positions[start:start + BATCH]
                batch = ids.repeat(len(chunk), 1)
                for row, position in enumerate(chunk):
                    batch[row, position] = tokenizer.mask_token_id
                logits = model(input_ids=batch).logits
                for row, position in enumerate(chunk):
                    distribution = torch.log_softmax(logits[row, position], dim=-1)
                    total += float(distribution[ids[0, position]])

        results.append({
            "design_id": record["design_id"],
            "longueur": len(sequence),
            "ESM2_PLL_par_residu": round(total / len(positions), 4),
            "ESM2_PLL_total": round(total, 2),
        })
        print(f"[{index}/{len(sequences)}] {record['design_id'][-34:]:<36} "
              f"PLL/residu {results[-1]['ESM2_PLL_par_residu']:8.4f}", flush=True)
    return json.dumps(results)


@app.local_entrypoint()
def score() -> None:
    import csv

    master = Path("out/master_rank.csv")
    if not master.is_file():
        raise SystemExit(f"{master} absent — lancer rank_designs.py d'abord")
    rows = list(csv.DictReader(master.open(newline="")))
    sequences = [
        {"design_id": r["design_id"] if r["type"] != "mutant"
         else f"{r['design_id']}__{r['mutations']}",
         "sequence": r["sequence"].strip().upper()}
        for r in rows
    ]
    seen, unique = set(), []
    for item in sequences:
        if item["sequence"] in seen:
            continue
        seen.add(item["sequence"])
        unique.append(item)

    print(f"{len(unique)} sequences uniques, modele {MODEL}")
    print(f"GPU {GPU} | budget ${BUDGET_USD:.2f}")
    print("RAPPEL : metrique DESCRIPTIVE, elle n'entre dans aucun classement.")
    results = json.loads(pll.remote(unique))
    out = Path("out/esm2_pll.csv")
    with out.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(results[0]))
        writer.writeheader()
        writer.writerows(results)
    values = [r["ESM2_PLL_par_residu"] for r in results]
    print()
    print(f"-> {len(results)} lignes dans {out}")
    print(f"   PLL/residu : {min(values):.4f} a {max(values):.4f}, "
          f"median {sorted(values)[len(values) // 2]:.4f}")
