#!/usr/bin/env python3
"""Re-scoring orthogonal des designs par Boltz-2 sur Modal.

    modal run modal_boltz2.py::selfcheck            # À FAIRE AVANT TOUTE DÉPENSE
    modal run modal_boltz2.py::rescore              # les 23 WT
    modal run modal_boltz2.py::rescore --designs egfr-dIII-prod01_denovo_l94_...

POURQUOI UN SECOND PRÉDICTEUR. BindCraft optimise ses designs PAR DESCENTE DE GRADIENT À
TRAVERS AlphaFold2. Les i_pTM et i_pAE qu'il rapporte sont donc des scores « in-sample » :
le générateur a eu accès au juge pendant l'entraînement de chaque trajectoire. L'analogie ML
est directe — c'est une erreur d'apprentissage, pas une erreur de test. Demander à un modèle
d'architecture et de poids INDÉPENDANTS s'il place le binder au même endroit est le seul
moyen de savoir si la pose est réelle ou si elle exploite les biais d'AF2.

Boltz-2 est indépendant d'AF2 : architecture de type diffusion sur les coordonnées, poids
entraînés séparément. Son module d'AFFINITÉ n'est PAS utilisé ici — il est calibré pour les
petites molécules, pas pour les interfaces protéine–protéine.

MSA : PROFONDE POUR LA CIBLE, VIDE POUR LE BINDER. La MSA de la cible est précalculée une
fois par fetch_msa.py et embarquée dans l'image ; aucun appel réseau n'est fait depuis le
conteneur GPU. Le binder est de novo, sans homologue naturel : une recherche d'homologues sur
lui ne ramènerait que du bruit, et la nouveauté de séquence est exigée par le règlement.

TROIS ÉCHANTILLONS DE DIFFUSION et non trois graines de bout en bout : le tronc (MSA +
représentation de paires) est déterministe à graine fixée, donc le relancer trois fois
coûterait trois fois le calcul pour un tronc identique. Ce qui varie entre poses, et ce
qu'on veut échantillonner, est l'étape de diffusion. Écart assumé au plan, noté ici.
"""

from __future__ import annotations

import json
import os
from pathlib import Path

from modal import App, Image, Volume

# ----------------------------------------------------------------------------------------
# Budget — le timeout Modal est le seul vrai plafond de dépense
# ----------------------------------------------------------------------------------------

GPU = os.environ.get("GPU", "L40S")

# Même garde volontaire que modal_bindcraft2.py : un GPU dont le tarif mesuré n'est pas
# dans cette table fait échouer le chargement, pour qu'un budget en dollars reste un budget.
USD_PER_HOUR = {"L40S": 1.95}
BUDGET_USD = float(os.environ.get("BUDGET_USD", 4.0))

# Nombre de poses tirées par complexe à l'étape de diffusion.
DIFFUSION_SAMPLES = int(os.environ.get("DIFFUSION_SAMPLES", 3))

# Pas de recyclage exotique : les valeurs par défaut de Boltz-2 sont celles sur lesquelles
# il est évalué, et s'en écarter rendrait les scores incomparables à la littérature.
RECYCLING_STEPS = 3

OUTPUTS = "/outputs"
WEIGHTS = "/weights"


def budget_timeout_seconds() -> int:
    rate = USD_PER_HOUR.get(GPU)
    if rate is None:
        raise ValueError(
            f"tarif horaire inconnu pour GPU={GPU} : ajouter la valeur mesuree dans "
            f"USD_PER_HOUR avant de lancer, sinon le budget n'est pas un plafond"
        )
    seconds = int(3600 * BUDGET_USD / rate)
    return max(300, min(seconds, 86400))


TIMEOUT = budget_timeout_seconds()

# ----------------------------------------------------------------------------------------
# Image
# ----------------------------------------------------------------------------------------

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git")
    # Les wheels torch de PyPI pour linux x86_64 embarquent CUDA ; une L40S (compute
    # capability 8,9) est couverte. On épingle boltz pour que le run soit redecrit
    # exactement dans le dossier de méthodes.
    .pip_install("boltz==2.2.0", "numpy<3", "gemmi")
    .env({"BOLTZ_CACHE": WEIGHTS})
    # La MSA précalculée de la cible, embarquée : aucun appel à un serveur MSA distant
    # depuis le conteneur GPU, qui est la cause de blocage n°1 de cette étape.
    .add_local_file("inputs/egfr_dIII.a3m", remote_path="/root/inputs/egfr_dIII.a3m")
    # La MSA de la cible MURINE, pour la mesure de cross-reactivite (objectif n°2).
    .add_local_file("inputs/mEGFR_dIII.a3m", remote_path="/root/inputs/mEGFR_dIII.a3m")
)

app = App("boltz2-egfr")
weights_volume = Volume.from_name("boltz-weights", create_if_missing=True)
outputs_volume = Volume.from_name("bindcraft", create_if_missing=True)


# ----------------------------------------------------------------------------------------
# Validation avant dépense
# ----------------------------------------------------------------------------------------


@app.function(image=image, gpu=GPU, volumes={WEIGHTS: weights_volume}, timeout=1800)
def selfcheck() -> None:
    """Confirme un GPU CUDA visible et télécharge les poids UNE fois dans le Volume.

    Un build qui tombe sur le CPU ne se signale pas de lui-même : il est simplement cent
    fois plus lent. On vérifie donc explicitement, avant de payer quoi que ce soit.
    """
    import subprocess

    import torch

    report = {
        "torch": torch.__version__,
        "cuda_disponible": torch.cuda.is_available(),
        "n_gpu": torch.cuda.device_count(),
        "gpu": torch.cuda.get_device_name(0) if torch.cuda.is_available() else None,
        "capability": list(torch.cuda.get_device_capability(0))
        if torch.cuda.is_available() else None,
    }
    if not torch.cuda.is_available():
        raise RuntimeError(f"aucun GPU CUDA visible : {report}")

    version = subprocess.run(
        ["boltz", "--help"], capture_output=True, text=True, timeout=300
    )
    report["boltz_cli"] = "ok" if version.returncode == 0 else version.stderr[-200:]

    msa = Path("/root/inputs/egfr_dIII.a3m")
    report["msa_presente"] = msa.is_file()
    report["msa_sequences"] = msa.read_text().count(">") if msa.is_file() else 0

    cache = Path(WEIGHTS)
    cache.mkdir(parents=True, exist_ok=True)
    report["cache_avant"] = sorted(p.name for p in cache.iterdir())
    weights_volume.commit()
    print(json.dumps(report, indent=2))


# ----------------------------------------------------------------------------------------
# Prédiction
# ----------------------------------------------------------------------------------------


HUMAN_MSA = "/root/inputs/egfr_dIII.a3m"
MOUSE_MSA = "/root/inputs/mEGFR_dIII.a3m"


def write_yaml(directory: Path, name: str, target: str, binder: str,
               target_msa: str = HUMAN_MSA) -> Path:
    """Une entrée Boltz par complexe. `msa: empty` = mode séquence seule pour le binder.

    `target_msa` change avec l'espèce : la cible murine a sa propre MSA, et réutiliser celle
    de l'humain injecterait l'alignement de la mauvaise protéine dans la prédiction.
    """
    path = directory / f"{name}.yaml"
    path.write_text(
        "version: 1\n"
        "sequences:\n"
        "  - protein:\n"
        "      id: A\n"
        f"      sequence: {target}\n"
        f"      msa: {target_msa}\n"
        "  - protein:\n"
        "      id: B\n"
        f"      sequence: {binder}\n"
        "      msa: empty\n"
    )
    return path


@app.function(
    image=image,
    gpu=GPU,
    volumes={WEIGHTS: weights_volume, OUTPUTS: outputs_volume},
    timeout=TIMEOUT,
    cpu=4.0,
    memory=16384,
)
def predict(jobs: list[dict], run_name: str) -> str:
    """Prédit chaque complexe et renvoie les métriques de confiance.

    Les structures prédites sont écrites dans le Volume : la récupération de contacts de la
    phase 6 se calcule en local sur elles, pas ici, pour ne pas payer du GPU à faire de la
    géométrie.
    """
    import shutil
    import subprocess
    import time

    import torch

    if not torch.cuda.is_available():
        raise RuntimeError("pas de GPU CUDA — run interrompu avant toute depense utile")

    work = Path("/tmp/boltz_in")
    work.mkdir(parents=True, exist_ok=True)
    destination = Path(OUTPUTS) / "boltz" / run_name
    destination.mkdir(parents=True, exist_ok=True)

    results = []
    for index, task in enumerate(jobs, start=1):
        name = task["name"]
        started = time.time()
        for leftover in work.glob("*"):
            shutil.rmtree(leftover, ignore_errors=True) if leftover.is_dir() \
                else leftover.unlink()
        yaml_path = write_yaml(
            work, name, task["target"], task["binder"],
            task.get("target_msa", HUMAN_MSA),
        )
        out_dir = Path("/tmp/boltz_out") / name
        out_dir.mkdir(parents=True, exist_ok=True)

        command = [
            "boltz", "predict", str(yaml_path),
            "--out_dir", str(out_dir),
            "--cache", WEIGHTS,
            "--diffusion_samples", str(DIFFUSION_SAMPLES),
            "--recycling_steps", str(RECYCLING_STEPS),
            "--output_format", "mmcif",
            "--override",
            # Matrices PAE completes, necessaires au calcul d'ipSAE — metrique nommee
            # explicitement par le reglement du challenge. Le cout est quelques Mo par
            # complexe, ecrits dans le Volume.
            "--write_full_pae",
            # SANS CE DRAPEAU, RIEN NE TOURNE. boltz 2.2.0 appelle inconditionnellement un
            # noyau cuEquivariance pour la mise à jour multiplicative triangulaire du
            # pairformer (`boltz/model/layers/triangular_mult.py`), et `pip install boltz`
            # NE tire pas `cuequivariance_torch`. Le résultat observé le 5 octobre : exit
            # code 0, aucune prédiction, 6 secondes par complexe. `--no_kernels` retombe sur
            # l'implémentation PyTorch de référence — plus lente, mais c'est la même
            # fonction, et elle n'ajoute pas une dépendance à faire correspondre à la
            # version de CUDA.
            "--no_kernels",
        ]
        print(f"[{index}/{len(jobs)}] {name}", flush=True)
        completed = subprocess.run(command, capture_output=True, text=True)
        elapsed = time.time() - started
        predictions = list(out_dir.rglob("confidence_*.json"))

        # UN CODE RETOUR 0 NE PROUVE PAS QU'UNE PREDICTION A EU LIEU. Boltz-2 peut sortir
        # proprement sans rien produire — c'est arrive avec une MSA mal formee, qui le
        # faisait rendre 0 en 6 secondes et aucun fichier. On n'est donc jamais aveugle :
        # la sortie est imprimee des que la prediction manque, quel que soit le code.
        if completed.returncode != 0 or not predictions:
            tail = (completed.stdout or "")[-2000:] + "\n--- stderr ---\n" + \
                   (completed.stderr or "")[-2000:]
            print(f"    ECHEC ({elapsed:.0f}s, code {completed.returncode}, "
                  f"{len(predictions)} predictions) :\n{tail}", flush=True)
            results.append({"name": name, "ok": False, "erreur": tail[-1200:],
                            "secondes": round(elapsed, 1)})
            continue
        samples = []
        for confidence in sorted(predictions):
            payload = json.loads(confidence.read_text())
            samples.append({
                "fichier": confidence.name,
                "confidence_score": payload.get("confidence_score"),
                "ptm": payload.get("ptm"),
                "iptm": payload.get("iptm"),
                "complex_plddt": payload.get("complex_plddt"),
                "complex_iplddt": payload.get("complex_iplddt"),
                "complex_pde": payload.get("complex_pde"),
                "complex_ipde": payload.get("complex_ipde"),
            })

        # Les structures ET LES MATRICES partent dans le Volume : tout ce qui n'est pas
        # recopie ici est detruit avec le conteneur.
        #
        # ⚠️ ERREUR COMMISE LE 5 OCTOBRE : `--write_full_pae` a ete ajoute sans etendre
        # cette boucle, qui ne prenait que les `.cif` et les `confidence_*.json`. Les
        # matrices PAE ont donc ete calculees puis jetees, et le run a ete refait pour rien.
        # D'ou le glob generique plutot qu'une liste de suffixes a maintenir.
        kept = destination / name
        kept.mkdir(parents=True, exist_ok=True)
        wanted = ("*.cif", "*.pdb", "*.npz", "*.npy", "*.json")
        saved = 0
        for pattern in wanted:
            for item in out_dir.rglob(pattern):
                shutil.copy(item, kept / item.name)
                saved += 1
        outputs_volume.commit()

        print(f"    {len(samples)} echantillons en {elapsed:.0f}s "
              f"| {saved} fichiers conserves "
              f"| iptm {[s['iptm'] for s in samples]}", flush=True)
        results.append({"name": name, "ok": True, "secondes": round(elapsed, 1),
                        "echantillons": samples})

    summary = destination / "metrics.json"
    summary.write_text(json.dumps(results, indent=2))
    outputs_volume.commit()
    # Renvoyé en TEXTE et non en objet : modal désérialise la valeur de retour dans
    # l'environnement local, où torch n'est pas installé. Une chaîne JSON traverse sans
    # exiger quoi que ce soit en local.
    return json.dumps(results)


# ----------------------------------------------------------------------------------------
# Entrypoints locaux
# ----------------------------------------------------------------------------------------


def load_jobs(designs: str | None) -> list[dict]:
    """Lit les séquences depuis les !_Ranked.csv et la cible depuis le PDB d'entrée."""
    import csv

    import gemmi

    structure = gemmi.read_structure("inputs/6ARU_A_309-506.pdb")
    structure.setup_entities()
    target = gemmi.one_letter_code([r.name for r in structure[0]["A"]]).upper()

    wanted = set(designs.split(",")) if designs else None
    jobs = []
    for run in ("egfr-dIII-prod01", "egfr-dIII-prod02"):
        path = Path(f"out/{run}/3_Ranked/!_Ranked.csv")
        if not path.is_file():
            continue
        for row in csv.DictReader(path.open(newline="")):
            name = row["design"]
            if wanted and name not in wanted:
                continue
            jobs.append({"name": name, "target": target,
                         "binder": row["Binder_Sequence"].strip().upper()})
    return jobs


@app.local_entrypoint()
def rescore(designs: str | None = None, run_name: str = "rescore01") -> None:
    jobs = load_jobs(designs)
    if not jobs:
        raise SystemExit("aucun design a traiter")
    print(f"{len(jobs)} complexes, {DIFFUSION_SAMPLES} echantillons chacun")
    print(f"GPU {GPU} | budget ${BUDGET_USD:.2f} -> timeout {TIMEOUT}s "
          f"({TIMEOUT / 3600:.2f} h)")
    print(f"cible {len(jobs[0]['target'])} residus | run_name {run_name}")
    results = json.loads(predict.remote(jobs, run_name))
    ok = [r for r in results if r.get("ok")]
    print()
    print(f"-> {len(ok)}/{len(results)} complexes predits")
    Path("out").mkdir(exist_ok=True)
    Path(f"out/boltz_{run_name}.json").write_text(json.dumps(results, indent=2))
    print(f"   metriques dans out/boltz_{run_name}.json")


@app.local_entrypoint()
def diagnose(design: str | None = None) -> None:
    """Un seul complexe, sortie de boltz affichee. A lancer avant toute serie.

    Le run du 5 octobre a brule 7 minutes de L40S a produire zero prediction avec un code
    retour 0, faute d'avoir verifie un complexe d'abord. Cet entrypoint existe pour que ca
    n'arrive qu'une fois.
    """
    jobs = load_jobs(design)
    if not jobs:
        raise SystemExit("aucun design a traiter")
    jobs = jobs[:1]
    print(f"diagnostic sur {jobs[0]['name']}")
    print(f"cible {len(jobs[0]['target'])} residus | "
          f"binder {len(jobs[0]['binder'])} residus")
    results = json.loads(predict.remote(jobs, "diagnose"))
    print(json.dumps(results, indent=2)[:4000])


def load_mutant_jobs() -> list[dict]:
    """Les 12 séquences mutées, depuis out/mutants_acide.csv.

    Ces séquences n'ont JAMAIS été repliées : `mutants_acide.csv` ne contient que des
    chaînes de caractères, et les structures threadées sont des greffes de chaîne latérale
    sur le squelette du parent, pas des prédictions. Les faire prédire par Boltz-2 est donc
    la première fois qu'un modèle de structure voit ces séquences, et c'est ce qui rend
    calculable le COÛT STRUCTURAL de la phase 5c : la pose du mutant retrouve-t-elle les
    contacts de son parent, ou la mutation casse-t-elle l'interface ?
    """
    import csv

    import gemmi

    structure = gemmi.read_structure("inputs/6ARU_A_309-506.pdb")
    structure.setup_entities()
    target = gemmi.one_letter_code([r.name for r in structure[0]["A"]]).upper()

    jobs = []
    with open("out/mutants_acide.csv", newline="") as handle:
        for row in csv.DictReader(handle):
            sequence = row.get("Binder_Sequence_mutee", "").strip().upper()
            if not sequence:
                continue
            jobs.append({
                "name": f"{row['design']}__{row['mutation']}",
                "target": target,
                "binder": sequence,
            })
    return jobs


@app.local_entrypoint()
def rescore_mutants(run_name: str = "rescore_mut01") -> None:
    """Phase 5c : le coût structural des 12 mutations.

    Decision de la premiere nuit : le GPU etait alle aux 23 natifs seulement, les mutants
    ayant deja ete disqualifies par PROPKA, et les paires restaient incompletes. L'echeance
    ayant ete repoussee de 24 h, la depense se justifie et les paires peuvent etre fermees.
    """
    jobs = load_mutant_jobs()
    if not jobs:
        raise SystemExit("aucun mutant a traiter")
    print(f"{len(jobs)} mutants, {DIFFUSION_SAMPLES} echantillons chacun")
    print(f"GPU {GPU} | budget ${BUDGET_USD:.2f} -> timeout {TIMEOUT}s "
          f"({TIMEOUT / 3600:.2f} h)")
    for job in jobs:
        print(f"  {job['name'][-40:]:<42} {len(job['binder'])} aa")
    results = json.loads(predict.remote(jobs, run_name))
    ok = [r for r in results if r.get("ok")]
    print()
    print(f"-> {len(ok)}/{len(results)} mutants predits")
    Path(f"out/boltz_{run_name}.json").write_text(json.dumps(results, indent=2))


def mouse_sequence() -> str:
    lines = [
        line.strip()
        for line in Path("inputs/mEGFR_dIII.fasta").read_text().splitlines()
        if line.strip()
    ]
    return "".join(line for line in lines if not line.startswith(">")).upper()


@app.local_entrypoint()
def rescore_mouse(designs: str | None = None, run_name: str = "mouse01") -> None:
    """Objectif n°2 MESURE : les binders reconnaissent-ils l'EGFR murin ?

    Jusqu'ici la cross-reactivite n'etait qu'un proxy de sequence — la fraction des residus
    de cible contactes identiques chez la souris. Un proxy de sequence ne dit rien de la
    conformation locale murine. Ici on predit directement chaque binder contre le domaine III
    de Q01279, avec SA propre MSA, et on compare pose et confiance a celles obtenues sur
    l'humain.

    Le domaine III murin vient de mouse_target.py, qui le derive DEUX fois de maniere
    independante et refuse d'ecrire le fasta si les deux divergent. Il n'y a aucun indel
    entre humain et souris dans cette fenetre, donc la numerotation PDB 309-506 s'applique
    telle quelle aux deux especes et le mapping de contacts est l'identite.
    """
    human_jobs = load_jobs(designs)
    if not human_jobs:
        raise SystemExit("aucun design a traiter")
    target = mouse_sequence()
    jobs = [
        {"name": job["name"], "target": target, "binder": job["binder"],
         "target_msa": MOUSE_MSA}
        for job in human_jobs
    ]
    print(f"{len(jobs)} complexes contre le domaine III MURIN, "
          f"{DIFFUSION_SAMPLES} echantillons chacun")
    print(f"cible murine : {len(target)} residus (Q01279, UniProt 333-530)")
    print(f"GPU {GPU} | budget ${BUDGET_USD:.2f} -> timeout {TIMEOUT}s")
    results = json.loads(predict.remote(jobs, run_name))
    ok = [r for r in results if r.get("ok")]
    print()
    print(f"-> {len(ok)}/{len(results)} complexes murins predits")
    Path(f"out/boltz_{run_name}.json").write_text(json.dumps(results, indent=2))
