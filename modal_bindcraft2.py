# /// script
# requires-python = ">=3.12"
# dependencies = [
#     "modal>=1.0",
# ]
# ///
"""BindCraft 2.0 sur Modal — image, validation de build, et lancement de campagne.

Remplace l'entrypoint `modal_bindcraft.py` écrit pour BindCraft 1 au commit c0a48d5
(supprimé le 3 octobre, récupérable dans l'historique git).

Dépôt amont : https://github.com/PacesaLab/BindCraft2 — ce n'est PAS un tag de
martinpacesa/BindCraft, c'est un dépôt distinct.

L'image est portée de `containers/Dockerfile` de l'amont. Trois différences avec
BindCraft 1, toutes lues dans la source au commit épinglé :

  - PyRosetta, DSSP et DAlphaBall ont disparu. Zéro occurrence dans le dépôt. Plus de
    contrainte de licence académique, plus de `chmod` de binaires, et plus de pin
    `numpy<2.0` (il n'existait que pour PyRosetta).
  - Il n'y a aucune notion de TIMEOUT dans BindCraft 2.0. Le budget se pilote par
    `max_trajectories` et par le `timeout` de la fonction Modal, pas par une variable
    d'environnement interne.
  - Les poids ProteinMPNN sont livrés dans le paquet (77 Mo). Seuls les 5,3 Go d'AlphaFold
    sont téléchargés, et ils sont *bakés dans l'image* — choix reproduit de BindCraft 1.

Deux entrypoints, à nommer explicitement : `modal run` sans entrypoint explicite
n'exécute rien et sort en 0.

    modal run modal_bindcraft2.py::selfcheck
    modal run --detach modal_bindcraft2.py::design --run-name <nom> --max-trajectories <n>

Parallélisation : il n'y a PAS de sharding ici, contrairement à BindCraft 1. BindCraft 2.0
répartit lui-même des workers concurrents sur l'allocation visible.

    --workers 1                          # série, pour mesurer un temps par trajectoire
    --workers auto                       # 3 workers sur une L40S ici, pour produire
    GPU_COUNT=4 modal run --detach ...   # 4 cartes dans un conteneur, une seule campagne

⚠️ Pourquoi `--workers` est un argument et `GPU_COUNT` une variable d'environnement : Modal
ne propage pas l'environnement local au conteneur. Une variable lue au niveau module n'est
correcte que si elle sert dans un DÉCORATEUR, évalué en local à l'import — c'est le cas de
`GPU`, `GPU_COUNT` et `BUDGET_USD`. Tout ce qui est lu à l'exécution doit voyager comme
argument, sinon ça retombe sur le défaut en silence.
"""

from __future__ import annotations

import csv
import json
import os
import shlex
import shutil
import subprocess
from collections import Counter
from pathlib import Path

from modal import App, Image, Volume

# ----------------------------------------------------------------------------------------
# Épinglages. Tout chiffre ici doit être justifié dans NOTES.md.
# ----------------------------------------------------------------------------------------

# HEAD de PacesaLab/BindCraft2 au 3 octobre, "Update pdl1_cyclic_peptide.json".
BC2_COMMIT = "a8d0f2002df373842b86a3c20c5a060c5cfdf980"

# L40S : compute capability 8.9, au-dessus du plancher 7.5 de CUDA 13. 46 Go vérifiés.
GPU = os.environ.get("GPU", "L40S")

# Nombre de cartes dans UN SEUL conteneur. BindCraft 2.0 a `auto_multi_gpu` à true et
# répartit lui-même ses workers sur toute l'allocation visible, donc monter ce chiffre
# parallélise sans sharder.
#
# ⚠️ NE PAS revenir au sharding de BindCraft 1 (`shard-000`, `shard-001`…). Sur 2.0 chaque
# shard serait une campagne indépendante chassant son propre `number_of_final_designs` :
# N shards = N × les designs et N × le coût. Et le barreau de l'échelle de désespoir est
# « lu sur les tables de la campagne, donc chaque worker et une campagne reprise sont sur
# le même » — des shards séparés le compteraient chacun de leur côté.
GPU_COUNT = int(os.environ.get("GPU_COUNT", 1))

# ⚠️ `workers_per_gpu` N'EST PAS une variable d'environnement, et c'est délibéré.
#
# Modal ne propage PAS l'environnement local au conteneur. Une valeur lue par
# `os.environ` au niveau module n'est correcte que si elle sert dans un DÉCORATEUR, qui
# est évalué en local à l'import (c'est le cas de GPU, GPU_COUNT et BUDGET_USD). Tout ce
# qui est lu à l'exécution tourne dans le conteneur, où la variable est absente et
# retombe sur le défaut, en silence.
#
# Bug vécu le 3 octobre : `WORKERS=1 modal run ...` a produit une campagne à
# `"workers_per_gpu": "auto"`, donc 3 workers au lieu d'un, et le run « série » censé
# mesurer un temps par trajectoire ne mesurait rien. Run arrêté, ~$0,07.
#
# Donc : argument de fonction, sérialisé par Modal et visible dans la commande.
#     --workers 1       série, pour mesurer un temps par trajectoire
#     --workers auto     3 workers sur une L40S ici, pour produire
#
# `auto` résout à 7 puis se fait plafonner par la mémoire, dans
# `bindcraft/design_workers.py` :
#     mem_par_worker = 2.0 × (3.4 + 38000 × n_residus² / 1e9)
#     workers        = (libre_Go − 4) // mem_par_worker
# Calculé le 3 octobre pour notre cible (198 résidus) sur une L40S de 46 Go :
#     binder 55 → 253 résidus → 11,66 Go/worker → 3 workers
#     binder 95 → 293 résidus → 13,32 Go/worker → 3 workers
# On ne tombe à 1 worker que vers 430 résidus au total.

# Installation éditable à cette racine : `settings/` et `scaffolds/` vivent à la racine du
# dépôt et non dans le paquet, et le runtime les trouve relativement à lui. Une install
# classique ne copierait que le paquet et les orphelinerait.
BC2_ROOT = "/opt/bindcraft"

# `model_weights.SHIPPED_WEIGHTS` = <dir du paquet>/weights, soit ceci en install éditable.
# `alphafold_parameters()` y cherche `alphafold/` avant de télécharger : baker là rend le
# téléchargement inutile au runtime. Vérifié par `selfcheck`.
SHIPPED_WEIGHTS = f"{BC2_ROOT}/bindcraft/weights"

OUTPUTS = "/outputs"

# Modal attend "L40S" pour une carte, "L40S:4" pour quatre dans le même conteneur.
GPU_SPEC = GPU if GPU_COUNT == 1 else f"{GPU}:{GPU_COUNT}"

# ----------------------------------------------------------------------------------------
# Ressources du conteneur Modal. CE BLOC EST UN PLAFOND DE DÉPENSE, pas de la décoration.
# ----------------------------------------------------------------------------------------

# 4 cœurs et 24 Go par GPU : c'est la taille que le script Slurm de l'amont se donne
# lui-même (docs/source/installation.md). Ce n'est pas un chiffre posé.
#
# ⚠️ POURQUOI LA RAM COMPTE : BindCraft plafonne ses workers DEUX FOIS, pas une. Après le
# plafond mémoire GPU (3 workers ici), il applique dans `design_workers.py` :
#     host_memory_worker_ceiling = (MemAvailable_Go // 4,0) // n_gpu
# Lu sur /proc/meminfo. Sous 12 Go de RAM disponible, on retombe à 2 workers ; sous 8 Go, à
# 1 — et les 3 workers calculés sur la mémoire GPU seraient perdus en silence. 24 Go donnent
# un plafond hôte de 6, donc c'est bien la mémoire GPU qui décide, ce qu'on veut.
CPU_PER_GPU = 4.0
MEMORY_PER_GPU_MB = 24 * 1024

# Tarif L40S mesuré : $0,000542/s = $1,95/h. Le `timeout` Modal est le SEUL vrai plafond de
# dépense — `max_trajectories` ne protège pas d'un run bloqué. On le dérive donc d'un budget
# en dollars au lieu de le laisser au maximum de 24 h, qui vaudrait $47 sur une carte et
# $187 sur quatre.
USD_PER_HOUR = {"L40S": 1.95}
BUDGET_USD = float(os.environ.get("BUDGET_USD", 5.0))


def budget_timeout_seconds() -> int:
    """Convertit un budget en dollars en secondes de `timeout` Modal."""
    rate = USD_PER_HOUR.get(GPU)
    if rate is None:
        raise ValueError(
            f"tarif horaire inconnu pour GPU={GPU} : ajouter la valeur mesurée dans "
            f"USD_PER_HOUR avant de lancer, sinon le budget n'est pas un plafond"
        )
    seconds = int(3600 * BUDGET_USD / (rate * GPU_COUNT))
    if seconds < 300:
        # Le plancher dépasserait le budget demandé : le dire plutôt que de laisser croire
        # que $0,02 est un plafond tenu.
        floor_cost = 300 * rate * GPU_COUNT / 3600
        print(
            f"⚠️  BUDGET_USD={BUDGET_USD} donnerait {seconds}s, sous le plancher de 300s. "
            f"Le coût réel plafonnera donc vers ${floor_cost:.2f}, pas ${BUDGET_USD:.2f}."
        )
    return max(300, min(seconds, 86400))  # plancher 5 min, plafond Modal 24 h


DESIGN_TIMEOUT = budget_timeout_seconds()

# Cache de compilation XLA. BindCraft le pose par défaut sur `/tmp/bindcraft_xla_cache`,
# ÉPHÉMÈRE : chaque conteneur Modal recompile tout, et avec `length_bucket_size 32`,
# `[55,95]` rembourre vers 64 et 96, donc deux compilations par worker à chaque démarrage.
#
# Le mettre sur le Volume le rendrait persistant d'un appel à l'autre. MAIS c'est un système
# de fichiers réseau en FUSE avec jusqu'à 3 écrivains concurrents, et le gain n'est pas
# mesuré — l'amont pose aussi `JAX_PERSISTENT_CACHE_ENABLE_XLA_CACHES=none`, donc tout n'est
# pas sérialisé de toute façon. L'introduire dans le run qui doit MESURER le débit
# ajouterait une variable non contrôlée à la mesure.
#
# Éteint par défaut, activable par `--xla-cache` — un argument et non une variable
# d'environnement, pour la raison expliquée plus haut.
XLA_CACHE_DIR = f"{OUTPUTS}/.xla_cache"

# Fréquence des commits du Volume pendant un run. Un commit n'arrive qu'à la fin sans ça :
# un conteneur tué dur perdrait tout depuis le début. `resume` étant à true, un commit
# régulier rend un redémarrage presque gratuit.
COMMIT_EVERY_S = 300

# ----------------------------------------------------------------------------------------
# La cible et les biais de ciblage. Numérotation PDB, celle du fichier cible.
# ----------------------------------------------------------------------------------------

TARGET_PDB = "inputs/6ARU_A_309-506.pdb"
TARGET_NAME = "hEGFR_dIII"
TARGET_CHAINS = "A"

# Jeu arrêté le 3 octobre, motifs dans NOTES.md : étendue CA 16,73 Å, identité humain/souris
# 4/4, D323 porte la route pH 1 (carboxylate 75,0 Å²), H409 la route pH 2 (His conservée).
HOTSPOTS = "A318,A323,A406,A409"

# A359 est un His en numérotation PDB — vérifié par lecture directe du fichier cible, pas
# de mémoire. Il diverge en Arg chez la souris et porte 16 % de la surface apolaire de sa
# zone : le contacter mettrait en danger l'objectif n°2. BindCraft 1 ne pouvait que
# l'éviter par choix de fenêtre et espérer.
#
# A359 est sûr parce qu'il est à 13,50 Å CA du hotspot le plus proche (A323), donc hors du
# rayon de 8,0 Å des pertes coldspot de BindCraft 2.0 (`bindcraft/loss.py`, cutoff=8.0).
#
# ⚠️ A325 est volontairement ABSENT, contre ce que NOTES.md du 3 octobre en faisait un
# « candidat coldspot au même titre que H359 ». Mesuré le 3 octobre : A325 est à
# **6,36 Å CA de A323**, sous le rayon de 8,0 Å. Le déclarer coldspot ferait repousser le
# binder hors de D323, qui porte la route pH n°1 — soit sacrifier l'objectif le mieux
# classé pour écarter un risque glycanique sur un résidu qu'on a déjà retiré des hotspots.
# Ne pas le rajouter sans avoir d'abord mesuré le séquon dans PyMOL.
#
# ⚠️ Incomplet : CLAUDE.md §8 action 4 demande aussi d'écarter les résidus à moins de ~10 Å
# d'un séquon. Cette distance n'a jamais été mesurée depuis les hotspots retenus. Ne pas
# inventer la liste — la mesurer, puis l'ajouter ici, et vérifier chaque candidat contre le
# rayon de 8,0 Å comme ci-dessus.
COLDSPOTS = "A359"

# Catégorie minibinders d'Adaptyv : 40–100 inclus. 55–95 laisse une marge aux deux bornes.
BINDER_LENGTHS = [55, 95]

# Défauts partagés par les deux entrypoints. Ils sont ici et pas dans les signatures parce
# que `modal run ...::design` construit sa CLI depuis la signature de `design` et non depuis
# celle de `main` : un défaut présent seulement sur `main` rend `--n-designs` obligatoire sur
# la commande documentée. C'est arrivé le 3 octobre.
#
# 20 places maximum en Track 3, et la règle est que 8 designs défendables battent 20
# médiocres. 12 laisse de la matière au tri a posteriori sans gonfler le budget.
DEFAULT_MAX_TRAJECTORIES = 10
DEFAULT_N_DESIGNS = 12

# ----------------------------------------------------------------------------------------
# Image. Portée de containers/Dockerfile de l'amont.
# ----------------------------------------------------------------------------------------

# jax[cuda13] garde ses bibliothèques sous site-packages/nvidia/*/lib, où le loader ne
# regarde pas. Sans ce fichier, jax avertit une fois puis tourne sur le CPU, cent fois plus
# lentement, et rien ensuite ne le signale.
_LDCONF = (
    "import nvidia, pathlib; "
    "print(chr(10).join(sorted(str(p) "
    "for r in nvidia.__path__ for p in pathlib.Path(r).glob('*/lib'))))"
)

image = (
    Image.debian_slim(python_version="3.12")
    .apt_install("git", "build-essential", "ca-certificates")
    .env({"PYTHONUNBUFFERED": "1", "PIP_NO_CACHE_DIR": "1"})
    .run_commands(
        f"git clone https://github.com/PacesaLab/BindCraft2.git {BC2_ROOT}",
        f"cd {BC2_ROOT} && git checkout {BC2_COMMIT}",
    )
    .run_commands(f"python -m pip install -e '{BC2_ROOT}[cuda13]'")
    .run_commands(
        f'python -c "{_LDCONF}" > /etc/ld.so.conf.d/bindcraft-cuda.conf',
        "ldconfig",
        # Échoue au build plutôt qu'à la minute 40 d'une campagne si les wheels ne
        # s'alignent pas.
        "ldconfig -p | grep -q libcupti",
    )
    # 5,3 Go, une fois, dans la couche d'image. Placé après l'install et avant la
    # vérification : une invalidation de cache en amont ne doit pas le refaire pour rien.
    .run_commands(
        f"BINDCRAFT_WEIGHTS={SHIPPED_WEIGHTS} bindcraft fetch-weights",
        f"test -d {SHIPPED_WEIGHTS}/alphafold",
    )
    .run_commands(
        'python -c "import jax, bindcraft.proteinmpnn; print(\'jax\', jax.__version__)"',
        "bindcraft --help > /dev/null",
        # pip signale un succès pour un environnement auquel il manque un module. Ceci
        # nomme chaque module et chaque checkpoint.
        "python -m bindcraft.selfcheck cuda13 --shipped-only",
    )
    .add_local_dir("inputs", remote_path="/root/inputs")
)

app = App("bindcraft2")
volume = Volume.from_name("bindcraft", create_if_missing=True)


# ----------------------------------------------------------------------------------------
# Réglages de campagne
# ----------------------------------------------------------------------------------------


def campaign_settings(
    run_name: str, max_trajectories: int, n_designs: int, workers: str
) -> dict:
    """Construit les réglages de la campagne.

    Les seuils laissés de côté viennent de `settings/core/default.json` de l'amont, lu au
    commit épinglé. ⚠️ Ils sont sur [0,1], pas sur 0–100 ni en Å : `i_pAE` 0,35,
    `i_pTM` 0,70, `Unbound_Binder_pLDDT` 0,80, `Interface_Residues` 7.
    """
    return {
        "campaign_name": run_name,
        "project_folder": f"{OUTPUTS}/{run_name}",
        "modality": "binder",
        "binder_lengths": BINDER_LENGTHS,
        "number_of_final_designs": n_designs,
        # Le seul garde de budget interne à BindCraft 2.0 : il n'y a pas de TIMEOUT.
        "max_trajectories": max_trajectories,
        # Écrit explicitement pour que `campaign_metadata.json` garde la trace de la
        # concurrence sous laquelle la mesure a été prise. Vient d'un ARGUMENT et non d'une
        # variable d'environnement : voir le commentaire sur workers_per_gpu plus haut.
        "workers_per_gpu": workers,
        # Adaptyv exprime en acellulaire : pas de cystéines libres. Ce réglage les interdit
        # à la source plutôt que de les filtrer après coup.
        "aa_bias": {"C": 0},
        "targets": [
            {
                "name": TARGET_NAME,
                "target_path": f"/root/{TARGET_PDB}",
                "chains": TARGET_CHAINS,
                "hotspots": HOTSPOTS,
                "coldspots": COLDSPOTS,
                "weight": 1.0,
            }
        ],
    }


def _run(command: list[str], cwd: str | None = None, env: dict | None = None) -> None:
    print(f"$ {shlex.join(command)}", flush=True)
    subprocess.run(command, cwd=cwd, env=env, check=True)


# ----------------------------------------------------------------------------------------
# Entrypoints
# ----------------------------------------------------------------------------------------


@app.function(image=image, gpu=GPU, volumes={OUTPUTS: volume}, timeout=600)
def selfcheck() -> None:
    """Valide le build avant toute dépense. ~1 min de GPU.

    Un build raté coûte le run entier, et un exit code 0 ne prouve pas qu'un GPU a été vu :
    jax tombe sur le CPU en silence. Cette fonction échoue si ce n'est pas un GPU CUDA.
    """
    import jax

    devices = jax.devices()
    print(f"jax {jax.__version__} | backend {jax.default_backend()} | devices {devices}")
    if jax.default_backend() != "gpu":
        raise RuntimeError(f"jax tourne sur {jax.default_backend()}, pas sur gpu : {devices}")

    from bindcraft.model_weights import SHIPPED_WEIGHTS as runtime_weights
    from bindcraft.model_weights import alphafold_parameters, missing_model_weights

    # L'emplacement baké doit être celui que le runtime interroge, sinon la campagne
    # retéléchargerait 5,3 Go à chaque conteneur.
    assert str(runtime_weights) == SHIPPED_WEIGHTS, (runtime_weights, SHIPPED_WEIGHTS)
    params = alphafold_parameters(download=False)
    if not params:
        raise RuntimeError(f"poids AF2 introuvables sous {SHIPPED_WEIGHTS}")
    print(f"poids AF2 : {params}")

    problems = missing_model_weights(params, str(runtime_weights / "proteinmpnn" / "weights_neutral"))
    if problems:
        raise RuntimeError("checkpoints manquants :\n  " + "\n  ".join(problems))
    print("checkpoints : les 7 modèles AF2 et les 3 variantes ProteinMPNN sont complets")

    # La cible est-elle bien montée et lisible ?
    target = Path("/root") / TARGET_PDB
    n_ca = sum(1 for line in target.read_text().splitlines() if line.startswith("ATOM") and line[12:16].strip() == "CA")
    print(f"cible {target} : {n_ca} résidus")

    print("\nBUILD VALIDE")


# Script exécuté par chaque processus du test de concurrence. Il appelle la VRAIE fonction
# d'écriture de BindCraft, pas une imitation : c'est ce chemin de code qu'on veut éprouver.
_CONCURRENCY_WORKER = '''
import sys
from bindcraft.campaign_output import append_campaign_metrics
csv_path, worker, rows = sys.argv[1], int(sys.argv[2]), int(sys.argv[3])
for index in range(rows):
    append_campaign_metrics(csv_path, {"worker": worker, "index": index})
print(f"worker {worker}: {rows} lignes ecrites", flush=True)
'''


@app.function(image=image, volumes={OUTPUTS: volume}, timeout=900, cpu=4.0, memory=8192)
def volume_concurrency_check(n_processes: int = 3, rows_each: int = 60) -> None:
    """Est-ce que des workers concurrents s'écrasent sur le Volume Modal ?

    `append_campaign_metrics` fait un read-modify-write du CSV ENTIER sous
    `fcntl.flock` : il relit toutes les lignes, réécrit tout dans un `.partial`, puis
    `os.replace`. Si le flock ne protège pas sur un Volume monté en FUSE, deux processus
    relisent le même état et le dernier écrase la ligne de l'autre.

    Les workers de BindCraft étant des subprocess d'UN SEUL conteneur, le flock devrait
    être arbitré par le noyau sur le même mount. Ce test le vérifie plutôt que de le
    supposer. CPU seul, donc le coût est négligeable.
    """
    import tempfile

    folder = Path(OUTPUTS) / ".concurrency_check"
    if folder.exists():
        shutil.rmtree(folder)
    folder.mkdir(parents=True)
    csv_path = folder / "rows.csv"

    script = Path(tempfile.gettempdir()) / "_concurrency_worker.py"
    script.write_text(_CONCURRENCY_WORKER)

    expected = n_processes * rows_each
    print(
        f"{n_processes} processus x {rows_each} lignes sur {csv_path}\n"
        f"attendu si flock protege : {expected} lignes",
        flush=True,
    )

    processes = [
        subprocess.Popen(
            ["python", str(script), str(csv_path), str(worker), str(rows_each)],
            cwd=BC2_ROOT,
        )
        for worker in range(n_processes)
    ]
    codes = [process.wait() for process in processes]
    if any(codes):
        raise RuntimeError(f"un processus a echoue : codes {codes}")

    with csv_path.open(newline="") as handle:
        rows = list(csv.DictReader(handle))
    per_worker = Counter(row["worker"] for row in rows)
    leftovers = sorted(p.name for p in folder.glob("*.partial"))

    print(f"\nlignes trouvees : {len(rows)} / {expected} attendues")
    for worker in sorted(per_worker, key=lambda value: int(value)):
        print(f"  worker {worker} : {per_worker[worker]} / {rows_each}")
    print(f"fichiers .partial residuels : {leftovers or 'aucun'}")

    volume.commit()

    if len(rows) != expected:
        raise RuntimeError(
            f"ECRASEMENT DETECTE : {expected - len(rows)} lignes perdues. "
            f"fcntl.flock ne protege pas sur ce Volume — il ne faut PAS faire tourner "
            f"plusieurs workers avec project_folder sur le Volume."
        )
    print("\nPAS D'ECRASEMENT : flock protege le read-modify-write sur ce Volume.")


@app.function(
    image=image,
    gpu=GPU_SPEC,
    volumes={OUTPUTS: volume},
    # Dérivé de BUDGET_USD : c'est le vrai plafond de dépense.
    timeout=DESIGN_TIMEOUT,
    cpu=CPU_PER_GPU * GPU_COUNT,
    memory=MEMORY_PER_GPU_MB * GPU_COUNT,
    # Un réessai sur une fonction GPU de plusieurs heures doublerait la facture en silence.
    retries=0,
    # Une seule campagne, un seul conteneur. Garde contre un fan-out accidentel.
    max_containers=1,
)
def design(
    run_name: str,
    max_trajectories: int = DEFAULT_MAX_TRAJECTORIES,
    n_designs: int = DEFAULT_N_DESIGNS,
    workers: str = "auto",
    xla_cache: bool = False,
) -> None:
    """Lance une campagne. `resume` est à true par défaut dans BindCraft 2.0, donc un
    rappel sur le même `run_name` reprend là où le précédent s'est arrêté."""
    import threading

    settings = campaign_settings(run_name, max_trajectories, n_designs, workers)
    folder = Path(settings["project_folder"])
    folder.mkdir(parents=True, exist_ok=True)

    # Les réglages résolus sont écrits à côté des sorties : la campagne reste auditable
    # sans relire ce fichier source.
    #
    # ⚠️ Garde de reproductibilité. `resume` est à true, donc un rappel sur le même
    # `run_name` REPREND la campagne. Si on le rappelait avec d'autres paramètres, ce
    # fichier serait réécrit et ne décrirait plus le run qui a produit les lignes déjà
    # présentes : les tables deviendraient inexplicables. On refuse plutôt que d'écraser.
    config = folder / "settings.json"
    if config.exists():
        previous = json.loads(config.read_text())
        drift = {
            key: (previous.get(key), settings.get(key))
            for key in set(previous) | set(settings)
            if previous.get(key) != settings.get(key)
        }
        if drift:
            raise RuntimeError(
                f"{config} existe déjà avec d'autres réglages, et `resume` reprendrait la "
                f"campagne : les tables mélangeraient deux configurations.\n"
                + "\n".join(f"  {key}: {was!r} -> {now!r}" for key, (was, now) in sorted(drift.items()))
                + f"\nUtiliser un --run-name neuf, ou supprimer {folder} si la reprise est voulue."
            )
        print(f"reprise de {folder} avec des réglages identiques", flush=True)
    config.write_text(json.dumps(settings, indent=2))
    print(json.dumps(settings, indent=2), flush=True)
    # Le budget n'est PAS imprimé ici : BUDGET_USD est une variable d'environnement locale,
    # donc sa valeur dans le conteneur est le défaut, pas celle demandée. Le `timeout` lui
    # est correct — il vient du décorateur, évalué en local. La ligne de budget est imprimée
    # par `main`, en local, où la valeur est vraie.
    volume.commit()

    # Commits périodiques : sans eux, un conteneur tué dur perdrait tout le run.
    done = threading.Event()

    def commit_loop() -> None:
        # Une exception non rattrapée tuerait ce thread daemon en silence et on perdrait
        # les commits périodiques sans le savoir. On la signale et on continue.
        while not done.wait(COMMIT_EVERY_S):
            try:
                volume.commit()
                print(f"[commit periodique] {folder}", flush=True)
            except Exception as failure:  # noqa: BLE001 - on veut tout voir, pas tomber
                print(f"[commit periodique ECHEC] {failure!r}", flush=True)

    committer = threading.Thread(target=commit_loop, daemon=True)
    committer.start()

    environment = dict(os.environ)
    if xla_cache:
        Path(XLA_CACHE_DIR).mkdir(parents=True, exist_ok=True)
        environment["JAX_COMPILATION_CACHE_DIR"] = XLA_CACHE_DIR
        print(f"cache XLA sur le Volume : {XLA_CACHE_DIR} (non mesuré)", flush=True)

    try:
        _run(["bindcraft", "design", str(config)], cwd=BC2_ROOT, env=environment)
    finally:
        done.set()
        committer.join(timeout=30)
        volume.commit()
        print(f"volume commité : {folder}", flush=True)


@app.local_entrypoint()
def main(
    run_name: str,
    max_trajectories: int = DEFAULT_MAX_TRAJECTORIES,
    n_designs: int = DEFAULT_N_DESIGNS,
    workers: str = "auto",
    xla_cache: bool = False,
) -> None:
    print(
        f"GPU {GPU_SPEC} | workers_per_gpu {workers} | "
        f"cpu {CPU_PER_GPU * GPU_COUNT} | ram {MEMORY_PER_GPU_MB * GPU_COUNT // 1024} Go\n"
        f"run {run_name} | max_trajectories {max_trajectories} | n_designs {n_designs}\n"
        f"budget ${BUDGET_USD:.2f} → timeout {DESIGN_TIMEOUT}s "
        f"({DESIGN_TIMEOUT / 3600:.2f} h) — c'est le plafond de dépense"
    )
    if workers == "auto" and max_trajectories <= 4:
        print(
            "\n⚠️  workers_per_gpu=auto donne 3 workers sur une L40S pour cette cible, donc "
            f"ces {max_trajectories} trajectoires partiront en parallèle. Le temps mural ne "
            "donnera PAS un temps par trajectoire.\n"
            "    Pour calibrer : --workers 1\n"
            "    Pour produire : --workers auto et monter max_trajectories.\n"
        )
    design.remote(
        run_name=run_name,
        max_trajectories=max_trajectories,
        n_designs=n_designs,
        workers=workers,
        xla_cache=xla_cache,
    )
