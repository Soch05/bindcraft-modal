# NOTES — BindCraft sur Modal

## Environnement local (23/09)

- MacBook Air 2018, Intel x86_64, macOS (Darwin 23.6).
- venv : `uv venv --python 3.12` → CPython 3.12.13, dans `.venv/`.
- `modal` : **1.5.5**.
- **Accroc :** `uv pip install modal` échoue seul. `modal` tire `cbor2>=6`, qui compile
  une extension Rust et n'a pas de wheel pour macOS x86_64 → `error: can't find Rust
  compiler`. Corrigé en épinglant **`cbor2==5.9.0`** (wheel disponible). Le pin est dans
  `pyproject.toml` — ne pas l'enlever, sinon l'install casse à nouveau.

## Cible de démo

`inputs/PDL1.pdb` — 923 lignes, 921 atomes, **115 résidus, chaîne A unique**.
Avec un binder de 50-130 résidus, le complexe plafonne à ~245 résidus, très en dessous
du seuil de ~550 sur 32 Go (règle 7). La mémoire GPU n'est donc pas limitante sur cette
cible, et le choix du GPU se fait sur le seul critère vitesse/prix.

## Décisions d'architecture

**Base :** `modal_bindcraft.py` de `hgbrian/biomodals` (1157 lignes), copié puis adapté
par edits chirurgicaux — 5 zones touchées, la logique BindCraft intacte.
BindCraft épinglé au commit `c0a48d595d4976694aa979438712ac94c16620bb`.

**Poids AF2 : laissés dans l'Image, contrairement à l'architecture cible de CLAUDE.md.**
Décision prise sciemment. La couche `aria2c` qui télécharge les 5,3 Go est placée *avant*
`set_up_pyrosetta()` et avant le `uv_pip_install` final de `jax[cuda]`/`numpy`/
`matplotlib`. Donc ajuster les pins qu'on est le plus susceptible de toucher (règles 2 et 8)
**ne réinvalide pas** la couche des poids — le risque que CLAUDE.md veut éviter est déjà
neutralisé par l'ordre des couches. Déplacer vers un Volume aurait coûté une fonction de
download one-shot plus le patch des chemins que BindCraft attend en dur
(`/root/bindcraft/params`), pour économiser un téléchargement de ~1 min en datacenter.
À revoir si on finit par modifier `apt_install` ou le premier pin `numpy<2.0`.

**Volume pour les sorties : obligatoire, pas optionnel.** La référence *retourne* les
fichiers par valeur et les écrit en local depuis le `local_entrypoint`. Ce design est
**incompatible avec `--detach`** (règle 4) : le client se déconnecte, la valeur de retour
ne parvient à personne, et `modal volume get` n'a rien à lire. Adaptation :

- Volume `bindcraft` monté sur `/outputs`, `design_path = /outputs/<run_name>/`.
  BindCraft écrit donc directement dans le Volume — un run tué ou crashé laisse ses
  résultats partiels, ce qui est précisément ce qu'on veut pour diagnostiquer.
- `volume.commit()` dans un `finally`, via un wrapper fin autour de la fonction d'origine
  (renommée `_bindcraft`) pour éviter de réindenter 1000 lignes. La doc Modal ne promet la
  visibilité hors conteneur qu'après un commit explicite ; on ne s'appuie pas sur un
  commit implicite non documenté.
- Le retour ne contient plus qu'un inventaire (`n_files` + liste), pas les octets.

## Versions épinglées dans l'image

Héritées de la référence, non modifiées à ce stade :

| Composant | Pin |
|---|---|
| python (image) | 3.11 |
| numpy | `<2.0`, posé **avant** tout le reste et re-posé après PyRosetta |
| jax | `jax[cuda]<0.7.0` — évite la suppression de `wraps` en 0.7.0 |
| matplotlib | `==3.8.1` — BindCraft issue #4 |
| BindCraft | commit `c0a48d5` |
| ColabDesign | `git+main` (⚠️ **flottant**, contraire à la règle 2 — à épingler si un échec apparaît ici) |
| AF2 params | `alphafold_params_2022-12-06.tar` |

## Garde-fous déjà présents dans la référence

Vérifié en lisant le code, pas supposé :

- `TIMEOUT` est en **minutes**, défaut 300 → `timeout=18000s` = 5 h. Généreux, pas trop court.
  Le commentaire du code note qu'un timeout trop haut rend le provisionnement GPU plus dur.
- `max_trajectories` existe déjà comme paramètre → plafond disponible. ⚠️ **Corrigé après le
  run `test1` :** il plafonne les trajectoires *réussies*, pas les tentatives. Voir la section
  du run.
- `enable_rejection_check` + `acceptance_rate` : le script s'arrête si le taux d'acceptation
  s'effondre. ⚠️ **Corrigé après `test1` :** inopérant sur un run court, parce que
  `start_monitoring` vaut **600** — le contrôle ne démarre qu'après 600 trajectoires.
  L'affirmation initiale « pas de risque de boucle qui brûle les crédits » était fausse :
  elle n'est vraie qu'avec un `--max-trajectories` explicite.

## Coûts attendus

Du docstring de la référence, pour **3** designs sur PDL1 :

| GPU | Coût | Durée |
|---|---|---|
| A10G | $2 | 1,5 h |
| A100 | $3 | 1 h |
| H100 | $4 | 40 min |

À `--number-of-final-designs 1`, compter ~⅓ : **~$1, ~20-30 min**. Sous le seuil de la
règle 9. **GPU retenu : L40S** (48 Go, défaut de la référence) — surdimensionné pour cette
cible, mais généralement plus rapide à provisionner qu'un A100.

## Smoke test `check_gpu` — PASSÉ (23/09)

Workspace Modal : `soch05`.

```
GPU=L40S .venv/bin/modal run modal_bindcraft.py::check_gpu
```

Sortie réelle, premier build (~10 min) :

```
NVIDIA-SMI 580.95.05   Driver Version: 580.95.05   CUDA Version: 13.0
NVIDIA L40S    3MiB / 46068MiB
jax 0.6.2 | jaxlib 0.6.2 | numpy 1.26.4
jax.devices() -> [CudaDevice(id=0)]
Built image im-7noEgKlToD1tPPeGN48xzI
```

**Versions effectivement résolues** (à noter : `jax[cuda]<0.7.0` donne **0.6.2**, et
`numpy<2.0` donne **1.26.4**. Ce sont les versions à réépingler à l'identique si un
run futur casse) :

| Composant | Résolu |
|---|---|
| jax / jaxlib | 0.6.2 |
| numpy | 1.26.4 |
| driver NVIDIA | 580.95.05, CUDA 13.0 |
| GPU | L40S, 46 Go |

Poids AF2 présents dans l'image : 16 fichiers — les 5 modèles en 3 variantes
(`params_model_N.npz`, `_ptm`, `_multimer_v3`) + LICENSE.
Les deux JSON de settings par défaut présents et non modifiés.

**Cache de build vérifié :** 2ᵉ exécution de la même commande = **12 s** de bout en bout,
aucune ligne `Built image`. L'image et les 5,3 Go de poids viennent du cache. Le choix de
laisser les poids dans l'Image est donc validé en pratique.

## Journal des runs

| Date | Commande | GPU | Durée | Coût | Trajectoires | Acceptés |
|---|---|---|---|---|---|---|
| 23/09 | `check_gpu` (1er build) | L40S | ~10 min | ~0 (GPU qq s) | — | — |
| 23/09 | `check_gpu` (cache) | L40S | 12 s | ~0 | — | — |
| 23/09 | run `test1` (ci-dessous) | L40S | **47 min 25 s** | **$1,54** | 4 lancées / 3 réussies | **1** |

## Run `test1` — PDL1, PASSÉ (23/09)

```bash
GPU=L40S .venv/bin/modal run --detach modal_bindcraft.py \
  --input-pdb inputs/PDL1.pdb --number-of-final-designs 1 \
  --max-trajectories 3 --run-name test1
```

> **Périmé depuis l'ajout de `parallel` (noté le 01/10).** Cette commande telle quelle
> n'exécute plus rien : le fichier déclare deux `local_entrypoint`, Modal refuse de choisir,
> **et sort en code 0**. Écrire `modal_bindcraft.py::main`. Conservée ici telle qu'elle a
> réellement tourné le 23/09, le journal n'étant pas réécrit.

App `ap-OoBXPM6BtVeYmf4qGauJUA`. **47 min 25 s**, 78 fichiers, coût **$1,54**
(2845 s × $0,000542/s). Terminé sans exception, app en état `stopped`.

### Baseline par trajectoire — le chiffre à réutiliser

| # | Binder | Résultat | Durée |
|---|---|---|---|
| 1 | `l112_s900583` | ✓ pLDDT 0,83 | 8 min 24 |
| 2 | `l94_s179902` | ✓ pLDDT 0,89 | 5 min 38 |
| 3 | `l55_s851460` | ✗ rejetée | 3 min 21 |
| 4 | `l81_s625098` | ✓ pLDDT 0,95 | 5 min 22 |

⚠️ **Ne pas dimensionner avec le chiffre ci-dessous** — voir « Correction de baseline » dans
la section parallélisation : c'est la durée de la descente de gradient seule, et le cycle
complet coûte **9 min**. Ce run s'arrêtait au premier design accepté, donc il exécutait peu
de cycles MPNN.

~5,5 min par trajectoire sur L40S, plus un surcoût unique de ~3 min sur la
première : elle paie la **compilation JIT de JAX**, que les suivantes réutilisent. Ne pas
dimensionner sur les 8 min 24 de la première, c'est un artefact de démarrage.

### Deux pièges découverts en lisant les logs

1. **`--max-trajectories` plafonne les trajectoires RÉUSSIES, pas les tentatives.**
   4 lancées pour un plafond de 3, parce que la n°3 a été rejetée et n'est pas comptée
   (`check_n_trajectories` compte les fichiers produits). Le coût réel peut donc dépasser
   le plafond × durée unitaire. À intégrer dans toute estimation future.
2. **`start_monitoring: 600`** : le garde-fou `enable_rejection_check` ne s'active qu'après
   **600** trajectoires. Il ne protège donc pas un run court. Sans `--max-trajectories`,
   le seul frein est le timeout de 5 h, soit ~$10 sur L40S.

Noté aussi : un avertissement `Modal Client → Modal Worker Heartbeat attempt failed` est
apparu en cours de run. Le client local perd le contact, le job continue côté Modal.
C'est exactement ce que `--detach` garantit — validé en conditions réelles.

### Design accepté

`final_design_stats.csv` — 1 ligne, 232 colonnes. `Accepted/Ranked/1_PDL1_l81_s625098_mpnn3_model1.pdb`

| Métrique | Valeur | Seuil par défaut |
|---|---|---|
| Design | `PDL1_l81_s625098_mpnn3` (81 aa) | |
| Average_pLDDT | 0,93 | > 0,8 |
| Average_pTM | 0,89 | > 0,55 |
| Average_i_pTM | 0,86 | > 0,5 |
| Average_i_pAE | 0,16 | < 0,35 |
| Average_ShapeComplementarity | 0,61 | > 0,6 |
| Average_dG | −46,84 | < 0 |
| Average_dSASA | 1840,18 | > 1 |
| Average_n_InterfaceResidues | 20 | > 7 |
| Average_n_InterfaceHbonds | 8 | > 3 |
| Average_n_InterfaceUnsatHbonds | 3,5 | < 4 |
| Average_Surface_Hydrophobicity | 0,33 | < 0,35 |

Séquence : `ALVTIDENAPVTYETVPKVIGRISRAAMGLSAEQMREVNYKIVEIWETASHEIHKGETEKTKELILEVVE…`

### Lire `failure_csv.csv` — comment faire et ce qu'il dit

Une seule ligne de données, 64 colonnes = un compteur par filtre. Les rejets portent sur les
**séquences MPNN** (jusqu'à 20 par trajectoire, évaluées sur plusieurs modèles AF2), pas sur
les trajectoires — d'où 156 rejets pour 19 designs MPNN dans `mpnn_design_stats.csv`.

```python
import csv
r = list(csv.reader(open('failure_csv.csv')))
nz = [(h, int(v)) for h, v in zip(r[0], r[1]) if v.strip() not in ('', '0')]
for h, v in sorted(nz, key=lambda x: -x[1]): print(f"{v:>3}x  {h}")
```

Résultat sur `test1` — 9 filtres sur 64 ont rejeté au moins une fois :

| Rejets | Filtre |
|---|---|
| 38 | `i_pAE` |
| 35 | `i_pTM` |
| 30 | `pTM` |
| 20 | `pLDDT` |
| 14 | `ShapeComplementarity` |
| 12 | `n_InterfaceUnsatHbonds` |
| 5 | `Surface_Hydrophobicity` |
| 1 | `Trajectory_logits_pLDDT` |
| 1 | `Trajectory_Clashes` |

**Lecture :** le goulot est la **confiance de l'interface** (`i_pAE`, `i_pTM`, `pTM`), pas la
géométrie ni la chimie de surface. C'est le mode d'échec attendu de BindCraft : AF2 ne croit
pas assez au complexe. Les filtres physiques (clashes, hydrophobicité) ne rejettent presque
rien. C'est une information sur la difficulté de la cible, pas un réglage à corriger.

### Contenu rapatrié

```
out/test1/
├── Trajectory/   (6 fichiers)  + Animation/ Clashing/ LowConfidence/ Plots/ Relaxed/
├── MPNN/        (38 fichiers)  + Binder/ Relaxed/ Sequences/
├── Accepted/    (12 fichiers)  + Animation/ Pickle/ Plots/ Ranked/
├── Rejected/
├── trajectory_stats.csv       (3 lignes)
├── mpnn_design_stats.csv     (19 lignes)
├── final_design_stats.csv     (1 ligne)
└── failure_csv.csv
```

## Critères d'acceptation — état

- [x] `jax.devices()` → GPU CUDA visible : `[CudaDevice(id=0)]` sur L40S
- [x] Image se construit sans erreur, 2e build caché (10 min → 12 s)
- [x] Poids AF2 ne se retéléchargent pas (cachés avec la couche de l'Image)
- [x] Run détaché sur `PDL1.pdb` sans exception (47 min 25 s, app `stopped`)
- [x] `modal volume get bindcraft test1 ./out/` ramène les sorties (78 fichiers)
- [x] `Trajectory/`, `MPNN/`, `Accepted/`, `final_design_stats.csv`, `failure_csv.csv` existent
- [x] On sait lire `failure_csv.csv` : goulot = confiance d'interface (i_pAE 38, i_pTM 35, pTM 30)
- [x] Durée et coût du run notés ici : 47 min 25 s, $1,54

**Les 8 critères d'acceptation sont remplis. L'étape « test » est passée.**

Bonus non requis : 1 design a passé tous les filtres par défaut, sans qu'aucun seuil
n'ait été touché.

## Parallélisation — palier 1 validé (24/09/2026)

⚠️ **Extension de périmètre.** La parallélisation est listée dans les anti-objectifs de
`CLAUDE.md`. Ce travail a été fait sur demande explicite de l'utilisateur, après que le
conflit ait été signalé. `CLAUDE.md` n'a pas été mis à jour et décrit donc un périmètre
plus étroit que l'état réel du dépôt.

### Pourquoi c'est une nécessité et non une optimisation

500 trajectoires demandent **~75 GPU-heures**, donc c'est **impossible** en un run, le
timeout étant de 5 h (~33 trajectoires max). Le coût en GPU-heures est le *même* en
parallèle ; ce qui change, c'est le temps mural et la faisabilité.

### ⚠️ Correction de baseline — les 5,5 min étaient trompeurs

**La baseline de ~5,5 min/trajectoire tirée de `test1` était la durée de la descente de
gradient seule, pas du cycle complet.** Mesuré sur `par-test2` : **9,0 min par trajectoire
complète**, soit un facteur 1,6.

La raison est structurelle. Dans `test1`, `number_of_final_designs=1` arrêtait le run dès le
premier design accepté, donc peu de cycles MPNN étaient exécutés. En mode shard ce paramètre
est neutralisé à `10**9`, si bien que **chaque** trajectoire réussie enchaîne le cycle MPNN
complet : ~20 séquences générées, chacune repliée par AF2 puis scorée par PyRosetta. C'est ce
cycle, et non la descente de gradient, qui domine le temps.

Conséquence sur les estimations, à corriger partout :

| | Estimé (faux) | Réel |
|---|---|---|
| Par trajectoire | 5,5 min | **9,0 min** |
| `par-test2` (12 traj) | $1,90 | **$3,52** (×1,85) |
| 500 trajectoires | ~$92 | **~$147** |

**Pour dimensionner : compter 9 min et $0,29 par trajectoire**, pas 5,5 min.

### Architecture : trois éléments

- **`bindcraft_shard`** — un `design_path` par shard (`/outputs/<run>/shard-NNN/`).
  C'est ce qui rend le sharding sûr : les 4 CSV sont construits sous `design_path`, donc
  des chemins distincts garantissent qu'aucun fichier n'est écrit par deux conteneurs.
  Les volumes Modal sont *last-write-wins* : un chemin partagé perdrait des données
  **en silence**.
- **`aggregate`** — **sans `gpu=`** : concaténer des CSV sur un GPU serait du gaspillage.
  Appelle `volume.reload()`, sans quoi il ne verrait rien de ce que les shards ont écrit.
  Refait le ranking sur l'**union** (N shards produisent chacun leur « rank 1 »).
- **`parallel`** — `starmap(..., return_exceptions=True)` pour qu'un shard mort n'emporte
  pas les autres.

### Deux pièges coûteux, traités dans le code

1. **`number_of_final_designs` doit être neutralisé dans les shards** (mis à `10**9`).
   La boucle s'arrête dès que `accepted_designs >= number_of_final_designs` : le laisser à 1
   ferait quitter chaque shard à son premier succès, payant N démarrages et N compilations
   JIT pour à peine plus qu'un run simple.
2. **Le surcoût JIT se paie par shard** (~3 min de GPU payées à ne rien produire). Il dicte
   le découpage. Gaspillage = `3/(3+9k)` avec k trajectoires par shard :

| Trajectoires/shard | Gaspillage JIT |
|---|---|
| 1 | 25 % |
| 5 | 6 % |
| 10 | 3 % |
| 20 | 1,6 % |

   *(Un tableau antérieur annonçait 35 % à k=1 ; il était calculé sur la baseline erronée de
   5,5 min. Corrigé sur 9 min.)*

### Dimensionner les shards : deux effets de nature différente, et un plafond dur

- **Le JIT gaspille de l'argent** — des GPU-minutes payées pour compiler. Argument pour de
  gros shards.
- **Le déséquilibre gaspille du temps, pas de l'argent.** Modal facture à la seconde et par
  conteneur : le shard qui finit en 29 min arrête de facturer à 29 min. Les 17 min d'écart
  avec le plus lent ne coûtent rien, elles retardent seulement l'agrégation, qui attend tout
  le monde. Argument pour de gros shards aussi, mais pour une autre raison.
- **Plafond : le timeout de 5 h.** À 9 min/trajectoire, un shard plafonne à ~33 trajectoires.
  Avec les 56 % de déséquilibre observés, un shard dimensionné pour 20 trajectoires (180 min)
  peut réellement en prendre 280 — proche des 300. Il faut donc de la marge.

| Trajectoires/shard | Verdict |
|---|---|
| 1-3 | gaspillage JIT, variance ingérable |
| **15-20** | **zone recommandée** — JIT ~2 %, marge confortable au timeout |
| 25-33 | possible, mais un shard lent risque de mourir au timeout |
| > 33 | impossible sans augmenter `TIMEOUT` |

Pour 500 trajectoires : **25-33 shards × 15-20 trajectoires**, ~$147, temps mural ~3 h —
et l'orchestration côté serveur devient obligatoire à cette durée.

### Résultat du palier 1 : `par-test`, 2 shards × 1 trajectoire

```bash
GPU=L40S .venv/bin/modal run modal_bindcraft.py::parallel \
  --input-pdb inputs/PDL1.pdb --n-shards 2 --trajectories-per-shard 1 --run-name par-test
```

App `ap-z5GmI837qeAfVTG0d5z2xh`, **2 tasks simultanées** confirmées. `2 ok, 0 failed`.
Durées : shard-000 **7 min 59**, shard-001 **12 min 33** → temps mural 12 min 33,
coût **$0,67** (20,5 GPU-min).

**Les seeds diffèrent bien entre conteneurs** — c'était le point critique, car des seeds
identiques rendraient la parallélisation inutile. Observé : `PDL1_l101_s151500` et
`PDL1_l105_s353332`. `np.random` non seedé ([modal_bindcraft.py:371](modal_bindcraft.py#L371))
s'initialise sur l'entropie de l'OS, donc aucun patch n'est nécessaire. Vérifié, pas supposé.

Agrégation correcte : `trajectory_stats` 2 lignes (1/shard), `mpnn_design_stats` 4 lignes
(2/shard), `failure_csv` sommé à 109 rejets sur 7 filtres. Profil identique à `test1` —
`i_pAE` 36, `i_pTM` 33, `pTM` 28 — ce qui confirme la reproductibilité du diagnostic.

Trou du palier 1, **comblé par le palier 2** : aucun design accepté sur ces 2 trajectoires,
donc le chemin de code du ranking global n'avait pas été exercé.

### Palier 2 : `par-test2`, 3 shards × 3 trajectoires — ranking validé

```bash
GPU=L40S .venv/bin/modal run modal_bindcraft.py::parallel \
  --input-pdb inputs/PDL1.pdb --n-shards 3 --trajectories-per-shard 3 --run-name par-test2
```

App `ap-gSqGKjjq8qj8AW6t4N5fDY`, 3 tasks simultanées, `3 ok, 0 failed`.
**12 trajectoires tentées pour un quota de 9 réussies** — reconfirme que le quota porte sur
les réussies. 25 designs MPNN évalués, **8 acceptés**, 244 rejets sur 8 filtres.

Durées : **29 min 46 / 32 min 10 / 46 min 21**. Temps mural 46 min 21, coût **$3,52**
(108,3 GPU-min).

**Le ranking global est correct, vérifié sur les valeurs et non sur la présence des fichiers :**

| Rang | Shard | Average_i_pTM |
|---|---|---|
| 1 | shard-000 | 0,83 |
| 2 | shard-000 | 0,83 |
| 3 | shard-000 | 0,81 |
| 4 | shard-000 | 0,80 |
| 5 | shard-001 | 0,77 |
| 6 | shard-001 | 0,76 |
| 7 | shard-002 | 0,74 |
| 8 | shard-002 | 0,72 |

Décroissance stricte, les trois shards entrelacés dans un classement unique, aucun design
tombé dans le cas par défaut du `sort_key` (qui retourne `len(order)` sur clé absente — un
mapping cassé aurait donné un ordre arbitraire **sans erreur ni avertissement**).
Ce contrôle est à refaire après toute modification de `aggregate`.

### Relecture avant gel du code (24/09/2026) — un bug trouvé et corrigé

Le code est gelé avant la compétition ; relecture faite à cette occasion.

**Bug corrigé : la colonne `Rank` du `final_design_stats.csv` agrégé était fausse.** Chaque
shard classe ses propres designs 1..n, donc la concaténation portait des rangs dupliqués —
trois `Rank=1` et trois `Rank=2` sur `par-test2`. `Accepted/Ranked/` était correct, mais
quiconque triait le CSV par `Rank` obtenait un ordre arbitraire. Symptôme silencieux :
aucune erreur, juste des données fausses.

Correction : `final_design_stats.csv` est maintenant traité **après** le ranking et
renuméroté contre le même `order` que les PDB. Vérifié sur `par-test2` — `Rank` 1→8
séquentiel, `i_pTM` décroissant, et les deux vues (CSV et `Accepted/Ranked/`) s'accordent
rang par rang.

Validé **sans GPU** en rejouant `aggregate` seul sur les données déjà produites :

```bash
.venv/bin/modal run modal_bindcraft.py::aggregate --run-name par-test2
```

C'est la manière de tester l'agrégation pour quelques centimes : `aggregate` n'a pas de
`gpu=`, donc il ne consomme que du CPU.

Deux durcissements au passage :

- `bindcraft_shard` lève une `ValueError` explicite si `run_name` manque, au lieu d'un
  `KeyError` opaque.
- `aggregate` affiche un `WARNING` si un PDB accepté est absent de `mpnn_design_stats.csv`.
  Ces designs retombent sur `len(order)` et se classent en dernier **sans erreur** ; sans
  cet avertissement, un mapping cassé passerait inaperçu. Aucun cas sur `par-test2`.

Non corrigé volontairement : pandas émet des `PerformanceWarning` (« DataFrame is highly
fragmented ») en insérant la colonne `Shard` dans des frames à 232 colonnes. C'est de la
performance, pas de la justesse, et négligeable à cette échelle. Modifier le code juste avant
un gel pour un avertissement cosmétique serait un risque de régression gratuit.

Vérifié aussi : `number_of_final_designs` ne sert qu'à la condition d'arrêt, donc le `10**9`
ne casse aucun calcul ; `main` et `bindcraft` sont inchangés, leur signature intacte ; les
six points d'entrée sont toujours découverts par Modal.

### `par-test3` (24/09) — run tué par une panne réseau locale

Test de non-régression après la correction du `Rank`, lancé en 2 shards × 2 trajectoires.
**Le run a échoué**, exit code 1, mais **pas à cause du code** :

```
socket.gaierror: [Errno 8] nodename nor servname provided, or not known
modal.exception.ConnectionError
Modal Client → Modal Worker heartbeat attempts have been failing for over 14.98 minutes
```

Échec de résolution DNS **sur le laptop** : la machine a perdu le réseau (veille, couvercle
fermé, coupure) pendant le run. Les heartbeats ont échoué 15 min, puis le client a abandonné
et l'app a été arrêtée.

**C'est la démonstration en conditions réelles du verrou déjà identifié** : l'orchestration
vit dans le `local_entrypoint`, donc sur le laptop. Ce n'est plus un risque théorique.

#### Manœuvre de récupération — à connaître, elle évite de repayer le GPU

Le `volume.commit()` dans le `finally` de `bindcraft_shard` a sauvé le travail : les deux
shards avaient leurs 4 CSV et leurs dossiers dans le Volume. Seul `aggregate` n'avait pas
tourné. Il suffit de le rejouer seul, **sans GPU** :

```bash
.venv/bin/modal run modal_bindcraft.py::aggregate --run-name <run>
```

Le travail GPU déjà payé est récupéré pour quelques centimes. C'est le bénéfice concret
d'avoir séparé `aggregate` des shards et de l'avoir laissé sans `gpu=`.

Résultat : 3 trajectoires réussies (au lieu de 4, les shards ayant été tués), 28 designs
MPNN, 91 rejets, **1 design accepté**. Aucun `WARNING` d'orphelin.

#### Ce que ce test valide, et ce qu'il ne valide pas

- ✅ Le flux `parallel` → shards → CSV fonctionne, et `aggregate` tourne sans erreur sur des
  données fraîches.
- ❌ **Le classement multi-shard n'est pas exercé** : un seul design accepté, donc `Rank=1`
  est trivialement correct. Ce test ne rejoue pas le cas qui avait révélé le bug.

**La correction du `Rank` reste néanmoins validée**, et par le bon test : elle est
entièrement contenue dans `aggregate`, pas dans les shards. Le rejeu sur `par-test2`
(3 shards, 8 designs acceptés, `Rank` 1→8 vérifié, CSV et `Accepted/Ranked` d'accord rang
par rang) exerce exactement le chemin de code modifié. Un flux complet n'y ajouterait que la
confirmation que les shards écrivent leurs CSV — ce qui était déjà établi par `par-test2`.

### Déséquilibre entre shards : 56 %

29 min 46 pour le plus rapide contre 46 min 21 pour le plus lent. Le temps mural étant dicté
par le plus lent, on paie de l'attente. Cause : à 3-4 trajectoires par shard, la variance du
nombre de tentatives et du succès MPNN est énorme. Avec 10-20 trajectoires par shard, la loi
des grands nombres lisse ça — raison supplémentaire de ne pas faire de shards minuscules.

### Limite connue : `--detach` et le mode parallèle

Modal avertit que le mode détaché ne garde en vie que **la dernière fonction déclenchée**.
En mode parallèle on en déclenche N+1 (N shards puis `aggregate`), donc `--detach` est
piégeux. Le palier 1 a tourné attaché. Pour un vrai run long il faudra une fonction
orchestratrice côté serveur (qui appelle `starmap` depuis un conteneur Modal) plutôt qu'un
`local_entrypoint` qui orchestre depuis le laptop.

## Risque identifié sur le run réel

`--number-of-final-designs 1` est un critère d'arrêt sur le **résultat**, pas sur l'effort :
le pipeline boucle jusqu'à ce qu'un design passe tous les filtres. L'estimation de ~$1 /
30 min suppose qu'un design passe vite. Si aucun ne passe, le run continue jusqu'à
l'arrêt par `enable_rejection_check` ou jusqu'au timeout de **5 h** — ce qui sortirait
largement du seuil « quelques dollars » de la règle 9.

D'où l'usage de `--max-trajectories` comme plafond dur sur le premier run. Le test valide
la plomberie, et CLAUDE.md dit explicitement que zéro design accepté ne l'invalide pas.

---

## Carte d'épitope EGFR (30/09/2026) — `egfr_epitope_map.py`

Premier run décisionnel du Challenge 1. Local, CPU, coût nul. Aucune dépense GPU ne doit
précéder le choix du patch : la cross-réactivité souris se joue ici, pas au filtrage.

Dépendances ajoutées à `pyproject.toml` : `biopython>=1.83` (1.88 installé),
`numpy>=1.26` (2.5.3 installé). Aucune n'était déclarée ni installée.

### UniProt n'annote pas le domaine III

P00533 n'a **qu'une** feature de type `Domain` : `712-979 Protein kinase`. Aucun
« Receptor L-domain ». Le script échouait donc sur `SystemExit` à l'étape 1.

Les L-domaines n'existent que comme `Repeat` marquées « Approximate » : 75-300 et
390-600 UniProt. La seconde vaut **366-576 mature**, décalée d'une cinquantaine de
résidus du domaine III structural. Inutilisable comme borne.

Décision : **définir la région structurellement**, depuis l'empreinte du Fab cétuximab
(chaînes B et C de 6ARU), et ne garder la `Repeat` qu'en colonne informative
(`uniprot_repeat2`). Aucune borne codée de mémoire — règle de CLAUDE.md respectée.

`REGION_RADIUS = 30.0` Å autour du centroïde de l'empreinte. Choisi comme le plus petit
rayon rendant la sélection **contiguë en séquence** : 311-511 avec 1 discontinuité,
contre 7 à 25 Å. Critère structural, pas une constante à la main.

### Faits établis depuis les fichiers

| | |
|---|---|
| Offset PDB → UniProt | **+24**, 99,7 % de concordance (609 résidus observés, chaîne A 4-612) |
| 6ARU | X-ray, **3,20 Å**, `_refine.ls_d_res_high` |
| Conformation | **repliée (tethered)**, contacts 242-253 ↔ 563-595 = bras de dimérisation ↔ domaine IV |
| Empreinte cétuximab | 24 résidus, 349-473, contacts lourds < 4,5 Å |
| N-glycosylation | 13 sites humains, **0 O-linked** ; les 10 sites murins sont un **sous-ensemble strict** des humains → masquer sur l'humain couvre la souris |

Conformation retenue pour le design : **repliée**, alignée sur la structure de référence
fournie par les organisateurs. La SASA étant calculée sur la chaîne A entière,
l'occlusion par les autres domaines dans cet état est prise en compte.

### Contrôle cétuximab — passé

| Ensemble | Identité humain/souris |
|---|---|
| Protéine entière | 90,6 % |
| Région 311-511 | 87,2 % |
| **Empreinte cétuximab** | **70,8 %** |

L'empreinte est appauvrie en conservation par rapport aux deux références. Divergences :
353, 418, 443, 467, 468, 471, 473.

L'attendu est **inversé** par rapport à l'intuition : le cétuximab ne reconnaissant pas
l'EGFR murin, une fonction de score saine doit classer son épitope **bas**. C'est le cas —
le patch E472, qui contient 4 résidus de l'empreinte, sort dernier (67 % identité,
4 divergences). Le masque de conservation mesure quelque chose.

Un contrôle qui aurait montré l'empreinte fortement conservée aurait signalé un offset
faux ou un alignement cassé, pas une bonne nouvelle.

### Deux corrections apportées au scoring

1. **Le seuil glycane ne portait que sur l'ancre.** `dist_glycan > GLYCAN_EXCLUSION`
   filtrait l'ancre, alors que le patch était rapporté par le minimum sur ses *membres*.
   Une ancre à 20 Å pouvait porter des membres à 6 Å. Le seuil porte désormais sur le
   patch entier. **Effet : 7 patches sur 9 écartés** — E320 (6 Å), D323 (7 Å), E489 (5 Å),
   E397 (10 Å), E495 (11 Å), E472 (11 Å), E400 (12 Å).
2. **Déduplication sur le recouvrement des membres** (`MAX_PATCH_OVERLAP = 0.5`) et non
   sur la distance entre ancres. À `PATCH_RADIUS = 11`, deux ancres séparées de 8 Å
   partagent encore les deux tiers de leur patch : E320 et D323 partageaient 6 membres
   sur 9 tout en étant comptés comme distincts.

Clé de tri laissée volontairement inchangée à `(-frac_ident, -n)`. Colonnes `n_hydro` et
`n_acidic` ajoutées pour mesurer la tension avant tout score composite.

### Résultat : le funnel s'effondre à un seul site

```
9 ancres acides conservées et exposées
  → 7 écartées par le seuil glycane appliqué au patch
  → 2 survivantes (E431, D434)
  → 1 site distinct après déduplication (elles partagent 5 membres sur 9)
```

Site retenu, **E431** : 9 résidus exposés, 100 % identité humain/souris, 0 divergence,
glycane le plus proche à 16 Å, hors empreinte Fab.
Patch : `E431 K430 G458 E400 R403 K455 S428 D434 T459`.
Hotspots suggérés : `A431,A430,A458,A400`.

### Le problème, et il n'est pas dans le code

**Aucun des 8 sites candidats n'avait de contenu apolaire** (0 à 2 hydrophobes exposés).
Le site survivant est à **0 hydrophobe**. Tous les sites ancrés sur un Asp/Glu conservé
sont des surfaces polaires et chargées — le cas le plus défavorable au design de novo.

La tension designabilité / ancrage acide n'est donc pas un arbitrage : il n'y a pas
d'option apolaire dans cet ensemble. Et un seul site ne fait pas une campagne : pas de
diversité d'épitope possible, donc rien à clusteriser à la sélection finale.

**Décision à prendre avant toute dépense GPU** — la conjonction de contraintes est trop
serrée. Trois relâchements possibles, aucun encore appliqué :

- autoriser les ancres `similar` et non seulement `identical` (l'identité régionale est
  de 87 %, le critère « identique » est peu discriminant et coûte des candidats) ;
- recalibrer `GLYCAN_EXCLUSION = 12.0` Å, seuil jamais calibré et qui à lui seul écarte
  7 patches sur 9 ;
- découpler : choisir le site sur la designabilité, et traiter l'ancrage acide comme
  départageur plutôt que comme filtre dur.

Le funnel tel quel est publiable — c'est un résultat négatif documenté — mais il ne
produit pas de lot soumissible.

---

## Refonte de la carte d'épitope (01/10/2026) — `egfr_epitope_map.py`

Second passage sur le script. Le run du 30/09 concluait à un funnel effondré à un seul
site, sans contenu apolaire, et posait l'incompatibilité ancrage acide / designabilité
comme hypothèse. **Cette conclusion était un artefact d'échantillonnage, et elle est
maintenant réfutée par la mesure.**

### Le défaut : l'énumération était le filtre

`rank_patches` itérait sur les ancres acides conservées et exposées, puis ramassait les
membres dans un rayon. « Être centré sur un Asp/Glu conservé » n'était donc pas un
critère appliqué à un échantillon de sites — c'était la **définition** du site. Conclure
« 0 hydrophobe sur les 8 patches » revenait à constater que des disques centrés sur des
résidus chargés sont polaires. Tautologie, pas mesure.

Correction : les centres sont désormais **tous les résidus exposés du domaine III**.
L'ancrage acide devient deux colonnes comptées parmi les membres (`n_acidic`,
`n_acidic_cons`). 9 ancres → **98 centres**.

### Bornes du domaine III : trois sources interrogées, aucune codée en dur

| Source | Segments « L-domain » sur P00533 | Verdict |
|---|---|---|
| CATH via PDBe `/mappings/cath/6aru` | — | **inutilisable** : la classification ne couvre que les chaînes B/C du Fab, rien sur la chaîne A |
| Pfam `PF01030` (InterPro) | 57-167 et **361-480** UniProt | **tronqué** : la description Pfam déclare *« missing the first 50 amino acid residues of the domain »* |
| CATH-Gene3D `G3DSA:3.80.20.20` (InterPro) | 25-213 et **333-530** UniProt | **retenu** |

**Retenu : CATH-Gene3D, domaine III = 333-530 UniProt = 309-506 PDB.** Motif : c'est la
même classification structurale que CATH, projetée sur la séquence, et elle couvre le
domaine entier là où Pfam l'ampute (+28 au N-term, −50 au C-term par rapport à Gene3D).

Règle de sélection du segment, **géométrique et non nominative** : parmi les segments
d'une source, on retient celui qui contient le plus de résidus de l'empreinte du Fab
cétuximab — l'épitope du cétuximab étant dans le domaine III. Aucun indice, aucun nom,
aucune borne en dur. Le domaine I (référence de la face, 25-213 UniProt = 1-189 PDB) est
désigné par exclusion depuis la même source.

**Corroboration rétrospective** : la région structurelle du 30/09, définie par un rayon
de 30 Å autour du centroïde de l'empreinte, donnait PDB 311-511 = UniProt 335-535. À 2-5
résidus près de la borne CATH-Gene3D. L'heuristique du rayon était juste ; elle est
maintenant remplacée par une source citable. `REGION_RADIUS` et `region_mask` supprimés.

### Numérotation : l'offset n'est plus une inférence

`_struct_ref_seq` du mmCIF déclare pour la chaîne A : P00533, auth **1-616 ↔ db 25-640**,
soit **offset +24**. Le balayage par maximisation de concordance donne **+24** à 99,7 %.
Les deux routes sont conservées et le script **s'arrête en cas de désaccord**.

`_struct_ref_seq_dif` déclare **8 écarts** sur la chaîne A :

| auth | cristal | UniProt | type | position |
|---|---|---|---|---|
| 516 | LYS | ASN | `conflict` (UniProt : *Sequence conflict*, Ref. 1 CAA25240) | hors domaine III |
| 610 | ARG | GLU | `conflict` | hors domaine III |
| 617-622 | HIS ×6 | — | `expression tag` | hors domaine III |

Ces deux conflits expliquent exactement le 99,7 % du balayage (607/609). Ils comptent
parce que `aa_human` est lu dans la **structure** tandis que le statut humain/souris est
dérivé de la séquence **UniProt** : aux positions concernées la table serait incohérente.
**Aucun ne tombe dans le domaine III** — impact réel nul, mais le script lève désormais
une alerte bruyante si c'était le cas.

### Glycanes : filtre dur retiré

**2 NAG seulement sont modélisés sur la chaîne A** de 6ARU (auth 709, 710), pour 1 sur C,
2 sur D, 2 sur E, + MAN×5 et BMA×1. À 3,20 Å un arbre glycanique est flexible et mal
ordonné : l'absence de densité n'est pas l'absence de glycane. Mesurer l'occlusion par la
SASA en gardant les NAG est donc **inapplicable** sur cette structure, et le masquage sur
l'annotation UniProt reste le bon choix pour l'expérience — la cible mesurée sera
glycosylée.

**13 séquons N-linked annotés, dont 5 dans le domaine III** : 352, 361, 413, 444, 528
UniProt. La densité est le vrai problème, pas la sévérité du seuil.

`GLYCAN_EXCLUSION` (filtre dur, écartait 7 patches sur 9) devient `GLYCAN_REF = 12.0`,
**sans recalibrage**, utilisé comme rampe linéaire d'atténuation : pénalité = 0 au contact
du séquon, 1 au-delà de 12 Å. Forme posée, non calibrée, affichée en colonne
(`glyc_penalty`, `score_glyc_pen`) et dans un second classement. Un patch n'est plus
supprimé par une constante jamais justifiée ; il est classé plus bas, motif lisible.

### SASA ventilée par élément

`ShrakeRupley` passe au niveau atomique. C et S = apolaire, N et O = polaire. Colonnes
`sasa_apolar`, `sasa_polar`, `apolar_frac` en Å². **Clé de tri = SASA apolaire absolue**,
pas le ratio, pas un comptage de `LIVFMWY` — un comptage rend invisibles les tiges
aliphatiques des Lys/Arg, qui exposent réellement du carbone. `n_hydro` conservé en
colonne pour comparer l'ancienne métrique à la nouvelle.

Réserve mesurée : **corrélation taille de patch / SASA apolaire absolue = 0,64**. Le tri
absolu classe donc en partie la taille. À garder en tête au moment de choisir.

### Colonne `face`

Le site de liaison de l'EGF est ménagé entre domaines I et III. Axe orienté du centroïde
du domaine III vers celui du domaine I ; produit scalaire positif = `ligand`, négatif =
`externe`. Répartition : **97 ligand / 101 externe** sur 198 résidus du domaine III.

### Résultats

```
198 résidus dans le domaine III, 87,4 % identité humain/souris
 98 exposés → 98 centres
 93 patches à ≥ 6 membres
```

Distribution des tailles avant toute retenue : min 5, médiane 9, max 16, moyenne 9,1.
Seuls **5 patches sur 98 (5 %) sont sous le minimum de 6** — le seuil hérité ne fait
quasiment rien dans ce régime, contrairement au précédent.

Déduplication : **aucun seuil fixé**, par décision. Le recouvrement avec l'union des
patches mieux classés est saturé (médiane 1,00), ce qui est attendu avec 98 centres
chevauchants. Balayage fourni à la place :

| seuil | 0,1 | 0,2 | 0,3 | 0,4 | 0,5 | 0,6 | 0,7 | 0,8 | 0,9 |
|---|---|---|---|---|---|---|---|---|---|
| sites | 7 | 8 | 9 | 11 | 13 | 14 | 16 | 21 | 26 |

### Ce que la mesure tranche

**L'incompatibilité ancrage acide / designabilité est réfutée.**

- **64 patches sur 93** portent au moins une ancre acide conservée.
- **15 patches** combinent `apolar_frac ≥ 0,55` **et** `n_acidic_cons ≥ 1`.
- Le premier au classement apolaire, C482, sort à 639 Å² apolaires et `apolar_frac` 0,57,
  avec 2 ancres acides conservées.

Le run du 30/09 ne voyait aucun site apolaire parce qu'il n'en avait énuméré aucun. La
tension posée comme hypothèse structurale n'existe pas dans les données : sur le domaine
III de l'EGFR, surface apolaire et ancre acide conservée coexistent largement.

Conséquence directe : **le découplage envisagé — soumettre des binders simples pour le
classement, garder le pH comme analyse méthodologique — n'a plus lieu d'être.** Les deux
objectifs sont poursuivables sur les mêmes sites.

Note de prudence sur le premier du classement : `C482` est une cystéine, donc très
probablement engagée dans un pont disulfure de la cible. Rien d'éliminatoire — c'est la
cible, pas le binder — mais le centre n'est pas un point d'ancrage à traiter naïvement.

### Supprimé

`annotated_repeat()` et la colonne `uniprot_repeat2` (l'annotation Repeat d'UniProt est
établie inutilisable, le fait est consigné, le code qui le réaffiche ne l'est plus) ;
`check_tethered()` (37 lignes re-dérivant à chaque run un fait déjà consigné : contacts
242-253 ↔ 563-595) ; la distance morte de `fab_footprint()` (la fonction renvoyait
`dict[int, float]` dont aucun des quatre appelants ne lisait le float).

### Ce qui n'est PAS décidé, volontairement

Aucun seuil de retenue n'est appliqué : ni déduplication, ni filtre glycane, ni critère
de conservation, ni critère de face. Le script sort les colonnes et les distributions.
La sélection se fait à la lecture, et sera consignée séparément.

### Pénalité glycane retirée (01/10, même journée)

`glyc_penalty` et `score_glyc_pen` supprimés, second classement supprimé. `min_glyc`
reste, **colonne brute en Å, ni filtre ni pondération**. Un seul classement, sur la SASA
apolaire absolue.

Motif, mesuré avant de trancher : la forme `clamp(min_glyc/12, 0, 1)` appliquée
multiplicativement à la SASA apolaire annulait **22 patches sur 93** (pénalité exactement
0 à `min_glyc = 0`) — le filtre dur, mais sans ligne de journal, donc en pire. Et
**77 patches sur 93** étaient pénalisés : ce n'était pas une correction marginale, ça
pilotait le classement, avec des écarts de rang jusqu'à −33 (C482 : rang 1 → 34). La pente
imposait de surcroît un taux de change jamais choisi : 1 Å de distance au séquon = 1/12 de
la SASA apolaire, soit 50 Å² sur un patch à 600 Å². Les distances se lisent à la main sur
la liste courte.

### [A] Empreinte cétuximab — la calibration enfin faite

C'est le seul point d'ancrage disponible : un site où une protéine se lie réellement.

| | |
|---|---|
| Patches touchant l'empreinte | **38 sur 93** |
| SASA apolaire, médiane tous patches | 352 Å² |
| SASA apolaire, médiane patches d'empreinte | **388 Å²** |
| Recouvrement maximal | **S468**, 9 des 24 résidus d'empreinte, **rang 34**, 402 Å², 40,2 Å²/membre |

**Ce que ça dit de l'échelle** : un binder protéique fonctionnel occupe un site à ~390-400 Å²
de SASA apolaire et ~40 Å²/membre. Les 639 Å² de C482 sont donc largement au-dessus du
point de calibration, et la gamme 400 Å² est démontrée suffisante — ce n'est pas une
estimation, c'est un cétuximab. Le haut du classement n'est pas un seuil à atteindre.

Note : les patches d'empreinte sont légèrement **au-dessus** de la médiane mais pas en tête
(rangs 6, 9, 13, 17, 34, 51, 57, 60, 67, 69, 72, 73). L'affinité réelle du cétuximab ne se
lit donc pas dans la SASA apolaire, ce qui borne ce que cette métrique prétend prédire.

**Défaut révélé par cette lecture — la colonne `face` ne mesure pas ce qu'elle annonce.**
Tous les patches d'empreinte sortent à `face = 0,00`, c'est-à-dire face externe. Or
l'épitope du cétuximab chevauche la surface de liaison de l'EGF sur le domaine III. La
cause est la conformation : **6ARU est replié, donc les domaines I et III sont écartés et
le site de liaison du ligand est démonté.** L'axe centroïde III → centroïde I ne suit pas
la face ligand dans cet état. `face` mesure « côté tourné vers le domaine I en conformation
repliée », ce qui n'est pas « face ligand ». À ne pas utiliser comme critère tant que ce
n'est pas refait sur une structure étendue — hors périmètre pour l'instant.

### [B] Densité apolaire par membre — le tri absolu classe bien la taille

| | |
|---|---|
| Corrélation `n` / SASA apolaire **absolue** | **+0,637** |
| Corrélation `n` / SASA apolaire **par membre** | **−0,182** |
| Taille médiane du top-10 absolu | **13,5** |
| Taille médiane du top-10 par membre | **7,5** |
| Intersection des deux top-10 | **3 / 10** (T358, S356, P361) |

Réponse nette : **oui, le tri absolu favorise systématiquement les gros patches.** Les deux
classements ne décrivent pas le même ensemble.

Et le classement par densité fait remonter exactement ce que le challenge demande — des
patches petits, denses, **et** doublement qualifiés :

| rang densité | centre | n | Å²/membre | ident. | acidC | glyc | rang absolu |
|---|---|---|---|---|---|---|---|
| 1 | H359 | 6 | 80,9 | 0,83 | 0 | 5,8 | 12 |
| 2 | S356 | 7 | 80,4 | 0,71 | 0 | 7,8 | 4 |
| 3 | T358 | 8 | 78,3 | 0,88 | 1 | 5,8 | 2 |
| 4 | L325 | 7 | 60,0 | **1,00** | 1 | 7,3 | 27 |
| 6 | T330 | 8 | 57,2 | 0,88 | **2** | 5,8 | 15 |
| 8 | D323 | 8 | 53,7 | **1,00** | **2** | 7,3 | 23 |
| 9 | G317 | 9 | 49,0 | **1,00** | **2** | 7,3 | 20 |
| 10 | K333 | 6 | 48,7 | **1,00** | **2** | 4,8 | 68 |

Quatre patches à **100 % d'identité humain/souris avec 2 ancres acides conservées**, dont
trois invisibles dans le top-10 absolu (rangs 23, 20, 68). G317 / D323 / K333 / T330 / L325
occupent PDB 317-333 = **UniProt 341-357**, soit la portion N-terminale du domaine III que
**Pfam aurait amputée** (PF01030 démarre à 361). Le choix de CATH-Gene3D se paie
directement ici.

À noter : D323 et E320 étaient les ancres écartées le 30/09 par le filtre glycane dur
(6 et 7 Å). Le retrait du filtre les ressuscite, cohérent.

### [C] Cystéines pontées — réserve légitime, effet négligeable

25 ponts disulfure annotés sur P00533, dont **6 touchant le domaine III** : (329,333),
(337,362), (470,499), (506,515), (510,523), (526,535) UniProt. Soit 10 cystéines pontées
dans le domaine, PDB 309, 313, 338, 446, 475, 482, 486, 491, 499, 502.

**21 patches sur 93** contiennent au moins une cystéine pontée. Sur le top-5 apolaire :

| centre | cysSS | apolaire | dont cysSS | part | corrigé | rang corrigé |
|---|---|---|---|---|---|---|
| C482 | 1 | 639 | 15 | **2,3 %** | 624 | 2 |
| T358 | 0 | 627 | 0 | 0,0 % | 627 | 1 |
| Q480 | 1 | 596 | 15 | 2,5 % | 581 | 3 |
| S356 | 0 | 563 | 0 | 0,0 % | 563 | 4 |
| V481 | 1 | 557 | 15 | 2,6 % | 542 | 5 |

Réponse : **non, le haut du classement ne tire pas sa SASA apolaire de soufres pontés.**
La contribution plafonne à 15 Å², soit 2,3-2,6 %. Le soufre de C482 est largement enfoui
dans son pont avec C491 (UniProt 506-515). Défalquer ne change qu'une permutation
C482 ↔ T358. Ma réserve était fondée sur le principe et sans portée quantitative — dit
franchement, elle ne méritait pas le rang qu'elle occupait dans mon message.

Le reste de la réserve tient quand même, pour une autre raison : un centre sur cystéine
pontée est un résidu structurellement contraint, pas un point d'accroche à solliciter.

### Déduplication : toujours aucun seuil

Le balayage ne montre aucune coupure naturelle (continuum 7 → 26 sites de 0,1 à 0,9).
La liste se lit à la main. Aucun seuil écrit dans le code.

### Lever le confond taille / troncature (01/10) — hypothèse réfutée

Les centres restent dans le domaine III, les membres sont désormais repris sur **toute la
chaîne A** (327 résidus exposés au lieu de 98). Colonnes `*_full` en parallèle des colonnes
masquées, plus `n_truncated`.

**Hypothèse testée** : les cinq patches 317-333 auraient une SASA absolue amputée par le
masque, leur densité restant intacte. **Faux. `n_truncated = 0` pour les dix patches du
top-10 absolu ET pour les dix du top-10 densité.** Aucun des vingt n'est tronqué. La
faible SASA absolue des patches 317-333 est réelle, pas un artefact de bord.

Mais le mécanisme existe, sur d'autres patches : **15 sur 93 sont tronqués**, troncature
médiane 2 résidus, et quatre le sont lourdement — ils étaient sous-évalués :

| centre | n → n_full | apolaire masquée → full | écart |
|---|---|---|---|
| R310 | 6 → 11 | 229 → 454 Å² | **+98 %** |
| K311 | 9 → 12 | 349 → 536 Å² | +54 % |
| C502 | 10 → 17 | 399 → **609 Å²** | +53 % |
| N337 | 8 → 11 | 356 → 497 Å² | +40 % |

Les centres concernés sont ceux des vrais bords : 310, 311, 336, 337 d'un côté, 497, 501,
502, 503 de l'autre, plus quelques-uns dont la sphère de 11 Å franchit la borne (372, 375,
397, 398, 428, 483, 484). C502 à 609 Å² full serait dans les trois premiers ; masqué il
n'apparaissait pas. Le classement absolu doit donc se lire sur les colonnes `_full`.

### Ce qui explique vraiment l'écart absolu / densité

Voisinage à 11 Å, résidus totaux du domaine III contre résidus exposés :

| centre | exposés | total | frac. exposée | Å²/membre |
|---|---|---|---|---|
| C482 | 15 | 30 | 0,50 | 42,6 |
| Q480 | 16 | 24 | 0,67 | 37,2 |
| R470 | 16 | 27 | 0,59 | 33,6 |
| H359 | 6 | **11** | 0,55 | 80,9 |
| S356 | 7 | **11** | 0,64 | 80,4 |
| D323 | 8 | **12** | 0,67 | 53,7 |
| K333 | 6 | **13** | 0,46 | 48,7 |
| L325 | 7 | 22 | **0,32** | 60,0 |
| T330 | 8 | 21 | **0,38** | 57,2 |

La fraction exposée est comparable dans les deux groupes (0,46-0,67). Ce qui diffère, c'est
le **nombre total de résidus dans la sphère** : 11-13 pour H359/S356/D323/K333 contre 24-30
pour C482/Q480/R470. À rayon fixe, un voisinage peu peuplé signifie que la surface
**s'incurve en s'éloignant** — protubérance convexe.

**Conséquence à marquer comme hypothèse, non mesurée ici** : une protubérance convexe est
*défavorable* au design de novo, pas favorable. Un binder maximise la surface enfouie, ce
qui demande une cible plutôt concave ou plate ; une crête étroite en offre peu. Si cela se
confirme, la métrique de densité sélectionne **contre** la designabilité, et mon
commentaire de la lecture [B] — « la densité est mieux alignée » — était prématuré.

L325 et T330 sont un cas distinct : voisinage peuplé (22, 21) mais peu exposé (0,32, 0,38),
donc plutôt une crevasse ou une arête qu'une protubérance.

### Plancher 400 Å² puis tri sur apolar_frac

`FLOOR_APOLAR = 400.0`, adossé à la calibration cétuximab — seul seuil du fichier appuyé
sur une mesure externe. Appliqué sur `sasa_apolar_full`. **40 patches passent, 53 tombent.**
K333 (292 Å²) tombe comme prévu.

| rg | centre | n_full | apol_full | apfr_full | ident | acidC | glyc | dFab |
|---|---|---|---|---|---|---|---|---|
| 1 | H359 | 6 | 486 | 0,73 | 0,83 | 0 | 5,8 | 8,3 |
| 2 | T358 | 8 | 627 | 0,69 | 0,88 | 1 | 5,8 | 6,0 |
| 3 | S356 | 7 | 563 | 0,68 | 0,71 | 0 | 7,8 | 0,0 |
| 4 | **L325** | 7 | 420 | 0,62 | **1,00** | 1 | 7,3 | 0,0 |
| 5 | P361 | 9 | 523 | 0,61 | 0,78 | 1 | 5,8 | 0,0 |
| 6 | G479 | 10 | 410 | 0,61 | 0,50 | 0 | 11,4 | 4,3 |
| 7 | V481 | 13 | 557 | 0,60 | 0,61 | 1 | 9,9 | 0,0 |
| 8 | **D323** | 8 | 430 | 0,58 | **1,00** | **2** | 7,3 | 6,0 |
| 9 | V500 | 11 | 491 | 0,57 | 0,82 | 1 | 5,0 | 9,2 |
| 10 | C482 | 15 | 639 | 0,57 | 0,60 | 2 | 5,0 | 5,7 |
| 11 | Q480 | 16 | 596 | 0,57 | 0,62 | 1 | 6,0 | 0,0 |
| 12 | S468 | 10 | 402 | 0,57 | 0,50 | 0 | 7,6 | 0,0 |

Les plus proches du plancher par en dessous : K455 (392), K454 (379), P362 (374), T406
(374), K336 (367), N444 (364). G317 passe le plancher à 441 Å² mais sort du top-12 sur
`apolar_frac`.

**Seuls L325 et D323 cumulent** plancher franchi, `apolar_frac` ≥ 0,58, **100 % d'identité
humain/souris** et au moins une ancre acide conservée — D323 en ayant deux.

### Colonne `face` supprimée, remplacée par `dist_fab`

`face` reposait sur un axe **inféré** : centroïde du domaine III vers centroïde du
domaine I, censé repérer la face de liaison du ligand puisque le site de l'EGF est ménagé
entre ces deux domaines. **Elle ne repérait rien.** 6ARU est en conformation repliée : les
domaines I et III sont écartés et le site de l'EGF est démonté, donc l'axe ne suit pas la
face ligand. Le contrôle l'a prouvé — les douze patches recouvrant l'empreinte du cétuximab
sortaient tous à `face = 0,00`, face externe, alors que cet épitope chevauche le site de
l'EGF.

Remplacement par une **mesure** : `dist_fab`, distance de l'atome d'ancrage du résidu au
plus proche résidu de l'empreinte du Fab, 0 Å pour un résidu de l'empreinte lui-même. Le
cétuximab compétitionne l'EGF, son empreinte marque donc la surface ligand-compétitive.
Proxy mesuré plutôt qu'axe inféré, et dérivé d'un calcul déjà présent dans le script.

Distribution sur le domaine III : min 0,0, médiane **11,4**, max 28,9 Å. Agrégé par patch
en `min_dist_fab` et `mean_dist_fab`. `face_axis()` et `other_l_domain()` supprimés.

### Calibration du voisinage, et reclassement sur les colonnes `_full` (01/10)

**Correction de cadrage.** L'entrée précédente concluait « hypothèse réfutée » sur
`n_truncated = 0`. Mauvaise lecture : l'hypothèse était **mal ciblée**, pas fausse. La
troncature ne pénalise pas les patches 317-333 — elle en **cache d'autres entièrement**,
parce que le classement était calculé sur les colonnes masquées. Reclassé sur
`sasa_apolar_full`, quatre patches invisibles entrent dans le top-20.

### Top-20 reclassé sur `sasa_apolar_full`

| rang | centre | n | n_tot | f_exp | trc | apol_full | ap/res | apfr | ident | acidC | glyc | dFab |
|---|---|---|---|---|---|---|---|---|---|---|---|---|
| 1 | C482 | 15 | 31 | 0,48 | 0 | 639 | 42,6 | 0,57 | 0,60 | 2 | 5,0 | 5,7 |
| 2 | T358 | 8 | 18 | 0,44 | 0 | 627 | 78,3 | 0,69 | 0,88 | 1 | 5,8 | 6,0 |
| **3** | **C502** | 17 | 29 | 0,59 | **7** | **609** | 35,8 | 0,43 | 0,71 | 3 | **0,0** | 15,8 |
| 4 | Q480 | 16 | 24 | 0,67 | 0 | 596 | 37,2 | 0,57 | 0,62 | 1 | 6,0 | 0,0 |
| 5 | S356 | 7 | 11 | 0,64 | 0 | 563 | 80,4 | 0,68 | 0,71 | 0 | 7,8 | 0,0 |
| 6 | V481 | 13 | 21 | 0,62 | 0 | 557 | 42,8 | 0,60 | 0,61 | 1 | 9,9 | 0,0 |
| 7 | R470 | 16 | 27 | 0,59 | 0 | 537 | 33,6 | 0,46 | 0,56 | 1 | 0,0 | 0,0 |
| **8** | **K311** | 12 | 22 | 0,55 | **3** | **536** | 44,7 | 0,50 | 0,83 | 0 | **0,0** | 15,6 |
| 9 | T450 | 13 | 29 | 0,45 | 0 | 529 | 40,7 | 0,55 | 0,77 | 2 | 11,4 | 0,0 |
| 10 | P361 | 9 | 16 | 0,56 | 0 | 523 | 58,1 | 0,61 | 0,78 | 1 | 5,8 | 0,0 |
| **13** | **N337** | 11 | 20 | 0,55 | **3** | 497 | 45,2 | 0,52 | 0,82 | 0 | **0,0** | 17,8 |
| **18** | **S501** | 11 | 20 | 0,55 | **1** | 458 | 41,7 | 0,52 | 0,82 | 1 | **0,0** | 12,2 |

**Les quatre patches révélés sont tous à `min_glyc = 0,0`** — par construction, un de leurs
membres *est* un séquon N-linked (PDB 328, 337, 504 selon le cas). Et leur `dFab` est de
12 à 18 Å, donc loin de la surface ligand-compétitive. Ils n'étaient pas cachés par hasard :
ce sont les bords du domaine, là où la chaîne passe aux domaines II et IV, et ces bords sont
glycosylés. C502 à 609 Å² n'est pas un gain net.

### Calibration : la population du voisinage ne discrimine rien

Pas de mesure de concavité — pas le temps de la calibrer. À la place, le seul calibrateur
disponible : l'empreinte du cétuximab. Résidus totaux du domaine III dans les 11 Å.

| ensemble | n | min | p25 | médiane | p75 | max |
|---|---|---|---|---|---|---|
| tous les patches | 93 | 9 | 16 | **19** | 22 | 30 |
| patches d'empreinte | 38 | **11** | 16 | **20** | 24 | 29 |

**Les deux distributions sont confondues.** L'empreinte couvre toute la gamme, de 11 à 29.
Trois de ses patches tombent dans la bande « suspecte » ≤ 13 résidus : **S356 (11 total,
563 Å² apolaires), N473 (11), A477 (13)**. S356 est donc à la fois un voisinage parmi les
plus clairsemés du lot *et* une partie d'une surface qui lie réellement une protéine.

**Verdict, selon la règle posée avant la mesure : la courbure locale est écartée comme
critère.** Elle ne sépare pas une surface liable d'une surface non liable. Mon avertissement
de l'entrée précédente — « la densité sélectionne peut-être contre la designabilité » — est
donc levé sans mesure supplémentaire. Enrichissement résiduel trop faible pour agir :
13 patches sur 93 ont un voisinage ≤ 13, dont 3 d'empreinte (23 %) contre 41 % d'empreinte
dans le lot entier, sur n = 13.

### Classement plancher 400 Å² puis `apolar_frac_full`, top-20

Tête : H359 (0,73), T358 (0,69), S356 (0,68), **L325 (0,62 / 100 % ident / 1 acidC)**,
P361 (0,61), G479 (0,61), V481 (0,60), **D323 (0,58 / 100 % / 2 acidC)**, V500 (0,57),
C482 (0,57), Q480 (0,57), S468 (0,57), **G317 (0,57 / 100 % / 2 acidC)**, T330 (0,56 /
0,88 / 2 acidC), N452 (0,56), P494, E489, T450, T478, T464.
40 patches franchissent le plancher, 53 tombent.

**Les quatre patches à 100 % d'identité humain/souris avec ancre acide conservée** —
L325, D323, G317 et T330 à 88 % — franchissent tous le plancher et sortent entre les rangs
4 et 14 sur la fraction apolaire. Ce sont les seuls candidats qui servent les objectifs 1 et
2 sans compromis, et aucun n'est tronqué, aucun n'est sur un séquon (glyc 5,8-7,3 Å), tous
sont à 0-8 Å de l'empreinte du cétuximab.

### Diversité d'épitope : 3 sites disjoints, et un seul est propre (01/10)

Objectif : répartir le compute sur 3-4 sites spatialement distincts pour pouvoir
basculer un quota si un site ne produit rien. Disjonction mesurée en **fraction de
membres partagés**, jamais en distance entre centres — à `PATCH_RADIUS = 11`, deux centres
à 15 Å partagent encore la moitié de leurs membres.

Site de référence **désigné par les données** et non par des bornes : parmi les 40 patches
au-dessus du plancher, les composantes connexes des patches à identité parfaite, puis la
plus grande, étendue à ses satellites (`GROUP_LINK = 0.5`, valeur **posée**). Résultat :
7 patches, 20 membres — **D323, E320, G317, H359, L325, T330, T358**. C'est bien le site
317-330, obtenu sans le nommer.

Clé de tri des sites alignée sur l'**ordre des objectifs du règlement** : ancre acide
conservée (pH) > identité humain/souris (souris) > surface apolaire (affinité). Ordonner par
identité d'abord — ce que j'avais fait en première passe — démote T450 et N449 qui portent
des ancres acides. Erreur corrigée.

Seuil de disjonction non fixé, balayé : **τ = 0,00 et 0,10 donnent les mêmes 3 sites**,
τ = 0,25 en ajoute un quatrième. Recouvrements croisés tous à 0,00.

| | site 1 | site 2 | site 3 |
|---|---|---|---|
| représentant | **C502** | **K375** | **N449** |
| apol_full | 609 | 445 | 477 |
| apolar_frac | **0,43** | 0,47 | 0,50 |
| identité h/s | 0,71 | **1,00** | 0,77 |
| ancres acides cons. | **3** | 2 | 1 |
| min_glyc | **0,0** | 5,1 | **11,0** |
| dFab | 15,8 | **17,6** | **0,0** |
| patches du groupe | 6 | 3 | **8** |
| union | 27 membres | 20 | **30** |
| hors domaine III | **7** (507-530, dom. IV) | **6** (289-308, dom. II) | **0** |
| séquon dans le groupe | **C502, S501, R503** | **N337, R310** | **aucun** |

### Verdict franc

**Il y a bien 3 sites disjoints, mais un seul des trois alternatifs est propre.**

- **Site 3 (N449)** est la seule vraie alternative : 8 patches — le groupe le plus robuste
  du lot —, 30 membres **tous dans le domaine III**, **aucun séquon**, glycane à 11 Å, et
  `dFab = 0` donc sur la surface ligand-compétitive. Faiblesses réelles : **une seule**
  ancre acide conservée, identité 0,77, fraction apolaire 0,50. Il sert l'objectif 3 et
  à moitié l'objectif 1.
- **Site 2 (K375)** est le meilleur sur les objectifs 1+2 après la référence — identité
  **1,00** et 2 ancres acides — mais deux de ses trois patches portent un séquon, son
  glycane le plus proche est à 5,1 Å, il est à 17,6 Å de l'empreinte du cétuximab, et un
  tiers de son union est dans le **domaine II**. C'est un site à risque assumé, pas un
  site de repli.
- **Site 1 (C502) est à écarter.** Trois de ses six patches portent un séquon, un quart
  de son union est dans le **domaine IV**, et sa fraction apolaire est la plus basse du
  lot au-dessus du plancher (0,43). Sa SASA absolue de 609 Å² — la plus haute — est
  précisément l'artefact que les colonnes `_full` ont révélé : ce qui était caché était
  glycosylé.

**Conclusion à assumer : le domaine III de l'EGFR offre un bon site, un site exploitable,
et un site à risque.** Pas quatre sites équivalents. Répartition proposée pour jeudi :
la référence 317-330 en principal, N449 en secondaire, K375 sur le slot « à risque » que
le §7 de CLAUDE.md demande de garder. C502 non alloué.

---

## Relecture du règlement à la source (01/10/2026)

Page relue : <https://proteinbase.com/competitions/anthropic-adaptyv-2026/challenges/egfr>.
`challenge-01-egfr.md` est une synthèse datée du 30/09 ; cette entrée confirme ou corrige.

### Confirmé

| point | valeur à la source |
|---|---|
| Colonnes CSV | `name`, `sequence`, `molecule_class` |
| `molecule_class` | `protein`, `nanobody`, `scfv`, `fab_kappa`, `fab_lambda` |
| Plafond | Track 1 : 40. **Tracks 2 et 3 : 20** |
| Ordre du CSV | *« Submit your designs as a CSV ordered by how you would rank your molecules (top row higher). »* |
| Longueur | 10-250 aa |
| Catégories | microbinders < 40 ; **minibinders 40-100 inclus** ; grands binders > 100 ; nanobodies ; anticorps |
| Clôture | **4 octobre, 23:59 AoE** |

### Le piège : la page énonce les objectifs dans l'ordre INVERSE de leur poids

- Section « Three objectives » : **1. affinité, 2. cross-réactivité souris, 3. sélectivité pH.**
- Section « How designs are ranked » : **1. sélectivité pH, 2. cross-réactivité souris, 3. affinité.**

C'est l'ordre du **classement** qui compte, et il met le pH en premier. CLAUDE.md §1 était
donc juste, mais pour une raison qui n'était pas écrite : quiconque lit la seule section
« Three objectives » conclut que l'affinité prime. À ne pas relire de mémoire.

**Conséquence directe : la clé de tri des sites est validée.** `n_acidic_cons` →
`frac_ident` → `sasa_apolar` suit l'ordre du classement. La correction du 01/10, qui
plaçait l'ancre acide avant l'identité, était la bonne.

### Un durcissement à ne pas enjoliver

Formulation de l'objectif pH à la source : *« you must design a binder that binds human
EGFR at pH 6.5 and shows **no detectable binding** at pH 7.4. »* C'est un switch binaire,
plus exigeant que le « décalage de KD » que CLAUDE.md §7 donne comme attente réaliste.

La tension est réelle et reste à assumer telle quelle dans le dossier : les binders
pH-dépendants publiés obtiennent typiquement un facteur quelques-uns sur le KD, pas un
tout-ou-rien. On ne prétendra pas atteindre « no detectable binding » ; on mesurera et on
rapportera ce qu'on obtient.

## Commande Modal : `::main` obligatoire (01/10)

`modal_bindcraft.py` déclare deux `local_entrypoint` (`main`, `parallel`). Sans suffixe,
Modal refuse de choisir, **n'exécute rien, et sort en code 0** :

```
Error: Specify a Modal Function or local entrypoint to run.
...
[exited with code 0]
```

Corrigé dans CLAUDE.md §5. La commande de `test1` au 23/09 est annotée sur place, pas
réécrite. Un échec silencieux à code 0 est ce qui ne doit pas rester dans un fichier de
commandes : dans un script d'automatisation il passe pour un succès.

## Le plafond en trajectoires ne freine pas les tentatives

Lu dans la source BindCraft au commit `c0a48d5`, `functions/generic_utils.py` :

```python
def check_n_trajectories(design_paths, advanced_settings):
    n_trajectories = [f for f in os.listdir(design_paths["Trajectory/Relaxed"])
                      if f.endswith('.pdb')]
    if advanced_settings["max_trajectories"] is not False and \
       len(n_trajectories) >= advanced_settings["max_trajectories"]:
        return True
```

Le comptage porte sur **`Trajectory/Relaxed` uniquement**. Une trajectoire qui finit en
`LowConfidence` ou `Clashing` ne consomme pas le quota — tout en ayant consommé du GPU.
Donc `--max-trajectories N` plafonne les trajectoires **relaxées**, pas les tentatives.

**Le vrai frein de budget est `TIMEOUT`** (variable d'environnement, minutes, défaut 300),
appliqué comme `timeout=TIMEOUT * 60` sur la fonction Modal. CLAUDE.md §2 disait « tout run
sans `--max-trajectories` est un bug de budget » : c'est insuffisant. Un run **avec**
`--max-trajectories` mais un `TIMEOUT` à 300 peut brûler 5 h de GPU sur une cible où rien ne
relaxe. Le plafond en trajectoires reste utile comme garde secondaire, pas comme budget.

Autre fait vérifié dans `generic_utils.py` : **`create_dataframe` ne touche pas un CSV
existant** (`if not os.path.exists(csv_file)`). Les stats s'accumulent donc d'un run à
l'autre sous le même `run_name` — c'est ce qui rend la reprise possible.

---

## Run de fumée `smoke-G317` — EGFR domaine III (01/10/2026)

Premier run GPU sur la vraie cible. App `ap-JcN5ZalbdRyeDLQkJGRV4c`, L40S, détaché.

```bash
GPU=L40S .venv/bin/modal run --detach modal_bindcraft.py::main \
  --input-pdb inputs/6ARU_A_309-506.pdb --target-chains A \
  --target-hotspot-residues "A318,A320,A323,A325" \
  --lengths 50,130 --number-of-final-designs 100 \
  --max-trajectories 3 --run-name smoke-G317
```

Cible : `inputs/6ARU_A_309-506.pdb`, 198 résidus, produite par `prepare_target.py`.
Hotspots : les deux ancres acides D323/E320 et le cœur apolaire I318/L325 du patch
représentatif G317, tous à 100 % d'identité humain/souris.

### Deux leçons de commande

Première tentative sans `::main` : Modal refuse de choisir entre les deux
`local_entrypoint`, n'exécute rien, **et sort en code 0**. Aucune dépense. Corrigé dans
CLAUDE.md §5.

Seconde, de ma part : la sortie était pipée dans `tail -60`, ce qui **a détruit le log du
run**. Il n'en restait que les 60 dernières lignes. Rediriger vers un fichier, toujours.

### Chiffres mesurés

| grandeur | valeur |
|---|---|
| Mur total | **60 min 44 s** (3644 s) |
| Coût | **$1,97** (3644 s × $0,000542/s, tarif L40S confirmé) |
| Tentatives | **6** (3 relaxées + 2 LowConfidence + 1 Clashing) |
| Temps par tentative, tout compris | **10,1 min** (boot, échecs, MPNN et PyRosetta inclus) |
| Temps par trajectoire relaxée seule | 7 min 52 s, 8 min 12 s, 9 min 24 s → **moyenne 8,49 min** |
| `p_relax` | **3/6 = 0,50** |
| Designs acceptés | **2** |
| Modes de liaison indépendants | **1** — les 2 acceptés sont `mpnn1` et `mpnn2` de la *même* trajectoire |
| Coût par design accepté | **$0,99** |
| Coût par mode indépendant | **$1,97** |

Le temps par trajectoire dépend de la longueur du binder : 62 aa → 7 min 52 s, 63 aa →
8 min 12 s, 83 aa → 9 min 24 s. Environ +1,5 min pour +20 résidus. Pour la plage 55-95
(moyenne 75), compter ~8,7 min par trajectoire relaxée.

**La troncature a payé.** 8,49 min par trajectoire sur 198 résidus, contre 9,0 min mesurés
sur PD-L1 qui n'en fait que 115. Mon estimation a priori de 14-23 min était pessimiste d'un
facteur ~2 : la cible tronquée ne coûte pas plus cher que la démo.

### Profil de rejet — `i_pAE` domine massivement

`failure_csv.csv`, comptages cumulés par filtre :

| filtre | rejets |
|---|---|
| **`i_pAE`** | **20** |
| `pLDDT` | 6 |
| `i_pTM` | 3 |
| `Trajectory_one-hot_pLDDT` | 2 |
| `Trajectory_final_pLDDT` | 2 |
| `Trajectory_logits_pLDDT` | 1 |
| `Trajectory_Clashes` | 1 |
| tous les autres (pTM, pAE, dG, dSASA, SC, PackStat, hydrophobicité, H-bonds non satisfaites, RMSD…) | **0** |

`i_pAE` rejette plus que tous les autres réunis. C'est le seul levier qui compte si le
rendement doit monter : les deux designs acceptés sortent à `i_pAE` 0,17-0,18, donc les
rejetés sont très au-dessus. À vérifier contre la valeur réelle dans `default_filters.json`
avant de toucher à quoi que ce soit — **aucune modification de seuil n'a été faite**.

### Les deux designs acceptés

Tous deux issus de la trajectoire `l83_s142379`, donc **frères, pas indépendants**.

| | `_l83_s142379_mpnn1` | `_l83_s142379_mpnn2` |
|---|---|---|
| Longueur | **83 aa** | **83 aa** |
| Average_pLDDT | 0,93 | 0,93 |
| Average_i_pTM | 0,83 | 0,83 |
| Average_i_pAE | 0,18 | 0,17 |
| ShapeComplementarity | 0,67 | 0,69 |
| dG | −47,73 | −54,78 |
| dSASA | 1842,85 | 1969,24 |
| Surface_Hydrophobicity | 0,15 | 0,17 |
| Unsat H-bonds | 3,0 | 3,5 |
| **His dans la séquence** | **3** | **3** |

```
mpnn1 MKKLSKGEEVVEKVKKEAEELKEKLEKGELSLEEVEEKWVEIWKEAEKEAPETFHKISEVEYEFQLWLHHKRIEERKKKEEEE
mpnn2 MKKLSKGEKVVEEVRKKTEELKKRLEEGKLSIEEVEKEWVKIWKEAEKEAPETFHKISEVEYEFQLWLHHKKIEERKKKEEEE
```

**Deux observations qui comptent pour le challenge :**

1. **83 aa : dans la catégorie minibinders (40-100) sans l'avoir cherché.** Le run utilisait
   `--lengths 50,130`. La production passera à `55,95` pour garantir la catégorie.
2. **3 His chacun, et `InterfaceAAs` de la trajectoire `l83` donne `H: 3` — trois His à
   l'interface, sans aucun biais de composition appliqué.** La trajectoire `l63` en avait
   aussi 3. C'est inattendu, et c'est une donnée pour l'objectif n°1 : la baseline produit
   déjà des His d'interface. Reste à vérifier si elles sont appariées à un Asp/Glu de la
   cible — ce que le run ne dit pas et qu'il faudra mesurer sur les structures.

### Fin du run : annulation externe, cause non attribuée

Arrêt à 60 min 44 s sur `RemoteError: Function call was cancelled by user or a failure.`
Log distant :

```
2026-10-01T16:41:26+0000 Received a cancellation signal while processing input (...)
2026-10-01T16:41:26+0000 Successfully canceled input (...)
```

**Ni timeout, ni OOM, ni exception applicative.** Un signal d'annulation externe. Je ne
l'attribue pas : aucune commande lancée localement pendant le run n'annule, et le log ne dit
pas qui a demandé l'annulation. À surveiller sur le prochain run ; si ça se reproduit, c'est
un facteur de dimensionnement.

Le run était encore au travail à l'arrêt — dernière ligne utile
`Unmet filter conditions for ..._l62_s857853_mpnn1`, donc en pleine évaluation MPNN de la
troisième trajectoire. Les 3 relaxées du plafond étaient atteintes, mais la boucle ne teste
le plafond qu'au début de l'itération suivante.

### Acquis : une annulation commite le volume

**Tous les artefacts ont survécu** — 4 CSV, 3 trajectoires relaxées, 2 LowConfidence,
1 Clashing, 2 acceptés, 2 MPNN. Le `finally: volume.commit()` du wrapper a tourné malgré
l'annulation. C'est la moitié de la réponse à « un run interrompu perd-il tout ». Un kill par
**timeout** reste à vérifier séparément : même mécanisme probable, non testé.

### Arborescence confirmée

Un répertoire par `run_name` sous `/outputs`, à côté de `test1`, `par-test`, `par-test2`,
`par-test3` :

```
smoke-G317/{trajectory,mpnn_design,final_design}_stats.csv  failure_csv.csv
           Trajectory/{Relaxed,LowConfidence,Clashing,Plots,Animation}
           MPNN/  Accepted/  Rejected/
```

Un `run_name` par site suffit donc à séparer les sites en production.

### Dimensionnement de la production, chiffres réels

Formule posée précédemment, remplie. `r` = $1,95/h (L40S), `t_att` = 10,1 min tout compris,
`p_acc_indep` = 1 mode indépendant pour 6 tentatives.

```
mode independant       = 6 tentatives = 60,7 min = $1,97
20 modes independants  = 120 tentatives = 20,2 GPU-h = ~$39
reparti sur 3 sites    = ~6,7 GPU-h par site
```

Découpage retenu : **`TIMEOUT=135` (2,25 h), `k = 3` appels par site, 3 sites** →
20,25 GPU-h, **~$40**. Mur ~2,5 h si les 9 appels sont concurrents, ~6,8 h s'ils sont
séquentiels par site. Les deux tiennent très largement dans la fenêtre de 30 h.

Avec une marge ×2,5 pour absorber un `p_acc` plus faible sur les sites K375 et N449 :
~50 GPU-h, **~$100**, mur inchangé. C'est le dimensionnement à retenir.

`--number-of-final-designs` reste haut (100) et `--max-trajectories` généreux : le frein est
`TIMEOUT`, par construction. Le coût devient alors déterministe — `S · k · T · r` ne dépend
d'aucun taux de passage, ce que le plafond en trajectoires ne garantissait pas.

---

## Deux bugs trouvés par relecture (02/10/2026)

### Bug 1 — `member_set_full` fuit dans le CSV

`write_csv` reçoit `drop=("member_set", "members", "members_full")` mais pas
`member_set_full`, ajouté après. La colonne 41 de `egfr_patches.csv` contient donc un
`frozenset({...})` Python sur les 93 lignes.

Deux défauts, dont le second est le grave : c'est illisible, et **l'ordre d'itération d'un
`set` Python n'est pas stable entre exécutions**, donc le fichier n'est pas reproductible
octet à octet. Un CSV de ce dépôt doit l'être.

La colonne n'est pas supprimée — la jointure patches → résidus est utile et c'est elle qui
permet de dériver les hotspots. Elle est reformatée : entiers triés, séparés par des
points-virgules, sous le nom explicite `member_resnums_full`.

### Bug 2 — le filtre de taille mal ciblé, et sa vraie nature

`MIN_PATCH_MEMBERS` filtrait sur `n` **masqué** alors que tous les classements sont passés
aux colonnes `_full`. Trois patches écartés à tort : **C309 (5 masqués / 15 full, 638 Å²
apolaires)**, N504 (5/10, 333 Å²), E495 (5/7, 262 Å²).

**Hypothèse testée** : « un patch centré près d'une borne a peu de membres masqués par
construction, donc le filtre est un filtre de bord déguisé ». **Partiellement réfutée.**

| mesure | valeur |
|---|---|
| corrélation distance-du-centre-à-la-borne / `n` masqué | **−0,095** |
| `n` masqué médian, centres à ≤ 10 résidus d'une borne (14 patches) | **9,0** |
| `n` masqué médian, centres éloignés (84 patches) | **9,0** |
| sous le seuil de 6, près du bord | **2/14 (14 %)** |
| sous le seuil de 6, loin du bord | **3/84 (3,6 %)** |

Le comptage masqué n'est donc **pas** déprimé près des bornes en général : médianes
identiques, corrélation nulle. Mais parmi les patches qui tombent sous le seuil, les patches
de bord sont **sur-représentés d'un facteur ~4**. Et les 5 écartés se séparent proprement
selon la distance du centre à la borne :

| patch | dist. borne | n / n_full | trc | verdict |
|---|---|---|---|---|
| C309 | **0** | 5 / 15 | 10 | écarté à tort |
| N504 | **2** | 5 / 10 | 5 | écarté à tort |
| E495 | **11** | 5 / 7 | 2 | écarté à tort |
| F357 | 48 | 5 / 5 | 0 | **correctement écarté** |
| H409 | 97 | 5 / 5 | 0 | **correctement écarté** |

**La nature du bug, énoncée précisément** : le filtre n'annonce pas ce qu'il mesure, mais de
façon *sélective* et non systématique. Il prétend écarter les patches pauvres ; il écarte
les patches pauvres **et** les patches de bord riches, sans distinguer. C'est la même famille
d'erreur que la tautologie d'échantillonnage du §5 — un critère corrélé à autre chose que ce
qu'il annonce — mais à l'inverse de celle-là, le mécanisme ici ne s'applique qu'à une
minorité de cas. Dire « filtre de bord déguisé » surinterprète ; dire « mauvaise colonne »
sous-interprète. La formulation juste : **un filtre de taille qui, sur les patches de bord,
mesure la troncature au lieu de la taille.**

### Prédiction, écrite AVANT la re-exécution

1. Le filtre passe à `n_full`. Le lot passe de **93 à 96 patches**.
2. **E495 (262 Å²) et N504 (333 Å²) n'atteignent pas `FLOOR_APOLAR = 400`** : ils entrent
   dans le lot mais pas dans les 40 passants du plancher, donc **aucun effet sur l'analyse
   de sites**.
3. **C309 (638 Å²) franchit le plancher** : 40 → **41 passants**. Il se classerait **rang 2**
   du classement principal, entre C482 (639) et T358 (627).
4. **Je prédis que C309 ne forme pas un quatrième site, mais rejoint celui de K375.** L'union
   du site K375 contient déjà 309, 310, 311, 312 et descend jusqu'à 289 — C309 est au même
   endroit, la jonction domaine II / domaine III. Recouvrement attendu élevé, donc fusion.
5. **Je prédis que le regroupement en 3 sites disjoints tient**, C309 étant absorbé plutôt
   qu'inséré. Risque identifié : si C309 porte ≥ 3 ancres acides conservées, la clé pH-first
   le place avant C502 et il devient le représentant du site, ce qui changerait l'étiquette
   du site 1 sans changer sa composition.
6. **Je prédis d'écarter C309 quand même, par argument et non par rang** : 10 de ses 15
   membres sont hors du domaine III, donc ~2/3 de sa surface apolaire est portée par le
   domaine II et la jonction II/III. Le §4 a déjà tranché que la surface dépendant de
   l'arrangement inter-domaines en conformation repliée est celle dont on ne se fie pas —
   c'est l'argument qui a supprimé la colonne `face`. C309 est le cas limite exact de cette
   règle. Son rang 2 est un artefact de la conformation, pas une propriété de l'épitope.
7. C309 est de plus centré sur une **cystéine pontée** (309 = UniProt 333, apparié à 329),
   donc un centre structurellement contraint.

### Bug 3 — jointure lossy, corrigée dans le même geste

`egfr_residues.csv` ne contenait que les 198 résidus du domaine III, alors que
`member_resnums_full` puise dans les 327 exposés de la chaîne entière. La jointure
patches → résidus perdait donc des membres, **et le plus là où ça compte** : C502 perdait
7 membres sur 17, C309 en perdrait 10 sur 15. Les hotspots des patches de bord étaient
indérivables.

`egfr_residues.csv` couvre maintenant **toute la chaîne A (609 résidus)**, la colonne
`in_domain3` distinguant l'appartenance. Les agrégats imprimés restent calculés sur le
domaine III.

### Résultat de la re-exécution, confronté à la prédiction

Prédiction écrite et commitée en `6648868` **avant** la correction.

| # | prédiction | résultat | verdict |
|---|---|---|---|
| 1 | lot 93 → 96 patches | **96** | ✓ |
| 2 | E495 (262 Å²) et N504 (333 Å²) sous le plancher, aucun effet | passants 40 → **41**, donc seul C309 entre | ✓ |
| 3 | C309 franchit le plancher, **rang 2** à 638 Å² | **rang 2**, entre C482 (639) et T358 (627) | ✓ |
| 4 | C309 **fusionne** avec le site K375 (recouvrement élevé) | recouvrement **0,33** (5/15 membres) | **✗ mécanisme faux** |
| 5 | le regroupement en 3 sites disjoints tient | C502, K375, N449 inchangés à τ = 0,00 et 0,10 ; K463 quatrième à 0,25 | ✓ |
| 5b | risque : si C309 a ≥ 3 ancres acides il devient représentant du site 1 | C309 a **2** ancres, C502 en a 3 → C502 garde le slot | ✓ risque non matérialisé |
| 6 | C309 à écarter par argument, ~2/3 de sa surface hors domaine | **70 %** de sa SASA apolaire portée par les 10 membres hors domaine III | ✓ |
| 7 | C309 centré sur une cystéine pontée | `cys_bridged = True`, UniProt 333, apparié à 329 | ✓ |

Le site de référence est **inchangé** : 7 patches, 20 membres, D323 / E320 / G317 / H359 /
L325 / T330 / T358. C309 a une identité de 0,80, il n'entre donc pas dans la composante à
identité parfaite qui amorce la référence.

### La divergence, et elle compte

J'avais prédit que C309 serait **absorbé** par le site K375 — recouvrement élevé, fusion.
Le recouvrement mesuré est de **0,33**, soit un tiers. La conclusion survit (C309 n'est pas
un quatrième site) mais **pas pour la raison prédite** : il est rejeté parce que 0,33 dépasse
le seuil de disjonction à τ = 0,10 et τ = 0,25, pas parce qu'il décrit le même site.

**Conséquence qui n'était pas dans la prédiction : à τ = 0,50, C309 deviendrait un
quatrième site disjoint.** Son exclusion de la liste des sites dépend donc d'un seuil
délibérément non fixé. Ce n'est pas une position robuste, et c'est exactement pourquoi
l'arbitrage devait être argumenté plutôt que lu dans un rang.

### Décision sur C309 : écarté, et l'argument est accablant

Composition, membres triés par SASA apolaire, `*` = hors domaine III :

```
*Y292  68,8  identical     *K304  67,6  identical    *A289  65,2  different
 R310  61,6  identical     *E306  59,2  similar      *P308  53,1  identical
 V312  48,3  identical      K311  44,7  identical    *G288  33,8  identical
*K303  29,0  identical     *D290  26,6  identical     C309  25,7  identical
*G307  22,9  identical     *E293  21,7  identical     N337   9,7  different
```

Quatre des cinq membres les plus apolaires sont **hors du domaine III**. Au total **70 % de
la SASA apolaire de C309 est portée par le domaine II et la jonction II/III** — surface dont
l'exposition dépend de l'arrangement inter-domaines en conformation repliée, c'est-à-dire
exactement la catégorie que le §4 déclare non fiable et qui a fait supprimer la colonne
`face`. Son rang 2 mesure la conformation de 6ARU, pas une propriété de l'épitope.

Trois défauts s'ajoutent, chacun suffisant :
- centre sur **cystéine pontée** (C309 ↔ C329 UniProt), résidu structurellement contraint ;
- **N337 parmi les membres à `min_glyc = 0,0`** : un séquon dans le patch ;
- **A289 en statut `different`** au rang 3 par surface apolaire, donc l'objectif 2 est touché
  sur l'un des membres qui portent le plus de surface.

C309 est écarté. Le bug qui le masquait était réel et devait être corrigé ; le patch qu'il
révèle n'est pas exploitable. Les deux énoncés tiennent ensemble.

### Correction d'une affirmation antérieure sur N449

J'avais écrit que `N473`, statut `different`, était « un des 4 plus exposés » du patch N449.
**Faux** : il est au **rang 6 sur 13** par SASA apolaire, à 33,8 Å². La conclusion tient —
une divergence humain/souris dans le patch pèse sur l'objectif 2 — mais l'argument est plus
faible qu'écrit, et il ne disqualifie pas N449 à lui seul.

### État des CSV après correction

| | avant | après |
|---|---|---|
| `egfr_patches.csv` | 93 lignes, `frozenset({...})` en colonne 41 | **96 lignes**, `member_resnums_full` en entiers triés séparés par `;` |
| `egfr_residues.csv` | 198 lignes, domaine III seul | **609 lignes**, chaîne A entière, `in_domain3` distingue |
| jointure patches → résidus | lossy (C502 perdait 7 membres sur 17) | **complète** |
| reproductibilité octet à octet | non (ordre d'itération de `set`) | **oui** |

---

## Accessibilité des carboxylates des ancres acides (02/10/2026) — `carboxylate_access.py`

Motif : `egfr_epitope_map.py` retient une ancre acide sur `rel_sasa`, la SASA du **résidu
entier**. Or ce qui doit être atteignable pour un pont salin His–acide, c'est le
**carboxylate**. Un Asp au CB exposé mais aux OD1/OD2 rentrants passe le seuil `exposed` en
étant inatteignable. Le script sépare les deux, et mesure en plus l'**orientation**.

### La calibration prévue a échoué

L'idée était d'étalonner sur les Asp/Glu de l'empreinte du Fab de cétuximab — des acides
qu'une protéine vient réellement contacter. **L'empreinte n'en contient qu'un : E472.** n = 1,
aucune distribution exploitable.

C'est en soi un fait à noter : **l'épitope du cétuximab, 24 résidus, ne porte qu'un seul
Asp/Glu.** Un binder protéique qui fonctionne sur le domaine III le fait donc sur une surface
quasi dépourvue d'acides. C'est une mise en garde sur la stratégie d'ancrage acide, pas une
réfutation — mais elle mérite d'être dans le dossier.

### Distribution de référence, SASA du carboxylate seul

| ensemble | n | min | p25 | médiane | p75 | max |
|---|---|---|---|---|---|---|
| chaîne A entière | 65 | 1,1 | 27,9 | 47,1 | 70,7 | 94,3 |
| domaine III | 19 | 3,2 | 27,9 | 39,6 | 59,5 | 92,1 |

**Aucun carboxylate n'a une SASA nulle** — minimum 1,1 Å² sur 65. Le critère « SASA > 0 »
ne discrimine donc rien sur cette structure.

### Les ancres mesurées

`carbox` = SASA des deux oxygènes, `part` = sa fraction dans la SASA du résidu,
`angle` = angle entre CB→carboxylate et le vecteur sortant local. Angle faible = pointe vers
le solvant ; angle élevé = longe la surface ou rentre.

| site | ancre | carbox (Å²) | CB | part | angle | centile dom. III |
|---|---|---|---|---|---|---|
| **G317** | **D323** | **75,0** | 16,9 | **55 %** | **33°** | **89** |
| **G317** | **E320** | 30,0 | 25,4 | **28 %** | **103°** | 32 |
| N449 | E472 | 47,1 | 21,7 | 46 % | 86° | 63 |
| K375 | E400 | 72,9 | 4,8 | 76 % | 49° | 84 |
| K375 | E397 | 36,4 | 1,2 | 78 % | 19° | 42 |
| C502 | E530 | 82,5 | 4,8 | 86 % | 27° | 95 |
| C502 | E510 | 64,3 | 8,5 | 78 % | 52° | 74 |
| C502 | E489 | 27,9 | 0,0 | 44 % | 35° | 26 |

### Seuil proposé, et son statut

Faute d'étalon empirique, le seuil est dérivé d'un **argument géométrique**, à traiter comme
tel — posé, non calibré.

1. **Nécessaire : SASA > 0.** La sonde de Shrake-Rupley fait 1,40 Å, soit un rayon de
   molécule d'eau : une SASA non nulle signifie qu'une eau peut s'approcher, donc qu'un
   donneur de liaison hydrogène peut le faire. **Les 65 carboxylates passent** : non
   discriminant ici.
2. **Insuffisant, parce qu'un imidazole n'est pas une eau.** Le cycle His mesure environ
   4,5 × 4,0 Å, soit une section de l'ordre de **20-25 Å²**. Pour qu'un cycle approche au
   lieu d'une seule eau, le carboxylate doit présenter de l'ordre de sa propre section en
   surface accessible. D'où un plancher **≥ 20-25 Å²**.
3. **La fraction est plus comparable que l'aire absolue.** Le carboxylate d'un Glu est porté
   plus loin du squelette que celui d'un Asp et sort donc systématiquement plus. La fraction
   `carbox / résidu` dit si c'est le groupe fonctionnel qui est exposé ou seulement la tige :
   **≥ 40 %** retenu.
4. **L'orientation est le vrai discriminant ici**, et elle est sans dimension :
   **≤ 60°** pour un carboxylate qui pointe vers le solvant, **≥ 90°** pour un qui longe la
   surface.

**Critère composite retenu pour un pont salin His–carboxylate direct :**
`carbox >= 25 A2` **et** `part >= 40 %` **et** `angle <= 60°`.

Application :

| ancre | aire | fraction | angle | verdict |
|---|---|---|---|---|
| **D323** | 75,0 ✓ | 55 % ✓ | 33° ✓ | **ancre réelle, les trois critères** |
| **E400** | 72,9 ✓ | 76 % ✓ | 49° ✓ | ancre réelle |
| E510 | 64,3 ✓ | 78 % ✓ | 52° ✓ | ancre réelle, **mais hors domaine III** (510 > 506) |
| E530 | 82,5 ✓ | 86 % ✓ | 27° ✓ | ancre réelle, **mais hors domaine III** |
| E397 | 36,4 ✓ | 78 % ✓ | 19° ✓ | ancre réelle, aire modeste |
| E489 | 27,9 ✓ | 44 % ✓ | 35° ✓ | ancre réelle, aire modeste |
| E472 | 47,1 ✓ | 46 % ✓ | **86° ✗** | **douteuse** : longe la surface |
| **E320** | 30,0 ✓ | **28 % ✗** | **103° ✗** | **nominale** : échoue sur deux des trois |

### Conséquence sur le site G317

**D323 est une excellente ancre** — 89e centile du domaine III, 55 % de la SASA du résidu
portée par le carboxylate, pointant vers le solvant à 33°.

**E320 est une ancre nominale.** Son aire de 30 Å² n'est pas nulle, mais seulement 28 % de la
SASA du résidu vient du carboxylate — l'essentiel est la tige — et surtout l'angle de **103°**
signifie que le groupe longe la surface au lieu d'en sortir. Le pipeline le comptait comme
ancre parce que `rel_sasa` regarde le résidu entier.

Le site G317 garde donc **une** ancre acide conservée réellement exploitable, pas deux. Même
chose pour N449, dont l'unique ancre E472 est douteuse sur l'orientation (86°). K375 en garde
**deux** (E400 et E397), ce qui en fait, sur ce seul critère, le site le mieux doté — mais ses
autres défauts tiennent : séquons, domaine II, `dFab` 17,6 Å.

### Limites de la mesure

- SASA sur 6ARU, conformation **repliée**, 3,20 Å.
- **Un seul rotamère cristallographique.** Un carboxylate peut tourner en solution ; l'angle
  de 103° de E320 est la valeur d'un modèle, pas une contrainte permanente. C'est la limite
  la plus sérieuse de cette analyse.
- Le vecteur sortant est approximé par centroïde local → carboxylate, sur un voisinage de
  12 Å. C'est un proxy, pas une normale de surface calculée.
- Les trois seuils (25 Å², 40 %, 60°) sont **posés**, pas calibrés, faute d'étalon.

---

## Étendue spatiale de l'empreinte du cétuximab (02/10/2026) — `footprint_extent.py`

Objet : situer l'ordre de grandeur de `PATCH_RADIUS` contre une empreinte réelle. **Lecture
seule** — aucune constante modifiée, pipeline non relancé, aucun CSV réécrit.

Mesuré sur les **24 résidus** de l'empreinte (critère `fab_footprint`, contacts lourds sous
4,5 Å, PDB 349-473), avec les mêmes atomes d'ancrage qu'`enumerate_patches` : CB, ou CA en
l'absence de CB.

### Valeurs mesurées

| grandeur | valeur |
|---|---|
| Diamètre de l'empreinte (distance max entre deux résidus) | **34,9 Å** — paire 350 ↔ 473 |
| Rayon de la plus petite sphère englobante, centre libre | **17,7 Å** (diamètre 35,4 Å) |
| **Couverture à `PATCH_RADIUS = 11 Å`** | **médiane 8/23 autres résidus**, min 2, max 14, moyenne 8,0 |
| **Fraction médiane de l'empreinte capturée** par un patch de 11 Å | **40 %** (9 résidus sur 24, centre inclus) |
| Rayon minimal couvrant les 24, centre contraint à être l'un d'eux | **19,1 Å** depuis le résidu 441, soit **1,7 ×** `PATCH_RADIUS` |

Les cinq meilleurs centres pour une couverture totale : 441 (19,1 Å), 418 (20,4), 440 (20,6),
417 (22,1), 438 (22,9).

### Ce que dit le point 3

**Un patch de 11 Å capture environ 40 % d'une empreinte réelle.** L'unité « site » du pipeline
est donc nettement plus petite que la surface qu'une protéine occupe effectivement sur cette
cible. Le rapport est d'environ 2,5 en nombre de résidus.

La dispersion de la couverture — de 2/23 à 14/23 selon le centre choisi — est elle-même
informative : les résidus des extrémités (350, 353, 473, à 2-3 voisins) capturent très peu,
ceux du milieu (417, 438, 440, à 13-14) beaucoup. Combiné au fait que le diamètre (34,9 Å)
égale pratiquement le diamètre de la sphère englobante (35,4 Å), cela décrit une empreinte
**allongée**, dont l'extension est fixée par son axe long. Un modèle de patch sphérique en
rend mal compte quel que soit son rayon — remarque structurale, pas une proposition.

### Deux réserves, à lire avec les chiffres

1. **Un Fab fait ~50 kDa sur deux chaînes et couvre une zone plus large qu'un minibinder de
   83 aa.** Cette mesure **surestime** donc le rayon pertinent pour le cas présent. Les
   binders visés font 55-95 résidus, soit une interface attendue plus petite que celle d'un
   Fab, et le facteur 1,7 n'est pas transposable tel quel.
2. **C'est un ordre de grandeur, pas une calibration.** La mesure ne distingue pas 9 Å de
   11 Å de 13 Å : les trois donneraient une couverture partielle du même genre. Elle situe
   l'échelle, elle ne désigne pas une valeur.

### Statut de `PATCH_RADIUS`

**Inchangé : 11,0 Å, POSÉ, NON CALIBRÉ.** Aucune valeur nouvelle n'est proposée. L'inventaire
des constantes de [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) §4 reste exact sur ce point,
et cette mesure ne le modifie pas — elle ajoute seulement qu'on connaît désormais l'échelle
d'une empreinte réelle sur cette cible, ce qui n'était pas le cas.

---

## Trois éliminations à l'intérieur du site de référence (02/10/2026)

Lecture seule sur `data/egfr_patches.csv` et `data/egfr_residues.csv`. Déclenché par une
question simple — « en quoi T358 est-il mauvais ? » — dont la réponse élimine trois choses.

### Élimination 1 — l'écart de rang entre G317 et T358 n'a plus de fondement

T358 sort 6ᵉ sur 7 du site de référence uniquement parce que la clé de tri compte
`n_acidic_cons` d'abord : 1 pour T358 contre 2 pour G317.

Mais **l'unique ancre conservée de T358 est D323** — exactement celle de G317. Les deux
patches la partagent, avec L325.

| | ancres annoncées | ancres **réelles** (critère carboxylate) |
|---|---|---|
| G317 | E320 + D323 | **D323** |
| T358 | D323 | **D323** |

La seconde ancre de G317, E320, a été mesurée **nominale** : 28 % de part, angle 103°. Donc
**les deux patches ont la même et unique ancre pH réelle, la même molécule.** L'écart 2 contre
1 qui place l'un au rang 1 et l'autre au rang 6 repose entièrement sur un résidu dont le
carboxylate ne sort pas.

Ce qui reste comme différence réelle entre eux n'est pas l'objectif 1 mais l'arbitrage
objectif 2 contre objectif 3.

### Élimination 2 — H359 est éliminé comme cible de contact

T358 a 42 % plus de surface apolaire que G317 (627 contre 441 Å²). Mais :

| | apolaire | membres |
|---|---|---|
| T358 complet | 627 Å² | 8 |
| **T358 sans H359** | **529 Å²** | 7 |
| G317 | 441 Å² | 9 |

**H359 porte 97,8 Å², soit 16 % de la surface apolaire de T358**, et c'est son 3ᵉ membre le
plus exposé — donc très probablement contacté. Or c'est précisément lui qui diverge :
**His chez l'humain, Arg chez la souris.**

La substitution n'est pas conservative : elle remplace un résidu titrable par un résidu
définitivement chargé positif. Un contact construit sur H359 ne se retrouve pas sur la
protéine murine. **C'est le mode de défaillance de l'objectif 2**, et l'objectif 2 est classé
au-dessus de l'objectif 3.

Conséquence chiffrée : **l'avantage de surface de T358 passe de +42 % à +20 %** une fois H359
défalqué. Il survit, mais il est deux fois moins grand que la colonne brute ne le suggère.
H359 est à exclure de tout jeu de hotspots.

**Note annexe, à traiter comme hypothèse** : H359 est une histidine **de la cible**, à
l'interface, donc titrable dans la fenêtre 6,5-7,4. C'est une route pH alternative à celle de
CLAUDE.md §7, qui suppose l'His portée par le binder. Mais elle est **absente chez la souris**
(Arg), donc l'exploiter échangerait l'objectif 2 contre l'objectif 1. Non retenue, consignée.

### Élimination 3 — « G317 ou T358 » est une fausse alternative

| | |
|---|---|
| recouvrement T358 / G317 | **25 %** — partagent D323 et L325 |
| union | **15 membres**, **926 Å²** apolaires |
| identité de l'union | **14/15 = 0,93** |
| ancres conservées de l'union | **E320 et D323** |
| diamètre de l'union | **28,9 Å** |

Les deux sont **deux patches du même site**, tous deux dans le groupe de référence de
7 patches. Choisir l'un contre l'autre revient à choisir où centrer les hotspots à l'intérieur
d'un site, pas à choisir entre deux sites. La question était mal posée, la mienne comprise.

L'union est géométriquement viable : **28,9 Å de diamètre, soit 83 % de l'étendue mesurée de
l'empreinte du cétuximab** (34,9 Å). Le run de fumée a produit des dSASA de ~1900 Å², donc
l'ordre de grandeur n'est pas absurde.

### Ce que ça fait apparaître : F357

Meilleurs candidats apolaires de l'union, par SASA apolaire :

| résidu | apolaire | statut | glycane |
|---|---|---|---|
| **F357** | **155,8** | identical | 11,8 |
| **L325** | 111,1 | identical | 7,3 |
| ~~H359~~ | 97,8 | **different → R** | 10,8 |
| **I318** | 83,3 | identical | 15,0 |
| P361 | 71,3 | identical | 8,5 |
| K322 | 70,0 | identical | 13,6 |
| T406 | 66,4 | identical | 19,9 |
| T358 | 56,8 | identical | 7,8 |

**F357 est le résidu le plus apolaire de tout le site de référence** — 155,8 Å², soit plus de
trois fois E320 — `identical`, à 11,8 Å d'un séquon. **Il n'était dans aucun des jeux de
hotspots discutés jusqu'ici**, parce qu'il appartient au patch T358 et pas au patch G317.
C'est l'oubli que ces trois éliminations mettent au jour.

Jeu de hotspots que cela suggère : `A323,A325,A357,A318` — l'ancre réelle, les deux plus gros
apolaires conservés, et un quatrième solide, **en évitant H359**.

**Non vérifié, et bloquant avant de l'adopter** : que ces quatre résidus pointent vers la même
face. À 28,9 Å d'étendue, l'union est proche de la limite de ce qu'un binder de 83 aa peut
enfouir, et rien ne garantit qu'un seul binder atteigne les quatre. Se tranche dans PyMOL, ou
par deux runs courts comparés — ce qui reste de toute façon le seul moyen de départager deux
jeux de hotspots.

**Aucun hotspot n'est changé, aucune constante n'est touchée, le jeu en vigueur reste
`A318,A320,A323,A325`.**

---

## Jeu de hotspots retenu (03/10/2026)

**`A318,A323,A325,A406,A409`** — décision prise, motifs ci-dessous.

### Le critère qui manquait : l'étendue

Aucune des analyses précédentes n'avait vérifié qu'un jeu de hotspots soit atteignable par
**un seul** binder. Ni les miennes, ni celles proposées par ailleurs. C'est le contrôle qui a
tranché.

Mesuré sur 6ARU chaîne A, distances **CA**, après vérification que la chaîne existe
(4727 atomes, objet contenant A, B, C, D, E) et que chaque résidu porte exactement un CA —
une sélection vide ne lève aucune erreur et donnerait silencieusement une valeur fausse.

```
          318     323     406     409
318      0,00    7,68   12,25   16,52
323      7,68    0,00   16,61   16,73
406     12,25   16,61    0,00    9,05
409     16,52   16,73    9,05    0,00
```

Recoupé entre **PyMOL 3 headless** (`cmd.get_distance` sur `chain A and resi N and name CA`)
et **Biopython** via `hotspot_distances.py` : accord exact à 0,1 Å près. L'écart aurait
signalé un problème de lecture de la numérotation auteur du mmCIF.

### Trois faits que la mesure établit

**1. Retirer `A318` ne gagne aucun angström.** L'étendue maximale est de **16,73 Å dans les
deux cas**, fixée par 323↔409 qui survit au retrait. Un jeu à 3 perdrait les 83,3 Å²
apolaires de I318 pour zéro gain géométrique. La règle de décision posée — seuil à 24 Å sur
318↔409, mesuré à 16,52 — était satisfaite, mais la vraie raison est plus forte : le
troisième résidu est gratuit.

**2. `A325` s'ajoute aussi sans coût géométrique.** L325↔409 = 14,7 Å, donc le jeu à 5 garde
une étendue de **16,73 Å inchangée** et gagne 111,1 Å² apolaires.

**3. `F357` est exclu, et c'est l'arbitrage de fond.** F357↔409 = **24,0 Å**, F357↔406 =
27,1 Å : l'ajouter porterait l'étendue à 27,1 Å. **F357 et H409 sont mutuellement
exclusifs.** Le choix est donc entre les 155,8 Å² de F357 — la meilleure ancre apolaire de la
carte — et la His conservée H409 avec sa sécurité glycane. **H409 retenu, parce que
l'objectif 1 est classé avant l'objectif 3.**

### Le bon décompte de la surface d'empilement

Compter la SASA apolaire de tous les membres du jeu surévalue ce qu'un binder empile : D323
est là pour son carboxylate, pas pour ses 30 Å² de tige. Décomposition par rôle :

| rôle | résidus | surface d'empilement |
|---|---|---|
| ancres apolaires pures | I318 (83,3) + T406 (66,4) + **L325 (111,1)** | **261 Å²** |
| double usage | H409, cycle imidazole | +82 Å² |
| fonction pure | D323 — valeur = carboxylate 75,0 Å² | — |

**L'ajout de L325 fait donc passer la surface d'empilement garantie de 150 à 261 Å², soit
+74 %** — et non +42 % comme le suggérait le total brut de tous les membres. C'est ce
recadrage qui a emporté la décision.

Décomposition atomique de H409, pour justifier le « double usage » : sur ses 82,1 Å² de
carbone, **44,7 Å² viennent du seul CE1**, le carbone du cycle imidazole entre les deux
azotes, plus CD2 (7,2) et CG (4,8). C'est un cycle aromatique exposé, il empile réellement.
Et ses azotes ND1 (20,8 Å²) et NE2 (13,1 Å²) portent le mécanisme pH. Les deux rôles sont
réels et simultanés.

### Fiche du jeu retenu

| | |
|---|---|
| étendue CA | **16,73 Å** — la plus compacte de tous les jeux examinés |
| identité humain/souris | **5/5** |
| surface d'empilement | 261 Å² garantis, ~343 avec le cycle de H409 |
| ancre pH acide | **D323** — carboxylate 75,0 Å², part 0,55, angle 33°, 89ᵉ centile du domaine |
| His conservée de la cible | **H409** — mécanisme inversé : un Asp sur le binder |
| glycane | min **7,3** (L325), médian 15,0 |
| dFab | min **0,0** (H409) — sur la surface ligand-compétitive |

Composition : I318 (83,3 apol / glyc 15,0), D323 (30,2 / 12,1), L325 (111,1 / 7,3),
T406 (66,4 / 19,9), H409 (82,1 / 21,1). Les cinq `identical`.

### Pourquoi ce jeu et pas mes deux zones

C'est un **hybride** que je n'avais pas proposé : I318 et D323 viennent de la zone A,
T406 et H409 de la zone B, et **T406 est la charnière** qui les relie. Il est plus compact
que la zone A (16,7 contre 19,8 Å) et c'est **le seul jeu examiné qui porte les deux
mécanismes pH** — l'acide pour une His du binder, et la His conservée pour un Asp du binder.

### Réserves

- **Le glycane de L325 à 7,3 Å est le seul vrai risque du jeu.** Les quatre autres résidus
  sont entre 12,1 et 21,1 Å d'un séquon. Et `min_glyc` est mesuré au CB du séquon, pas à
  l'arbre glycanique : il sous-estime l'occlusion. Le séquon candidat est UniProt 352 =
  **PDB 328**, trois résidus après L325 — non vérifié.
- **Aucune corréférence de face n'a été vérifiée.** 16,7 Å dit que les résidus sont proches,
  pas qu'ils regardent du même côté.
- **Les hotspots sont un biais, pas une contrainte.** À `PATCH_RADIUS = 11` ne capturant que
  40 % d'une empreinte réelle, les trajectoires dériveront. La stratégie retenue est
  délibérément de lancer ce seul jeu et de **trier les designs a posteriori** selon la zone
  atteinte, plutôt que de lancer trois campagnes.
- **Un seul rotamère, à 3,20 Å, en conformation repliée.** Toutes les SASA et tous les angles
  en héritent.
- **La troncature 309-506 est vérifiée sûre pour les cinq** : SASA identique à 0,0 Å² près
  entre chaîne A entière et domaine III isolé, contrôle étendu le 03/10 aux résidus de la
  zone B qui n'avaient jamais été testés.

### Révision du 3 octobre : `A325` écarté, jeu final à quatre

**Jeu retenu : `A318,A323,A406,A409`.** L325 est retiré pour la raison qui avait été
identifiée comme son seul point faible : **son glycane à 7,3 Å**.

Motif, et il est méthodologiquement le bon : `min_glyc` est mesuré au **CB du séquon**, pas à
l'arbre glycanique, qui s'étend bien au-delà et reste flexible. La colonne **sous-estime donc
l'occlusion de façon systématique** — 7,3 Å n'est pas une marge, c'est un risque. Et le séquon
candidat (UniProt 352 = PDB 328) n'a jamais été vérifié dans PyMOL. Retenir L325 revenait à
parier sur une vérification non faite.

L'échange, chiffré :

| | jeu à 5 | **jeu à 4 retenu** |
|---|---|---|
| ancres apolaires pures | I318 83 + T406 66 + L325 111 = **261 Å²** | I318 83 + T406 66 = **150 Å²** |
| avec le cycle de H409 (double usage) | 343 Å² | **232 Å²** |
| glycane min | **7,3** | **12,1** |
| glycane médian | 15,0 | **17,4** |
| glycane max | 21,1 | 21,1 |
| identité | 5/5 | **4/4** |
| étendue CA | 16,73 Å | **16,73 Å** — inchangée |
| dFab min | 0,0 | 0,0 |

**Le coût est réel : −43 % de surface d'empilement garantie**, de 261 à 150 Å². Le gain est de
sortir le seul résidu à risque glycanique du jeu : le minimum passe de 7,3 à **12,1 Å** et la
médiane de 15,0 à **17,4 Å**. Plus aucun hotspot sous 12 Å d'un séquon.

Deux conséquences à noter :

- **L'étendue ne change pas** (16,73 Å, fixée par 323↔409), donc retirer L325 ne libère aucune
  place géométrique.
- **`F357` ne redevient pas accessible** : F357↔H409 = 24,0 Å, inchangé. Le retrait de L325
  n'ouvre rien de ce côté.
- L'interface reposera donc largement sur le cycle imidazole de H409 (82 Å², double usage) et
  sur ce que les trajectoires iront chercher **au-delà** des hotspots. C'est cohérent avec la
  stratégie de tri a posteriori, et ça la rend plus nécessaire qu'avant.

`L325` devient un **candidat coldspot** pour BindCraft 2.0, au même titre que `H359` : s'il est
proche d'un glycane, autant l'écarter explicitement plutôt que de seulement ne pas le nommer.

CLAUDE.md §5 et §8 mis à jour en conséquence. Les deux décisions successives — ajouter L325 le
3/10 puis le retirer le même jour — sont conservées telles quelles dans ce journal : la
première reposait sur le gain de surface, la seconde sur le fait que `min_glyc` sous-estime
l'occlusion. Les deux sont défendables, la seconde est la prudente.
