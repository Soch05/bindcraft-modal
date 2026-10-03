# CLAUDE.md — Adaptyv × Anthropic, Challenge 1 : binder conditionnel anti-EGFR

Fichier de contexte projet. À lire en entier avant toute action sur ce repo.
Le règlement condensé est dans [challenge-01-egfr.md](challenge-01-egfr.md) ; il a été
**relu à la source le 1er octobre**, voir §1. Le journal complet est [NOTES.md](NOTES.md),
~1900 lignes : toute décision, mesure et erreur y est datée.

**État au 3 octobre : la carte d'épitope est terminée, le jeu de hotspots est choisi, et le
générateur change. On repart sur BindCraft 2.0.**

---

## 1. L'objectif, et pourquoi il n'est pas celui qu'on croit

Challenge 1, **Track 3**, participant solo, auto-financé. Trois objectifs.

**Piège vérifié à la source** : la page du challenge les énonce dans un ordre et les
classe dans l'autre.

| section de la page | ordre |
|---|---|
| « Three objectives » | affinité, souris, pH |
| « How designs are ranked » | **pH, souris, affinité** |

**C'est l'ordre du classement qui compte** : sélectivité pH, puis cross-réactivité souris,
puis affinité. Quiconque ne lit que la première section conclut l'inverse. Tout arbitrage se
tranche dans cet ordre.

1. **Sélectivité pH** — formulation exacte : *« binds human EGFR at pH 6.5 and shows no
   detectable binding at pH 7.4 »*. C'est un switch binaire, plus exigeant que le décalage de
   KD qu'on peut raisonnablement viser (§7). Ne pas enjoliver cet écart dans le dossier.
2. **Cross-réactivité souris** — la même séquence doit reconnaître P00533 et Q01279.
3. **Affinité** — sur l'ectodomaine humain.

**Conséquence structurante : l'affinité est le critère le moins bien classé, et c'est le seul
que BindCraft optimise.** Une campagne menée par défaut produit un bon résultat sur le n°3 et
rien sur les n°1 et n°2.

**Deuxième conséquence : la méthode est notée.** En Track 3 les soumissions sont mises en
commun et un modèle sélectionne sur qualité prédite, nouveauté du design et **nouveauté de la
méthode**, avec ~375 places pour ~1500 designs. Le dossier de méthodes est le canal de
sélection, pas un livrable annexe.

**Échéance dure : 4 octobre 23:59 AoE = dimanche 5 octobre 13h59 Paris.** Aujourd'hui
3 octobre. Ce chiffre prime sur toute considération d'élégance.

Tout est publié en open data sous ODC-BY, résultats négatifs compris. Chaque fichier du dépôt
est écrit en supposant qu'un tiers le lira.

**Critère de succès minimal** : un CSV de soumission reproductible depuis un commit, avec pour
chaque séquence sa trajectoire de génération, ses métriques, et la raison de sa sélection —
y compris l'argument pH.

---

## 2. Contraintes dures

| Contrainte | Conséquence |
|---|---|
| **~1,5 jour** | Privilégier le chemin court qui produit une soumission. Un pipeline inachevé vaut zéro. |
| Machine locale = MacBook Air 2018, **Intel x86_64, pas de GPU CUDA** | **Aucun modèle ne tourne en local.** Jamais proposer d'exécuter AlphaFold2, BindCraft, ProteinMPNN, Boltz ou PyRosetta ici. Le local sert à écrire du code, parser des CSV, aligner des séquences, mesurer des distances, tracer des figures. |
| Python local | venv **3.12 via `uv`**. Ne jamais cibler 3.13. Le pin `cbor2==5.9.0` dans `pyproject.toml` est obligatoire : sans lui `uv pip install modal` tente de compiler une extension Rust et échoue sur cette machine. Ne pas l'enlever. |
| Compute GPU | **Modal** uniquement. Colab est un anti-objectif : ~10× plus lent à hardware équivalent, sessions qui meurent. |
| **Budget : `max_trajectories` + le `timeout` Modal** | **Révisé le 3 octobre sur la source de 2.0 : `TIMEOUT` n'existe plus.** Zéro occurrence de `timeout` dans `bindcraft/` hors appels réseau. Le mécanisme de BindCraft 1 — `check_n_trajectories` ne comptant que `Trajectory/Relaxed`, donc une trajectoire `LowConfidence`/`Clashing` brûlant du GPU hors quota — n'a plus de code correspondant. Le budget se pilote désormais par `max_trajectories` (réglage de campagne) et par le `timeout` de la fonction Modal, qui est le vrai plafond de dépense. |
| Reprise | `resume` est à **`true` par défaut** dans `settings/core/default.json`. Un rappel sur le même `run_name` reprend la campagne. C'est ce qui rend l'architecture en appels courts sûre, et ça rend sans objet la question « un kill par `TIMEOUT` commite-t-il le volume ». |
| Licence | **Plus de contrainte PyRosetta.** BindCraft 2.0 ne dépend plus de PyRosetta, DSSP ni DAlphaBall — zéro occurrence dans le dépôt amont au commit épinglé. Le relax est interne et en JAX (`relax_steps`, `relax_learning_rate`), et `relax_accepted_designs` est à `false` par défaut. |

---

## 3. Stack

### Génération : BindCraft 2.0 — tout est à réétablir

**Décision du 2 octobre : passage à BindCraft 2.0**, pour deux fonctionnalités qui règlent
des problèmes identifiés et non traitables sur l'ancienne version :

- **coldspots** — permet d'écarter explicitement un résidu. Deux usages immédiats : `H359`,
  qui diverge en Arg chez la souris et porte 16 % de la surface apolaire de sa zone, et les
  résidus proches d'un séquon. Jusqu'ici on ne pouvait que choisir une fenêtre qui les évite
  et espérer qu'un binder ne les contacte pas.
- **design multicible** — change la nature de l'objectif n°2. La cross-réactivité souris
  cesse d'être un filtre a posteriori pour devenir une cible d'optimisation : on conçoit
  contre P00533 **et** Q01279 simultanément. C'est aussi l'argument le plus défendable sur
  l'axe « nouveauté de la méthode ».

**Forme réelle des deux, vérifiée dans la source le 3 octobre.** Les deux passent par la même
liste `targets`, une entrée par cible, clés `name, target_path, chains, hotspots, coldspots,
weight, objective` (`bindcraft/settings.py:38`) :

```json
"targets": [
  {"name": "hEGFR_dIII", "target_path": "...", "chains": "A",
   "hotspots": "A318,A323,A406,A409", "coldspots": "A359", "weight": 1.0},
  {"name": "mEGFR_dIII", "target_path": "...", "chains": "A", "weight":  1.0},
  {"name": "offTarget",  "target_path": "...", "chains": "A", "weight": -0.5}
]
```

- `coldspots` accepte des **plages** : `"131-134,139,196-200"`. Poser des coldspots active
  `weights_coldspot_repel = 1.0` et pose `max_coldspot_contact_final = 0.05` comme filtre de
  sortie (`bindcraft/settings.py:192-195`). Le run logge `target=… coldspots=… residues=N` :
  **c'est le point de vérification** que la plage a bien été résolue.
- un **poids négatif** est un détargeting — c'est comme ça qu'on sélectionne *contre* une
  cible, et plusieurs poids positifs demandent au même binder de marcher sur toutes.
  Exemples de référence : `examples/pdl1_ortholog_pair.json` et
  `examples/pdl1_crossreactive_detarget.json`, qui sont littéralement notre cas de figure.

**Numérotation : vérifiée, pas de mémoire.** Lecture directe de `inputs/6ARU_A_309-506.pdb` le
3 octobre : `PDB 359 = HIS` et `PDB 409 = HIS`, donc `H359` et `H409` sont bien en
numérotation **PDB**, la même que les hotspots et que le fichier cible. Les six His de la
cible sont aux PDB 334, 346, 359, 394, 409, 483.

**Prérequis non vérifié** : le multicible demande une structure du domaine III **murin**.
Aucune recherche n'a été faite sur l'existence d'une structure expérimentale de Q01279 ;
à défaut, un modèle AlphaFold, avec les réserves que ça implique pour tout calcul de SASA.

**Coldspots encore incomplets** : seul `A359` est câblé dans
[modal_bindcraft2.py](modal_bindcraft2.py). Les résidus à moins de ~10 Å d'un séquon
(§8 action 4) **n'ont jamais été mesurés depuis les hotspots retenus**. Ne pas inventer la
liste — la mesurer.

**⚠️ Ce qui n'est PLUS valide.** L'ancienne version était épinglée au commit `c0a48d5`, et
tous les chiffres de débit de [NOTES.md](NOTES.md) y sont attachés :

| mesuré sur `c0a48d5` | valeur | statut sur 2.0, au 3 octobre |
|---|---|---|
| temps par trajectoire relaxée, cible 198 résidus | 8,49 min | **à re-mesurer** |
| temps par tentative, tout compris | 10,1 min | **à re-mesurer** |
| coût, 6 tentatives sur L40S | $1,97 | **à re-mesurer** |
| taux de relaxation | 3/6 | **caduc** : `relax_accepted_designs` est à `false` par défaut, le relax n'est plus PyRosetta |
| `check_n_trajectories` ne compte que `Relaxed` | établi | **caduc** : plus de code correspondant, voir §2 |
| filtre dominant : `i_pAE`, 20 rejets sur 29 | établi | **à re-mesurer** — mais les noms de filtres ont changé |
| noms de colonnes de `failure_csv.csv` | établis | **caduc** : voir la structure des sorties ci-dessous |
| seuils de `default_filters.json` | non lus | **lus**, dans `settings/core/default.json` — valeurs en §6 |

**Ce qui est établi sur 2.0** (lu dans la source au commit épinglé, 3 octobre) :

| | |
|---|---|
| Dépôt | **`PacesaLab/BindCraft2`** — un dépôt distinct, *pas* un tag de `martinpacesa/BindCraft` |
| Commit épinglé | **`a8d0f2002df373842b86a3c20c5a060c5cfdf980`** |
| Version du paquet | `1.0.1`, `requires-python >= 3.12` |
| Pins | `jax>=0.11,<0.12` via l'extra **`cuda13`**, `numpy` **non épinglé** (le pin `<2.0` n'existait que pour PyRosetta) |
| Plancher GPU | CUDA 13 exige une compute capability **≥ 7,5**. L40S = 8,9, donc bon. |
| Poids ProteinMPNN | **livrés dans le paquet**, 77 Mo, 3 variantes × 4 modèles. Rien à télécharger. |
| Poids AF2 | 5,3 Go, 7 modèles (`model_1..5_multimer_v3`, `model_1_ptm`, `model_2_ptm`). Trouvés par `BINDCRAFT_AF2_PARAMS`, sinon sous `<paquet>/weights/alphafold`, sinon téléchargés. |
| CLI | `bindcraft design <settings.json>`, plus `rank`, `filter`, `score`, `fetch-weights`, `archive` |
| Config | **JSON**, pas de CLI de paramètres. 235 réglages documentés dans `settings/core/reference.json`. |
| Validation | `python -m bindcraft.selfcheck cuda13` nomme chaque module et chaque checkpoint manquant |

Le build Modal est porté de `containers/Dockerfile` de l'amont, dans
[modal_bindcraft2.py](modal_bindcraft2.py). Deux pièges que ce Dockerfile documente et qu'il
faut garder :

- les wheels `jax[cuda13]` gardent leurs bibliothèques sous `site-packages/nvidia/*/lib`,
  **où le loader ne regarde pas**. Sans le fichier `ld.so.conf.d` qui les déclare, jax
  avertit une fois puis tourne sur le CPU, cent fois plus lentement, et rien ensuite ne le
  signale. Le build échoue exprès sur `ldconfig -p | grep -q libcupti` ;
- l'installation doit être **éditable** à la racine du dépôt : `settings/` et `scaffolds/`
  vivent à la racine et non dans le paquet, et le runtime les trouve relativement à lui. Une
  install classique ne copierait que le paquet et les orphelinerait.

**Valider le build avant toute dépense** (§8) : `modal run modal_bindcraft2.py::selfcheck`.

### Structure des sorties — établie au run `egfr-dIII-cal01` du 3 octobre

Lue dans `bindcraft/campaign_output.py` **et** confirmée par le log du run. Trois étages, pas
un dossier plat :

```
/outputs/<run_name>/
├── 1_Trajectories/!_Trajectories.csv     une ligne par trajectoire
├── 2_Refolded/!_Refolded.csv             candidats ProteinMPNN repliés
├── 3_Ranked/!_Ranked.csv                 réécrit à CHAQUE design accepté
├── accepted.csv
├── summary.csv                           écrit à la FIN de la campagne
├── .campaign_state.json                  état de campagne, pas de processus
├── campaign_metadata.json                réglages réellement utilisés — à garder
└── settings.json                          écrit par modal_bindcraft2.py
```

`summary.csv` a les colonnes `campaign, scope, metric, samples, mean, std, min, max`.

**Rien de commun avec BindCraft 1** — plus de `Accepted/Ranked/`, `Trajectory/Relaxed/`,
`Rejected/`, `failure_csv.csv`, `final_design_stats.csv` ni `trajectory_stats.csv`. Vérifié
le 3 octobre : **aucun script suivi du dépôt ne référence ces anciens noms**, donc rien à
réécrire côté code. Les occurrences restantes sont dans NOTES.md, qui est un journal et doit
les garder.

⚠️ **Correction d'une erreur de ma part** : une version antérieure de ce fichier annonçait
`trajectories.csv`, `candidates.csv`, `accepted.csv`. C'était déduit de la liste de
constantes `CAMPAIGN_OUTPUT_NAMES` et non de la structure réelle. Le run a tranché.

Le nom d'un design encode campagne, modalité, longueur et hash de recette :
`egfr-dIII-cal01_denovo_l69_876123b3938b459b`.

### Exécution : Modal

Volume `bindcraft` monté sur `/outputs`, un répertoire par `run_name`. GPU par défaut `L40S`
(46 Go vérifiés), surchargeable par la variable d'environnement `GPU`. Tarif L40S mesuré :
**$0,000542/s = $1,95/h**.

**Parallélisation : pas de sharding, BindCraft 2.0 le fait lui-même.** `workers_per_gpu` vaut
`auto`, ce qui donne **2 workers concurrents** sur une L40S pour notre cible — **observé** dans
le log du run `prod01`, qui annonce 13,1 et 14,6 Go par worker.

⚠️ **Correction du 3 octobre** : j'avais annoncé 3 workers, calculés sur le nombre de résidus
**brut** (253–293). `length_bucket_size = 32` arrondit (cible + binder) au multiple de 32
supérieur, donc le calcul porte sur **288–320 rembourrés** :

| binder | bruts | rembourré | Go/worker |
|---|---|---|---|
| 55–64 | 253–262 | **288** | 13,10 |
| 65–95 | 263–293 | **320** | 14,58 |

`(45 − 4) // 14,58 = 2`. **La concurrence rapporte donc au mieux 2×, pas 3×.**

**Levier de débit qui en découle** : la plage de longueurs pilote le nombre de workers par le
seau de rembourrage. `[55, 95]` donne 2 workers ; `[55, 64]` tiendrait dans le seul seau de
288 et en donnerait 3. À arbitrer contre la diversité de longueurs voulue dans la soumission.

Le partage des longueurs entre workers est **inégal** : sur `prod01`, worker 0 tire 31
longueurs (65–95) et worker 1 seulement 10 (55–64), mais dans un seau plus rapide. L'amont
prévient que « les groupes de longueurs plus rapides peuvent apparaître plus souvent dans les
résultats » — à surveiller dans la distribution de longueurs du lot final.

Deux variables dans `modal_bindcraft2.py` : `--workers` (→ `workers_per_gpu`, mettre `1` pour
mesurer un temps par trajectoire) et `GPU_COUNT` (→ `gpu="L40S:N"`, N cartes dans **un**
conteneur, `auto_multi_gpu` répartissant seul).

**⚠️ Ne pas restaurer le sharding de BindCraft 1** (`shard-000`, `shard-001`…) : sur 2.0
chaque shard serait une campagne indépendante chassant son propre
`number_of_final_designs` — N shards = N × les designs et N × le coût — et l'échelle de
désespoir est comptée au niveau campagne, pas du processus.

Les poids AF2 sont **dans l'Image et non dans le Volume** — choix de l'ancien build,
reproduit, et que l'amont recommande lui aussi (`--build-arg ALPHAFOLD_PARAMETERS=bake`).
La couche de téléchargement est placée après l'install et avant la vérification finale, pour
qu'une invalidation de cache en amont ne refasse pas les 5,3 Go pour rien.

Le volume ne contient plus qu'un répertoire : `smoke-G317`, le run de fumée EGFR du
1er octobre. Les quatre runs de démo PD-L1 de BindCraft 1 (`test1`, `par-test`, `par-test2`,
`par-test3`) ont été supprimés du volume et du local le 3 octobre. `smoke-G317` est conservé
parce qu'il porte les structures acceptées sur lesquelles prototyper la mesure
d'appariement His–acide (§8 action 6) et parce que CLAUDE.md §6 cite son observation des
3 His d'interface.

### Re-scoring orthogonal — jamais câblé

Un prédicteur **indépendant d'AF2** (Boltz-2 / Chai-1), pour éviter que le filtre valide ce
que le générateur a optimisé. Les organisateurs pointent
https://github.com/anthropics/uplifting-biomolecular-modeling. **Non fait, et c'est la plus
grosse lacune restante du pipeline.**

### Analyse locale

biopython **1.88** et numpy **2.5.3** installés dans le venv. PyMOL disponible comme binaire
(`/usr/local/bin/pymol`, headless via `pymol -cq`) mais **pas** comme module Python.
openpyxl, reportlab et pypdf sont utilisés en dépendances jetables via `uv run --with`, et
volontairement non déclarés dans `pyproject.toml` — les scripts concernés ne sont pas des
étapes du pipeline.

### Le trou dans la stack

**Rien là-dedans ne connaît le pH.** AF2 comme ProteinMPNN ignorent les états de
protonation ; ils ne voient qu'une identité de résidu. Aucune boucle d'optimisation ne
poussera vers un binder conditionnel. La sélectivité pH doit être **imposée par construction**
puis **vérifiée à part**, jamais espérée du générateur.

---

## 4. État réel du dépôt

Tout ce qui suit est suivi par git et existe :

```
.
├── CLAUDE.md
├── challenge-01-egfr.md          règlement condensé, relu à la source le 01/10
├── NOTES.md                      journal, ~1900 lignes — la source de vérité
├── docs/ARCHITECTURE.md          pipeline, inventaire de TOUTES les constantes
├── data/LECTURE.md               les 41 colonnes du CSV, une par une
│
├── egfr_epitope_map.py           carte d'épitope, local CPU — TERMINÉ
├── prepare_target.py             extrait le PDB cible depuis 6ARU
├── carboxylate_access.py         accessibilité des carboxylates des ancres acides
├── footprint_extent.py           étendue de l'empreinte du Fab, lecture seule
├── hotspot_distances.py          distances CA entre hotspots candidats
├── build_workbook.py             classeur Excel de travail, 4 feuilles
├── modal_bindcraft2.py           entrypoint Modal pour BindCraft 2.0 — BUILD VALIDÉ 03/10
├── analyze_campaign.py           funnel, distributions, chronométrage, coût, matière pH
│
├── data/egfr_patches.csv         96 patches × 41 colonnes
├── data/egfr_residues.csv        609 résidus × 18 colonnes
├── data/6aru_chainA_conserv.pdb  B-factor = conservation, pour PyMOL
├── inputs/6ARU_A_309-506.pdb     LA CIBLE : domaine III, 198 résidus
├── pyproject.toml  uv.lock
└── out/                          résultats rapatriés (gitignored)
```

Gitignorés et régénérables : `data/*.cif`, `data/*.json` (caches réseau),
`data/*.xlsx` et `data/*.pdf` (rendus dérivés des CSV).

**Supprimé le 3 octobre, vestiges de BindCraft 1** — tout est récupérable dans l'historique
git, et les entrées de NOTES.md qui s'y réfèrent sont conservées comme journal :

| supprimé | ce que c'était |
|---|---|
| `modal_bindcraft.py` | entrypoint de 1429 lignes pour `c0a48d5`, avec sa logique de sharding et son setup PyRosetta. Remplacé par `modal_bindcraft2.py`. |
| `inputs/PDL1.pdb` | cible de la démo du 23 septembre, plus utilisée |
| `out/test1`, `out/par-test2`, `out/par-test3` | 134 Mo de sorties de démo PD-L1 |
| volume : `test1`, `par-test`, `par-test2`, `par-test3` | les mêmes, côté Modal |

**Conservé** : `out/smoke-G317` (28 Ko de CSV) et `smoke-G317` sur le volume, qui porte en
plus les structures acceptées. C'est le run de fumée EGFR du 1er octobre, la source de
l'observation des 3 His d'interface citée en §6 et la matière de l'action 6 du §8. Ses
*chiffres de débit* restent caducs comme tous ceux de `c0a48d5`.

**`NOTES.md` est le journal et doit être tenu à jour à chaque run** : commande exacte, GPU,
durée, coût, tentatives, acceptés, décision. C'est la matière première du dossier de méthodes,
donc du Track 3.

---

## 5. Cible et hotspots — ÉTABLIS, ne pas rediscuter sans mesure

| | |
|---|---|
| Humain | UniProt **P00533**, ectodomaine 25–645 |
| Souris | UniProt **Q01279** |
| Structure | PDB **6ARU**, chaîne **A**, conformation **repliée**, 3,20 Å |
| Offset PDB → UniProt | **+24**, lu dans `_struct_ref_seq` **et** retrouvé par balayage |
| Domaine III | **333–530 UniProt = 309–506 PDB**, bornes CATH-Gene3D `G3DSA:3.80.20.20` |
| **Fichier cible** | **`inputs/6ARU_A_309-506.pdb`** — 198 résidus, chaîne A seule, 0 hétéroatome |
| Longueur binder | **55–95 aa** → catégorie **minibinders** (40–100 inclus) |
| Designs | **20 max** (Track 3) |
| CSV | ordonné par classement, meilleur en première ligne ; `name`, `sequence`, `molecule_class` = `protein` |

### Le jeu de hotspots retenu

```
A318,A323,A406,A409
```

Décidé le 3 octobre, motifs complets dans NOTES.md. Résumé :

| | |
|---|---|
| étendue CA | **16,73 Å** — la plus compacte de tous les jeux examinés |
| identité humain/souris | **4/4** |
| surface d'empilement | **150 Å²** d'ancres apolaires pures (I318 83 + T406 66), 232 Å² avec le cycle de H409 |
| ancre pH acide | **D323** — carboxylate 75,0 Å², part 0,55, angle 33°, 89ᵉ centile |
| His conservée de la cible | **H409** — mécanisme pH inversé, cycle imidazole à 44,7 Å² sur le seul CE1 |
| glycane | min **12,1 Å**, médian 17,4, max 21,1 |
| dFab | min **0,0 Å** (H409) — sur la surface ligand-compétitive |

**Trois faits mesurés qui fondent ce jeu :**
- retirer `A318` ne gagnerait aucun angström — l'étendue est fixée par 323↔409, donc le
  quatrième résidu est gratuit et un jeu à 3 serait strictement pire ;
- **`A325` a été envisagé puis écarté.** Il s'ajoutait sans coût géométrique et apportait
  +111 Å² (150 → 261, soit +74 %), mais son glycane à **7,3 Å** aurait été le seul point
  faible du jeu. `min_glyc` étant mesuré au CB du séquon et non à l'arbre glycanique, il
  **sous-estime l'occlusion** : 7,3 Å est un risque réel, et le séquon candidat
  (UniProt 352 = PDB 328) n'a jamais été vérifié. Écarté pour cette raison le 3 octobre ;
- **`F357` est exclu** : 24,0 Å de H409, donc mutuellement exclusifs — et retirer L325 ne le
  rend pas accessible. Le choix entre les 155,8 Å² de F357 et la His conservée H409 est
  tranché en faveur de H409, **parce que l'objectif 1 passe avant le 3**.

**Ce que le jeu coûte** : 150 Å² d'ancres apolaires pures est modeste. L'interface reposera
largement sur le cycle imidazole de H409 (82 Å², double usage) et sur ce que les trajectoires
iront chercher au-delà des hotspots. C'est cohérent avec la stratégie de tri a posteriori (§6).

**Non vérifié** : la corréférence de face des quatre résidus. 16,73 Å dit qu'ils sont proches,
pas qu'ils regardent du même côté.

**La troncature 309–506 est mesurée sûre** pour les cinq hotspots : SASA identique à 0,0 Å²
près entre chaîne A entière et domaine III isolé. Là où elle mord, c'est aux bornes
elles-mêmes — 506 gagne 125,8 Å², 309 en gagne 122,3.

### Règles qui restent valides

**Numéros de résidus : jamais de mémoire.** Tout hotspot cité doit venir des CSV, d'un script
du dépôt, ou d'une lecture directe du fichier.

**Piège sur l'épitope recommandé** : le cétuximab ne reconnaît pas l'EGFR murin, donc
l'épitope le plus documenté du domaine III est celui qui met en danger l'objectif n°2. Le
patch a été choisi sur la conservation mesurée. Fait associé, à verser au dossier :
**l'empreinte du cétuximab, 24 résidus, ne porte qu'un seul Asp/Glu** — un binder protéique
fonctionne donc sur une surface quasi dépourvue d'acides.

**Contraintes d'expression** (Adaptyv exprime en acellulaire, mesure par BLI/SPR) — des
filtres, pas des préférences : pas de cystéines libres (BindCraft les omet par défaut, garder
ce réglage) ; pas de dépendance à une glycosylation ni à un chaperon ; surface peu hydrophobe,
pas de longues extrémités désordonnées. **Un design qui ne s'exprime pas produit zéro
information** — l'expressibilité prime sur l'affinité prédite.

---

## 6. Pipeline

**cible+hotspots (FAIT) → générer → filtrer → switch pH → re-scorer orthogonalement →
diversifier → soumettre**

### Le switch pH — deux mécanismes, pas un

L'histidine est le seul acide aminé canonique dont le pKa (~6,0–6,5 en solution libre,
décalable par l'environnement) tombe entre les deux pH mesurés. Asp/Glu sont à ~4, Lys/Arg
au-dessus de 10. À pH 6,5 une fraction notable des His est protonée donc chargée ; à 7,4 elles
sont majoritairement neutres.

**Route 1 — His du binder contre acide de la cible.** C'est `D323`, carboxylate validé
(75,0 Å², 33° sortant). Polarité visée : **gain** de liaison à pH 6,5, le pont salin n'existant
que sous forme protonée. La polarité inverse (His enfouie près d'un Arg/Lys) donnerait une
perte à 6,5, soit l'opposé.

**Route 2 — acide du binder contre His conservée de la cible.** C'est `H409`, découverte le
2 octobre. **Plus facile à concevoir** : placer un Asp ou Glu sur une surface de binder est
trivial comparé à placer une His avec le bon pKa et la bonne géométrie. Et H409 est
`identical`, donc la route sert aussi l'objectif n°2.

Le jeu de hotspots retenu porte **les deux**, et c'est le seul examiné dans ce cas.

**Honnêteté à ne pas enjoliver** : une His isolée donne rarement un basculement franc. Les
binders pH-dépendants publiés en alignent plusieurs, et l'effet obtenu est typiquement un
décalage de KD d'un facteur quelques-uns, pas un tout-ou-rien — alors que le règlement demande
« no detectable binding ». Un lot dont la dépendance au pH est modeste mais **mesurée et
argumentée** vaut mieux qu'une affirmation de switch binaire non étayée.

**Jamais mesuré** : qu'une His d'un design soit effectivement à portée de pont salin de D323,
ou un Asp de H409. Le run de fumée a produit des designs portant 3 His d'interface chacun
**sans aucun biais appliqué**, mais personne n'a vérifié l'appariement. C'est l'objectif n°1 du
challenge et c'est le trou le plus béant du dossier.

### Métriques à logger pour **chaque** design, y compris rejeté

`design_id, seed, trajectory, sequence, length, i_pTM, i_pAE, pLDDT_binder, dG, dSASA,
shape_complementarity, n_hotspot_contacts, unsat_hbonds, surface_hydrophobicity,
n_interface_his, his_acidic_pairs, asp_his409_pairs, epitope_conservation_frac,
min_dist_glycan, filters_passed, reject_reason, run_id, version`

Les champs d'appariement sont l'ajout qui rend le lot défendable sur les objectifs 1 et 2 ;
sans eux la soumission ne peut argumenter que de l'affinité.

### Seuils

⚠️ **Piège d'échelle, désormais confirmé par lecture** : BindCraft normalise pLDDT et pAE sur
[0,1] dans ses fichiers de filtres, alors que la littérature les cite en 0–100 et en Å.
Vérifier l'échelle avant toute comparaison ou tout seuil copié d'un papier.

**Les filtres de sortie réels de 2.0**, lus dans `settings/core/default.json` au commit
épinglé le 3 octobre (le fichier `default_filters.json` de BindCraft 1 n'existe plus) :

| filtre | seuil | sens |
|---|---|---|
| `Unbound_Binder_pLDDT` | **0,70** | plus haut est mieux — ⚠️ `default.json` dit 0,80, mais le preset `binder` l'écrase par `min_monomer_plddt_final: 0.7`. Valeur lue dans `campaign_metadata.json` du run `cal01`, qui est la source autoritative. |
| `pTM` | **0,55** | plus haut est mieux |
| `i_pTM` | **0,70** | plus haut est mieux |
| `i_pAE` | **0,35** | plus bas est mieux |
| `Backbone_Clashes` | **0** | plus bas est mieux |
| `Interface_Residues` | **7** | plus haut est mieux |

Tous les quatre premiers sont sur **[0,1]**. Poser des coldspots ajoute
`max_coldspot_contact_final = 0,05`.

Il existe par ailleurs des planchers **par étage de trajectoire**, qui coupent avant d'arriver
aux filtres de sortie : `min_plddt_screen` 0,60, `min_plddt_refine` 0,60, `min_plddt_anneal`
0,65, `min_plddt_harden` 0,65, `min_plddt_final` 0,70, `min_iptm_anneal` / `_harden` /
`_mutate` 0,50, `min_iptm_final` 0,70. C'est un pipeline par étages, pas un filtre unique en
bout de chaîne comme sur BindCraft 1 — donc le « profil de rejet » n'a plus la même forme.

Sur `c0a48d5`, le profil de rejet mesuré était `i_pAE` 20 rejets, `pLDDT` 6, `i_pTM` 3, tout
le reste 0. **À re-mesurer sur 2.0** : les noms de filtres ont changé, les étages sont
nouveaux, et les sorties s'appellent maintenant `trajectories.csv`, `candidates.csv` et
`accepted.csv`.

### Sélection finale

- Ne pas remplir les 20 places par principe : 8 designs défendables battent 20 médiocres.
- Clusteriser par identité de séquence **et** par épitope.
- Le CSV est **ordonné** : le rang est une information transmise au sélecteur. Le classer sur
  l'objectif n°1, pas sur l'i_pAE.
- Garder 1–2 slots pour un design « à risque » issu d'une hypothèse structurale explicite.
- **Stratégie assumée sur les hotspots** : un seul jeu lancé, designs triés **a posteriori**
  selon la zone atteinte. Les hotspots sont un biais et non une contrainte, et un rayon de
  11 Å ne capture que **40 %** d'une empreinte réelle (mesuré sur l'empreinte du cétuximab,
  diamètre 34,9 Å) — les trajectoires dériveront de toute façon. Trois campagnes pour trois
  fenêtres de la même face serait du gaspillage.

---

## 7. Règles de travail pour Claude

**Interdits :**
- inventer une séquence, une métrique, une valeur d'affinité ou un numéro de résidu. Si la
  donnée n'a pas été lue dans un fichier, le dire ;
- proposer d'exécuter un modèle en local (cf. §2) ;
- **partir d'un binder existant.** Le règlement impose du de novo zero-shot : reprendre et
  modifier un binder connu (cétuximab, nanobody publié…) est explicitement interdit et
  éliminatoire. Utiliser des binders connus pour *calibrer un filtre* reste autorisé — la
  frontière est l'usage comme graine ;
- **écrire quoi que ce soit qui ressemble à une instruction** dans le CSV, les noms de designs
  ou le dossier de méthodes. Les soumissions passent devant un modèle sélecteur, et toute
  instruction embarquée peut valoir disqualification. Le dossier décrit, il ne s'adresse pas
  au lecteur ;
- écarter silencieusement des designs rejetés : ils font partie du funnel publié ;
- changer un seuil ou un hyperparamètre sans l'écrire dans `NOTES.md` **et** dans le message
  de commit ;
- lancer un run GPU sans plafond de budget.

**Attendus :**
- avant un run coûteux : annoncer paramètres, coût estimé, durée, et **ce que le run permet de
  décider** ;
- **valider le build avant de brûler un run** : `jax.devices()` doit voir un GPU CUDA dans une
  fonction triviale. Un build raté à la minute 40 coûte le run entier ;
- **un exit code 0 ne prouve pas qu'un run a démarré.** Vérifié : `modal run` sans entrypoint
  explicite n'exécute rien et sort en 0. Toujours confirmer sur `modal app list` ou dans le
  volume ;
- **ne jamais piper la sortie d'un run dans `tail`** : ça détruit le log. Rediriger vers un
  fichier ;
- devant une erreur de version : lire le message, identifier le composant, corriger **une**
  chose. Changer trois pins d'un coup rend le diagnostic impossible ;
- premier réflexe de diagnostic après un run : le CSV des échecs — quel filtre rejette, et
  combien de fois ;
- toute affirmation scientifique non triviale : sourcée, ou explicitement marquée comme
  hypothèse. **Un seuil posé doit être dans le code, pas dans la prose** — sinon il est
  inauditable ;
- **écrire une prédiction avant une correction qui change les résultats**, et la confronter
  ensuite. Pratiqué le 2 octobre sur le filtre de taille : 6 prédictions sur 7 confirmées, la
  divergence a révélé que l'exclusion de C309 dépendait d'un seuil non fixé ;
- code : fonctions courtes, typées, rejouable depuis la CLI avec les mêmes arguments ;
- après chaque run : entrée datée dans `NOTES.md`.

**Sécurité :** le token Modal ne va ni dans le dépôt ni dans un fichier suivi par git
(`modal token set` uniquement). `out/` est gitignored.

**Ton** : direct. Signaler les erreurs de raisonnement, les seuils arbitraires et les impasses
de méthode sans les emballer. Pas de validation de complaisance.

**Pédagogie** : je suis en formation biotech + IA. Quand un choix repose sur un concept
(backprop à travers AF2, pAE vs pLDDT, hallucination vs diffusion, pKa et protonation),
expliquer le *pourquoi* en une ou deux phrases, par analogie ML quand c'est possible — puis
avancer.

---

## 8. Prochaines actions, dans l'ordre

1. [x] **Installer et valider BindCraft 2.0 sur Modal.** FAIT le 3 octobre.
       `modal run modal_bindcraft2.py::selfcheck` →
       `jax 0.11.2 | backend gpu | devices [CudaDevice(id=0)]`, les 7 modèles AF2 et les
       3 variantes ProteinMPNN complets, cible lue à 198 résidus. Reste non établi : la
       **structure des sorties**, qui ne se verra qu'au premier vrai run.
2. [ ] **Re-mesurer le débit** sur `inputs/6ARU_A_309-506.pdb` avec
       `A318,A323,A406,A409` et `binder_lengths [55,95]` : temps par trajectoire, taux
       d'acceptation, profil de rejet par étage, coût réel. Tous les chiffres de
       `c0a48d5` sont caducs (§3). Lancer petit — `--max-trajectories 3` — pour fixer le
       coût par trajectoire avant d'engager un budget.
3. [x] ~~Vérifier si un kill par `TIMEOUT` commite le volume.~~ **Sans objet** : `TIMEOUT`
       n'existe pas dans 2.0, et `resume` est à `true` par défaut (§2). L'architecture en
       appels courts est sûre par construction. $0,20 économisés.
4. [ ] **Coldspots** : `A359` est **câblé** dans `modal_bindcraft2.py`. Restent les résidus à
       moins de ~10 Å d'un séquon, **jamais mesurés depuis les hotspots retenus** : c'est une
       mesure locale à écrire, pas une liste à deviner. Vérifier dans le log du run la ligne
       `target=… coldspots=… residues=N` qui dit combien ont été résolus.
5. [ ] **Multicible** : trouver ou modéliser le domaine III de Q01279. Sans ça, pas de
       multicible, et l'objectif n°2 reste un filtre a posteriori.
6. [ ] **Mesurer l'appariement His–acide** sur les structures des designs acceptés : une His du
       binder à portée de D323, ou un Asp à portée de H409. C'est l'objectif n°1 et il n'a
       jamais été mesuré.
7. [ ] Câbler le prédicteur orthogonal. Jamais fait.
8. [ ] Rédiger le dossier de méthodes **en parallèle des runs**, pas à la fin. NOTES.md en est
       la matière première ; `docs/ARCHITECTURE.md` §4 contient déjà l'inventaire des
       constantes avec leur statut de calibration, qui est la partie la plus défendable.

**Marge** : la clôture est dimanche 5 à 13h59 Paris. Viser samedi matin laisse ~24 h ; samedi
soir n'en laisse que ~15. La marge se compte contre dimanche 13h59, pas contre minuit.

---

## 9. Ce qui est mesuré, et ce qui ne l'est pas

Garde-fou contre la sur-confiance. Détail complet dans `docs/ARCHITECTURE.md` §4.

**Adossé à une mesure externe** : les bornes du domaine III (CATH-Gene3D), l'offset +24
(`_struct_ref_seq` et balayage, concordants), les 13 séquons et 25 ponts disulfure (UniProt),
le plancher de 400 Å² apolaires (médiane de l'empreinte du cétuximab), la sûreté de la
troncature (0,0 Å² d'écart).

**Posé, non calibré** : `PATCH_RADIUS = 11 Å` — capture seulement 40 % d'une empreinte réelle.
`MIN_REL_SASA = 0,20` — la convention couvre 0,20–0,25. Les trois critères d'ancre réelle
(25 Å², 40 %, 60°) — la calibration a échoué, l'empreinte du cétuximab ne contenant qu'un seul
Asp/Glu. `GROUP_LINK = 0,5`. Les paramètres d'alignement (−11/−1).

**Non mesuré du tout** : la convexité locale, la corréférence de face des hotspots,
l'appariement His–acide, le comportement en conformation **étendue**. Ce dernier point est la
limite la plus profonde : 6ARU est replié, toute SASA calculée ici en hérite, et c'est ce qui a
fait supprimer une colonne `face` qui donnait des résultats faux.
