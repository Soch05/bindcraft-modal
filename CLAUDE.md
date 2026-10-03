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
| **Budget : le frein est `TIMEOUT`, pas le plafond de trajectoires** | Établi sur la source BindCraft : `check_n_trajectories` ne compte que les `.pdb` de `Trajectory/Relaxed`. Une trajectoire qui finit en `LowConfidence` ou `Clashing` **ne consomme pas le quota tout en ayant consommé du GPU**. `--max-trajectories` reste un garde secondaire utile, mais le budget se pilote par `TIMEOUT` (minutes, défaut 300). **À revérifier sur 2.0.** |
| Licence | PyRosetta en **usage académique**. Ne pas configurer ce projet pour un usage commercial. |

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

**Prérequis non vérifié** : le multicible demande une structure du domaine III **murin**.
Aucune recherche n'a été faite sur l'existence d'une structure expérimentale de Q01279 ;
à défaut, un modèle AlphaFold, avec les réserves que ça implique pour tout calcul de SASA.

**⚠️ Ce qui n'est PLUS valide.** L'ancienne version était épinglée au commit `c0a48d5`, et
tous les chiffres de débit de [NOTES.md](NOTES.md) y sont attachés :

| mesuré sur `c0a48d5` | valeur | statut sur 2.0 |
|---|---|---|
| temps par trajectoire relaxée, cible 198 résidus | 8,49 min | **à re-mesurer** |
| temps par tentative, tout compris | 10,1 min | **à re-mesurer** |
| coût, 6 tentatives sur L40S | $1,97 | **à re-mesurer** |
| taux de relaxation | 3/6 | **à re-mesurer** |
| `check_n_trajectories` ne compte que `Relaxed` | établi | **à revérifier** |
| filtre dominant : `i_pAE`, 20 rejets sur 29 | établi | **à revérifier** |
| noms de colonnes de `failure_csv.csv` | établis | **à revérifier** |
| seuils de `default_filters.json` | non lus | **à lire** |

Le build Modal, les pins `jax`/`numpy`, l'emplacement des poids AF2 et la structure des
sorties sont également à réétablir. **Valider le build avant toute dépense** (§8).

### Exécution : Modal

Volume `bindcraft` monté sur `/outputs`, un répertoire par `run_name`. GPU par défaut `L40S`
(46 Go vérifiés), surchargeable par la variable d'environnement `GPU`. Tarif L40S mesuré :
**$0,000542/s = $1,95/h**.

Sur l'ancien build, les poids AF2 étaient dans l'Image et non dans le Volume — décision
délibérée documentée dans NOTES.md, la couche `aria2c` étant placée avant PyRosetta et avant
les pins. Si 2.0 garde une structure d'image comparable, reproduire ce choix.

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
├── modal_bindcraft.py            entrypoint Modal — POUR c0a48d5, à refaire pour 2.0
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

`inputs/PDL1.pdb` est la cible de la démo du 23 septembre, **plus utilisée** — conservée
seulement parce que le run `test1` de NOTES.md s'y réfère.

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

⚠️ **Piège d'échelle** : BindCraft normalise pLDDT et pAE sur [0,1] dans ses fichiers de
filtres, alors que la littérature les cite en 0–100 et en Å. Vérifier l'échelle avant toute
comparaison ou tout seuil copié d'un papier.

Sur `c0a48d5`, le profil de rejet mesuré était **`i_pAE` 20 rejets, `pLDDT` 6, `i_pTM` 3,
tout le reste 0**. Si le rendement doit monter, c'est le seul levier qui compte — mais
**à revérifier sur 2.0** avant d'y toucher, et contre le `default_filters.json` réel.

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

1. [ ] **Installer et valider BindCraft 2.0 sur Modal.** Build, pins, poids AF2, structure des
       sorties. `jax.devices()` doit voir un GPU CUDA avant toute dépense.
2. [ ] **Re-mesurer le débit** sur `inputs/6ARU_A_309-506.pdb` avec
       `A318,A323,A406,A409` et `--lengths 55,95` : temps par tentative, taux de
       relaxation, taux d'acceptation, profil de rejet, coût réel. Tous les chiffres de
       `c0a48d5` sont caducs (§3).
3. [ ] **Vérifier si un kill par `TIMEOUT` commite le volume.** Une annulation le fait, c'est
       établi ; un timeout n'est pas testé. ~$0,20 sur un run `TIMEOUT=6`. Conditionne
       l'architecture en appels courts.
4. [ ] **Coldspots** : écarter `H359` (diverge en Arg) et les résidus à moins de ~10 Å d'un
       séquon. C'est la première fonctionnalité de 2.0 à exploiter.
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
