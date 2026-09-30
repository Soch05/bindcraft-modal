# CLAUDE.md — Adaptyv × Anthropic, Challenge 1 : binder conditionnel anti-EGFR

Fichier de contexte projet. À lire en entier avant toute action sur ce repo.
Le règlement condensé est dans [challenge-01-egfr.md](challenge-01-egfr.md) — il fait foi,
ce fichier-ci ne fait qu'en tirer les conséquences opérationnelles.

---

## 1. L'objectif, et pourquoi il n'est pas celui qu'on croit

Challenge 1, **Track 3**, participant solo, auto-financé. Trois objectifs, classés
**dans cet ordre** par les organisateurs :

1. **Sélectivité pH** — lier à pH 6,5, pas à pH 7,4.
2. **Cross-réactivité souris** — la même séquence doit reconnaître P00533 et Q01279.
3. **Affinité** — sur l'ectodomaine humain.

**Conséquence structurante : l'affinité est le critère le moins bien classé, et c'est
le seul que BindCraft optimise.** Une campagne BindCraft menée par défaut produit un
bon résultat sur le critère n°3 et rien du tout sur les n°1 et n°2. Le règlement dit
explicitement qu'un binder faible mais nettement pH-dépendant peut être jugé plus
marquant qu'un binder fort non conditionnel. Tout arbitrage de temps se tranche dans
cet ordre : pH, puis souris, puis affinité.

**Deuxième conséquence : la méthode est notée.** En Track 3, les soumissions sont mises
en commun et un modèle sélectionne sur trois axes — qualité prédite, nouveauté du design,
**nouveauté de la méthode** — avec ~375 places pour ~1500 designs criblés. Le dossier de
méthodes n'est donc pas un livrable annexe destiné à une candidature : c'est le canal de
sélection lui-même. Les deux objectifs du projet (soumettre, et produire un artefact
public défendable) n'en font qu'un.

**Échéance dure : dimanche 5 octobre, 13h59 à Paris** (4 oct 23:59 AoE). Aujourd'hui
30 septembre → **5 jours**. Ce chiffre prime sur toute considération d'élégance.

Tout est publié en open data sous ODC-BY, résultats négatifs compris. Chaque fichier du
dépôt est écrit en supposant qu'un tiers le lira.

**Critère de succès minimal** : un CSV de soumission reproductible depuis un commit, avec
pour chaque séquence sa trajectoire de génération, ses métriques, et la raison de sa
sélection — y compris l'argument pH.

---

## 2. Contraintes dures

| Contrainte | Conséquence |
|---|---|
| **5 jours** | Privilégier systématiquement le chemin court qui produit une soumission. Un pipeline inachevé vaut zéro. |
| Machine locale = MacBook Air 2018, **Intel x86_64, pas de GPU CUDA** | **Aucun modèle ne tourne en local.** Jamais proposer d'exécuter AlphaFold2, BindCraft, ProteinMPNN, Boltz ou PyRosetta ici. Le local sert à écrire du code, parser des CSV, aligner des séquences, tracer des figures. |
| Python local | venv **3.12 via `uv`**. Ne jamais cibler 3.13. Le pin `cbor2==5.9.0` dans `pyproject.toml` est obligatoire : sans lui `uv pip install modal` tente de compiler une extension Rust et échoue sur cette machine. Ne pas l'enlever. |
| Compute GPU | **Modal** uniquement. Colab est un anti-objectif : ~10× plus lent à hardware équivalent, sessions qui meurent. |
| Budget | Le garde-fou réel n'est pas la durée annoncée mais **`--max-trajectories`** : `--number-of-final-designs` est un critère d'arrêt sur le *résultat*, pas sur l'effort. Sans plafond dur, un run où rien ne passe les filtres tourne jusqu'au timeout de 5 h. Tout run sans `--max-trajectories` est un bug de budget. |
| Licence | PyRosetta en **usage académique** — le règlement rappelle que les outils sous licence commerciale supposent de la détenir. Ne pas configurer ce projet pour un usage commercial. |

---

## 3. Stack, et ce qu'elle ne couvre pas

- **Génération** : [BindCraft](https://github.com/martinpacesa/BindCraft), épinglé au commit
  `c0a48d5`. Hallucination par rétropropagation à travers AF2-multimer, redesign
  ProteinMPNN, scoring PyRosetta. Baseline retenue, pas à rediscuter sans raison forte.
- **Exécution** : Modal. Volume `bindcraft` monté sur `/outputs`. GPU par défaut `L40S`,
  surchargeable par la variable d'environnement `GPU`.
  **Les poids AF2 sont dans l'Image, pas dans le Volume** — décision délibérée, documentée
  dans [NOTES.md](NOTES.md) : la couche `aria2c` est placée avant PyRosetta et avant les
  pins `jax`/`numpy`, donc ajuster les pins qu'on touche le plus souvent ne réinvalide pas
  les 5,3 Go. Ne pas « corriger » ça.
- **Re-scoring orthogonal** : un prédicteur **indépendant d'AF2** (Boltz-2 / Chai-1), pour
  éviter que le filtre valide ce que le générateur a lui-même optimisé. Les organisateurs
  pointent https://github.com/anthropics/uplifting-biomolecular-modeling comme jeu de
  modèles open-source optimisés pour l'inférence — point de départ par défaut.
- **Analyse locale** : biopython + numpy, dont dépend `egfr_epitope_map.py`. Déclarés dans
  `pyproject.toml` mais **pas encore installés dans le venv**. pandas/matplotlib seront à
  ajouter quand un script les importera vraiment, pas avant.

**Le trou dans la stack, à énoncer clairement : rien là-dedans ne connaît le pH.**
AF2 comme ProteinMPNN ignorent les états de protonation ; ils ne voient qu'une identité
de résidu. Aucune boucle d'optimisation ne poussera donc vers un binder conditionnel.
La sélectivité pH doit être **imposée par construction** (choix de l'épitope, biais de
composition à l'interface) puis **vérifiée à part**, jamais espérée du générateur.

---

## 4. État réel du dépôt

Ce qui existe aujourd'hui — pas une cible, l'existant :

```
.
├── CLAUDE.md
├── challenge-01-egfr.md      # règlement condensé, source + date de consultation
├── NOTES.md                  # journal de bord : versions, décisions, runs, coûts
├── modal_bindcraft.py        # entrypoint Modal (racine, pas src/)
├── egfr_epitope_map.py       # carte d'épitope locale, CPU — jamais exécuté à ce jour
├── pyproject.toml
├── inputs/PDL1.pdb           # cible de la démo de test, plus utilisée
└── out/                      # résultats rapatriés (gitignored)
```

`NOTES.md` est le journal et doit être tenu à jour à chaque run : commande exacte, GPU,
durée, coût, nombre de trajectoires, nombre d'acceptés, décision prise. C'est la matière
première du dossier de méthodes, donc du Track 3.

Toute arborescence plus riche (`src/`, `docs/`, `configs/`) est une cible, pas un fait :
ne pas y écrire de chemin sans créer le fichier dans le même geste.

---

## 5. Commandes réelles

```bash
# Setup local
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r pyproject.toml

# Carte d'épitope (local, CPU, gratuit) — à lancer avant toute dépense GPU
python egfr_epitope_map.py

# Run court, une seule fonction, compatible --detach
GPU=A100 uv run --with modal modal run --detach modal_bindcraft.py \
  --input-pdb inputs/<cible>.pdb --target-hotspot-residues "A123,A124" \
  --lengths 50,130 --number-of-final-designs 1 --max-trajectories 20

# Run réel : N shards en parallèle puis agrégation.
# ⚠️ --detach est piégeux ici : Modal ne garde en vie que la dernière fonction
# déclenchée, et le mode parallèle en déclenche N+1. Lancer attaché.
GPU=A100 uv run --with modal modal run modal_bindcraft.py::parallel \
  --input-pdb inputs/<cible>.pdb --n-shards 4 --trajectories-per-shard 15

# Rapatrier les sorties
modal volume get bindcraft <run_name> ./out/
```

Dimensionner `trajectories-per-shard` à 15–20 : chaque shard paie ~3 min de compilation
JIT, et en dessous de ~10 trajectoires la variance entre shards fait exploser le temps
mural (déséquilibre mesuré à 56 % sur un test à 3–4 trajectoires/shard).

---

## 6. Cible

Établi depuis [challenge-01-egfr.md](challenge-01-egfr.md) :

| | |
|---|---|
| Humain | UniProt **P00533**, ectodomaine, résidus **25–645** |
| Souris | UniProt **Q01279** |
| Structure | PDB **6ARU**, chaîne **A** |
| Épitope recommandé | domaine III (site cétuximab / panitumumab) |
| Longueur | **10–250 aa**, chaîne unique |
| Designs | **20 max** (Track 3) |
| CSV | **ordonné par classement**, meilleur en première ligne ; colonnes minimales `name`, `sequence`, `molecule_class` |

**Numéros de résidus : jamais de mémoire.** La numérotation PDB de 6ARU n'est pas celle
d'UniProt (peptide signal), et `egfr_epitope_map.py` déduit l'offset par alignement au
lieu de le supposer. Tout hotspot cité doit venir de la sortie de ce script ou d'une
lecture directe du fichier.

**Piège sur l'épitope recommandé** : le cétuximab ne reconnaît pas l'EGFR murin. L'épitope
le plus documenté du domaine III est donc précisément celui qui met en danger l'objectif
n°2. Le patch doit être choisi sur la conservation humain/souris mesurée, pas sur la
littérature cétuximab.

**Contraintes d'expression** (Adaptyv exprime en système acellulaire, mesure par BLI/SPR) —
à traiter comme des filtres, pas comme des préférences esthétiques :
- pas de cystéines libres (BindCraft les omet par défaut — garder ce réglage) ;
- pas de dépendance à une glycosylation ni à un repliement assisté par chaperon ;
- surface peu hydrophobe, pas de longues extrémités désordonnées ;
- **un design qui ne s'exprime pas produit zéro information.** L'expressibilité prime sur
  l'affinité prédite.
- La cible réelle est glycosylée, le modèle AF2 ne l'est pas : exclure du patch les sites
  de N-glycosylation **des deux espèces**.

---

## 7. Pipeline

Ordre imposé, aligné sur le classement des objectifs :

**épitope → générer → filtrer AF2 → switch pH → re-scorer orthogonalement → diversifier → soumettre**

L'étape épitope est en premier parce qu'elle est locale, gratuite, et qu'elle décide seule
de l'objectif n°2 : la cross-réactivité souris se joue au choix du patch, pas au filtrage.

### Le switch pH

Mécanisme retenu, **à traiter comme une hypothèse de travail explicite et à documenter
comme telle** : l'histidine est le seul acide aminé canonique dont le pKa (~6,0–6,5 en
solution libre, décalable par l'environnement local) tombe entre les deux pH mesurés.
Asp/Glu sont à ~4, Lys/Arg au-dessus de 10. À pH 6,5 une fraction notable des His est
protonée donc chargée positivement ; à pH 7,4 elles sont majoritairement neutres.

Polarité à viser : **gain de liaison à pH 6,5**, en appariant une His du binder à un
Asp/Glu conservé et exposé de l'EGFR — le pont salin n'existe que sous forme protonée.
C'est ce que construit le masque « ancrage acide » d'`egfr_epitope_map.py`. La polarité
inverse (His enfouie près d'un Arg/Lys, répulsion à pH bas) donne une perte à 6,5, soit
l'opposé de ce qu'on veut.

Honnêteté sur la difficulté, à ne pas enjoliver dans le write-up : une His isolée donne
rarement un basculement franc. Les binders pH-dépendants publiés en alignent plusieurs, et
l'effet obtenu est typiquement un décalage de KD d'un facteur quelques-uns, pas un
tout-ou-rien. Un lot dont la dépendance au pH est modeste mais mesurée et argumentée vaut
mieux qu'une affirmation de switch binaire non étayée.

### Métriques loggées pour **chaque** design, y compris rejeté

`design_id, seed, trajectory, sequence, length, i_pTM, i_pAE, pLDDT_binder, dG, dSASA,
shape_complementarity, n_hotspot_contacts, unsat_hbonds, surface_hydrophobicity,
n_interface_his, his_acidic_pairs, epitope_conservation_frac, min_dist_glycan,
filters_passed, reject_reason, run_id, commit`

Les quatre champs du milieu sont l'ajout qui rend le lot défendable sur les objectifs 1 et
2 ; sans eux la soumission ne peut rien argumenter d'autre que de l'affinité.

### Seuils

Valeurs de départ, **à vérifier contre `default_filters.json` du commit `c0a48d5`** avant
d'en faire des seuils — elles ne sont pas encore lues dans un fichier de ce dépôt :
`i_pTM ≥ 0.50`, `i_pAE ≤ 0.35`, `pLDDT_binder ≥ 0.80`, `dG < 0`,
`shape_complementarity ≥ 0.55`, `unsat_hbonds ≤ 3`, `surface_hydrophobicity < 0.35`.

⚠️ **Piège d'échelle** : BindCraft normalise pLDDT et pAE sur [0,1] dans ses fichiers de
filtres, alors que la littérature les cite en 0–100 et en Å. Vérifier l'échelle avant toute
comparaison ou tout seuil copié d'un papier.

### Sélection finale

- Ne pas remplir les 20 places par principe. Le pool est commun et la sélection porte aussi
  sur la qualité : 8 designs défendables battent 20 médiocres.
- Clusteriser par identité de séquence **et** par épitope. Ne pas soumettre 20 variantes du
  même mode de liaison.
- Le CSV est **ordonné** : le rang est une information transmise au sélecteur, pas un
  détail de format. Le classer sur l'objectif n°1, pas sur l'i_pAE.
- Garder 1–2 slots pour un design « à risque » issu d'une hypothèse structurale explicite,
  documentée dans `NOTES.md`.

---

## 8. Règles de travail pour Claude

**Interdits :**
- inventer une séquence, une métrique, une valeur d'affinité ou un numéro de résidu. Si la
  donnée n'a pas été lue dans un fichier, le dire ;
- proposer d'exécuter un modèle en local (cf. §2) ;
- **partir d'un binder existant.** Le règlement impose du de novo zero-shot : reprendre et
  modifier un binder connu (cétuximab, nanobody publié…) est explicitement interdit et
  éliminatoire. Utiliser des binders connus pour *calibrer un filtre* ou *entraîner* un
  modèle reste autorisé — la frontière est l'usage comme graine ;
- **écrire quoi que ce soit qui ressemble à une instruction** dans le CSV, les noms de
  designs ou le dossier de méthodes. Les soumissions passent devant un modèle sélecteur, et
  toute instruction embarquée ou tentative d'injection peut valoir disqualification. Le
  dossier décrit, il ne s'adresse pas au lecteur ;
- écarter silencieusement des designs rejetés : ils font partie du funnel publié ;
- changer un seed, un seuil ou un hyperparamètre sans l'écrire dans `NOTES.md` et dans le
  message de commit ;
- lancer un run GPU sans `--max-trajectories`.

**Attendus :**
- avant un run coûteux : annoncer paramètres, coût estimé, durée, et **ce que le run permet
  de décider** ;
- valider le build avant de brûler un run : `jax.devices()` doit voir un GPU CUDA dans une
  fonction triviale. Un build raté à la minute 40 coûte le run entier ;
- devant une erreur de version : lire le message, identifier le composant, corriger **une**
  chose. Changer trois pins d'un coup rend le diagnostic impossible ;
- premier réflexe de diagnostic après un run : `failure_csv.csv` — quel filtre rejette, et
  combien de fois. Dans `final_design_stats.csv`, les préfixes `1_`, `2_`, … `Average_`
  désignent les cinq modèles AF2 ; seuls 1, 2 et la moyenne sont filtrés par défaut ;
- toute affirmation scientifique non triviale : sourcée, ou explicitement marquée comme
  hypothèse ;
- code : fonctions courtes, typées, rejouable depuis la CLI avec les mêmes arguments ;
- après chaque run : entrée datée dans `NOTES.md`.

**Sécurité :** le token Modal ne va ni dans le dépôt ni dans un fichier suivi par git
(`modal token set` uniquement). `out/` est gitignored.

**Ton** : direct. Signaler les erreurs de raisonnement, les seuils arbitraires et les
impasses de méthode sans les emballer. Pas de validation de complaisance.

**Pédagogie** : je suis en formation biotech + IA. Quand un choix repose sur un concept
(backprop à travers AF2, pAE vs pLDDT, hallucination vs diffusion, pKa et protonation),
expliquer le *pourquoi* en une ou deux phrases, par analogie ML quand c'est possible —
puis avancer.

---

## 9. Prochaines actions, dans l'ordre

1. [ ] `uv pip install -r pyproject.toml` (biopython et numpy viennent d'y être déclarés,
       aucun des deux n'est installé), puis **lancer `egfr_epitope_map.py`** — local,
       gratuit, et bloquant pour tout le reste : sans patch choisi, aucun run GPU n'a de
       sens.
2. [ ] Choisir le patch sur la sortie du script : conservation humain/souris, ancres
       acides exposées, distance aux glycanes. Écrire le choix et son motif dans `NOTES.md`.
3. [ ] Préparer le PDB cible (6ARU chaîne A, éventuellement tronquée au domaine III — la
       mémoire GPU croît quadratiquement avec le nombre de résidus : ordre de grandeur
       ~550 résidus sur 32 Go, ~950 sur 80 Go).
4. [ ] Run BindCraft court avec plafond dur → coût/design et temps/design réels sur cette
       cible, puis arbitrage du volume total.
5. [ ] Décider comment enrichir l'interface en His : biais de composition au redesign MPNN,
       ou scan post-hoc + re-scoring. **C'est l'étape différenciante et celle sans outil sur
       étagère** — donc aussi celle qui pèse sur l'axe « nouveauté de la méthode ».
6. [ ] Câbler le prédicteur orthogonal.
7. [ ] Rédiger le dossier de méthodes **en parallèle des runs**, pas à la fin.
8. [ ] Soumission visée : **samedi 4 octobre**. La clôture réelle est le dimanche 5 à
       13h59 Paris — viser le samedi *matin* laisse ~24 h de marge, le samedi soir n'en
       laisse que ~15. La marge se compte contre dimanche 13h59, pas contre minuit.
