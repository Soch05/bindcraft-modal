# Architecture — `egfr_epitope_map.py`

Carte d'épitope de l'EGFR humain pour le Challenge 1 Adaptyv × Anthropic. Local, CPU,
coût nul. Le script **ne sélectionne pas** de site : il produit des colonnes et des
distributions.

- Lecture des sorties : [`data/LECTURE.md`](../data/LECTURE.md)
- Journal des décisions, runs et erreurs : [`NOTES.md`](../NOTES.md)
- Contexte et règles du projet : [`CLAUDE.md`](../CLAUDE.md)

Dépendances : biopython 1.88, numpy 2.5.3, Python 3.12.

---

## 1. Flux de données

```mermaid
flowchart TD
    subgraph sources["Sources distantes (cache disque dans data/, gitignoré)"]
        UP["UniProt REST<br/>P00533, Q01279<br/>séquences, séquons,<br/>ponts disulfure"]
        RCSB["RCSB<br/>6ARU.cif<br/>coordonnées,<br/>_struct_ref_seq"]
        IP["InterPro API<br/>Pfam PF01030<br/>CATH-Gene3D 3.80.20.20"]
        PDBE["PDBe API<br/>mappings/cath/6aru"]
    end

    UP --> ALIGN["align_orthologs<br/>mapping position → (aa souris, statut)"]
    RCSB --> LOAD["load_chain<br/>chaîne A, hétéroatomes détachés<br/>SASA atomique"]
    RCSB --> OFF["scan_offset + deposited_offset<br/>offset +24, assertion d'accord"]
    IP --> DOM["resolve_domain_iii"]
    PDBE --> DOM
    LOAD --> FOOT["fab_footprint<br/>24 résidus < 4,5 Å"]
    FOOT --> DOM
    DOM --> |"domaine III<br/>333-530 UniProt<br/>309-506 PDB"| TABLE

    ALIGN --> TABLE["build_table<br/>1 ligne par résidu observé"]
    LOAD --> TABLE
    OFF --> TABLE
    UP --> TABLE
    FOOT --> TABLE

    TABLE --> ENUM["enumerate_patches<br/>98 centres × 2 jeux de membres"]
    LOAD --> ENUM
    ENUM --> STATS["patch_stats<br/>masqué + _full"]
    STATS --> CSV1["data/egfr_patches.csv<br/>93 lignes, 41 colonnes"]
    TABLE --> CSV2["data/egfr_residues.csv<br/>198 lignes"]
    TABLE --> PDB["data/6aru_chainA_conserv.pdb<br/>B-factor = conservation"]
    STATS --> REPORTS["classements et rapports<br/>stdout"]
```

Le cache est un simple « le fichier existe → ne pas retélécharger ». Un rejeu complet ne
fait aucun appel réseau. Les caches `data/*.json` et `data/*.cif` sont gitignorés
délibérément : ils sont reproductibles depuis la source.

## 2. Chaîne de transformation d'un résidu

```mermaid
flowchart LR
    A["résidu observé<br/>6ARU chaîne A"] --> B["anchor_atom<br/>CB, ou CA si absent"]
    A --> C["ShrakeRupley niveau atome<br/>sonde 1,40 Å, 100 points"]
    C --> D["split_sasa<br/>C,S → apolaire<br/>N,O → polaire"]
    D --> E["rel_sasa = total / MAX_ASA<br/>Tien et al. 2013"]
    E --> F{"exposed ?<br/>rel_sasa ≥ 0,20"}
    F -->|non| G["hors de tout patch"]
    F -->|oui| H["membre candidat"]
    B --> I["distances : centre, séquons,<br/>empreinte Fab"]
    I --> H
    H --> J{"dans le domaine III ?"}
    J -->|oui| K["membre masqué<br/>+ centre possible"]
    J -->|non| L["membre _full seulement"]
    K --> M["patch_stats<br/>sommes, comptages, minima"]
    L --> M
    M --> N["tri sur sasa_apolar_full"]
    N --> O["groupement par<br/>recouvrement de membres"]
```

Un résidu non exposé ne disparaît pas : il compte dans `n_total`, le dénominateur du taux
d'exposition.

## 3. Fonctions, dans l'ordre d'appel

### Accès aux sources

| fonction | prend | rend | décision portée |
|---|---|---|---|
| `fetch` | url, chemin | chemin local | aucune — cache « existe ou non » |
| `fetch_json` | url, nom | dict ou `None` | **tolère l'absence d'une source** : renvoie `None` au lieu d'échouer, pour que PDBe muet n'arrête pas le script |
| `load_uniprot` | accession | entrée UniProt | aucune |
| `uniprot_features` | entrée, type | liste (desc, début, fin) | ignore les features sans bornes numériques |
| `disulfide_cys` | entrée humaine | positions UniProt | aucune — lit les 25 `Disulfide bond` |

### Bornes de domaine

| fonction | prend | rend | décision portée |
|---|---|---|---|
| `interpro_domains` | base, requête | segments UniProt | aucune |
| `gene3d_superfamily_domains` | — | segments de `G3DSA:3.80.20.20` | **filtre sur la superfamille** par suffixe d'accession |
| `pdbe_cath_domains` | — | segments en numérotation auteur | **ne retient que la chaîne cible** ; renvoie vide sur 6ARU, CATH ne classant que le Fab |
| `pick_domain` | segments, empreinte | un segment | **règle de désignation** : le segment contenant le plus de résidus de l'empreinte du Fab. Géométrique, jamais nominative ni indicielle |
| `resolve_domain_iii` | empreinte, offset | (début, fin) UniProt | **choix de la source : CATH-Gene3D**. Pfam PF01030 déclare omettre ~50 résidus en tête du domaine ; PDBe CATH ne couvre pas la chaîne A. Arrête le script si Gene3D est muet |

### Alignement et structure

| fonction | prend | rend | décision portée |
|---|---|---|---|
| `align_orthologs` | 2 séquences | mapping position → (aa, statut) | **aligne les précurseurs entiers**, puis découpe. Découper d'abord supposerait les bornes des deux côtés. Classe en `identical` / `similar` / `different` |
| `fab_footprint` | modèle | ensemble de numéros PDB | **appelé avant de détacher le Fab**. Double usage : contrôle positif du masque de conservation, et système de désignation du domaine |
| `deposited_offset` | chemin du cif | offset, liste d'écarts | **autorité externe** : lit `_struct_ref_seq`. Arrête si la correspondance n'est pas un décalage unique |
| `scan_offset` | résidus, séquence | offset | inférence par maximisation de concordance ; arrête sous 95 % |
| `load_chain` | séquence humaine | résidus, offset, empreinte, écarts | **assertion d'accord entre les deux routes de numérotation** — arrêt en cas de désaccord. Détache les hétéroatomes **avant** la SASA |
| `report_difs` | écarts, offset, bornes | — | **alerte bruyante** si un écart séquence/structure tombe dans le domaine III : `aa_human` vient de la structure et le statut souris de la séquence UniProt, ces positions seraient incohérentes |
| `cetuximab_control` | empreinte, offset, mapping | — | **contrôle positif inversé** : le cétuximab ne reconnaissant pas l'EGFR murin, son épitope doit sortir **peu** conservé. 70,8 % contre 87,4 % sur le domaine. Une empreinte très conservée signalerait un alignement cassé |

### Table et patches

| fonction | prend | rend | décision portée |
|---|---|---|---|
| `anchor_atom` | résidu | atome | **CB plutôt que CA** : approxime la direction de la chaîne latérale |
| `split_sasa` | résidu | (apolaire, polaire) | **ventilation par élément** : C,S contre N,O |
| `build_table` | résidus, offset, mapping, bornes, empreinte, séquons, cystéines | liste de lignes | arrête si aucun séquon ou aucun résidu d'empreinte n'est observable — refus du repli silencieux sur une distance infinie |
| `patch_stats` | membres | dict d'agrégats | **aucun masque appliqué ici** : appelée deux fois, sur les membres masqués puis complets |
| `enumerate_patches` | lignes, résidus | 98 patches | **centres = résidus exposés du domaine III ; membres = toute la chaîne A**. Produit `n_total`, `frac_exposed`, `n_truncated` |
| `reference_site` | passants du plancher | membres, noms | **désignation du site de référence par les données** : composantes connexes des patches à identité parfaite, la plus grande, étendue à ses satellites |
| `site_report` | patches, plancher, bornes | — | **disjonction en fraction de membres partagés**, jamais en distance entre centres. Clé alignée sur l'ordre des objectifs : ancre acide, puis identité, puis surface |

### Rapports

`size_distribution`, `overlap_distribution`, `dedup_sweep`, `show_ranking`,
`footprint_report`, `density_report`, `truncation_report`, `floor_report`,
`neighbourhood_report`, `cysteine_report`. Aucun ne filtre ; tous impriment.
`write_csv`, `write_bfactor_pdb` écrivent les sorties.

---

## 4. Inventaire des constantes et critères binaires

**C'est la partie qui compte.** Tout ce qui, en changeant, changerait la sortie. État des
lieux, sans proposition de correction.

### Identifiants — non arbitraires

| nom | valeur | origine |
|---|---|---|
| `HUMAN_AC` | `P00533` | [`challenge-01-egfr.md`](../challenge-01-egfr.md) |
| `MOUSE_AC` | `Q01279` | idem |
| `PDB_ID` / `PDB_CHAIN` | `6ARU` / `A` | idem |
| `FAB_CHAINS` | `("B", "C")` | lu dans 6ARU.cif |
| `PFAM_AC` | `PF01030` | accession Pfam « Receptor L domain » |
| `GENE3D_SF` | `3.80.20.20` | superfamille CATH « Receptor L-domain » |

### Seuils géométriques et physiques

| nom | valeur | origine | statut de calibration |
|---|---|---|---|
| `MIN_REL_SASA` | `0.20` | **convention du domaine** — seuil usuel d'exposition relative | **non calibré sur ce projet.** Détermine quels résidus sont membres ; décale tout en aval |
| `PATCH_RADIUS` | `11.0` Å | **posé arbitrairement** | **non calibré.** Fixe la taille de l'unité « site ». Détermine aussi le recouvrement entre patches, donc tout le groupement |
| `CONTACT_CUTOFF` | `4.5` Å | **convention du domaine** — contact lourd | non calibré, mais l'empreinte obtenue (24 résidus, 349-473) est cohérente avec un Fab |
| sonde SASA | `1.40` Å, 100 points | **défaut Biopython** `ShrakeRupley` | non touché. Rayon = molécule d'eau |
| `MAX_ASA` | 20 valeurs | **fichier cité** : Tien et al. 2013, valeurs théoriques | publié, pas recalculé ici |
| `APOLAR_ELEMENTS` / `POLAR_ELEMENTS` | `{C,S}` / `{N,O}` | **posé** — ventilation par électronégativité | **équivalence posée** : un carbone de tige de Lys compte comme un carbone de Leu. Les hydrogènes sont absents de la structure (3,20 Å) |

### Seuils de sélection et de classement

| nom | valeur | origine | statut |
|---|---|---|---|
| `MIN_PATCH_MEMBERS` | `6` | **posé** pour un régime d'énumération qui n'existe plus (9 ancres acides) | **obsolète de fait** : ne concerne plus que 5 patches sur 98. La distribution complète est imprimée avant application |
| `FLOOR_APOLAR` | `400.0` Å² | **mesure externe** : médiane 388 Å² des 38 patches recouvrant l'empreinte du cétuximab | **le seul seuil du fichier adossé à une mesure.** Et c'est une référence d'échelle, pas un seuil de suffisance — un Fab enfouit plus qu'un mini-binder |
| `GROUP_LINK` | `0.5` | **posé** | **non calibré.** Définit quand deux patches sont « le même site » |
| `DISJOINT_SWEEP` | `(0.0, 0.1, 0.25)` | **délibérément non fixé** | balayage au lieu d'un seuil. 0,00 et 0,10 donnent les mêmes 3 sites |
| `DEDUP_SWEEP` | 0,1 → 0,9 | **délibérément non fixé** | le balayage ne montre aucune coupure naturelle : continuum de 7 à 26 sites |
| `HYDROPHOBIC` | `LIVFMWY` | **posé** | conservé **uniquement** pour comparaison avec `sasa_apolar`, qui l'a remplacé comme clé |

### Paramètres d'alignement

| paramètre | valeur | origine | statut |
|---|---|---|---|
| matrice | BLOSUM62 | convention | non calibré |
| ouverture de gap | `-11` | **convention** (réglage BLASTP usuel) | non calibré ; le défaut Biopython serait −1 |
| extension de gap | `-1` | idem | idem |
| gaps terminaux | `0.0` | **posé** — les précurseurs entiers n'ont pas à être pénalisés aux bouts | non calibré |
| critère `similar` | `BLOSUM62[a,b] > 0` | **posé** | **non calibré.** N'affecte pas `frac_ident`, qui ne compte que `identical` |
| fenêtre de balayage d'offset | `range(-60, 61)` | **posé** | large devant le peptide signal de 24 résidus |
| fiabilité d'offset | `frac < 0.95` → arrêt | **posé** | la valeur observée est 99,7 % (607/609) |

### Critères binaires et seuils d'alerte

| critère | valeur | rôle | statut |
|---|---|---|---|
| alerte conservation de l'empreinte | `frac >= 0.90` | déclenche l'avertissement « masque de conservation inopérant » | **posé.** Valeur observée 70,8 %, l'alerte ne se déclenche pas |
| identité parfaite | `frac_ident_full >= 1.0` | définit les patches qui amorcent le site de référence | strict par construction |
| voisinage clairsemé | `n_total <= 13` | **reporting uniquement** | **posé puis écarté comme critère** : l'empreinte couvre les deux régimes |
| séquon parmi les membres | `min_glyc == 0.0` | signalement | exact par construction |
| connexité du site de référence | recouvrement `> 0` | composantes connexes des patches parfaits | sans seuil, volontairement |
| clé de tri des sites | `n_acidic_cons` → `frac_ident` → `sasa_apolar` | ordre des objectifs du règlement | **choix documenté**, pas une mesure |

### Ce qui n'est **pas** un paramètre

Les bornes du domaine III (333-530 UniProt) viennent de CATH-Gene3D. L'offset (+24) vient
de `_struct_ref_seq` et du balayage. Les 13 séquons, les 25 ponts disulfure et les deux
conflits de séquence viennent d'UniProt et du mmCIF. Aucune de ces valeurs n'est écrite
dans le script.

---

## 5. Trois erreurs trouvées par la mesure

### `Repeat` UniProt inutilisable comme borne de domaine

P00533 n'a **qu'une** feature `Domain` : la kinase, 712-979. Aucun « Receptor L-domain ».
Les L-domaines n'existent que comme `Repeat` marquées « Approximate » : 75-300 et 390-600.
La seconde vaut 366-576 en numérotation mature, décalée d'une cinquantaine de résidus du
domaine III structural. Le script échouait dessus. Remplacée d'abord par un rayon de 30 Å
autour du centroïde de l'empreinte — qui donnait 335-535, à 2-5 résidus de la vraie borne —
puis par CATH-Gene3D. **La fausse borne a été détectée en comparant deux sources, pas en
relisant le code.**

### Funnel effondré sur une tautologie d'échantillonnage

`rank_patches` itérait sur les ancres acides conservées et exposées pour produire les
centres, puis ramassait les membres. « Être centré sur un Asp/Glu conservé » n'était donc
pas un critère appliqué à un échantillon de sites : c'était la **définition** du site. Le
funnel tombait à 1 site, et la conclusion « 0 hydrophobe sur les 8 patches » ne mesurait
rien d'autre que le fait qu'un disque centré sur un résidu chargé est polaire.

Après passage à 98 centres — tous les résidus exposés du domaine III — **64 patches sur 93
portent une ancre acide conservée** et 15 combinent `apolar_frac ≥ 0,55` avec au moins une
ancre. L'incompatibilité « ancrage acide contre designabilité », posée comme hypothèse
structurale, n'existait pas dans les données.

### Pénalité glycane annulant 22 patches sans ligne de journal

Le filtre dur `dist_glycan > 12 Å` écartait 7 patches sur 9 dans l'ancien régime. Remplacé
par une pénalité continue `clamp(min_glyc / 12, 0, 1)` multipliant la SASA apolaire. Mesure
faite avant de s'en servir : la pénalité vaut **exactement 0** à `min_glyc = 0`, donc elle
**annulait 22 patches sur 93** — le filtre dur, mais sans trace. Et 77 sur 93 étaient
pénalisés, avec des écarts de rang jusqu'à −33. Sa pente imposait de surcroît un taux de
change jamais choisi : 1 Å de distance au séquon contre 1/12 de la SASA apolaire, soit 50 Å²
sur un patch à 600 Å².

Retirée. `min_glyc` est une colonne brute en Å, lue à la main sur la liste courte.

---

## 6. Rejouer

```bash
uv venv --python 3.12 && source .venv/bin/activate
uv pip install -r pyproject.toml
python egfr_epitope_map.py
```

Premier appel : ~2 Mo téléchargés dans `data/`. Rejeux : aucun réseau. Durée de l'ordre de
la dizaine de secondes sur CPU. Aucun GPU, aucun modèle.
