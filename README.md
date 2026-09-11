# PNTCLOUD2CONVERTER — E57/RCP → Gaussian Splat (SuperSplat)

Convertit un nuage de points géomatique/BIM coloré (scan laser `.e57`, ou
export `.pts`/`.xyz`/`.las` depuis Autodesk ReCap) en un fichier `.ply` de
**3D Gaussian Splatting**, directement chargeable dans
[SuperSplat](https://superspl.at/editor) (l'éditeur/visualiseur web de
PlayCanvas).

## Ce que fait — et ne fait pas — cet outil

Il ne s'agit **pas** d'un entraînement 3DGS classique (COLMAP + optimisation
par gradient à partir de photos). Sans photographie appariée, il n'y a pas
d'information *view-dependent* à apprendre. À la place, l'outil génère des
**splats de surface synthétiques** directement depuis le nuage de points
existant :

- position + couleur → prises telles quelles (ou dérivées de l'intensité si
  pas de RGB),
- normale locale → estimée par PCA sur le voisinage (ou fournie si le
  fichier source la contient déjà),
- échelle + orientation du splat → dérivées de la densité locale et de la
  normale (aplatissement en disque, comme un "surfel"),
- opacité → quasi-pleine en zone dense, plus douce en zone clairsemée.

Le résultat est un nuage de splats fixes, sans coefficients d'harmoniques
sphériques d'ordre supérieur (pas de rendu dépendant du point de vue), qui
se charge, s'édite (recadrage, suppression, export) et se rend nativement
dans SuperSplat.

## Structure du projet

```
splatconv/            package Python partagé par le CLI et l'API web
  io/readers.py        lecteurs .e57 / .pts / .xyz / .las + garde-fou .rcp
  processing/           recentrage, normales, échelle/rotation/opacité
  pipeline.py           orchestration complète (utilisée par CLI et API)
  ply_writer.py         écriture du .ply binaire (ordre de champs exact)
backend/               API FastAPI (upload, conversion async, progression, téléchargement)
frontend/              UI drag & drop en HTML/JS vanilla
cli/convert.py         script CLI autonome, pour un usage batch
scripts/make_test_cube.py  génère un petit nuage de test + son .ply de validation
tests/                 tests pytest (writer PLY + pipeline complet)
test_data/             .ply d'exemple déjà généré, pour tester SuperSplat tout de suite
```

## Installation

```bash
python3 -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
```

### Dépendance système pour `pye57` / `libE57Format`

`pye57` embarque généralement des wheels précompilées pour Linux/macOS/Windows
récents, donc `pip install pye57` suffit dans la plupart des cas. Si
l'installation échoue et compile depuis les sources, il faut au préalable :

- **Debian/Ubuntu** : `sudo apt-get install cmake build-essential libxerces-c-dev`
- **macOS (Homebrew)** : `brew install cmake xerces-c`
- **Windows** : Visual Studio Build Tools + CMake, ou utiliser WSL.

### À propos d'Open3D (optionnel)

Le brief original prévoit Open3D (`estimate_normals`,
`orient_normals_consistent_tangent_plane`, `voxel_down_sample`). Open3D reste
supporté et utilisé automatiquement **s'il est installé**
(`pip install open3d`, listé en commentaire dans `requirements.txt` — c'est
un paquet volumineux qui n'installe pas toujours proprement selon la
plateforme). **S'il n'est pas installé**, `splatconv` bascule
automatiquement sur une implémentation numpy/scipy pure et fonctionnellement
équivalente :

- normales : PCA vectorisée sur le voisinage k-NN + orientation cohérente
  par propagation sur l'arbre couvrant minimal (algorithme de Hoppe et al.),
- sous-échantillonnage voxel : regroupement par cellule de grille + moyenne
  position/couleur.

Aucune action n'est nécessaire pour profiter de ce repli : c'est le
comportement par défaut de ce dépôt (Open3D n'est pas dans les dépendances
obligatoires).

## Limite connue : fichiers `.rcp`

Le format `.rcp` (Autodesk ReCap) est **propriétaire et non documenté
publiquement** ; il n'existe pas de parseur open-source fiable. Cet outil
**ne tente pas** de le lire directement — un fichier `.rcp` uploadé déclenche
un message explicite (UI et CLI) demandant de l'exporter d'abord depuis
ReCap Pro :

> Menu **Export → E57** (recommandé), ou PTS/XYZ/LAS.

Réimportez ensuite le fichier exporté ici.

## Usage — application web

```bash
python run.py
# puis ouvrir http://127.0.0.1:8000
```

1. Glissez-déposez le fichier source.
2. Ajustez les paramètres si besoin (valeurs par défaut ci-dessous).
3. Cliquez sur *Convertir*, suivez la barre de progression.
4. Téléchargez le `.ply` (et, si besoin plus tard pour du géoréférencement,
   le sidecar `.offset.json` associé).
5. Chargez le `.ply` dans https://superspl.at/editor.

`docker-compose up` fonctionne aussi (voir `Dockerfile` / `docker-compose.yml`).

## Usage — CLI (batch)

```bash
python cli/convert.py scan.e57 --output splat.ply --voxel-size 0.02 --scale-factor 1.5
python cli/convert.py --help   # liste complète des paramètres
```

## Paramètres et valeurs par défaut

| Paramètre | Défaut | Rôle |
|---|---|---|
| `--k-neighbors` | 10 | k pour la distance moyenne aux voisins (échelle) et l'estimation des normales |
| `--scale-factor` | 1.5 | multiplicateur appliqué à la distance moyenne aux voisins, pour combler les trous sans trop flouter |
| `--anisotropy` | 0.3 | ratio rayon-axe-normal / rayon-tangentiel — aplatit le splat en disque le long de la normale |
| `--opacity-dense` | 0.97 | opacité en zone dense |
| `--opacity-sparse` | 0.5 | opacité plancher en zone clairsemée (densité locale plus faible ⇒ opacité réduite, rendu plus doux) |
| `--voxel-size` | désactivé | taille de voxel pour sous-échantillonnage optionnel (Open3D ou repli numpy) |
| `--center-mode` | `centroid` | recentrage par centroïde ou par min de la bounding box |
| `--include-f-rest` | désactivé | écrit 45 coefficients `f_rest_*` (SH degré 1-3) à zéro, pour les lecteurs stricts qui les exigeraient — SuperSplat n'en a pas besoin |

Ces trois derniers points restaient ouverts dans le brief initial ; les
valeurs ci-dessus sont des défauts raisonnables **à ajuster sur vos scans
réels** — tout est exposé en paramètre, dans l'UI comme en CLI, plutôt que
figé dans le code.

## Format de sortie PLY

Binaire little-endian, un élément `vertex` par splat :

```
x y z  nx ny nz  f_dc_0 f_dc_1 f_dc_2  opacity  scale_0 scale_1 scale_2  rot_0 rot_1 rot_2 rot_3
```

(ordre exact de l'implémentation de référence Inria 3DGS, repris par
SuperSplat). Pas de `f_rest_*` par défaut : sans photographie il n'y a rien
de *view-dependent* à y stocker, et leur omission réduit fortement la taille
du fichier.

- `f_dc_i = (couleur_i − 0.5) / 0.28209479177387814` (constante SH degré 0)
- `opacity` stockée en espace logit : `ln(a / (1 − a))`
- `scale_i` stockée en log : `ln(rayon_i)`
- `rot_*` : quaternion normalisé `(w, x, y, z)`, alignant l'axe local le
  plus fin du splat avec la normale locale.

## Recentrage / géoréférencement

Les coordonnées suisses (MN95/LV95) valent plusieurs millions de mètres ;
stockées en `float32` telles quelles, elles provoquent du jitter visible
dans un rendu WebGL. Chaque conversion soustrait donc un offset
(centroïde ou min de bounding box) et l'enregistre dans
`<sortie>.offset.json` :

```json
{"x": 2600123.4, "y": 1200456.7, "z": 512.3, "mode": "centroid"}
```

Pour retrouver les coordonnées monde d'origine : `xyz_monde = xyz_ply + offset`.

## Gros nuages

- Sous-échantillonnage voxel optionnel (`--voxel-size`) avant génération des
  splats.
- Les fichiers `.e57` multi-scans sont traités scan par scan (limite
  naturelle de mémoire par scan plutôt que par fichier entier).
- Un garde-fou refuse de traiter, sans sous-échantillonnage, un nuage de
  plus de 30 millions de points (configurable via `SplatParams`), avec un
  message clair invitant à activer `--voxel-size`.
- Un avertissement est affiché (UI et CLI) si le nombre de splats final
  dépasse ~8 millions — limite pratique de fluidité du rendu WebGL dans
  SuperSplat sur un GPU standard.

### Limites connues sur le "streaming"

L'estimation des normales et de l'échelle locale (k plus proches voisins)
nécessite, par construction, l'ensemble du nuage en mémoire — il n'y a pas
de vrai traitement "out-of-core" ligne par ligne ici. Pour un fichier `.e57`
multi-scans, chaque scan est néanmoins lu et libéré indépendamment, ce qui
borne le pic mémoire à la taille du plus gros scan plutôt qu'à la taille du
fichier entier. Pour un unique scan de plusieurs dizaines de millions de
points, utilisez `--voxel-size` pour réduire la mémoire nécessaire avant le
calcul des normales.

## Test rapide (avant un vrai scan)

```bash
python scripts/make_test_cube.py --out test_data/test_cube.splat.ply
```

Génère un cube de points colorés (une couleur par face) et le fait passer
par le pipeline réel de conversion — pratique pour vérifier que le fichier
s'ouvre et s'affiche correctement dans SuperSplat avant de tester sur un
scan volumineux. Un exemple déjà généré est fourni dans `test_data/`.

## Tests automatisés

```bash
pip install pytest
pytest tests/
```

Couvre : ordre exact des propriétés du header PLY, mode de repli
`f_rest_*`, pipeline complet sur un nuage synthétique, recentrage, rejet
`.rcp` avec message explicite, rejet des extensions inconnues, réduction du
nombre de points par sous-échantillonnage voxel.

## Gestion des erreurs

Toutes les erreurs prévisibles (fichier corrompu, format non supporté,
`.rcp` non pris en charge, nuage vide, mémoire insuffisante) sont des
exceptions typées (`splatconv.errors`) portant un message prêt à afficher
tel quel, aussi bien côté CLI (code de sortie non nul + message sur
stderr) que côté API web (réponse HTTP 422 avec `{"detail": "..."}`,
affichée dans un bandeau d'erreur par le frontend).
