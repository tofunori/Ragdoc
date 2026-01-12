# RAGDOC - Gestionnaire d'Indexation Chroma DB

Interface complète et moderne pour gérer l'indexation des articles de recherche glaciaire.

## Installation ✅

Les fichiers suivants ont été installés dans `C:\Users\thier\bin\`:
- `ragdoc.bat` - Launcher principal
- `ragdoc-tui.bat` - TUI Textual uniquement
- Le chemin est déjà dans votre PATH Windows

## Utilisation

### Mode TUI (Terminal User Interface) - **Recommandé**

```bash
ragdoc
```

Ou explicitement :
```bash
ragdoc-tui
```

**Contrôles** :
- `↑ ↓` : Naviguer dans le menu
- `Enter` : Exécuter l'action sélectionnée
- `Q` : Quitter
- `H` : Afficher l'aide

**Interface** :
```
┌──────────────────────────────────────────────────────────────────┐
│          RAGDOC - Indexation Manager                      [Time] │
├──────────────────────┬──────────────────────────────────────────┤
│ ▶ Afficher Statut    │  Statistiques d'indexation                │
│   Indexer (+)        │                                            │
│   Réindexer (--f)    │  📄 Articles          110                  │
│   Nettoyer           │  📦 Chunks            10,281               │
│   Monitoring         │  📊 Moyenne           93.5                │
│   Reset BD           │                                            │
│   Delete BD          │  Hash MD5  ✓ 110/110                      │
│   Fix HNSW           │  Dates     ✓ 110/110                      │
│   Quitter            │                                            │
├──────────────────────┼──────────────────────────────────────────┤
│ Logs                                                              │
│ ✓ Indexation complétée                                           │
│ ℹ Prêt pour les commandes...                                    │
├──────────────────────┴──────────────────────────────────────────┤
│ ↑↓ Navigation | Enter Exécuter | Q Quitter | H Aide             │
└──────────────────────────────────────────────────────────────────┘
```

### Mode CLI - Interface en ligne de commande

Utiliser avec des arguments pour automatisation/scripts :

```bash
# Afficher statistiques
ragdoc status

# Indexation incrémentale
ragdoc index

# Réindexation complète
ragdoc index --force

# Nettoyer documents supprimés
ragdoc index --delete-missing

# Monitoring continu
ragdoc monitor

# Réinitialiser collections
ragdoc reset

# Supprimer complètement chroma_db_fresh/
ragdoc delete

# Corriger corruption HNSW
ragdoc fix-hnsw

# Menu interactif CLI
ragdoc
```

## Fonctionnalités

### Statistiques Détaillées
- Nombre d'articles indexés
- Nombre total de chunks
- Moyenne chunks/article
- Vérification des métadonnées (MD5, dates)
- Répartition par modèle d'embedding
- Fichiers sources disponibles

### Opérations d'Indexation
- **Indexation incrémentale** : Ajoute les nouveaux fichiers, ignore les inchangés
- **Réindexation complète** : Force la réindexation de tous les documents
- **Nettoyage** : Supprime les chunks des documents qui n'existent plus
- **Monitoring** : Surveillance continue avec mise à jour toutes les 10s

### Gestion de Base de Données
- **Reset** : Vide les collections (garde les fichiers)
- **Delete** : Suppression physique de `chroma_db_fresh/`
- **Fix HNSW** : Correctiondes problèmes d'index HNSW

## Architecture

```
D:\Claude Code\ragdoc-mcp\
├── ragdoc.bat                 # Launcher (installé dans bin)
├── ragdoc-cli.py              # Interface CLI (colorama)
├── ragdoc-tui.py              # Interface TUI (Textual + Rich)
├── ragdoc-tui.bat             # Launcher TUI direct
├── scripts/
│   ├── index_incremental.py   # Core indexation
│   ├── indexing_config.py     # Configuration
│   └── ...
├── articles_markdown/         # Fichiers sources (.md)
└── chroma_db_fresh/           # Base de données Chroma
```

## Commandes Rapides

| Commande | Fonction |
|----------|----------|
| `ragdoc` | Lancer la TUI |
| `ragdoc status` | Voir statistiques |
| `ragdoc index` | Indexer nouveaux fichiers |
| `ragdoc index --force` | Réindexer tout |
| `ragdoc monitor` | Surveillance continue |

## Configuration

Les paramètres d'indexation sont dans `scripts/indexing_config.py`:
- Taille chunks : 2000 chars (400 overlap)
- Modèle par défaut : `voyage-context-3`
- Modèle gros documents : `voyage-3-large` (>50K chars)
- Collection Chroma : `zotero_research_context`
- Base de données : `chroma_db_fresh/`

## Dépendances

```bash
pip install chromadb colorama textual rich
```

### Variables d'environnement requises (`.env`)
```bash
VOYAGE_API_KEY=votre_clé
COHERE_API_KEY=votre_clé
```

## Dépannage

### La TUI ne se lance pas
```bash
# Vérifier les dépendances
pip install --upgrade textual rich

# Lancer en CLI
ragdoc status
```

### Corruption HNSW
```bash
ragdoc fix-hnsw
```

### Base de données corrompue
```bash
# Supprimer et recréer
ragdoc delete
ragdoc index
```

## Support

Pour plus d'informations sur les commandes spécifiques :
```bash
ragdoc -h        # Aide CLI
ragdoc           # Menu TUI avec descriptions
```
