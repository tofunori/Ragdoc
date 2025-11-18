# Nouvelle fonctionnalité : Extraction automatique des métadonnées

## Résumé

Ragdoc peut maintenant extraire automatiquement **l'auteur**, **la date** et **le titre** de vos fichiers Markdown lors de l'indexation. Ces métadonnées sont affichées dans tous les outils MCP (recherche, liste de documents, contenu de document).

## Modifications apportées

### 1. Nouveau module : `scripts/metadata_extractor.py`

Module complet d'extraction de métadonnées supportant :
- **YAML frontmatter** (format recommandé)
- **Extraction depuis le contenu** (premières lignes du texte)
- **Extraction depuis le nom de fichier** (dates uniquement)
- **Normalisation automatique des dates** au format ISO

**Formats de date supportés :**
- ISO : `2024-01-15`, `2024/01/15`
- Européen : `15-01-2024`, `15/01/2024`
- Texte : `January 15, 2024`, `15 janvier 2024`
- Année seule : `1982`

### 2. Scripts d'indexation modifiés

**Fichiers modifiés :**
- `scripts/index_contextualized_adaptive.py`
- `scripts/index_contextualized_incremental.py`

**Changements :**
- Import du module `metadata_extractor`
- Extraction des métadonnées avant traitement de chaque document
- Ajout des champs `author`, `date`, `title` aux métadonnées de chaque chunk
- Affichage des métadonnées pendant l'indexation

### 3. Serveur MCP mis à jour : `src/server.py`

**Outils modifiés :**

#### `semantic_search_hybrid()` et `search_by_source()`
Affichent maintenant :
```
[1] Rerank Score: 0.9234 | Hybrid: 0.8567
    Source: Warren_1982.md
    Title: Effect of Black Carbon on Glacier Albedo
    Author: Warren, S. G., & Wiscombe, W. J.
    Date: 1982
    Position: chunk 42/120
    Rankings: BM25 #3, Semantic #1
```

#### `list_documents()`
Affiche maintenant :
```
[1] Warren_1982.md
    Title: Effect of Black Carbon on Glacier Albedo
    Author: Warren, S. G., & Wiscombe, W. J.
    Date: 1982
    Chunks: 120
```

#### `get_document_content()`
Formats `markdown` et `chunks` incluent désormais :
```markdown
# Effect of Black Carbon on Glacier Albedo

**Author:** Warren, S. G., & Wiscombe, W. J.
**Date:** 1982
**Source:** Warren_1982.md
...
```

### 4. Documentation

**Nouveau fichier : `docs/METADATA_EXTRACTION_GUIDE.md`**
- Guide complet d'utilisation
- Exemples de formats supportés
- Instructions de réindexation
- API pour développeurs
- Dépannage

### 5. Exemples de fichiers

Deux exemples créés dans `articles_markdown/` :

1. **`example_glacier_study.md`** : Avec YAML frontmatter
2. **`Warren_1982.md`** : Extraction depuis le contenu

## Utilisation

### 1. Ajouter des métadonnées à vos fichiers

**Option A : YAML frontmatter (recommandé)**
```markdown
---
title: "Titre de l'article"
author: "Nom de l'auteur"
date: "2024-01-15"
---

# Contenu...
```

**Option B : Dans le contenu**
```markdown
# Titre de l'article

Author: Nom de l'auteur
Date: 2024-01-15

Contenu...
```

### 2. Indexer ou réindexer

```bash
# Indexation incrémentale (nouveaux et modifiés seulement)
python scripts/index_contextualized_incremental.py

# Forcer réindexation de tous les documents
python scripts/index_contextualized_incremental.py --force
```

### 3. Utiliser via Claude Desktop

Les métadonnées apparaissent automatiquement dans :
- Résultats de recherche
- Liste de documents
- Contenu de document

Aucune modification nécessaire dans vos requêtes !

## Compatibilité

### Rétrocompatibilité

- ✅ **Documents sans métadonnées** : Fonctionnent normalement, métadonnées affichées comme `N/A`
- ✅ **Base de données existante** : Aucune migration requise
- ✅ **Anciens scripts** : Continuent de fonctionner (métadonnées optionnelles)

### Migration

Pour ajouter des métadonnées à des documents déjà indexés :

1. Ajoutez frontmatter YAML à vos fichiers `.md`
2. Réindexez : `python scripts/index_contextualized_incremental.py --force`
3. Les nouvelles métadonnées remplacent les anciennes

## Structure de données

### Métadonnées dans ChromaDB

Chaque chunk contient maintenant :
```python
{
    "source": "article.md",
    "chunk_index": 0,
    "total_chunks": 50,
    "author": "Dr. Jane Smith",        # NOUVEAU
    "date": "2024-01-15",              # NOUVEAU
    "title": "Article Title",          # NOUVEAU
    "doc_hash": "abc123...",
    "indexed_date": "2024-11-18T...",
    "model": "voyage-context-3",
    "embedding_strategy": "contextualized_full"
}
```

## Tests

### Test du module d'extraction

```bash
python scripts/metadata_extractor.py
```

Résultat attendu :
```
Testing MetadataExtractor...

Test 1 (YAML frontmatter):
  Author: Dr. Jane Smith
  Date: 2024-01-15
  Title: Glacier Dynamics Study

Test 2 (Text extraction):
  Author: Warren et al.
  Date: 1982
  Title: Impact of Black Carbon on Glacier Albedo

✓ Tests completed!
```

### Test d'indexation

```bash
# Créer le dossier si nécessaire
mkdir -p articles_markdown

# Copier un exemple
cp docs/examples/Warren_1982.md articles_markdown/

# Indexer
python scripts/index_contextualized_incremental.py
```

Résultat attendu :
```
[NEW] Warren_1982.md
      [META] Author: Warren, S. G., & Wiscombe, W. J., Date: 1982
   Warren_1982.md: 15,234 chars (~3,808 tokens)
      [->] Strategie: CONTEXTUALIZED
      [OK] 10 chunks, 10 embeddings
```

## Avantages

1. **Meilleure organisation** : Facilite la recherche et le tri des documents
2. **Citation facilitée** : Auteur et date disponibles immédiatement
3. **Traçabilité** : Savoir d'où viennent les informations
4. **Compatibilité académique** : Support des standards de citation
5. **Automatisation** : Extraction automatique, pas de saisie manuelle

## Limitations

1. **Précision variable** : L'extraction automatique dépend de la structure du document
2. **YAML recommandé** : Pour une fiabilité maximale, utiliser frontmatter YAML
3. **Longueur maximale** : Champs limités à 500 caractères
4. **Auteurs multiples** : Format libre, pas de parsing structuré

## Prochaines étapes possibles

Améliorations futures potentielles :
- [ ] Support des DOI et URLs
- [ ] Extraction de mots-clés
- [ ] Support de BibTeX
- [ ] Filtrage par auteur ou date dans les recherches
- [ ] Export de citations formatées
- [ ] Analyse de co-auteurs

## Support

- **Documentation complète** : `docs/METADATA_EXTRACTION_GUIDE.md`
- **Exemples** : `articles_markdown/example_glacier_study.md`, `Warren_1982.md`
- **Code source** : `scripts/metadata_extractor.py`

---

**Version** : 1.0
**Date** : 2024-11-18
**Auteur** : Claude AI
**Statut** : ✅ Production Ready
