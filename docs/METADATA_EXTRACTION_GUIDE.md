# Guide d'extraction des métadonnées

## Vue d'ensemble

Ragdoc extrait automatiquement les métadonnées (auteur, date, titre) de vos fichiers Markdown lors de l'indexation. Ces métadonnées sont ensuite disponibles dans tous les outils MCP.

## Formats supportés

### 1. YAML Frontmatter (recommandé)

Le format le plus fiable consiste à utiliser un frontmatter YAML au début de votre fichier :

```markdown
---
title: "Impact of Black Carbon on Glacier Albedo"
author: "Warren et al."
date: "1982-03-15"
---

# Introduction

Votre contenu ici...
```

**Champs supportés :**
- `title` : Titre du document
- `author` ou `authors` : Nom de l'auteur ou des auteurs
- `date` ou `published` : Date de publication

### 2. Extraction depuis le contenu

Si votre document n'a pas de frontmatter, Ragdoc tentera d'extraire les métadonnées depuis le début du texte :

```markdown
# Impact of Black Carbon on Glacier Albedo

Author: Warren et al.
Date: 1982

Cette étude examine l'impact du carbone noir...
```

**Patterns reconnus :**
- `Author:` ou `Auteur:` suivi du nom
- `Date:` ou `Published:` suivi de la date
- Le premier titre `#` peut être utilisé comme titre

### 3. Extraction depuis le nom de fichier

Pour la date uniquement, Ragdoc peut l'extraire du nom de fichier :

```
Warren_1982.md           → Date: 1982
2024-01-15_article.md    → Date: 2024-01-15
2009_RSE_Painter.md      → Date: 2009
```

## Formats de date supportés

Ragdoc normalise automatiquement les dates au format ISO (YYYY-MM-DD ou YYYY) :

- **ISO standard** : `2024-01-15`, `2024/01/15`
- **Format européen** : `15-01-2024`, `15/01/2024`
- **Format texte** : `January 15, 2024`, `Jan 15, 2024`
- **Année seule** : `1982`
- **Mois français** : `15 janvier 2024`

## Utilisation avec les outils MCP

### 1. Recherche hybride

Les métadonnées sont automatiquement affichées dans les résultats de recherche :

```python
semantic_search_hybrid("glacier albedo")
```

**Résultat :**
```
[1] Rerank Score: 0.9234 | Hybrid: 0.8567
    Source: Warren_1982.md
    Title: Impact of Black Carbon on Glacier Albedo
    Author: Warren et al.
    Date: 1982
    Position: chunk 42/120
    Rankings: BM25 #3, Semantic #1
```

### 2. Liste des documents

```python
list_documents()
```

**Résultat :**
```
[1] Warren_1982.md
    Title: Impact of Black Carbon on Glacier Albedo
    Author: Warren et al.
    Date: 1982
    Chunks: 120
```

### 3. Contenu de document

```python
get_document_content("Warren_1982.md", format="markdown")
```

**Résultat :**
```markdown
# Impact of Black Carbon on Glacier Albedo

**Author:** Warren et al.
**Date:** 1982
**Source:** Warren_1982.md
**Total chunks:** 120
...
```

## Exemple complet

### Créer un fichier avec métadonnées

Créez `/articles_markdown/example_paper.md` :

```markdown
---
title: "Glacier Dynamics in a Warming Climate"
author: "Dr. Jane Smith and Dr. John Doe"
date: "2024-01-15"
---

# Introduction

This study examines the impact of climate change on glacier dynamics...

## Methodology

We used remote sensing data from satellites...
```

### Indexer le document

```bash
python scripts/index_contextualized_incremental.py
```

**Sortie attendue :**
```
[NEW] example_paper.md
      [META] Author: Dr. Jane Smith and Dr. John Doe, Date: 2024-01-15
   example_paper.md: 45,231 chars (~11,307 tokens)
      [->] Strategie: CONTEXTUALIZED
      [OK] 30 chunks, 30 embeddings
```

### Rechercher le document

Via Claude Desktop ou l'API MCP :
```
Search for information about glacier dynamics
```

Les résultats afficheront automatiquement l'auteur et la date !

## Réindexation

Si vous modifiez les métadonnées d'un document existant, réindexez-le :

```bash
# Réindexer uniquement les documents modifiés
python scripts/index_contextualized_incremental.py

# Forcer la réindexation de tous les documents
python scripts/index_contextualized_incremental.py --force
```

## Limitations et conseils

### Limitations

1. **Extraction automatique** : La précision dépend de la structure du document
2. **Métadonnées multiples** : Si plusieurs auteurs ou dates sont trouvés, seul le premier est conservé
3. **Longueur maximale** : Les champs sont limités à 500 caractères

### Conseils

1. **Utilisez YAML frontmatter** pour une fiabilité maximale
2. **Format de date ISO** : Privilégiez `YYYY-MM-DD` pour éviter toute ambiguïté
3. **Noms d'auteurs clairs** : Évitez les formats trop complexes
4. **Vérification** : Utilisez `list_documents()` pour vérifier que les métadonnées sont correctement extraites

## Test du module d'extraction

Pour tester l'extraction de métadonnées sans indexer :

```bash
python scripts/metadata_extractor.py
```

Ce script exécute des tests sur différents formats de métadonnées.

## Migration de documents existants

Si vous avez déjà des documents indexés sans métadonnées, vous pouvez :

1. Ajouter des frontmatter YAML à vos fichiers Markdown
2. Réindexer avec `--force` :
   ```bash
   python scripts/index_contextualized_incremental.py --force
   ```

## Dépannage

### Les métadonnées ne s'affichent pas

1. Vérifiez le format du frontmatter YAML (doit commencer par `---`)
2. Assurez-vous que le document a été réindexé après ajout des métadonnées
3. Utilisez `get_document_content(source, format="chunks")` pour voir les métadonnées brutes

### Dates incorrectes

1. Utilisez le format ISO `YYYY-MM-DD` dans le frontmatter
2. Évitez les dates ambiguës comme `01/02/2024` (jour/mois ou mois/jour ?)

### Auteur non extrait

1. Assurez-vous que le champ est dans les 50 premières lignes
2. Utilisez un format clair : `Author: Nom Complet`
3. Ou ajoutez-le au frontmatter YAML

## API pour développeurs

### Utiliser le module metadata_extractor

```python
from scripts.metadata_extractor import extract_metadata

# Lire le fichier
with open('article.md', 'r') as f:
    content = f.read()

# Extraire les métadonnées
metadata = extract_metadata(content, 'article.md')

print(metadata['author'])  # "Dr. Jane Smith"
print(metadata['date'])    # "2024-01-15"
print(metadata['title'])   # "My Article Title"
```

### Métadonnées disponibles dans ChromaDB

Lors d'une requête, chaque chunk contient :

```python
{
    "source": "article.md",
    "chunk_index": 0,
    "total_chunks": 50,
    "author": "Dr. Jane Smith",  # Nouveau !
    "date": "2024-01-15",        # Nouveau !
    "title": "Article Title",    # Nouveau !
    "doc_hash": "abc123...",
    "indexed_date": "2024-11-18T...",
    "model": "voyage-context-3"
}
```

## Contribution

Pour améliorer l'extraction de métadonnées, modifiez :
- `scripts/metadata_extractor.py` : Logique d'extraction
- Ajoutez de nouveaux patterns dans `author_patterns` ou `date_patterns`
- Testez avec `python scripts/metadata_extractor.py`

---

**Besoin d'aide ?** Ouvrez une issue sur GitHub avec un exemple de votre fichier Markdown.
