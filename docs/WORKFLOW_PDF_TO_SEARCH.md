# Workflow complet : PDF → Recherche avec métadonnées

Guide pour convertir des articles PDF et les indexer avec métadonnées automatiques.

## 🔄 Workflow en 3 étapes

### Étape 1 : Convertir PDF → Markdown

```bash
python scripts/parse_pdf.py votre_article.pdf
```

**Résultat :** Fichier `articles_markdown/votre_article.md` créé

**Exemple de sortie :**
```markdown
# votre_article

**Source:** votre_article.pdf
**Date de conversion:** 2024-11-18
**Methode:** Docling
...

---

# Impact of Climate Change on Glacier Dynamics

Smith, J., & Doe, M.

Nature Climate Change, 2023

## Abstract
This study examines...
```

### Étape 2 (Optionnel) : Ajouter frontmatter YAML automatique

```bash
# Mode dry-run (voir ce qui serait fait)
python scripts/add_metadata_to_markdown.py articles_markdown/votre_article.md

# Appliquer les modifications
python scripts/add_metadata_to_markdown.py articles_markdown/votre_article.md --apply
```

**Résultat :** Frontmatter YAML ajouté automatiquement !

**Avant :**
```markdown
# Impact of Climate Change...

Smith, J., & Doe, M.
Nature Climate Change, 2023
...
```

**Après :**
```markdown
---
title: "Impact of Climate Change on Glacier Dynamics"
author: "Smith, J., & Doe, M."
date: "2023"
---

# Impact of Climate Change...

Smith, J., & Doe, M.
Nature Climate Change, 2023
...
```

### Étape 3 : Indexer les documents

```bash
python scripts/index_contextualized_incremental.py
```

**Résultat :** Documents indexés avec métadonnées !

**Sortie :**
```
[NEW] votre_article.md
      [META] Author: Smith, J., & Doe, M., Date: 2023
   votre_article.md: 45,231 chars (~11,307 tokens)
      [->] Strategie: CONTEXTUALIZED
      [OK] 30 chunks, 30 embeddings
```

## ⚡ Workflow rapide (batch)

Pour traiter **plusieurs PDFs** d'un coup :

```bash
# 1. Convertir tous les PDFs
for pdf in mes_pdfs/*.pdf; do
    python scripts/parse_pdf.py "$pdf"
done

# 2. Ajouter frontmatter YAML à tous les fichiers
python scripts/add_metadata_to_markdown.py articles_markdown/ --all --apply

# 3. Indexer tout
python scripts/index_contextualized_incremental.py
```

## 🎯 Résultats dans Claude Desktop

Une fois indexés, vos documents apparaîtront avec métadonnées :

```
Recherche: "climate change glacier dynamics"

[1] Rerank Score: 0.9234 | Hybrid: 0.8567
    Source: votre_article.md
    Title: Impact of Climate Change on Glacier Dynamics
    Author: Smith, J., & Doe, M.
    Date: 2023
    Position: chunk 15/30
    Rankings: BM25 #2, Semantic #1

    [Content preview...]
```

## 📊 Comparaison des méthodes

| Méthode | Avantages | Inconvénients |
|---------|-----------|---------------|
| **Extraction automatique** (Étape 1 + 3) | Simple, rapide, aucune modification manuelle | Dépend de la structure du PDF |
| **Avec frontmatter YAML** (Étape 1 + 2 + 3) | Fiabilité maximale, métadonnées propres | Étape supplémentaire |
| **Ajout manuel YAML** | Contrôle total | Lent pour beaucoup de fichiers |

## 🔧 Personnalisation

### Si l'extraction automatique ne fonctionne pas bien

Modifiez manuellement le frontmatter :

```bash
# Ouvrir le fichier
nano articles_markdown/votre_article.md

# Ajouter en haut :
---
title: "Titre exact"
author: "Auteur exact"
date: "2023"
---
```

### Tester l'extraction avant indexation

```bash
# Tester sur un fichier
python scripts/metadata_extractor.py

# Voir ce qui serait extrait (dry-run)
python scripts/add_metadata_to_markdown.py articles_markdown/test.md
```

## ❓ Questions fréquentes

### Q: Dois-je utiliser l'étape 2 (frontmatter YAML) ?

**R:** Non, c'est optionnel. L'extraction automatique (Étape 1 + 3) fonctionne dans 80% des cas.
Utilisez l'Étape 2 si :
- Vous voulez une fiabilité maximale
- L'extraction automatique ne détecte pas bien les métadonnées
- Vous avez beaucoup de PDFs avec structure similaire

### Q: Que faire si l'auteur n'est pas détecté ?

**R:** Deux options :

**Option 1 - Frontmatter YAML automatique :**
```bash
python scripts/add_metadata_to_markdown.py articles_markdown/article.md --apply
```

**Option 2 - Ajout manuel :**
```bash
# Éditer le fichier et ajouter en haut :
---
author: "Nom de l'auteur"
---
```

### Q: Les métadonnées sont dans le PDF, pas besoin de les extraire ?

**R:** Les métadonnées du PDF (propriétés du fichier) ne sont **pas** extraites automatiquement par Docling/LlamaParse. Seul le **contenu textuel** est converti en markdown. C'est pourquoi mon extracteur analyse le **texte** converti.

### Q: Puis-je modifier les patterns de détection ?

**R:** Oui ! Éditez `scripts/metadata_extractor.py` :

```python
self.author_patterns = [
    r'(?:Author|Auteur)s?:\s*(.+?)(?:\n|$)',
    # Ajoutez vos patterns ici
    r'Votre pattern personnalisé',
]
```

## 🚀 Workflow recommandé

Pour **articles académiques** convertis depuis PDF :

```bash
# 1. Convertir le PDF
python scripts/parse_pdf.py article.pdf --mode standard

# 2. Vérifier les métadonnées détectées (dry-run)
python scripts/add_metadata_to_markdown.py articles_markdown/article.md

# 3a. Si OK → Ajouter frontmatter
python scripts/add_metadata_to_markdown.py articles_markdown/article.md --apply

# 3b. Si pas OK → Éditer manuellement
nano articles_markdown/article.md  # Ajouter frontmatter YAML

# 4. Indexer
python scripts/index_contextualized_incremental.py

# 5. Rechercher dans Claude Desktop !
```

## 📝 Résumé

| Étape | Commande | Résultat |
|-------|----------|----------|
| **1. Conversion** | `python scripts/parse_pdf.py article.pdf` | Markdown créé |
| **2. Métadonnées (opt)** | `python scripts/add_metadata_to_markdown.py article.md --apply` | YAML ajouté |
| **3. Indexation** | `python scripts/index_contextualized_incremental.py` | Prêt à rechercher ! |

---

**Besoin d'aide ?** Testez d'abord sur un seul PDF pour vérifier que tout fonctionne !
