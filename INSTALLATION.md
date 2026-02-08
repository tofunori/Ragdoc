# 🚀 Installation du Hybrid Search - Guide Simple

## 3 options selon votre situation

### Option 1 : Installation automatique (RECOMMANDÉ) ⚡

**Si vous avez déjà une base ChromaDB avec des documents indexés :**

```bash
# Tout faire automatiquement
./install_hybrid_search.sh
```

Ça fait tout :
- ✅ Installe rank-bm25
- ✅ Teste le hybrid search
- ✅ Active en production
- ✅ Crée un backup de votre ancien serveur

---

### Option 2 : Installation manuelle (étape par étape) 📋

**Pour plus de contrôle :**

```bash
# 1. Vérifier les prérequis
python check_prerequisites.py

# 2. Installer rank-bm25 (si manquant)
pip install rank-bm25>=0.2.2

# 3. Tester le hybrid search
python quick_test_hybrid.py

# 4. Si le test passe, activer
python activate_hybrid_search.py

# 5. Redémarrer votre serveur MCP / Claude Desktop
```

---

### Option 3 : Première installation (pas de ChromaDB encore) 🆕

**Si vous n'avez pas encore de base ChromaDB :**

```bash
# 1. Vérifier ce qui manque
python check_prerequisites.py

# 2. Si pas de documents indexés, indexer d'abord
python index_hybrid_collection.py

# 3. Puis installer le hybrid search
./install_hybrid_search.sh
```

---

## Vérification rapide

Pour savoir quelle option choisir :

```bash
python check_prerequisites.py
```

Ce script vous dit exactement ce qui manque et quoi faire.

---

## Après installation

### 1. Redémarrer le serveur

**Si vous utilisez Claude Desktop :**
- Quitter Claude Desktop complètement (CMD+Q / Alt+F4)
- Relancer Claude Desktop

**Si vous lancez le serveur manuellement :**
```bash
# Arrêter l'ancien (Ctrl+C)
# Relancer
python src/server.py
```

### 2. Tester

Dans Claude Desktop :
```
Search for articles about black carbon on glaciers
```

Vous devriez voir dans les résultats :
```
[1] Rerank Score: 0.9234 | Hybrid: 0.7821
    Rankings: BM25 #3, Semantic #5    ← C'est le nouveau !
```

---

## En cas de problème

### Le test échoue

```bash
# Voir les détails du problème
python check_prerequisites.py

# Problèmes courants:
# - API key manquante → Ajouter dans .env
# - ChromaDB vide → Indexer vos documents d'abord
# - rank-bm25 manquant → pip install rank-bm25
```

### Revenir en arrière

```bash
# Restaurer l'ancien serveur (le chemin exact est donné lors de l'activation)
cp src/backups/server_backup_YYYYMMDD_HHMMSS.py src/server.py

# Redémarrer
```

---

## Fichiers créés

- ✅ `src/hybrid_retriever.py` - Logique hybrid search
- ✅ `src/server_hybrid.py` - Nouveau serveur MCP
- ✅ `src/server.py` - **Sera remplacé** lors de l'activation
- ✅ `test_hybrid_search.py` - Tests comparatifs
- ✅ `check_prerequisites.py` - Vérification des prérequis
- ✅ `quick_test_hybrid.py` - Test rapide
- ✅ `activate_hybrid_search.py` - Script d'activation
- ✅ `install_hybrid_search.sh` - Installation automatique

---

## Documentation détaillée

Pour plus de détails : **HYBRID_SEARCH_GUIDE.md**

---

## Support

En cas de problème, vérifiez :
1. Les logs d'erreur
2. Que ChromaDB a bien des documents indexés
3. Que les API keys sont configurées

Les backups sont automatiquement créés dans `src/backups/`
