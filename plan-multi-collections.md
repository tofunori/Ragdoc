# Plan: Support Multi-Collections pour ragdoc-mcp

## Objectif
Permettre plusieurs collections ChromaDB (ex: cryosphere, programming) avec recherche automatique dans toutes les collections et fusion des resultats via Cohere rerank.

## Decisions UX
- **Configuration**: YAML + CLI (database.yaml + commandes CLI)
- **Indexation**: Mode interactif (le CLI demande la collection cible)
- **Recherche**: Toutes collections par defaut, parametre optionnel pour filtrer

---

## Fichiers a modifier

### 1. `src/collection_manager.py` (NOUVEAU)
Gestionnaire central des collections avec lazy-loading.

```python
class CollectionManager:
    def __init__(self, chroma_client, embedding_function):
        self._chroma_client = chroma_client
        self._embedding_function = embedding_function
        self._retrievers: Dict[str, HybridRetriever] = {}

    def get_retriever(self, collection_name: str) -> HybridRetriever:
        """Lazy-load un retriever pour une collection."""
        if collection_name not in self._retrievers:
            collection = self._chroma_client.get_collection(name=collection_name)
            self._retrievers[collection_name] = HybridRetriever(
                collection=collection,
                embedding_function=self._embedding_function
            )
        return self._retrievers[collection_name]

    def list_collections(self) -> List[str]:
        """Liste toutes les collections disponibles."""
        return [c.name for c in self._chroma_client.list_collections()]

    def search_all(self, query: str, top_k: int, alpha: float,
                   collections: List[str] = None) -> List[Dict]:
        """Recherche dans toutes les collections actives."""
        collections = collections or ACTIVE_COLLECTIONS
        all_results = []

        for coll_name in collections:
            retriever = self.get_retriever(coll_name)
            results = retriever.search(query=query, top_k=50, alpha=alpha)
            for r in results:
                r['collection'] = coll_name  # Ajouter source
            all_results.extend(results)

        return all_results
```

### 2. `src/config.py`
Ajouter constantes multi-collections:

```python
# Multi-collections support
MULTI_COLLECTIONS_ENABLED = True
ACTIVE_COLLECTIONS: List[str] = []  # Charge depuis YAML
DEFAULT_COLLECTION = COLLECTION_NAME  # Backward compat
```

### 3. `config/database.yaml`
Nouvelle section:

```yaml
multi_collections:
  enabled: true
  default: "ragdoc_contextualized_v1"
  collections:
    - name: "ragdoc_contextualized_v1"
      description: "Cryosphere & glaciology papers"
      active: true
    - name: "programming_docs"
      description: "Programming documentation"
      active: true
```

### 4. `src/server.py`

**Modifications:**
- Remplacer `hybrid_retriever` global par `collection_manager`
- Modifier `init_clients()` pour creer CollectionManager
- Modifier `_perform_search_hybrid()`:
  1. Chercher dans toutes collections actives
  2. Agreger les resultats
  3. Cohere rerank sur le pool combine
  4. Afficher collection source dans output
- Ajouter parametre `collections: list = None` aux tools MCP
- Nouveau tool: `list_collections()`

**Exemple de signature modifiee:**
```python
@mcp.tool()
def semantic_search_hybrid(
    query: str,
    top_k: int = 10,
    alpha: float = 0.5,
    collections: list = None  # NOUVEAU - None = toutes
) -> str:
```

### 5. `scripts/index_incremental.py`

**Modifications:**
- Ajouter `--collection` flag optionnel
- Si pas specifie: mode interactif (demander via input())

```python
def main():
    parser.add_argument('--collection', default=None,
                        help="Collection cible (interactif si omis)")
    args = parser.parse_args()

    if args.collection is None:
        # Lister collections disponibles
        collections = list_available_collections()
        print("Collections disponibles:")
        for i, c in enumerate(collections, 1):
            print(f"  {i}. {c}")
        print(f"  {len(collections)+1}. [Creer nouvelle collection]")

        choice = input("Choisir collection: ")
        # ... handle choice ...
```

### 6. `ragdoc-cli.py`

**Nouvelles commandes:**
```bash
ragdoc collection list              # Lister collections
ragdoc collection create <name>     # Creer collection vide
ragdoc collection delete <name>     # Supprimer collection
ragdoc collection stats [name]      # Stats (toutes ou une)
```

**Modifier `index`:**
```bash
ragdoc index                        # Interactif: demande collection
ragdoc index --collection=<name>    # Specifie collection directement
```

---

## Ordre d'implementation

### Phase 1: Infrastructure (1-2h)
1. [ ] Modifier `config/database.yaml` - ajouter section multi_collections
2. [ ] Modifier `src/config.py` - charger config multi-collections
3. [ ] Creer `src/collection_manager.py`

### Phase 2: Serveur MCP (2-3h)
4. [ ] Modifier `src/server.py`:
   - [ ] Integrer CollectionManager
   - [ ] Modifier `_perform_search_hybrid()` pour multi-search
   - [ ] Ajouter param `collections` aux 6 tools existants
   - [ ] Ajouter tool `list_collections()`

### Phase 3: CLI & Indexation (1-2h)
5. [ ] Modifier `scripts/index_incremental.py` - support `--collection` + interactif
6. [ ] Modifier `ragdoc-cli.py` - commandes collection + mode interactif

### Phase 4: Tests (1h)
7. [ ] Tester backward compatibility (une seule collection)
8. [ ] Tester recherche multi-collections
9. [ ] Tester CLI interactif

---

## Flow de recherche multi-collections

```
Query: "albedo feedback"
         |
         v
+--------+--------+
|                 |
v                 v
[cryosphere]   [programming]
     |              |
     v              v
HybridRetriever  HybridRetriever
(BM25+Semantic)  (BM25+Semantic)
     |              |
     +------+-------+
            |
            v
    Agregation (~100 candidats)
            |
            v
    Cohere Rerank v3.5
            |
            v
    Top 10 avec collection source
```

---

## Output exemple

```
SEARCH RESULTS: albedo feedback
======================================================================

[1] Rerank Score: 0.9234 | Collection: cryosphere
    Source: 2009_RSE_Painter.md
    Position: chunk 12/45
    ...

[2] Rerank Score: 0.8891 | Collection: cryosphere
    Source: 1982_RGSP.md
    Position: chunk 3/28
    ...

[3] Rerank Score: 0.4521 | Collection: programming
    Source: python_optics.md
    Position: chunk 7/15
    ...
```

---

## Backward Compatibility

- Si une seule collection configuree: comportement identique a avant
- `COLLECTION_NAME` reste disponible comme alias du default
- Tous les tools MCP gardent leurs signatures (param `collections` optionnel avec default None)
- Aucun changement requis pour les utilisateurs existants

---

## Notes techniques

### Lazy-loading des retrievers
- Les HybridRetrievers sont crees a la demande (pas au demarrage)
- L'index BM25 est construit lors de la premiere requete sur chaque collection
- Avantage: demarrage rapide meme avec 10+ collections

### Limite de candidats
- 50 candidats max par collection avant rerank
- Cohere rerank sur pool combine (~100-150 candidats)
- Top 10 final retourne a l'utilisateur

### Gestion des erreurs
- Si une collection est inaccessible, log warning et continuer avec les autres
- Si toutes les collections echouent, retourner erreur explicite
