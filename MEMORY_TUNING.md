# RAGDOC MCP – Réduire l’usage RAM

Sur de gros corpus, la partie **BM25** peut consommer plusieurs Go (construction d’un index lexical en RAM).
Si tu utilises surtout la recherche sémantique (Voyage) + rerank, tu peux désactiver/retarder BM25 pour garder le MCP léger.

## Options (via variables d’environnement)

- `RAGDOC_LIGHT_MODE=true`  
  Désactive BM25 (recherche sémantique uniquement) et évite la construction BM25 en arrière‑plan.

- `RAGDOC_BM25_ENABLED=false`  
  Désactive BM25 explicitement.

- `RAGDOC_BM25_AUTO_BUILD=false`  
  N’essaie plus de construire l’index BM25 en arrière‑plan. (BM25 ne se construit alors que si tu forces `alpha=0.0`.)

- `RAGDOC_BM25_KEEP_DOCS=false`  
  Ne garde pas le texte complet des chunks en RAM côté BM25 (réduit la RAM, au prix de certains filtres BM25).

## Exemple (config MCP / Claude)

Dans ton config MCP (ex: `.claude.json` / `mcpServers`), ajoute dans `env` :

```json
{
  "env": {
    "RAGDOC_LIGHT_MODE": "true"
  }
}
```

Ensuite **redémarre** l’app qui lance le MCP (Claude/Cursor/etc.) pour que ça prenne effet.
