#!/usr/bin/env python3
"""
Script d'indexation hybride avec nouvelle collection Chroma distincte
Pipeline: TokenChunker → SemanticChunker → OverlapRefinery → Embeddings
"""

import os
import sys
import time
import hashlib
from pathlib import Path
from typing import List, Tuple
from datetime import datetime

# Ajouter le chemin des scripts
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from indexing_config import (
    MARKDOWN_DIR, CHROMA_DB_HYBRID_PATH,
    COLLECTION_HYBRID_NAME, COLLECTION_HYBRID_METADATA,
    CHONKIE_CHUNK_SIZE, CHONKIE_CHUNK_OVERLAP, CHONKIE_TOKENIZER
)
from dotenv import load_dotenv
import chromadb
import voyageai
from chonkie import TokenChunker, SemanticChunker, OverlapRefinery

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    print("ERREUR: VOYAGE_API_KEY non trouvé")
    sys.exit(1)

def _compute_doc_hash(content: str) -> str:
    """Compute MD5 hash of document content"""
    return hashlib.md5(content.encode()).hexdigest()

class HybridModelProcessor:
    """Classe de gestion des embeddings hybrides avec stratégies adaptatives"""

    def __init__(self, api_key: str):
        self.client = voyageai.Client(api_key=api_key)
        self.CONTEXT3_LIMIT = 25  # Marge sécurité (docs Voyage indiquent ~32 chunks max)
        self.LARGE_MODEL_LIMIT = 100

    def choose_strategy(self, num_chunks: int) -> dict:
        """Choix stratégique du modèle selon le nombre de chunks"""
        if num_chunks <= self.CONTEXT3_LIMIT:
            return {
                "model": "voyage-context-3",
                "method": "contextualized",
                "batch_size": num_chunks,
                "reason": "Petit document - Context-3 optimal"
            }
        elif num_chunks <= self.LARGE_MODEL_LIMIT:
            return {
                "model": "voyage-3-large",
                "method": "standard",
                "batch_size": min(50, num_chunks),
                "reason": "Moyen document - Voyage Large standard"
            }
        else:
            return {
                "model": "voyage-3-large",
                "method": "batched",
                "batch_size": 50,
                "reason": "Gros document - Large en batches"
            }

    def process_contextualized(self, chunk_texts: List[str]) -> List[List[float]]:
        """Traitement contextualized avec Voyage Context-3"""
        try:
            result = self.client.contextualized_embed(
                inputs=[chunk_texts],
                model="voyage-context-3",
                input_type="document"
            )
            return result.results[0].embeddings
        except Exception as e:
            print(f"      Erreur contextualized: {e}")
            return self.process_standard(chunk_texts, 25)

    def process_standard(self, chunk_texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Traitement standard avec Voyage Large"""
        all_embeddings = []
        for i in range(0, len(chunk_texts), batch_size):
            batch = chunk_texts[i:i + batch_size]
            try:
                result = self.client.embed(
                    texts=batch,
                    model="voyage-3-large"
                )
                all_embeddings.extend(result.embeddings)
                if len(batch) < batch_size:
                    print(f"         OK {len(result.embeddings)} embeddings (dernier batch)")
                else:
                    print(f"         OK {len(result.embeddings)} embeddings")
            except Exception as e:
                print(f"      Erreur batch standard: {e}")
                return []
        return all_embeddings

    def process_with_strategy(self, chunk_texts: List[str]) -> Tuple[str, List[List[float]], dict]:
        """Point d'entrée principal avec choix automatique de stratégie"""
        strategy = self.choose_strategy(len(chunk_texts))
        print(f"Strategie: {strategy['model']} ({strategy['method']}) - {strategy['reason']}")

        if strategy["method"] == "contextualized":
            embeddings = self.process_contextualized(chunk_texts)
        elif strategy["method"] == "standard":
            embeddings = self.process_standard(chunk_texts, strategy["batch_size"])
        else:
            embeddings = self.process_standard(chunk_texts, strategy["batch_size"])

        return strategy["model"], embeddings, strategy

def get_optimal_chunk_config(doc_size_estimate: int) -> Tuple[int, int, int]:
    """Configuration optimale selon taille du document"""
    if doc_size_estimate < 20000:
        return 1024, 180, 25
    elif doc_size_estimate < 80000:
        return 1536, 200, 20
    elif doc_size_estimate < 200000:
        return 2048, 250, 15
    else:
        return 2500, 300, 10

def process_hybrid_document(file_path: Path, hybrid_processor: HybridModelProcessor) -> dict:
    """Pipeline hybride complet avec Chonkie"""

    print(f"\nTraitement: {file_path.name}")
    print("-" * 50)

    # 1. Lecture et estimation
    try:
        # Essayer UTF-8 d'abord, puis latin-1 comme fallback
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
        except UnicodeDecodeError:
            with open(file_path, 'r', encoding='latin-1') as f:
                content = f.read()
            print(f"   WARNING: Utilisé encodage latin-1 pour {file_path.name}")
    except Exception as e:
        print(f"   ERREUR lecture: {e}")
        return {"error": str(e)}

    if not content.strip():
        print(f"   Document vide, skip")
        return {"error": "Document vide"}

    estimated_tokens = len(content.split()) * 1.3
    chunk_size, overlap, _ = get_optimal_chunk_config(estimated_tokens)

    print(f"   Tokens estimés: {estimated_tokens:,.0f}")
    print(f"   Configuration: chunk_size={chunk_size}, overlap={overlap}")

    # 2. Pipeline Chonkie MANUEL
    start_time = time.time()

    # Étape 1: Token Chunker (structure globale)
    print(f"   Token chunking (structure globale)...")
    token_chunker = TokenChunker(
        tokenizer="gpt2",
        chunk_size=chunk_size * 2,  # Plus gros pour structure markdown
        chunk_overlap=overlap
    )
    token_chunks = token_chunker.chunk(content)
    print(f"      OK {len(token_chunks)} chunks structure générés")

    # Étape 2: Semantic Chunker (cohérence thématique)
    print(f"   Semantic chunking (cohérence thématique)...")
    semantic_chunker = SemanticChunker(
        embedding_model="minishlab/potion-base-32M",
        threshold=0.75,
        chunk_size=chunk_size,
        min_sentences_per_chunk=2
    )

    # Traiter chaque token chunk avec semantic chunking
    all_semantic_chunks = []
    for i, token_chunk in enumerate(token_chunks):
        semantic_chunks = semantic_chunker.chunk(token_chunk.text)
        all_semantic_chunks.extend(semantic_chunks)
        if len(token_chunks) > 5 and i % 5 == 0:
            print(f"      Progression: {i+1}/{len(token_chunks)} token chunks traités")

    print(f"      OK {len(all_semantic_chunks)} chunks sémantiques générés")

    # Étape 3: Overlap Refinery (contexte adjacent)
    print(f"   Overlap refinery (contexte adjacent)...")
    overlap_refinery = OverlapRefinery(
        context_size=overlap,
        method="suffix",
        merge=True
    )
    final_chunks = overlap_refinery.refine(all_semantic_chunks)
    print(f"      OK {len(final_chunks)} chunks finaux avec overlap")

    chunking_time = time.time() - start_time
    print(f"   Temps chunking: {chunking_time:.2f}s")

    # 3. Embeddings hybrides
    print(f"   Embeddings hybrides...")
    chunk_texts = [chunk.text for chunk in final_chunks]

    embed_start = time.time()
    model_used, embeddings, strategy = hybrid_processor.process_with_strategy(chunk_texts)
    embed_time = time.time() - embed_start

    if not embeddings:
        print(f"   ERREUR: Aucun embedding généré")
        return {"error": "Échec génération embeddings"}

    print(f"   OK {len(embeddings)} embeddings générés en {embed_time:.2f}s")

    # 4. Métadonnées enrichies
    current_hash = _compute_doc_hash(content)
    current_date = datetime.now().isoformat()

    metadatas = [{
        "filename": file_path.name,
        "source": file_path.name,  # Ajout pour compatibilité MCP
        "title": file_path.stem,  # Titre depuis le nom de fichier
        "chunk_index": i,
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "model": model_used,
        "strategy": strategy["method"],
        "reason": strategy["reason"],
        "token_count": chunk.token_count if hasattr(chunk, 'token_count') else len(text.split()),
        "pipeline": "hybrid_token_semantic_overlap",
        # Champs requis pour le suivi
        "doc_hash": current_hash,
        "indexed_date": current_date,
        "total_chunks": len(final_chunks)
    } for i, (chunk, text) in enumerate(zip(final_chunks, chunk_texts))]

    total_time = time.time() - start_time
    print(f"   Temps total: {total_time:.2f}s")

    return {
        "chunks": final_chunks,
        "embeddings": embeddings,
        "metadatas": metadatas,
        "model_used": model_used,
        "strategy": strategy,
        "stats": {
            "original_tokens": estimated_tokens,
            "final_chunks": len(final_chunks),
            "chunking_time": chunking_time,
            "embedding_time": embed_time,
            "total_time": total_time
        }
    }

def create_hybrid_collection():
    """Créer et peupler la collection hybride"""

    print("INDEXATION HYBRIDE AVEC NOUVELLE COLLECTION CHROMA")
    print("=" * 60)

    # Initialiser les composants
    print("\n[1/5] Initialisation des composants...")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    hybrid_processor = HybridModelProcessor(VOYAGE_API_KEY)

    client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_HYBRID_NAME,
        metadata=COLLECTION_HYBRID_METADATA
    )

    print("   OK Voyage AI connecté")
    print("   OK HybridModelProcessor initialisé")
    print(f"   OK Collection '{COLLECTION_HYBRID_NAME}' créée/sélectionnée")

    # Lister les documents
    print("\n[2/5] Scan des documents markdown...")
    md_files = sorted(list(MARKDOWN_DIR.glob("*.md")))
    print(f"   Documents trouvés: {len(md_files)}")

    if not md_files:
        print("   Aucun document markdown trouvé. Vérifiez MARKDOWN_DIR.")
        return

    # Vérifier les déjà indexés
    existing = collection.get()
    existing_names = set(meta.get("filename", "") for meta in existing.get("metadatas", []))

    files_to_index = [f for f in md_files if f.name not in existing_names]

    print(f"   Documents déjà indexés: {len(existing_names)}")
    print(f"   Documents à indexer: {len(files_to_index)}")

    if not files_to_index:
        print("   Tous les documents sont déjà indexés dans la collection hybride.")
        return

    # Traitement
    print(f"\n[3/5] Traitement avec pipeline hybride...")
    all_results = []
    total_chunks = 0
    successful_docs = 0

    for i, file_path in enumerate(files_to_index, 1):
        print(f"\n   Document {i}/{len(files_to_index)}: {file_path.name}")

        result = process_hybrid_document(file_path, hybrid_processor)

        if "error" not in result:
            all_results.append(result)
            total_chunks += len(result["chunks"])
            successful_docs += 1

            # Ajouter à ChromaDB
            ids = [f"{file_path.stem}_chunk_{j:03d}" for j in range(len(result["chunks"]))]
            chunk_texts = [chunk.text for chunk in result["chunks"]]

            collection.add(
                ids=ids,
                embeddings=result["embeddings"],
                documents=chunk_texts,
                metadatas=result["metadatas"]
            )
            print(f"   OK Ajouté à ChromaDB: {len(result['chunks'])} chunks")

        else:
            print(f"   ERREUR: Document ignoré - {result['error']}")

    # Résumé final
    print(f"\n[4/5] Résumé de l'indexation")
    print("=" * 60)
    print(f"Documents traités: {successful_docs}/{len(files_to_index)}")
    print(f"Total chunks générés: {total_chunks}")
    print(f"Collection: '{COLLECTION_HYBRID_NAME}'")

    if all_results:
        avg_chunking_time = sum(r["stats"]["chunking_time"] for r in all_results) / len(all_results)
        avg_embedding_time = sum(r["stats"]["embedding_time"] for r in all_results) / len(all_results)
        avg_total_time = sum(r["stats"]["total_time"] for r in all_results) / len(all_results)

        print(f"Temps moyen chunking: {avg_chunking_time:.2f}s")
        print(f"Temps moyen embeddings: {avg_embedding_time:.2f}s")
        print(f"Temps moyen total: {avg_total_time:.2f}s")

        # Distribution des stratégies
        strategies = {}
        for result in all_results:
            strategy = result["strategy"]["method"]
            strategies[strategy] = strategies.get(strategy, 0) + 1

        print(f"Stratégies utilisées: {strategies}")

    # Validation finale
    print(f"\n[5/5] Validation de la collection")
    collection_count = collection.count()
    print(f"Chunks dans collection hybride: {collection_count}")

    # Comparaison avec collection originale
    try:
        original_collection = client.get_collection(name="zotero_research_context_v2")
        original_count = original_collection.count()
        print(f"Chunks dans collection originale: {original_count}")

        if original_count > 0:
            increase_ratio = (collection_count / original_count - 1) * 100
            print(f"Augmentation granularité: +{increase_ratio:.1f}% (plus de chunks = plus sémantique)")
    except Exception:
        print("Collection originale non trouvée pour comparaison")

    print(f"\nIndexation hybride terminée avec succès !")
    print(f"Nouvelle collection disponible: '{COLLECTION_HYBRID_NAME}'")

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Indexation hybride avec pipeline Chonkie")
    parser.add_argument("--force", action="store_true", help="Supprimer et recréer la collection")
    args = parser.parse_args()

    if args.force:
        print("MODE FORCE: Suppression de la collection existante...")
        client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
        try:
            client.delete_collection(name=COLLECTION_HYBRID_NAME)
            print(f"   Collection '{COLLECTION_HYBRID_NAME}' supprimée")
        except Exception as e:
            print(f"   Collection non trouvée ou déjà supprimée: {e}")

    create_hybrid_collection()