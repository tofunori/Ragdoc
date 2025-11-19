#!/usr/bin/env python3
"""
Test du pipeline hybride Chonkie avec syntaxe correcte
Basé sur l'architecture de l'utilisateur, mais avec vrais composants Chonkie
"""

import os
import sys
import time
from pathlib import Path
from typing import List, Tuple

# Ajouter le chemin des scripts
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from indexing_config import MARKDOWN_DIR
from dotenv import load_dotenv
import chromadb
import voyageai
from chonkie import TokenChunker, SemanticChunker, OverlapRefinery
import voyageai

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    print("ERREUR: VOYAGE_API_KEY non trouvé")
    sys.exit(1)

class HybridModelProcessor:
    """Classe de gestion des embeddings hybrides (adaptée du code utilisateur)"""

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

    def process_batched(self, chunk_texts: List[str], batch_size: int = 50) -> List[List[float]]:
        """Traitement batched pour très gros documents"""
        print(f"      ATTENTION: Gros document : traitement en batches de {batch_size}")
        return self.process_standard(chunk_texts, batch_size)

    def process_with_strategy(self, chunk_texts: List[str]) -> Tuple[str, List[List[float]], dict]:
        """Point d'entrée principal avec choix automatique de stratégie"""
        strategy = self.choose_strategy(len(chunk_texts))
        print(f"Strategie: {strategy['model']} ({strategy['method']}) - {strategy['reason']}")

        if strategy["method"] == "contextualized":
            embeddings = self.process_contextualized(chunk_texts)
        elif strategy["method"] == "standard":
            embeddings = self.process_standard(chunk_texts, strategy["batch_size"])
        else:
            embeddings = self.process_batched(chunk_texts, strategy["batch_size"])

        return strategy["model"], embeddings, strategy

def get_optimal_chunk_config(doc_size_estimate: int) -> Tuple[int, int, int]:
    """Configuration optimale selon taille du document (code utilisateur)"""
    if doc_size_estimate < 20000:
        return 1024, 180, 25
    elif doc_size_estimate < 80000:
        return 1536, 200, 20
    elif doc_size_estimate < 200000:
        return 2048, 250, 15
    else:
        return 2500, 300, 10

def process_hybrid_document(file_path: Path, hybrid_processor: HybridModelProcessor) -> dict:
    """Pipeline hybride avec syntaxe Chonkie correcte"""

    print(f"\nTraitement: {file_path.name}")
    print("-" * 50)

    # 1. Lecture et estimation
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
    except Exception as e:
        print(f"   ERREUR lecture: {e}")
        return {"error": str(e)}

    estimated_tokens = len(content.split()) * 1.3
    chunk_size, overlap, _ = get_optimal_chunk_config(estimated_tokens)

    print(f"   Tokens estimés: {estimated_tokens:,.0f}")
    print(f"   Configuration: chunk_size={chunk_size}, overlap={overlap}")

    # 2. Pipeline Chonkie MANUEL (syntaxe correcte)
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

    # 4. Embeddings hybrides
    print(f"   Embeddings hybrides...")
    chunk_texts = [chunk.text for chunk in final_chunks]

    embed_start = time.time()
    model_used, embeddings, strategy = hybrid_processor.process_with_strategy(chunk_texts)
    embed_time = time.time() - embed_start

    if not embeddings:
        print(f"   ERREUR: Aucun embedding généré")
        return {"error": "Échec génération embeddings"}

    print(f"   OK {len(embeddings)} embeddings générés en {embed_time:.2f}s")

    # 5. Métadonnées enrichies
    metadatas = [{
        "filename": file_path.name,
        "chunk_index": i,
        "text_preview": text[:200] + "..." if len(text) > 200 else text,
        "model": model_used,
        "strategy": strategy["method"],
        "reason": strategy["reason"],
        "token_count": chunk.token_count if hasattr(chunk, 'token_count') else len(text.split()),
        "pipeline": "hybrid_token_semantic_overlap"
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

def test_hybrid_pipeline():
    """Test du pipeline hybride sur quelques documents"""

    print("TEST PIPELINE HYBRIDE CHONKIE")
    print("=" * 60)

    # Initialiser les composants
    print("\n[1/4] Initialisation des composants...")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    hybrid_processor = HybridModelProcessor(VOYAGE_API_KEY)
    print("   OK Voyage AI connecté")
    print("   OK HybridModelProcessor initialisé")

    # Sélectionner quelques documents de test
    print("\n[2/4] Sélection des documents de test...")
    md_files = sorted(list(MARKDOWN_DIR.glob("*.md")))

    # Choisir 3 documents de tailles différentes
    test_files = []
    if len(md_files) >= 3:
        # Prendre le premier, un milieu, et un gros
        test_files = [md_files[0], md_files[len(md_files)//2], md_files[-1]]
    else:
        test_files = md_files[:3]

    print(f"   Documents sélectionnés: {len(test_files)}")
    for i, f in enumerate(test_files, 1):
        print(f"      {i}. {f.name}")

    # Traiter chaque document
    print("\n[3/4] Traitement avec pipeline hybride...")
    all_results = []
    total_chunks = 0

    for file_path in test_files:
        result = process_hybrid_document(file_path, hybrid_processor)
        if "error" not in result:
            all_results.append(result)
            total_chunks += len(result["chunks"])
        else:
            print(f"   ERREUR: Document {file_path.name} ignoré: {result['error']}")

    # Résumé
    print("\n[4/4] Résumé des résultats")
    print("=" * 60)
    print(f"Documents traités: {len(all_results)}/{len(test_files)}")
    print(f"Total chunks générés: {total_chunks}")

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

    print("\nTest du pipeline hybride terminé!")
    return all_results

if __name__ == "__main__":
    test_hybrid_pipeline()