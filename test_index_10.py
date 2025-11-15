#!/usr/bin/env python3
"""
Script de test pour indexer seulement 10 documents avec Chonkie
"""

import os
import sys
from pathlib import Path

# Ajouter le chemin des scripts
sys.path.insert(0, str(Path(__file__).parent / "scripts"))

from indexing_config import (
    MARKDOWN_DIR, CHROMA_DB_PATH, COLLECTION_NAME, COLLECTION_METADATA,
    CHONKIE_CHUNK_SIZE, CHONKIE_CHUNK_OVERLAP, CHONKIE_TOKENIZER
)
from dotenv import load_dotenv
import chromadb
import voyageai
from chonkie import TokenChunker

load_dotenv()

VOYAGE_API_KEY = os.getenv("VOYAGE_API_KEY")
if not VOYAGE_API_KEY:
    print("ERREUR: VOYAGE_API_KEY non trouvé")
    sys.exit(1)

def index_10_documents():
    """Indexer seulement les 10 premiers documents"""

    print("Test d'indexation de 10 documents avec Chonkie + Voyage Context-3")
    print("=" * 60)

    # Initialiser les clients
    print("\n[1/4] Initialisation des clients...")
    voyage_client = voyageai.Client(api_key=VOYAGE_API_KEY)
    client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata=COLLECTION_METADATA
    )
    print("   OK Clients initialises")

    # Récupérer les 10 premiers fichiers
    print("\n[2/4] Sélection des 10 premiers documents...")
    md_files = sorted(list(MARKDOWN_DIR.glob("*.md")))
    test_files = md_files[:10]

    print(f"   Fichiers: {len(test_files)} documents trouvés:")
    for i, md_file in enumerate(test_files, 1):
        print(f"      {i}. {md_file.name}")

    # Vérifier si déjà indexés
    existing = collection.get()
    existing_names = set(existing.get("metadatas", [{}])[0].get("filename", "")
                        for existing in existing.get("metadatas", []))

    # Filtrer les documents non indexés
    files_to_index = [f for f in test_files if f.name not in existing_names]

    print(f"\n   Documents à indexer: {len(files_to_index)}")
    if len(files_to_index) == 0:
        print("   Tous les documents sont déjà indexés")
        return

    # Configuration Chonkie
    print(f"\n[3/4] Configuration Chonkie:")
    print(f"   - Chunk size: {CHONKIE_CHUNK_SIZE} tokens")
    print(f"   - Overlap: {CHONKIE_CHUNK_OVERLAP} tokens")
    print(f"   - Tokenizer: {CHONKIE_TOKENIZER}")
    print(f"   - Ratio: {CHONKIE_CHUNK_OVERLAP/CHONKIE_CHUNK_SIZE:.1%}")

    chunker = TokenChunker(
        tokenizer=CHONKIE_TOKENIZER,
        chunk_size=CHONKIE_CHUNK_SIZE,
        chunk_overlap=CHONKIE_CHUNK_OVERLAP
    )
    print("   OK TokenChunker initialise")

    # Indexation
    print(f"\n[4/4] Indexation des documents...")
    model = "voyage-context-3"
    total_chunks = 0
    total_tokens = 0

    for i, md_file in enumerate(files_to_index, 1):
        print(f"\n   Document {i}/{len(files_to_index)}: {md_file.name}")

        # Lire le contenu
        try:
            with open(md_file, 'r', encoding='utf-8') as f:
                content = f.read()
        except Exception as e:
            print(f"      ERREUR lecture: {e}")
            continue

        if not content.strip():
            print(f"      Document vide, skip")
            continue

        # Chunking avec Chonkie
        chunks = chunker.chunk(content)
        chunk_texts = [chunk.text for chunk in chunks]

        print(f"      Chunks generes: {len(chunks)}")

        # Calculer les tokens totaux
        doc_tokens = sum(chunk.token_count for chunk in chunks)
        total_tokens += doc_tokens

        # Embeddings avec Voyage Context-3
        try:
            print(f"      Generation embeddings ({len(chunk_texts)} chunks)...")
            result = voyage_client.contextualized_embed(
                inputs=[chunk_texts],
                model=model,
                input_type="document"
            )
            embeddings = result.results[0].embeddings

            print(f"      OK Embeddings generes: {len(embeddings)} vecteurs")

            # Préparer les données
            ids = [f"{md_file.stem}_chunk_{j:03d}" for j in range(len(chunks))]
            metadatas = [{
                "filename": md_file.name,
                "chunk_index": j,
                "chunk_text": chunk.text[:200] + "..." if len(chunk.text) > 200 else chunk.text,
                "token_count": chunk.token_count,
                "model": model
            } for j, chunk in enumerate(chunks)]

            # Ajouter à ChromaDB
            collection.add(
                ids=ids,
                embeddings=embeddings,
                documents=chunk_texts,
                metadatas=metadatas
            )

            total_chunks += len(chunks)
            print(f"      OK Ajoute a ChromaDB: {len(chunks)} chunks")

        except Exception as e:
            print(f"      ERREUR embeddings: {e}")
            continue

    # Résumé
    print(f"\n" + "=" * 60)
    print(f"TEST D'INDEXATION TERMINE")
    print(f"   Documents traites: {len(files_to_index)}/{len(test_files)}")
    print(f"   Total chunks: {total_chunks}")
    print(f"   Total tokens: {total_tokens:,}")
    print(f"   Tokens/chunk moyen: {total_tokens/total_chunks:.0f}")

    # Vérifier la collection
    collection_count = collection.count()
    print(f"   Total dans collection: {collection_count} chunks")

    print(f"\nSysteme RAG teste avec succes !")

if __name__ == "__main__":
    index_10_documents()