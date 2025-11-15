#!/usr/bin/env python3
"""
Validation et test de l'indexation incrémentale.

Affiche:
- Statistiques complètes de la base Chroma
- État des hash MD5 (documents avec/sans hash)
- Répartition par modèle d'embedding
- Historique des derniers documents indexés
"""

import sys
from pathlib import Path
from datetime import datetime

import chromadb

# Importer la configuration
sys.path.insert(0, str(Path(__file__).parent))
from indexing_config import CHROMA_DB_PATH, COLLECTION_NAME


def test_incremental_indexing() -> None:
    """Valide et affiche l'état de l'indexation."""

    print("\n" + "=" * 70)
    print("VALIDATION - ÉTAT DE LA BASE CHROMA")
    print("=" * 70)

    try:
        # Connecter Chroma
        client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))
        collection = client.get_collection(name=COLLECTION_NAME)
        print(f"\nOK Collection '{COLLECTION_NAME}' chargee")
    except Exception as e:
        print(f"\nERREUR Erreur de connexion: {str(e)}")
        return

    # Récupérer tous les documents
    try:
        all_docs = collection.get(include=["metadatas"])
    except Exception as e:
        print(f"ERREUR Erreur lors de la recuperation: {str(e)}")
        return

    # Statistiques globales
    total_chunks = len(all_docs['ids'])
    unique_sources = set(m['source'] for m in all_docs['metadatas'] if m.get('source'))

    print(f"\nSTATISTIQUES GLOBALES:")
    print(f"   Total documents:         {len(unique_sources)}")
    print(f"   Total chunks:            {total_chunks}")
    if len(unique_sources) > 0:
        print(f"   Moyenne chunks/doc:      {total_chunks / len(unique_sources):.1f}")

    # Analyser par source
    docs_by_source = {}
    for i, metadata in enumerate(all_docs['metadatas']):
        source = metadata.get('source')
        if source not in docs_by_source:
            docs_by_source[source] = {
                'chunks': 0,
                'hash': metadata.get('doc_hash'),
                'indexed_date': metadata.get('indexed_date'),
                'model': metadata.get('model')
            }
        docs_by_source[source]['chunks'] += 1

    # Vérifier les hash MD5
    print(f"\nVERIFICATION DES HASH MD5:")
    docs_with_hash = sum(1 for d in docs_by_source.values() if d['hash'])
    docs_without_hash = len(docs_by_source) - docs_with_hash

    print(f"   Documents avec hash:     {docs_with_hash}/{len(docs_by_source)}")
    if docs_without_hash > 0:
        print(f"   ATTENTION Documents sans hash:  {docs_without_hash}")
        print(f"   Lancer: python migrate_add_hashes.py")

    # Vérifier les métadonnées d'indexation
    print(f"\nMETADONNEES D'INDEXATION:")
    docs_with_date = sum(1 for d in docs_by_source.values() if d['indexed_date'])
    print(f"   Documents datés:         {docs_with_date}/{len(docs_by_source)}")

    # Répartition par modèle
    print(f"\nREPARTITION PAR MODELE D'EMBEDDING:")
    models = {}
    for doc in docs_by_source.values():
        model = doc['model'] or 'unknown'
        models[model] = models.get(model, 0) + 1

    for model, count in sorted(models.items()):
        pct = (count / len(docs_by_source)) * 100
        print(f"   {model:25} {count:3d} documents ({pct:5.1f}%)")

    # Statistiques par taille
    print(f"\nREPARTITION PAR NOMBRE DE CHUNKS:")
    chunk_counts = sorted(set(d['chunks'] for d in docs_by_source.values()))
    for chunk_count in chunk_counts:
        docs_with_count = sum(1 for d in docs_by_source.values() if d['chunks'] == chunk_count)
        pct = (docs_with_count / len(docs_by_source)) * 100
        print(f"   {chunk_count:3d} chunks: {docs_with_count:3d} documents ({pct:5.1f}%)")

    # Derniers documents indexés
    print(f"\nHISTORIQUE DES INDEXATIONS:")
    docs_with_dates = [
        (source, data['indexed_date'])
        for source, data in docs_by_source.items()
        if data['indexed_date']
    ]

    if docs_with_dates:
        docs_with_dates.sort(key=lambda x: x[1], reverse=True)
        print(f"\n   5 derniers documents indexés:")
        for source, date in docs_with_dates[:5]:
            datetime_obj = datetime.fromisoformat(date)
            formatted_date = datetime_obj.strftime("%Y-%m-%d %H:%M:%S")
            print(f"      {formatted_date} - {source}")
    else:
        print(f"   ATTENTION Aucune date d'indexation trouvee")
        print(f"   Lancer: python migrate_add_hashes.py")

    # Recommandations
    print(f"\nRECOMMANDATIONS:")
    if docs_without_hash > 0:
        print(f"   1. Ajouter les hash MD5 manquants:")
        print(f"      python migrate_add_hashes.py")
    else:
        print(f"   OK Tous les documents ont un hash MD5")

    print(f"   2. Ajouter des documents:")
    print(f"      - Copier le markdown dans ../articles_markdown/")
    print(f"      - Lancer: python index_incremental.py")

    print(f"   3. Forcer une réindexation complète (si nécessaire):")
    print(f"      python index_incremental.py --force")

    print(f"   4. Nettoyer les documents supprimés:")
    print(f"      python index_incremental.py --delete-missing")

    # État global
    print(f"\n" + "=" * 70)
    if docs_without_hash == 0 and docs_with_date > 0:
        print(f"OK ETAT: READY FOR PRODUCTION")
    elif docs_without_hash > 0:
        print(f"ATTENTION ETAT: MIGRATION NEEDED (hash MD5 manquants)")
    else:
        print(f"ATTENTION ETAT: DATES D'INDEXATION MANQUANTES")
    print("=" * 70 + "\n")


if __name__ == "__main__":
    test_incremental_indexing()
