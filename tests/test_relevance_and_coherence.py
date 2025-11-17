#!/usr/bin/env python3
"""
Tests de Pertinence et Cohérence Contextuelle
Compare les bases HYBRID vs CONTEXTUALIZED sur 18 queries scientifiques
"""

import sys
import json
import time
from pathlib import Path
from collections import Counter, defaultdict
sys.path.insert(0, str(Path(__file__).parent.parent / "scripts"))

import chromadb
import voyageai
from indexing_config import (
    CHROMA_DB_HYBRID_PATH,
    CHROMA_DB_CONTEXTUALIZED_PATH,
    COLLECTION_HYBRID_NAME,
    COLLECTION_CONTEXTUALIZED_NAME
)


# ============================================================================
# QUERIES DE TEST (18 queries couvrant 5 thématiques)
# ============================================================================

TEST_QUERIES = {
    'black_carbon': [
        "How does black carbon deposition affect glacier surface albedo?",
        "What are the sources of black carbon on glaciers?",
        "BC concentration measurements techniques in snow",
        "Saharan dust vs black carbon impact on alpine glaciers"
    ],
    'modis_remote_sensing': [
        "MODIS albedo product validation accuracy",
        "Landsat vs MODIS for glacier albedo monitoring",
        "Sentinel-2 satellite albedo measurements",
        "Remote sensing of snow grain size"
    ],
    'energy_balance': [
        "Surface energy balance components on glaciers",
        "Turbulent heat flux parameterization",
        "Temperature index vs energy balance models",
        "Shortwave radiation and albedo feedback"
    ],
    'snow_algae': [
        "Snow algae impact on Greenland ice sheet darkening",
        "Biological darkening vs mineral dust",
        "Microbial communities on glacier surfaces"
    ],
    'wildfire_smoke': [
        "Wildfire smoke deposition on glaciers",
        "BC from biomass burning on North American glaciers",
        "Post-wildfire impact on snow albedo"
    ]
}


def search_database(query: str, database: str, top_k: int = 5):
    """
    Recherche dans une base de données (hybrid ou contextualized)

    Returns:
        dict: {
            'documents': [...],
            'metadatas': [...],
            'distances': [...],
            'time': float
        }
    """
    vo = voyageai.Client()

    if database == 'hybrid':
        client = chromadb.PersistentClient(path=str(CHROMA_DB_HYBRID_PATH))
        collection = client.get_collection(name=COLLECTION_HYBRID_NAME)

        # Standard embedding
        start = time.time()
        query_embd = vo.embed([query], model="voyage-3-large", input_type="query").embeddings[0]

        results = collection.query(
            query_embeddings=[query_embd],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        elapsed = time.time() - start

    elif database == 'contextualized':
        client = chromadb.PersistentClient(path=str(CHROMA_DB_CONTEXTUALIZED_PATH))
        collection = client.get_collection(name=COLLECTION_CONTEXTUALIZED_NAME)

        # Contextualized embedding
        start = time.time()
        query_embd = vo.contextualized_embed(
            inputs=[[query]],
            model="voyage-context-3",
            input_type="query"
        ).results[0].embeddings[0]

        results = collection.query(
            query_embeddings=[query_embd],
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )
        elapsed = time.time() - start

    else:
        raise ValueError(f"Unknown database: {database}")

    return {
        'documents': results['documents'][0],
        'metadatas': results['metadatas'][0],
        'distances': results['distances'][0],
        'time': elapsed
    }


def analyze_coherence(results: dict) -> dict:
    """
    Analyse la cohérence contextuelle des résultats

    Métriques:
    - document_diversity: Nombre de documents uniques dans top-K
    - same_doc_clustering: % de chunks venant du même document
    - top_source: Document le plus fréquent
    - top_source_count: Nombre de chunks du document principal
    """
    sources = [meta.get('source', 'unknown') for meta in results['metadatas']]
    source_counter = Counter(sources)

    total_chunks = len(sources)
    unique_docs = len(source_counter)
    top_source, top_count = source_counter.most_common(1)[0] if source_counter else ('', 0)

    same_doc_pct = (top_count / total_chunks * 100) if total_chunks > 0 else 0

    return {
        'document_diversity': unique_docs,
        'same_doc_clustering_pct': same_doc_pct,
        'top_source': top_source,
        'top_source_count': top_count,
        'all_sources': dict(source_counter)
    }


def run_query_comparison(query: str, top_k: int = 5):
    """
    Compare une query sur les deux bases de données
    """
    print(f"\n{'='*80}")
    print(f"QUERY: {query}")
    print(f"{'='*80}")

    # Recherche HYBRID
    print("\n[HYBRID] Searching...")
    try:
        results_hybrid = search_database(query, 'hybrid', top_k=top_k)
        coherence_hybrid = analyze_coherence(results_hybrid)

        print(f"   Time: {results_hybrid['time']:.2f}s")
        print(f"   Unique docs: {coherence_hybrid['document_diversity']}/{top_k}")
        print(f"   Top source: {coherence_hybrid['top_source']} ({coherence_hybrid['top_source_count']} chunks)")

    except Exception as e:
        print(f"   [ERROR] {e}")
        results_hybrid = None
        coherence_hybrid = None

    # Recherche CONTEXTUALIZED
    print("\n[CONTEXTUALIZED] Searching...")
    try:
        results_ctx = search_database(query, 'contextualized', top_k=top_k)
        coherence_ctx = analyze_coherence(results_ctx)

        print(f"   Time: {results_ctx['time']:.2f}s")
        print(f"   Unique docs: {coherence_ctx['document_diversity']}/{top_k}")
        print(f"   Top source: {coherence_ctx['top_source']} ({coherence_ctx['top_source_count']} chunks)")

    except Exception as e:
        print(f"   [ERROR] {e}")
        results_ctx = None
        coherence_ctx = None

    # Comparaison
    if results_hybrid and results_ctx:
        print(f"\n[COMPARISON]")
        print(f"   Coherence: HYBRID {coherence_hybrid['same_doc_clustering_pct']:.1f}% vs "
              f"CONTEXTUALIZED {coherence_ctx['same_doc_clustering_pct']:.1f}%")

        if coherence_ctx['same_doc_clustering_pct'] > coherence_hybrid['same_doc_clustering_pct']:
            print(f"   -> CONTEXTUALIZED has better document coherence!")

    return {
        'query': query,
        'hybrid': {
            'results': results_hybrid,
            'coherence': coherence_hybrid
        },
        'contextualized': {
            'results': results_ctx,
            'coherence': coherence_ctx
        }
    }


def run_full_test_suite(top_k: int = 5):
    """
    Lance la suite complète de tests
    """
    print("\n" + "="*80)
    print("TESTS DE PERTINENCE ET COHERENCE CONTEXTUELLE")
    print("="*80)
    print(f"\nTotal queries: {sum(len(queries) for queries in TEST_QUERIES.values())}")
    print(f"Top-K: {top_k}")

    all_results = defaultdict(list)

    for category, queries in TEST_QUERIES.items():
        print(f"\n\n{'#'*80}")
        print(f"CATEGORY: {category.upper().replace('_', ' ')}")
        print(f"{'#'*80}")

        for query in queries:
            result = run_query_comparison(query, top_k=top_k)
            all_results[category].append(result)

            # Petite pause pour ne pas surcharger l'API
            time.sleep(0.5)

    return dict(all_results)


def generate_summary_report(all_results: dict):
    """
    Génère un rapport de synthèse
    """
    print("\n\n" + "="*80)
    print("RAPPORT DE SYNTHESE")
    print("="*80)

    # Statistiques globales
    total_queries = sum(len(results) for results in all_results.values())

    hybrid_coherence_scores = []
    ctx_coherence_scores = []
    hybrid_diversity_scores = []
    ctx_diversity_scores = []

    ctx_better_count = 0
    hybrid_better_count = 0

    for category, results in all_results.items():
        for result in results:
            if result['hybrid']['coherence'] and result['contextualized']['coherence']:
                h_coh = result['hybrid']['coherence']['same_doc_clustering_pct']
                c_coh = result['contextualized']['coherence']['same_doc_clustering_pct']

                hybrid_coherence_scores.append(h_coh)
                ctx_coherence_scores.append(c_coh)

                hybrid_diversity_scores.append(result['hybrid']['coherence']['document_diversity'])
                ctx_diversity_scores.append(result['contextualized']['coherence']['document_diversity'])

                if c_coh > h_coh:
                    ctx_better_count += 1
                elif h_coh > c_coh:
                    hybrid_better_count += 1

    # Affichage statistiques
    print(f"\nTotal queries tested: {total_queries}")

    print(f"\n[COHERENCE SCORES] (% chunks from same document)")
    print(f"   HYBRID average:        {sum(hybrid_coherence_scores)/len(hybrid_coherence_scores):.1f}%")
    print(f"   CONTEXTUALIZED average: {sum(ctx_coherence_scores)/len(ctx_coherence_scores):.1f}%")

    print(f"\n[DOCUMENT DIVERSITY] (unique docs in top-5)")
    print(f"   HYBRID average:        {sum(hybrid_diversity_scores)/len(hybrid_diversity_scores):.2f} docs")
    print(f"   CONTEXTUALIZED average: {sum(ctx_diversity_scores)/len(ctx_diversity_scores):.2f} docs")

    print(f"\n[COHERENCE COMPARISON]")
    print(f"   CONTEXTUALIZED better: {ctx_better_count}/{total_queries} queries ({ctx_better_count/total_queries*100:.1f}%)")
    print(f"   HYBRID better:         {hybrid_better_count}/{total_queries} queries ({hybrid_better_count/total_queries*100:.1f}%)")

    # Résumé par catégorie
    print(f"\n[RESULTS BY CATEGORY]")
    for category, results in all_results.items():
        cat_ctx_better = sum(
            1 for r in results
            if r['contextualized']['coherence'] and r['hybrid']['coherence']
            and r['contextualized']['coherence']['same_doc_clustering_pct'] > r['hybrid']['coherence']['same_doc_clustering_pct']
        )
        print(f"   {category:25s}: CONTEXTUALIZED better in {cat_ctx_better}/{len(results)} queries")

    print("\n" + "="*80)


def export_results_json(all_results: dict, output_file: str = "test_results.json"):
    """
    Exporte les résultats en JSON pour analyse ultérieure
    """
    output_path = Path(__file__).parent / output_file

    # Convertir en format sérialisable
    export_data = {}
    for category, results in all_results.items():
        export_data[category] = []
        for result in results:
            export_data[category].append({
                'query': result['query'],
                'hybrid': {
                    'coherence': result['hybrid']['coherence'],
                    'time': result['hybrid']['results']['time'] if result['hybrid']['results'] else None
                },
                'contextualized': {
                    'coherence': result['contextualized']['coherence'],
                    'time': result['contextualized']['results']['time'] if result['contextualized']['results'] else None
                }
            })

    with open(output_path, 'w', encoding='utf-8') as f:
        json.dump(export_data, f, indent=2, ensure_ascii=False)

    print(f"\n[EXPORT] Results saved to: {output_path}")


if __name__ == "__main__":
    # Lancer la suite complète de tests
    all_results = run_full_test_suite(top_k=5)

    # Générer le rapport de synthèse
    generate_summary_report(all_results)

    # Exporter en JSON
    export_results_json(all_results)

    print("\n[DONE] Tests completed!")
