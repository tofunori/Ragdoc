#!/bin/bash
# Installation automatique du Hybrid Search

echo "================================================================================"
echo "INSTALLATION DU HYBRID SEARCH (BM25 + Semantic)"
echo "================================================================================"

# Vérifier qu'on est dans le bon répertoire
if [ ! -f "src/server_hybrid.py" ]; then
    echo "❌ Erreur: Ce script doit être lancé depuis le répertoire Ragdoc"
    exit 1
fi

# 1. Installer les dépendances
echo ""
echo "[1/3] Installation de rank-bm25..."
pip install rank-bm25>=0.2.2 || {
    echo "❌ Erreur lors de l'installation de rank-bm25"
    exit 1
}
echo "✅ rank-bm25 installé"

# 2. Test rapide
echo ""
echo "[2/3] Test du hybrid search..."
python quick_test_hybrid.py || {
    echo "❌ Le test a échoué. Vérifiez les erreurs ci-dessus."
    exit 1
}

# 3. Activation
echo ""
echo "[3/3] Activation du hybrid search..."
python activate_hybrid_search.py || {
    echo "❌ L'activation a échoué"
    exit 1
}

# Success
echo ""
echo "================================================================================"
echo "✅ HYBRID SEARCH INSTALLÉ ET ACTIVÉ AVEC SUCCÈS!"
echo "================================================================================"
echo ""
echo "🚀 Prochaines étapes:"
echo "  1. Redémarrer Claude Desktop (quitter et relancer)"
echo "  2. Faire une recherche pour tester"
echo "  3. Vérifier que vous voyez 'BM25 rank' et 'Semantic rank' dans les résultats"
echo ""
echo "📖 Documentation complète: HYBRID_SEARCH_GUIDE.md"
echo "================================================================================"
