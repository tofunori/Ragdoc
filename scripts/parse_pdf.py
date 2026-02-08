#!/usr/bin/env python3
"""
Convertir un PDF en Markdown structure via LlamaParse ou Docling.

Backends disponibles:
  - docling: 100% gratuit, local, open-source (DEFAULT)
  - llamaparse: Cloud-based, requiert API key et credits

Modes disponibles:
  - test: 3 premieres pages seulement
  - fast: Rapide sans OCR (Docling uniquement)
  - standard: Qualite equilibree avec formules LaTeX [DEFAULT]
  - premium: Qualite maximale (VLM pour Docling, agent pour LlamaParse)
"""

import os
import sys
import argparse
import time
from pathlib import Path
from dotenv import load_dotenv

# Charger les variables d'environnement
load_dotenv()

# Importer LlamaParse (optionnel)
LLAMAPARSE_AVAILABLE = False
try:
    from llama_parse import LlamaParse
    from llama_parse_config import get_config as get_llamaparse_config
    from llama_parse_config import print_available_modes as print_llamaparse_modes
    LLAMAPARSE_AVAILABLE = True
except ImportError:
    pass

# Importer Docling (optionnel)
DOCLING_AVAILABLE = False
try:
    from docling.document_converter import DocumentConverter, PdfFormatOption
    from docling.datamodel.base_models import InputFormat
    from docling_config import get_config as get_docling_config
    from docling_config import print_available_modes as print_docling_modes
    DOCLING_AVAILABLE = True
except ImportError:
    pass

# Verifier qu'au moins un backend est disponible
if not LLAMAPARSE_AVAILABLE and not DOCLING_AVAILABLE:
    print("ERREUR: Aucun backend de parsing disponible!")
    print("Installez au moins un:")
    print("  - Docling (gratuit): pip install docling")
    print("  - LlamaParse: pip install llama-parse")
    sys.exit(1)

# Configuration
OUTPUT_DIR = Path(__file__).parent.parent / "articles_markdown"


def convert_with_docling(pdf_path: Path, output_file: Path, mode: str) -> bool:
    """
    Convertir PDF avec Docling (gratuit, local, open-source).

    Args:
        pdf_path: Chemin vers le PDF
        output_file: Chemin de sortie markdown
        mode: Mode de parsing ('fast', 'standard', 'premium', 'test')

    Returns:
        bool: True si conversion reussie
    """
    if not DOCLING_AVAILABLE:
        print("[ERROR] Docling non installe. Installez avec: pip install docling")
        return False

    try:
        # Recuperer configuration
        config = get_docling_config(mode)
        pipeline_options = config["pipeline_options"]
        max_pages = config.get("max_pages")

        print(f"Backend:    Docling (gratuit, local, GPU-accelere)")
        print(f"Mode:       {mode.upper()}")
        print(f"Desc:       {config['description']}")
        print(f"Formules:   LaTeX {'ACTIVE' if pipeline_options.do_formula_enrichment else 'DESACTIVE'}")
        if max_pages:
            print(f"Pages:      {max_pages} premieres pages seulement")
        print()

        # Initialiser converter
        print("Initialisation de Docling avec GPU...")
        converter = DocumentConverter(
            format_options={
                InputFormat.PDF: PdfFormatOption(
                    pipeline_options=pipeline_options
                )
            }
        )

        # Convertir
        print(f"Parsing PDF ({pdf_path.name})...")
        print("(GPU acceleration activee - devrait etre rapide!)")
        result = converter.convert(str(pdf_path))

        # Extraire markdown
        markdown_text = result.document.export_to_markdown()

        # Limiter pages si necessaire
        if max_pages:
            # Approximation: limiter par nombre de lignes
            lines = markdown_text.split('\n')
            # Garder environ 200 lignes par page max
            max_lines = max_pages * 200
            if len(lines) > max_lines:
                markdown_text = '\n'.join(lines[:max_lines])
                markdown_text += f"\n\n... (Limite de {max_pages} pages atteinte)"

        # Ajouter header avec metadonnees
        header = f"""# {pdf_path.stem}

**Source:** {pdf_path.name}
**Chemin original:** {pdf_path.absolute()}
**Date de conversion:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Methode:** Docling (Open-Source, Gratuit, Local, GPU-Accelere)
**Mode:** {mode}
**Formules LaTeX:** {'Oui' if pipeline_options.do_formula_enrichment else 'Non'}

---

"""

        # Sauvegarder
        output_file.write_text(header + markdown_text, encoding='utf-8')

        file_size = output_file.stat().st_size / 1024  # KB
        print(f"\n[SUCCESS] Fichier converti et sauvegarde!")
        print(f"  Fichier: {output_file.name}")
        print(f"  Taille:  {file_size:.1f} KB")
        print(f"  Backend: Docling (100% gratuit, GPU-accelere)")
        print()
        return True

    except Exception as e:
        print(f"\n[ERROR] Erreur lors de la conversion Docling: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_with_llamaparse(pdf_path: Path, output_file: Path, mode: str) -> bool:
    """
    Convertir PDF avec LlamaParse (cloud, payant).

    Args:
        pdf_path: Chemin vers le PDF
        output_file: Chemin de sortie markdown
        mode: Mode de parsing ('test', 'economy', 'standard', 'premium')

    Returns:
        bool: True si conversion reussie
    """
    if not LLAMAPARSE_AVAILABLE:
        print("[ERROR] LlamaParse non installe. Installez avec: pip install llama-parse")
        return False

    try:
        # Recuperer configuration
        config = get_llamaparse_config(mode)

        print(f"Backend:    LlamaParse (cloud, credits requis)")
        print(f"Mode:       {mode.upper()}")
        print(f"Parse mode: {config.get('parse_mode', 'N/A')}")
        if config.get('max_pages'):
            print(f"Pages:      {config['max_pages']} premieres pages seulement")
        print()

        # Initialiser LlamaParse
        api_key = os.getenv("LLAMA_CLOUD_API_KEY")
        if not api_key:
            print("[ERROR] LLAMA_CLOUD_API_KEY non configuree dans .env")
            return False

        print("Connexion a LlamaParse...")
        parser = LlamaParse(
            api_key=api_key,
            result_type="markdown",
            **config  # Appliquer toute la configuration
        )

        print(f"Parsing PDF ({pdf_path.name})...")
        documents = parser.load_data(str(pdf_path))

        # Extraire le texte Markdown
        markdown_text = ""
        if isinstance(documents, list):
            for doc in documents:
                if hasattr(doc, 'text'):
                    markdown_text += doc.text
                else:
                    markdown_text += str(doc)
        elif hasattr(documents, 'text'):
            markdown_text = documents.text
        else:
            markdown_text = str(documents)

        # Ajouter header
        header = f"""# {pdf_path.stem}

**Source:** {pdf_path.name}
**Chemin original:** {pdf_path.absolute()}
**Date de conversion:** {time.strftime('%Y-%m-%d %H:%M:%S')}
**Methode:** LlamaParse (Cloud)
**Mode:** {mode}

---

"""

        # Sauvegarder
        output_file.write_text(header + markdown_text, encoding='utf-8')

        file_size = output_file.stat().st_size / 1024  # KB
        print(f"\n[SUCCESS] Fichier converti et sauvegarde!")
        print(f"  Fichier: {output_file.name}")
        print(f"  Taille:  {file_size:.1f} KB")
        print(f"  Backend: LlamaParse (credits consommes)")
        print()
        return True

    except Exception as e:
        print(f"\n[ERROR] Erreur lors de la conversion LlamaParse: {e}")
        import traceback
        traceback.print_exc()
        return False


def convert_pdf(pdf_path: str, output_file: str = None, mode: str = "standard", backend: str = "auto") -> bool:
    """
    Convertir un PDF en Markdown avec backend au choix.

    Args:
        pdf_path: Chemin vers le PDF a convertir
        output_file: Nom du fichier de sortie (optionnel)
        mode: Mode de parsing ('test', 'fast', 'standard', 'premium')
        backend: 'docling' (gratuit), 'llamaparse' (payant), ou 'auto' (defaut)

    Returns:
        bool: True si conversion reussie, False sinon
    """

    # Nettoyer les guillemets autour du chemin (input Windows)
    pdf_path = pdf_path.strip().strip('"').strip("'")
    pdf_path = Path(pdf_path)

    # Verifier que le PDF existe
    if not pdf_path.exists():
        print(f"[ERROR] Fichier PDF non trouve: {pdf_path}")
        return False

    # Verifier que c'est un PDF
    if pdf_path.suffix.lower() != ".pdf":
        print(f"[ERROR] Le fichier n'est pas un PDF: {pdf_path}")
        return False

    # Determiner le nom de sortie
    if output_file:
        output_file = Path(output_file)
    else:
        # Generer le nom depuis le PDF
        output_file = OUTPUT_DIR / f"{pdf_path.stem}.md"

    # Creer le repertoire de sortie
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

    # Auto-detection du backend
    if backend == "auto":
        if DOCLING_AVAILABLE:
            backend = "docling"  # Preferer Docling (gratuit)
        elif LLAMAPARSE_AVAILABLE:
            backend = "llamaparse"
        else:
            print("[ERROR] Aucun backend disponible!")
            return False

    # Afficher header
    print("\n" + "="*70)
    print("CONVERSION PDF -> MARKDOWN")
    print("="*70)
    print(f"Source:     {pdf_path.absolute()}")
    print(f"Sortie:     {output_file.absolute()}")
    print()

    # Router vers le bon backend
    if backend == "docling":
        return convert_with_docling(pdf_path, output_file, mode)
    elif backend == "llamaparse":
        return convert_with_llamaparse(pdf_path, output_file, mode)
    else:
        print(f"[ERROR] Backend inconnu: {backend}")
        print("Backends disponibles: docling, llamaparse")
        return False


def main():
    """Lancer la conversion."""

    # Detecter backends disponibles pour les choix
    backend_choices = ["auto"]
    if DOCLING_AVAILABLE:
        backend_choices.append("docling")
    if LLAMAPARSE_AVAILABLE:
        backend_choices.append("llamaparse")

    parser = argparse.ArgumentParser(
        description="Convertir un PDF en Markdown via Docling (gratuit) ou LlamaParse",
        prog="parse_pdf",
        epilog="Exemples:\n"
               "  python parse_pdf.py article.pdf                            # Auto (Docling si dispo)\n"
               "  python parse_pdf.py article.pdf --backend docling          # Gratuit, local, GPU\n"
               "  python parse_pdf.py article.pdf --backend llamaparse       # Cloud, payant\n"
               "  python parse_pdf.py article.pdf --mode test                # 3 pages seulement\n"
               "  python parse_pdf.py article.pdf --mode standard            # Standard + LaTeX\n"
               "  python parse_pdf.py --list-modes --backend docling         # Modes Docling\n"
               "  python parse_pdf.py --list-modes --backend llamaparse      # Modes LlamaParse",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )
    parser.add_argument(
        "file",
        nargs="?",
        help="Chemin vers le fichier PDF a convertir"
    )
    parser.add_argument(
        "--output",
        help="Nom du fichier de sortie (optionnel, genere depuis le PDF par defaut)"
    )
    parser.add_argument(
        "--backend",
        default="auto",
        choices=backend_choices,
        help="Backend de parsing (defaut: auto). "
             "docling=gratuit/local/GPU, llamaparse=cloud/payant, auto=detecte automatiquement"
    )
    parser.add_argument(
        "--mode",
        default="standard",
        help="Mode de parsing (defaut: standard). "
             "Docling: test/fast/standard/premium. LlamaParse: test/economy/standard/premium"
    )
    parser.add_argument(
        "--list-modes",
        action="store_true",
        help="Afficher tous les modes disponibles avec details"
    )
    args = parser.parse_args()

    # Si --list-modes, afficher et quitter
    if args.list_modes:
        backend = args.backend
        if backend == "auto":
            backend = "docling" if DOCLING_AVAILABLE else "llamaparse"

        if backend == "docling" and DOCLING_AVAILABLE:
            print_docling_modes()
        elif backend == "llamaparse" and LLAMAPARSE_AVAILABLE:
            print_llamaparse_modes()
        else:
            print(f"[ERROR] Backend '{backend}' non disponible")
            return 1
        return 0

    # Sinon, le fichier est requis
    if not args.file:
        parser.error("argument file requis (sauf avec --list-modes)")

    if convert_pdf(args.file, output_file=args.output, mode=args.mode, backend=args.backend):
        return 0
    else:
        return 1


if __name__ == "__main__":
    sys.exit(main())
