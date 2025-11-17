#!/usr/bin/env python3
"""
RAGDOC Menu - Interface menu simple avec pick
Navigation aux fleches + Enter pour selectionner
"""
# -*- coding: utf-8 -*-

import sys
import subprocess
from pathlib import Path

try:
    from pick import pick
    from colorama import init, Fore, Style
except ImportError:
    print("Dependances manquantes. Installez avec: pip install pick colorama")
    sys.exit(1)

init(autoreset=True)

# Couleurs
COLOR_SUCCESS = Fore.GREEN
COLOR_ERROR = Fore.RED
COLOR_WARNING = Fore.YELLOW
COLOR_INFO = Fore.CYAN
RESET = Style.RESET_ALL

CLI_DIR = Path(__file__).parent
SCRIPTS_DIR = CLI_DIR / "scripts"

# Import du gestionnaire de serveur
try:
    from chromadb_server_manager import server_manager
except ImportError:
    print(f"{COLOR_WARNING}Avertissement: chromadb_server_manager introuvable{RESET}")
    server_manager = None


def print_header(text: str):
    """Afficher un header"""
    print(f"\n{COLOR_INFO}{'='*70}{RESET}")
    print(f"{COLOR_INFO}{text.center(70)}{RESET}")
    print(f"{COLOR_INFO}{'='*70}{RESET}\n")


def print_success(text: str):
    """Afficher message succès"""
    print(f"{COLOR_SUCCESS}[OK] {text}{RESET}")


def print_error(text: str):
    """Afficher message erreur"""
    print(f"{COLOR_ERROR}[ERREUR] {text}{RESET}")


def print_info(text: str):
    """Afficher message info"""
    print(f"{COLOR_INFO}[INFO] {text}{RESET}")


def print_warning(text: str):
    """Afficher message warning"""
    print(f"{COLOR_WARNING}[WARN] {text}{RESET}")


def run_command(script_path: Path, *args, **kwargs) -> bool:
    """Lancer un script Python et afficher la progression"""
    try:
        # Utiliser conda run pour executer dans ragdoc-env
        cmd = ["conda", "run", "-n", "ragdoc-env", "python", str(script_path)] + list(args)
        result = subprocess.run(cmd, cwd=str(CLI_DIR), **kwargs)
        return result.returncode == 0
    except Exception as e:
        print_error(f"Erreur: {e}")
        return False


def action_status():
    """Afficher les statistiques"""
    print_header("STATISTIQUES D'INDEXATION")

    # Utiliser conda run pour exécuter avec ragdoc-env
    cmd = ["conda", "run", "-n", "ragdoc-env", "python", str(CLI_DIR / "ragdoc-cli.py"), "status"]
    result = subprocess.run(cmd, cwd=str(CLI_DIR))

    input("\nAppuyez sur Entrée pour continuer...")


def action_status_by_date():
    """Afficher les statistiques triées par date d'indexation"""
    print_header("STATISTIQUES D'INDEXATION - TRI PAR DATE")

    # Utiliser conda run pour exécuter avec ragdoc-env
    cmd = ["conda", "run", "-n", "ragdoc-env", "python", str(CLI_DIR / "ragdoc-cli.py"), "status", "--sort-by-date"]
    result = subprocess.run(cmd, cwd=str(CLI_DIR))

    input("\nAppuyez sur Entrée pour continuer...")


def action_status_menu():
    """Sous-menu pour afficher les statistiques avec différents tris"""
    while True:
        options = [
            "Tri alphabétique",
            "Tri par date",
            "Retour au menu principal"
        ]

        selected, index = pick(
            options,
            f"{COLOR_INFO}Voir statistiques - Choisir le tri :{RESET}",
            indicator=">",
            default_index=0
        )

        if selected == "Tri alphabétique":
            action_status()
        elif selected == "Tri par date":
            action_status_by_date()
        elif selected == "Retour au menu principal":
            break


def action_index():
    """Indexation avec contextualized embeddings"""
    print_header("INDEXATION CHROMA DB")
    if run_command(SCRIPTS_DIR / "index_contextualized_adaptive.py"):
        print_success("Indexation complétée")
    else:
        print_error("Indexation échouée")
    input("\nAppuyez sur Entrée pour continuer...")


def action_index_force():
    """Réindexation forcée avec contextualized embeddings"""
    print_header("REINDEXATION COMPLETE")
    if run_command(SCRIPTS_DIR / "index_contextualized_adaptive.py"):
        print_success("Réindexation complétée")
    else:
        print_error("Réindexation échouée")
    input("\nAppuyez sur Entrée pour continuer...")


def action_index_delete():
    """Nettoyage documents supprimés (non applicable en mode contextualized)"""
    print_header("NETTOYAGE DOCUMENTS SUPPRIMES")
    print_warning("Mode contextualized: Réindexation complète à chaque fois")
    if run_command(SCRIPTS_DIR / "index_contextualized_adaptive.py"):
        print_success("Réindexation complétée")
    else:
        print_error("Réindexation échouée")
    input("\nAppuyez sur Entrée pour continuer...")


def action_parse_pdf():
    """Convertir PDF en Markdown"""
    print_header("CONVERSION PDF -> MARKDOWN")

    pdf_path = input(f"{COLOR_INFO}Chemin du PDF: {RESET}").strip()

    if not pdf_path:
        print_error("Conversion annulée")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    # Nettoyer les guillemets autour du chemin (input Windows)
    pdf_path = pdf_path.strip('"').strip("'")

    # Verifier que le fichier existe
    from pathlib import Path
    if not Path(pdf_path).exists():
        print_error(f"Fichier non trouvé: {pdf_path}")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    if not Path(pdf_path).suffix.lower() == ".pdf":
        print_error("Le fichier doit être un PDF (.pdf)")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    print_info(f"Conversion de: {Path(pdf_path).name}")
    if run_command(SCRIPTS_DIR / "parse_pdf.py", pdf_path):
        print_success("Conversion PDF complétée")
        print_info(f"Fichier sauvegardé dans: articles_markdown/")
    else:
        print_error("Conversion échouée")

    input("\nAppuyez sur Entrée pour continuer...")


def action_monitor():
    """Monitoring continu"""
    print_header("MONITORING CONTINU - CHROMA DB")
    print_info("Ctrl+C pour arrêter")

    try:
        run_command(Path(__file__).parent / "monitor_indexation.py")
        print_success("Monitoring arrêté")
    except KeyboardInterrupt:
        print_warning("Monitoring interrompu")

    input("\nAppuyez sur Entrée pour continuer...")


def action_reset():
    """Réinitialiser base de données"""
    print_header("REINITIALISATION - SUPPRESSION COLLECTIONS")
    response = input(f"{COLOR_WARNING}Êtes-vous sûr ? (oui/non): {RESET}").strip().lower()

    if response not in ['oui', 'yes', 'y']:
        print_warning("Opération annulée")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    if run_command(Path(__file__).parent / "reset_chroma.py"):
        print_success("Reset complété")
    else:
        print_error("Reset échoué")

    input("\nAppuyez sur Entrée pour continuer...")


def action_delete():
    """Supprimer physiquement base de données"""
    print_header("SUPPRESSION PHYSIQUE - CHROMA DB")
    response = input(
        f"{COLOR_ERROR}ATTENTION: Cela supprimera COMPLETEMENT chroma_db_fresh/\n"
        f"Êtes-vous absolument sûr ? (oui/non): {RESET}"
    ).strip().lower()

    if response not in ['oui', 'yes', 'y']:
        print_warning("Opération annulée")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    if run_command(Path(__file__).parent / "delete_chroma.py"):
        print_success("Suppression complétée")
    else:
        print_error("Suppression échouée")

    input("\nAppuyez sur Entrée pour continuer...")


def action_fix_hnsw():
    """Corriger corruption HNSW"""
    print_header("CORRECTION CORRUPTION HNSW")
    response = input(f"{COLOR_WARNING}Êtes-vous sûr ? (oui/non): {RESET}").strip().lower()

    if response not in ['oui', 'yes', 'y']:
        print_warning("Opération annulée")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    if run_command(Path(__file__).parent / "fix_hnsw_chroma.py"):
        print_success("Fix HNSW complété")
    else:
        print_error("Fix HNSW échoué")

    input("\nAppuyez sur Entrée pour continuer...")


def action_delete_doc():
    """Supprimer un document spécifique"""
    print_header("SUPPRESSION DE DOCUMENT")

    try:
        # Afficher la liste des documents via ragdoc status
        print_info("Chargement de la liste des documents...")
        cmd_status = ["conda", "run", "-n", "ragdoc-env", "python", str(CLI_DIR / "ragdoc-cli.py"), "status"]
        subprocess.run(cmd_status, cwd=str(CLI_DIR))

        # Demander quel document supprimer
        print()
        doc_name = input(f"{COLOR_INFO}Nom du document à supprimer: {RESET}").strip()

        if not doc_name:
            print_error("Suppression annulée")
            input("\nAppuyez sur Entrée pour continuer...")
            return

        # Nettoyer le nom de document (enlever [N], (X chunks), etc.)
        import re
        # Enlever prefix [N]
        doc_name = re.sub(r'^\[\d+\]\s*', '', doc_name)
        # Enlever suffix (X chunks)
        doc_name = re.sub(r'\s*\(\d+\s+chunks?\).*$', '', doc_name)
        doc_name = doc_name.strip()

        # Demander confirmation avant d'appeler la CLI
        print()
        response = input(f"{COLOR_WARNING}Confirmer la suppression de '{doc_name}'? (oui/non): {RESET}").strip().lower()

        if response not in ['oui', 'yes', 'y']:
            print_warning("Suppression annulée")
            input("\nAppuyez sur Entrée pour continuer...")
            return

        # Appeler la CLI pour supprimer (avec --yes pour skip la confirmation redondante)
        cmd_delete = ["conda", "run", "-n", "ragdoc-env", "python", str(CLI_DIR / "ragdoc-cli.py"), "delete-doc", doc_name, "--yes"]
        result = subprocess.run(cmd_delete, cwd=str(CLI_DIR))

        if result.returncode == 0:
            print_success("Document supprimé avec succès")
        else:
            print_error("Échec de la suppression")

    except Exception as e:
        print_error(f"Erreur: {e}")

    input("\nAppuyez sur Entrée pour continuer...")


def action_server_start():
    """Démarrer le serveur ChromaDB"""
    print_header("DEMARRAGE SERVEUR CHROMADB")

    if not server_manager:
        print_error("Gestionnaire de serveur non disponible")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    success, message = server_manager.start()
    if success:
        print_success(message)
    else:
        print_error(message)

    input("\nAppuyez sur Entrée pour continuer...")


def action_server_stop():
    """Arrêter le serveur ChromaDB"""
    print_header("ARRET SERVEUR CHROMADB")

    if not server_manager:
        print_error("Gestionnaire de serveur non disponible")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    success, message = server_manager.stop()
    if success:
        print_success(message)
    else:
        print_error(message)

    input("\nAppuyez sur Entrée pour continuer...")


def action_server_status():
    """Afficher le statut du serveur ChromaDB"""
    print_header("STATUT SERVEUR CHROMADB")

    if not server_manager:
        print_error("Gestionnaire de serveur non disponible")
        input("\nAppuyez sur Entrée pour continuer...")
        return

    print(server_manager.get_status())
    input("\nAppuyez sur Entrée pour continuer...")


# Mapping actions
ACTIONS = {
    "Voir statistiques": action_status_menu,
    "Indexer": action_index,
    "Réindexer tout (--force)": action_index_force,
    "Nettoyer documents supprimés": action_index_delete,
    "Supprimer un document": action_delete_doc,
    "Convertir PDF": action_parse_pdf,
    "🟢 Démarrer serveur ChromaDB": action_server_start,
    "🔴 Arrêter serveur ChromaDB": action_server_stop,
    "📊 Statut serveur ChromaDB": action_server_status,
    "Monitoring continu": action_monitor,
    "Reset base de données": action_reset,
    "Supprimer base (delete)": action_delete,
    "Corriger HNSW": action_fix_hnsw,
}


def main():
    """Menu principal"""
    print_header("RAGDOC - Indexation Manager")

    while True:
        options = list(ACTIONS.keys()) + ["Quitter"]

        selected, index = pick(
            options,
            f"{COLOR_INFO}Que voulez-vous faire ?{RESET}",
            indicator=">",
            default_index=0
        )

        if selected == "Quitter":
            print_info("Au revoir!")
            break

        action = ACTIONS[selected]
        action()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{COLOR_WARNING}Interruption utilisateur{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{COLOR_ERROR}Erreur inattendue: {e}{RESET}")
        sys.exit(1)
