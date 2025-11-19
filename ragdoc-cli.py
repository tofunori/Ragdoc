#!/usr/bin/env python3
"""
RAGDOC CLI - Interface unifiée pour gestion d'indexation Chroma DB
Supports: indexation, monitoring, statistiques, réinitialisation
"""

import sys
import os
import argparse
import subprocess
import time
from pathlib import Path
from datetime import datetime
from typing import Optional

try:
    import chromadb
    from colorama import Fore, Back, Style, init
except ImportError:
    print("Erreur: dépendances manquantes")
    print("Installez: pip install chromadb colorama")
    sys.exit(1)

# Initialiser colorama
init(autoreset=True)

# Configuration - USING NEW CONFIG
sys.path.insert(0, str(Path(__file__).parent / "src"))
from config import ACTIVE_DB_PATH as CHROMA_DB_PATH, COLLECTION_NAME, MARKDOWN_DIR

# Import du gestionnaire de serveur
try:
    from chromadb_server_manager import server_manager
except ImportError:
    print("Avertissement: chromadb_server_manager introuvable")
    server_manager = None

# Couleurs
COLOR_SUCCESS = Fore.GREEN
COLOR_ERROR = Fore.RED
COLOR_WARNING = Fore.YELLOW
COLOR_INFO = Fore.CYAN
COLOR_HEADER = Fore.CYAN + Style.BRIGHT
RESET = Style.RESET_ALL


class RagdocCLI:
    """Interface CLI pour RAGDOC indexation"""

    def __init__(self):
        self.cli_dir = Path(__file__).parent
        self.scripts_dir = self.cli_dir / "scripts"

    def print_header(self, text: str):
        """Afficher un header"""
        print(f"\n{COLOR_HEADER}{'='*70}{RESET}")
        print(f"{COLOR_HEADER}{text.center(70)}{RESET}")
        print(f"{COLOR_HEADER}{'='*70}{RESET}\n")

    def print_success(self, text: str):
        """Afficher message succès"""
        print(f"{COLOR_SUCCESS}[OK] {text}{RESET}")

    def print_error(self, text: str):
        """Afficher message erreur"""
        print(f"{COLOR_ERROR}[ERREUR] {text}{RESET}")

    def print_warning(self, text: str):
        """Afficher message warning"""
        print(f"{COLOR_WARNING}[WARN] {text}{RESET}")

    def print_info(self, text: str):
        """Afficher message info"""
        print(f"{COLOR_INFO}[INFO] {text}{RESET}")

    def get_indexation_status(self, sort_by_date: bool = False) -> str:
        """Obtenir statistiques d'indexation complètes

        Args:
            sort_by_date: Si True, trier les articles par date d'indexation (plus récents en premier)
        """
        try:
            # Use PersistentClient for contextualized database
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

            collections = client.list_collections()

            if COLLECTION_NAME not in [c.name for c in collections]:
                return "Collection non trouvée. L'indexation n'a pas encore commencé."

            collection = client.get_collection(name=COLLECTION_NAME)
            all_docs = collection.get(include=["metadatas"])

            # Statistiques globales
            docs_by_source = {}
            for metadata in all_docs['metadatas']:
                source = metadata.get('source', 'unknown')
                if source not in docs_by_source:
                    docs_by_source[source] = {
                        'chunks': 0,
                        'hash': metadata.get('doc_hash'),
                        'indexed_date': metadata.get('indexed_date'),
                        'model': metadata.get('model'),
                        'title': metadata.get('title', 'N/A')
                    }
                docs_by_source[source]['chunks'] += 1

            total_chunks = len(all_docs['ids'])
            total_docs = len(docs_by_source)

            # Construire rapport
            header_suffix = " (TRI PAR DATE)" if sort_by_date else ""
            output = f"{COLOR_HEADER}ETAT DE L'INDEXATION - CHROMA DB{header_suffix}{RESET}\n"
            output += f"{COLOR_HEADER}{'='*70}{RESET}\n\n"

            # Stats globales
            output += f"{COLOR_INFO}STATISTIQUES GLOBALES:{RESET}\n"
            output += f"   Nombre d'articles: {COLOR_SUCCESS}{total_docs}{RESET}\n"
            output += f"   Nombre total de chunks: {COLOR_SUCCESS}{total_chunks}{RESET}\n"
            if total_docs > 0:
                avg_chunks = total_chunks / total_docs
                output += f"   Moyenne chunks/article: {COLOR_SUCCESS}{avg_chunks:.1f}{RESET}\n\n"

            # Métadonnées
            docs_with_hash = sum(1 for d in docs_by_source.values() if d['hash'])
            docs_with_date = sum(1 for d in docs_by_source.values() if d['indexed_date'])

            output += f"{COLOR_INFO}VERIFICATION DES METADONNEES:{RESET}\n"
            hash_status = COLOR_SUCCESS if docs_with_hash == total_docs else COLOR_WARNING
            date_status = COLOR_SUCCESS if docs_with_date == total_docs else COLOR_WARNING
            output += f"   Documents avec hash MD5: {hash_status}{docs_with_hash}/{total_docs}{RESET}\n"
            output += f"   Documents avec date: {date_status}{docs_with_date}/{total_docs}{RESET}\n\n"

            # Modèles
            models = {}
            for doc in docs_by_source.values():
                model = doc['model'] or 'unknown'
                models[model] = models.get(model, 0) + 1

            output += f"{COLOR_INFO}REPARTITION PAR MODELE:{RESET}\n"
            for model, count in sorted(models.items()):
                output += f"   {model}: {COLOR_SUCCESS}{count}{RESET}\n"

            # Répertoire source
            md_count = len(list(MARKDOWN_DIR.glob("*.md")))
            output += f"\n{COLOR_INFO}SOURCE - articles_markdown/:{RESET}\n"
            output += f"   Fichiers .md disponibles: {COLOR_SUCCESS}{md_count}{RESET}\n\n"

            # Liste des articles
            if total_docs > 0:
                output += f"{COLOR_INFO}ARTICLES INDEXES:{RESET}\n"

                # Fonction de tri par date
                def get_sort_key(item):
                    source, doc_info = item
                    date_str = doc_info.get('indexed_date')
                    if date_str:
                        try:
                            # Parser la date ISO 8601
                            return datetime.fromisoformat(date_str)
                        except (ValueError, TypeError):
                            # Si parsing échoue, mettre à la fin
                            return datetime.min
                    # Documents sans date vont à la fin
                    return datetime.min

                # Trier selon le mode demandé
                if sort_by_date:
                    # Tri par date (plus récents en premier)
                    sorted_docs = sorted(docs_by_source.items(), key=get_sort_key, reverse=True)
                else:
                    # Tri alphabétique par nom de fichier
                    sorted_docs = sorted(docs_by_source.items())

                for i, (source, doc_info) in enumerate(sorted_docs, 1):
                    output += f"   [{i}] {COLOR_SUCCESS}{source}{RESET}\n"
                    output += f"       Chunks: {doc_info['chunks']}\n"
                    output += f"       Modèle: {doc_info['model']}\n"
                    if doc_info['indexed_date']:
                        output += f"       Indexé: {doc_info['indexed_date']}\n"

            return output

        except Exception as e:
            return f"{COLOR_ERROR}ERREUR: {str(e)}{RESET}"

    def run_index(self, force: bool = False, delete_missing: bool = False):
        """Lancer l'indexation"""
        self.print_header("INDEXATION CHROMA DB")

        if force:
            self.print_warning("Mode FORCE - Tous les documents seront réindexés")
        if delete_missing:
            self.print_warning("Mode DELETE-MISSING - Les documents supprimés seront supprimés de la DB")

        try:
            # Utiliser chemin absolu du script incrémental
            script_path = (self.scripts_dir / "index_incremental.py").resolve()
            cmd = [sys.executable, "-u", str(script_path)]

            if force:
                cmd.append("--force")
            if delete_missing:
                cmd.append("--delete-missing")

            # Lancer depuis le répertoire du projet pour que .env soit trouvé
            subprocess.run(cmd, cwd=str(self.cli_dir.resolve()), check=True)

            self.print_success("Indexation complétée avec succès")
            time.sleep(0.5)
            print(self.get_indexation_status())
        except subprocess.CalledProcessError as e:
            self.print_error(f"Indexation échouée: {e}")
            return False
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            return False

        return True

    def run_reset(self):
        """Réinitialiser la base de données"""
        self.print_header("REINITIALISATION - SUPPRESSION COLLECTIONS")
        response = input(f"{COLOR_WARNING}Êtes-vous sûr? (oui/non): {RESET}").strip().lower()
        if response not in ['oui', 'yes', 'y']:
            self.print_warning("Opération annulée")
            return False

        try:
            script_path = (self.cli_dir / "reset_chroma.py").resolve()
            subprocess.run(["python", str(script_path)], cwd=str(self.cli_dir.resolve()), check=True)
            self.print_success("Base de données réinitialisée")
            return True
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            return False

    def run_delete(self):
        """Supprimer physiquement la base de données"""
        self.print_header("SUPPRESSION PHYSIQUE - CHROMA DB")
        response = input(
            f"{COLOR_ERROR}ATTENTION: Cela supprimera COMPLETEMENT chroma_db_fresh/\n"
            f"Êtes-vous absolument sûr? (oui/non): {RESET}"
        ).strip().lower()

        if response not in ['oui', 'yes', 'y']:
            self.print_warning("Opération annulée")
            return False

        try:
            script_path = (self.cli_dir / "delete_chroma.py").resolve()
            subprocess.run(["python", str(script_path)], cwd=str(self.cli_dir.resolve()), check=True)
            self.print_success("Base de données supprimée")
            return True
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            return False

    def run_fix_hnsw(self):
        """Corriger corruption HNSW"""
        self.print_header("CORRECTION CORRUPTION HNSW")
        response = input(f"{COLOR_WARNING}Êtes-vous sûr? (oui/non): {RESET}").strip().lower()

        if response not in ['oui', 'yes', 'y']:
            self.print_warning("Opération annulée")
            return False

        try:
            script_path = (self.cli_dir / "fix_hnsw_chroma.py").resolve()
            subprocess.run(["python", str(script_path)], cwd=str(self.cli_dir.resolve()), check=True)
            self.print_success("Correction HNSW complétée")
            return True
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            return False

    def run_monitor(self):
        """Lancer le monitoring continu"""
        self.print_header("MONITORING CONTINU - CHROMA DB")
        self.print_info("Ctrl+C pour arrêter")

        try:
            script_path = (self.cli_dir / "monitor_indexation.py").resolve()
            subprocess.run(["python", str(script_path)], cwd=str(self.cli_dir.resolve()))
        except KeyboardInterrupt:
            self.print_info("Monitoring arrêté")
        except Exception as e:
            self.print_error(f"Erreur: {e}")

    def run_parse_pdf(self, file: str, output: str = None, index: bool = False):
        """Convertir un PDF en Markdown"""
        self.print_header("CONVERSION PDF -> MARKDOWN")

        try:
            # Convertir le PDF
            script_path = (self.scripts_dir / "parse_pdf.py").resolve()
            cmd = ["python", str(script_path), file]

            if output:
                cmd.extend(["--output", output])

            result = subprocess.run(cmd, cwd=str(self.cli_dir.resolve()), check=True)

            if result.returncode == 0:
                self.print_success("Conversion PDF complétée")

                # Si --index demandé, indexer les nouveaux markdown
                if index:
                    self.print_info("Lancement de l'indexation...")
                    time.sleep(0.5)
                    self.run_index()
            else:
                self.print_error(f"Conversion échouée (code {result.returncode})")
                return False

        except subprocess.CalledProcessError as e:
            self.print_error(f"Conversion échouée: {e}")
            return False
        except Exception as e:
            self.print_error(f"Erreur: {e}")
            return False

        return True

    def run_server_start(self):
        """Démarrer le serveur ChromaDB"""
        self.print_header("DEMARRAGE SERVEUR CHROMADB")

        if not server_manager:
            self.print_error("Gestionnaire de serveur non disponible")
            return False

        success, message = server_manager.start()
        if success:
            self.print_success(message)
        else:
            self.print_error(message)
        return success

    def run_server_stop(self):
        """Arrêter le serveur ChromaDB"""
        self.print_header("ARRET SERVEUR CHROMADB")

        if not server_manager:
            self.print_error("Gestionnaire de serveur non disponible")
            return False

        success, message = server_manager.stop()
        if success:
            self.print_success(message)
        else:
            self.print_error(message)
        return success

    def run_server_restart(self):
        """Redémarrer le serveur ChromaDB"""
        self.print_header("REDEMARRAGE SERVEUR CHROMADB")

        if not server_manager:
            self.print_error("Gestionnaire de serveur non disponible")
            return False

        success, message = server_manager.restart()
        if success:
            self.print_success(message)
        else:
            self.print_error(message)
        return success

    def run_server_status(self):
        """Afficher le statut du serveur ChromaDB"""
        if not server_manager:
            self.print_error("Gestionnaire de serveur non disponible")
            return False

        print(server_manager.get_status())
        return True

    def delete_document(self, document_name: str, skip_confirmation: bool = False):
        """Supprimer un document spécifique de la base de données"""
        self.print_header("SUPPRESSION DE DOCUMENT")

        try:
            # Use PersistentClient for contextualized database
            client = chromadb.PersistentClient(path=str(CHROMA_DB_PATH))

            collection = client.get_collection(name=COLLECTION_NAME)

            # Nettoyer le nom de document (enlever [N], (X chunks), etc.)
            import re
            document_name = document_name.strip()
            # Enlever prefix [N]
            document_name = re.sub(r'^\[\d+\]\s*', '', document_name)
            # Enlever suffix (X chunks)
            document_name = re.sub(r'\s*\(\d+\s+chunks?\).*$', '', document_name)
            # Normaliser le nom (ajouter .md si manquant)
            if not document_name.endswith('.md'):
                document_name = f"{document_name}.md"

            # Chercher tous les chunks pour ce document
            results = collection.get(
                where={"source": document_name},
                include=["metadatas"]
            )

            if not results or not results.get('ids'):
                # Essayer avec 'filename' pour compatibilité
                results = collection.get(
                    where={"filename": document_name},
                    include=["metadatas"]
                )

            if not results or not results.get('ids'):
                self.print_error(f"Document '{document_name}' non trouvé dans la base")
                return False

            chunk_ids = results['ids']
            chunk_count = len(chunk_ids)

            # Confirmation
            self.print_warning(f"Document: {document_name}")
            self.print_warning(f"Chunks à supprimer: {chunk_count}")

            if not skip_confirmation:
                response = input(f"\n{COLOR_WARNING}Confirmer la suppression? (oui/non): {RESET}").strip().lower()
                if response not in ['oui', 'yes', 'y']:
                    self.print_warning("Suppression annulée")
                    return False

            # Supprimer tous les chunks
            collection.delete(ids=chunk_ids)

            self.print_success(f"Document '{document_name}' supprimé ({chunk_count} chunks)")
            return True

        except Exception as e:
            self.print_error(f"Erreur lors de la suppression: {e}")
            return False



def main():
    cli = RagdocCLI()

    parser = argparse.ArgumentParser(
        description="RAGDOC CLI - Gestionnaire d'indexation Chroma DB",
        prog="ragdoc"
    )

    subparsers = parser.add_subparsers(dest="command", help="Commandes disponibles")

    # Command: status
    status_parser = subparsers.add_parser("status", help="Afficher statistiques d'indexation")
    status_parser.add_argument("--sort-by-date", action="store_true", help="Trier par date d'indexation")

    # Command: index
    index_parser = subparsers.add_parser("index", help="Indexation (incrémentale par défaut)")
    index_parser.add_argument("--force", action="store_true", help="Réindexer tous les documents")
    index_parser.add_argument("--delete-missing", action="store_true", help="Supprimer documents supprimés")

    # Command: monitor
    subparsers.add_parser("monitor", help="Monitoring continu")

    # Command: reset
    subparsers.add_parser("reset", help="Réinitialiser collections")

    # Command: delete
    subparsers.add_parser("delete", help="Supprimer physiquement chroma_db_fresh/")

    # Command: delete-doc
    delete_doc_parser = subparsers.add_parser("delete-doc", help="Supprimer un document spécifique")
    delete_doc_parser.add_argument("document", help="Nom du document à supprimer (avec ou sans .md)")
    delete_doc_parser.add_argument("--yes", action="store_true", help="Skip confirmation prompt")

    # Command: fix-hnsw
    subparsers.add_parser("fix-hnsw", help="Corriger corruption HNSW")

    # Command: parse-pdf
    pdf_parser = subparsers.add_parser("parse-pdf", help="Convertir PDF en Markdown")
    pdf_parser.add_argument("file", help="Chemin vers le fichier PDF")
    pdf_parser.add_argument("--output", help="Nom du fichier de sortie (optionnel)")
    pdf_parser.add_argument("--index", action="store_true", help="Indexer apres conversion")

    # Command: server (avec sous-commandes)
    server_parser = subparsers.add_parser("server", help="Gestion du serveur ChromaDB")
    server_subparsers = server_parser.add_subparsers(dest="server_command", help="Commandes serveur")

    server_subparsers.add_parser("start", help="Demarrer le serveur ChromaDB")
    server_subparsers.add_parser("stop", help="Arreter le serveur ChromaDB")
    server_subparsers.add_parser("restart", help="Redemarrer le serveur ChromaDB")
    server_subparsers.add_parser("status", help="Afficher le statut du serveur")

    args = parser.parse_args()

    # Exécuter commande
    if args.command == "status":
        cli.print_header("STATISTIQUES D'INDEXATION")
        print(cli.get_indexation_status(sort_by_date=args.sort_by_date))
    elif args.command == "index":
        cli.run_index(force=args.force, delete_missing=args.delete_missing)
    elif args.command == "monitor":
        cli.run_monitor()
    elif args.command == "reset":
        cli.run_reset()
    elif args.command == "delete":
        cli.run_delete()
    elif args.command == "delete-doc":
        cli.delete_document(args.document, skip_confirmation=args.yes)
    elif args.command == "fix-hnsw":
        cli.run_fix_hnsw()
    elif args.command == "parse-pdf":
        cli.run_parse_pdf(args.file, output=args.output, index=args.index)
    elif args.command == "server":
        if args.server_command == "start":
            cli.run_server_start()
        elif args.server_command == "stop":
            cli.run_server_stop()
        elif args.server_command == "restart":
            cli.run_server_restart()
        elif args.server_command == "status":
            cli.run_server_status()
        else:
            server_parser.print_help()
    else:
        # Afficher l'aide si aucune commande
        parser.print_help()


if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        print(f"\n{COLOR_WARNING}Interruption utilisateur{RESET}")
        sys.exit(0)
    except Exception as e:
        print(f"\n{Fore.RED}Erreur inattendue: {e}{Style.RESET_ALL}")
        sys.exit(1)
