#!/usr/bin/env python3
"""
RAGDOC Menu - Interface moderne avec Rich
"""
# -*- coding: utf-8 -*-

import sys
import subprocess
import time
from pathlib import Path
from typing import Dict, List, Callable

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text
    from rich.layout import Layout
    from rich.prompt import Prompt, Confirm
    from rich import print as rprint
    import questionary
    from questionary import Choice, Separator
except ImportError:
    # Tenter de relancer avec l'environnement ragdoc-env si les dépendances manquent
    import os
    import sys
    from pathlib import Path
    
    RAGDOC_PYTHON = r"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe"
    
    if sys.executable != RAGDOC_PYTHON and Path(RAGDOC_PYTHON).exists():
        # Relancer le script avec le bon Python via subprocess pour gérer les espaces
        try:
            script_path = str(Path(__file__).resolve())
            # On reconstruit la commande: [python.exe, script.py, *args]
            cmd = [RAGDOC_PYTHON, script_path] + sys.argv[1:]
            
            # On utilise subprocess.call au lieu de execv pour éviter les soucis de parsing d'arguments Windows
            ret = subprocess.call(cmd)
            sys.exit(ret)
        except Exception as e:
            print(f"Erreur lors du relancement: {e}")
            # On continue pour afficher le message d'erreur standard

    print("Dependances manquantes. Installez avec: pip install rich questionary")
    print(f"Ou activez l'environnement: conda activate ragdoc-env")
    sys.exit(1)

# Initialiser Console
console = Console()

CLI_DIR = Path(__file__).parent
SCRIPTS_DIR = CLI_DIR / "scripts"

# Import du gestionnaire de serveur
try:
    from chromadb_server_manager import server_manager
except ImportError:
    console.print("[yellow]Avertissement: chromadb_server_manager introuvable[/yellow]")
    server_manager = None


def run_command(script_path: Path, *args, **kwargs) -> bool:
    """Lancer un script Python et afficher la progression"""
    try:
        # Utiliser directement le Python de ragdoc-env
        python_exe = r"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe"
        cmd = [python_exe, "-u", str(script_path)] + list(args)
        
        console.print(f"[dim]Exécution: {script_path.name}[/dim]")
        
        # Utiliser Popen pour streaming temps réel
        process = subprocess.Popen(
            cmd, 
            cwd=str(CLI_DIR), 
            stdout=subprocess.PIPE, 
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,  # Line buffered
            encoding='utf-8',
            errors='replace'
        )
        
        # Lire et afficher la sortie en temps réel
        for line in process.stdout:
            print(line, end='', flush=True)
            
        process.wait()
        return process.returncode == 0
    except Exception as e:
        console.print(f"[bold red]Erreur: {e}[/bold red]")
        return False


def print_header(text: str):
    """Afficher un header"""
    console.print(Panel(Text(text, justify="center", style="bold cyan"), border_style="cyan"))


def action_status():
    """Afficher les statistiques"""
    print_header("STATISTIQUES D'INDEXATION")
    python_exe = r"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe"
    cmd = [python_exe, str(CLI_DIR / "ragdoc-cli.py"), "status"]
    subprocess.run(cmd, cwd=str(CLI_DIR))
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_status_by_date():
    """Afficher les statistiques triées par date"""
    print_header("STATISTIQUES - TRI PAR DATE")
    python_exe = r"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe"
    cmd = [python_exe, str(CLI_DIR / "ragdoc-cli.py"), "status", "--sort-by-date"]
    subprocess.run(cmd, cwd=str(CLI_DIR))
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_index():
    """Indexation incrémentale"""
    print_header("INDEXATION INCREMENTALE")
    if run_command(SCRIPTS_DIR / "index_incremental.py"):
        console.print("[bold green]Indexation complétée[/bold green]")
    else:
        console.print("[bold red]Indexation échouée[/bold red]")
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_index_force():
    """Réindexation forcée"""
    print_header("REINDEXATION COMPLETE")
    if Confirm.ask("[yellow]Êtes-vous sûr de vouloir tout réindexer ?[/yellow]"):
        if run_command(SCRIPTS_DIR / "index_incremental.py", "--force"):
            console.print("[bold green]Réindexation complétée[/bold green]")
        else:
            console.print("[bold red]Réindexation échouée[/bold red]")
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_index_delete():
    """Nettoyage documents supprimés"""
    print_header("NETTOYAGE DOCUMENTS SUPPRIMES")
    console.print("[yellow]Cette option supprimera les chunks des documents absents du filesystem[/yellow]")
    if run_command(SCRIPTS_DIR / "index_incremental.py", "--delete-missing"):
        console.print("[bold green]Nettoyage complété[/bold green]")
    else:
        console.print("[bold red]Nettoyage échoué[/bold red]")
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_remove_lock():
    """Supprimer le fichier de verrouillage"""
    print_header("SUPPRESSION DU VERROU")
    lock_file = CLI_DIR / ".indexing.lock"
    if lock_file.exists():
        try:
            lock_file.unlink()
            console.print(f"[green]Verrou supprimé: {lock_file.name}[/green]")
        except Exception as e:
            console.print(f"[red]Erreur: {e}[/red]")
    else:
        console.print("[blue]Aucun fichier de verrouillage trouvé.[/blue]")
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_delete_doc():
    """Supprimer un document spécifique"""
    print_header("SUPPRESSION DE DOCUMENT")
    
    # Afficher status d'abord
    python_exe = r"C:\Users\thier\miniforge3\envs\ragdoc-env\python.exe"
    subprocess.run([python_exe, str(CLI_DIR / "ragdoc-cli.py"), "status"], cwd=str(CLI_DIR))
    
    doc_name = Prompt.ask("\n[cyan]Nom du document à supprimer[/cyan]")
    if not doc_name:
        return

    if Confirm.ask(f"[red]Confirmer la suppression de '{doc_name}' ?[/red]"):
        cmd = [python_exe, str(CLI_DIR / "ragdoc-cli.py"), "delete-doc", doc_name, "--yes"]
        subprocess.run(cmd, cwd=str(CLI_DIR))
    
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_parse_pdf():
    """Convertir PDF"""
    print_header("CONVERSION PDF -> MARKDOWN")
    pdf_path = Prompt.ask("[cyan]Chemin du PDF[/cyan]")
    if not pdf_path: return
    
    pdf_path = pdf_path.strip('"').strip("'")
    if not Path(pdf_path).exists():
        console.print(f"[red]Fichier non trouvé: {pdf_path}[/red]")
        Prompt.ask("Continuer...")
        return

    if run_command(SCRIPTS_DIR / "parse_pdf.py", pdf_path):
        console.print("[green]Conversion réussie[/green]")
    else:
        console.print("[red]Echec conversion[/red]")
    Prompt.ask("\n[bold]Appuyez sur Entrée pour continuer...[/bold]")


def action_server_start():
    if not server_manager: return
    print_header("DEMARRAGE SERVEUR")
    success, msg = server_manager.start()
    if success: console.print(f"[green]{msg}[/green]")
    else: console.print(f"[red]{msg}[/red]")
    Prompt.ask("Continuer...")


def action_server_stop():
    if not server_manager: return
    print_header("ARRET SERVEUR")
    success, msg = server_manager.stop()
    if success: console.print(f"[green]{msg}[/green]")
    else: console.print(f"[red]{msg}[/red]")
    Prompt.ask("Continuer...")


def action_server_force_stop():
    if not server_manager: return
    print_header("ARRET FORCE SERVEUR")
    success, msg = server_manager.force_kill()
    if success: console.print(f"[green]{msg}[/green]")
    else: console.print(f"[red]{msg}[/red]")
    Prompt.ask("Continuer...")


def action_server_status():
    if not server_manager: return
    print_header("STATUT SERVEUR")
    print(server_manager.get_status())
    Prompt.ask("Continuer...")


def action_monitor():
    print_header("MONITORING CONTINU")
    console.print("[dim]Ctrl+C pour arrêter[/dim]")
    try:
        run_command(Path(__file__).parent / "monitor_indexation.py")
    except KeyboardInterrupt:
        console.print("\n[yellow]Arrêté[/yellow]")
    Prompt.ask("Continuer...")


def action_reset():
    print_header("RESET BASE DE DONNEES")
    if Confirm.ask("[bold red]Êtes-vous sûr de vouloir TOUT réinitialiser ?[/bold red]"):
        run_command(Path(__file__).parent / "reset_chroma.py")
    Prompt.ask("Continuer...")


def action_delete_db():
    print_header("SUPPRESSION PHYSIQUE DB")
    if Confirm.ask("[bold red]ATTENTION: Suppression physique des fichiers. Sûr ?[/bold red]"):
        run_command(Path(__file__).parent / "delete_chroma.py")
    Prompt.ask("Continuer...")


def action_fix_hnsw():
    print_header("CORRECTION HNSW")
    if Confirm.ask("Lancer la réparation ?"):
        run_command(Path(__file__).parent / "fix_hnsw_chroma.py")
    Prompt.ask("Continuer...")


def main():
    while True:
        console.clear()
        
        # Titre
        console.print(Panel.fit(
            "[bold cyan]RAGDOC MANAGER[/bold cyan]\n[dim]Gestionnaire d'indexation et de base de données[/dim]",
            border_style="cyan",
            padding=(1, 4)
        ))
        
        # Définition du menu avec Questionary
        choices = [
            Separator("\n   📊 STATISTIQUES & MONITORING"),
            Choice("Voir statistiques", action_status),
            Choice("Voir statistiques (par date)", action_status_by_date),
            Choice("Monitoring continu", action_monitor),
            
            Separator("\n   🔄 INDEXATION"),
            Choice("Indexer (Incrémental)", action_index),
            Choice("Réindexer TOUT (--force)", action_index_force),
            Choice("Nettoyer supprimés", action_index_delete),
            Choice("Supprimer verrou (.lock)", action_remove_lock),
            
            Separator("\n   📄 DOCUMENTS"),
            Choice("Convertir PDF -> Markdown", action_parse_pdf),
            Choice("Supprimer un document", action_delete_doc),
            
            Separator("\n   🖥️ SERVEUR CHROMADB"),
            Choice("Statut Serveur", action_server_status),
            Choice("Démarrer Serveur", action_server_start),
            Choice("Arrêter Serveur", action_server_stop),
            Choice("Forcer arrêt Serveur", action_server_force_stop),
            
            Separator("\n   🛠️ MAINTENANCE"),
            Choice("Reset Collections", action_reset),
            Choice("Supprimer DB Physique", action_delete_db),
            Choice("Corriger HNSW", action_fix_hnsw),
            
            Separator("\n   🚪 QUITTER"),
            Choice("Quitter", "quit")
        ]
        
        # Style personnalisé pour Questionary
        custom_style = questionary.Style([
            ('qmark', 'fg:#00FFFF bold'),       # Token.QuestionMark
            ('question', 'bold'),               # Token.Question
            ('answer', 'fg:#00FFFF bold'),      # Token.Answer
            ('pointer', 'fg:#00FFFF bold'),     # Token.Pointer
            ('highlighted', 'fg:#00FFFF bold'), # Token.Selected
            ('selected', 'fg:#00FFFF'),         # Token.Selected
            ('separator', 'fg:#888888'),        # Token.Separator
            ('instruction', ''),                # Token.Instruction
            ('text', ''),                       # Token.Text
            ('disabled', 'fg:#858585 italic')   # Token.Disabled
        ])

        selection = questionary.select(
            "Que voulez-vous faire ?",
            choices=choices,
            style=custom_style,
            pointer=">",
            use_indicator=True,
            qmark=""
        ).ask()
        
        if selection == "quit" or selection is None:
            console.print("[cyan]Au revoir![/cyan]")
            break
            
        # Exécuter l'action
        console.clear()
        selection()

if __name__ == "__main__":
    try:
        main()
    except KeyboardInterrupt:
        console.print("\n[yellow]Interruption[/yellow]")
        sys.exit(0)
