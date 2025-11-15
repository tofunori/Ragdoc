#!/usr/bin/env python3
"""
Gestionnaire de serveur ChromaDB pour RAGDOC
Gère le démarrage, l'arrêt et le statut du serveur ChromaDB local
"""

import sys
import os
import subprocess
import time
import signal
from pathlib import Path
from typing import Optional, Tuple

try:
    import requests
except ImportError:
    print("Erreur: requests manquant. Installez avec: pip install requests")
    sys.exit(1)

# Configuration
PROJECT_DIR = Path(__file__).parent
PID_FILE = PROJECT_DIR / ".chroma_server.pid"
LOG_FILE = PROJECT_DIR / "chroma_server.log"
SERVER_HOST = "localhost"
SERVER_PORT = 8000
STARTUP_TIMEOUT = 15  # secondes

# Configuration Python
# Détecter le chemin de Python dans ragdoc-env
RAGDOC_PYTHON = Path.home() / "miniforge3" / "envs" / "ragdoc-env" / "python.exe"
if not RAGDOC_PYTHON.exists():
    # Fallback: essayer d'autres emplacements communs
    alt_paths = [
        Path.home() / "anaconda3" / "envs" / "ragdoc-env" / "python.exe",
        Path.home() / "miniconda3" / "envs" / "ragdoc-env" / "python.exe",
    ]
    for alt_path in alt_paths:
        if alt_path.exists():
            RAGDOC_PYTHON = alt_path
            break


class ChromaDBServerManager:
    """Gestionnaire du serveur ChromaDB"""

    def __init__(self):
        self.host = SERVER_HOST
        self.port = SERVER_PORT
        self.base_url = f"http://{self.host}:{self.port}"
        self.heartbeat_url = f"{self.base_url}/api/v1/heartbeat"

    def is_running(self) -> bool:
        """Vérifie si le serveur ChromaDB répond"""
        try:
            response = requests.get(self.heartbeat_url, timeout=2)
            return response.status_code == 200
        except Exception:
            return False

    def get_pid(self) -> Optional[int]:
        """Récupère le PID du serveur depuis le fichier"""
        try:
            if PID_FILE.exists():
                pid = int(PID_FILE.read_text().strip())
                # Vérifier si le processus existe vraiment
                if sys.platform == "win32":
                    # Windows: utiliser tasklist
                    result = subprocess.run(
                        ["tasklist", "/FI", f"PID eq {pid}"],
                        capture_output=True,
                        text=True
                    )
                    if str(pid) in result.stdout:
                        return pid
                else:
                    # Unix: envoyer signal 0
                    try:
                        os.kill(pid, 0)
                        return pid
                    except OSError:
                        pass
        except Exception:
            pass
        return None

    def save_pid(self, pid: int):
        """Sauvegarde le PID dans le fichier"""
        PID_FILE.write_text(str(pid))

    def delete_pid_file(self):
        """Supprime le fichier PID"""
        try:
            if PID_FILE.exists():
                PID_FILE.unlink()
        except Exception:
            pass

    def start(self) -> Tuple[bool, str]:
        """
        Démarre le serveur ChromaDB

        Returns:
            Tuple[bool, str]: (succès, message)
        """
        # Vérifier si déjà actif
        if self.is_running():
            return True, f"[OK] Serveur ChromaDB deja actif ({self.base_url})"

        # Vérifier si un PID existe (processus zombie?)
        old_pid = self.get_pid()
        if old_pid:
            return False, f"[ERREUR] Fichier PID existe ({old_pid}) mais serveur ne repond pas. Essayez 'ragdoc server stop' d'abord."

        try:
            # Préparer la commande
            server_script = PROJECT_DIR / "run_chroma_server.py"

            if not server_script.exists():
                return False, f"[ERREUR] Script serveur introuvable: {server_script}"

            # Lancer le serveur en arrière-plan
            log_file = open(LOG_FILE, "w")

            # Préparer la commande - utiliser le Python de ragdoc-env
            cmd = [str(RAGDOC_PYTHON), str(server_script)]

            if sys.platform == "win32":
                # Windows: CREATE_NO_WINDOW
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    creationflags=subprocess.CREATE_NO_WINDOW,
                    cwd=str(PROJECT_DIR)
                )
            else:
                # Unix: détacher du terminal
                process = subprocess.Popen(
                    cmd,
                    stdout=log_file,
                    stderr=subprocess.STDOUT,
                    start_new_session=True,
                    cwd=str(PROJECT_DIR)
                )

            # Sauvegarder le PID
            self.save_pid(process.pid)

            # Attendre que le serveur réponde
            print(f"[INFO] Demarrage du serveur ChromaDB (PID: {process.pid})...")
            print(f"[INFO] Attente de la disponibilite ({STARTUP_TIMEOUT}s max)...")

            for i in range(STARTUP_TIMEOUT):
                time.sleep(1)
                if self.is_running():
                    return True, f"[OK] Serveur ChromaDB demarre avec succes!\n  URL: {self.base_url}\n  PID: {process.pid}\n  Logs: {LOG_FILE}"

                # Vérifier si le processus est mort
                if process.poll() is not None:
                    self.delete_pid_file()
                    log_content = LOG_FILE.read_text() if LOG_FILE.exists() else "Logs non disponibles"
                    return False, f"[ERREUR] Le serveur s'est arrete immediatement\nLogs:\n{log_content}"

            # Timeout
            return False, f"[WARN] Timeout: le serveur ne repond pas apres {STARTUP_TIMEOUT}s\n  Verifiez les logs: {LOG_FILE}"

        except Exception as e:
            self.delete_pid_file()
            return False, f"[ERREUR] Erreur lors du demarrage: {e}"

    def find_pid_by_port(self) -> Optional[int]:
        """
        Trouve le PID du processus utilisant le port ChromaDB
        Utile quand le fichier PID est désynchronisé
        """
        try:
            if sys.platform == "win32":
                # Windows: utiliser netstat
                result = subprocess.run(
                    ["netstat", "-ano"],
                    capture_output=True,
                    text=True
                )
                for line in result.stdout.splitlines():
                    if f":{self.port}" in line and "LISTENING" in line:
                        parts = line.split()
                        if parts:
                            try:
                                return int(parts[-1])
                            except (ValueError, IndexError):
                                continue
            else:
                # Unix: utiliser lsof
                result = subprocess.run(
                    ["lsof", "-ti", f":{self.port}"],
                    capture_output=True,
                    text=True
                )
                if result.returncode == 0 and result.stdout.strip():
                    return int(result.stdout.strip().split()[0])
        except Exception:
            pass
        return None

    def stop(self) -> Tuple[bool, str]:
        """
        Arrête le serveur ChromaDB

        Returns:
            Tuple[bool, str]: (succès, message)
        """
        pid = self.get_pid()

        # Si pas de PID valide mais serveur actif, chercher le PID via le port
        if not pid and self.is_running():
            pid = self.find_pid_by_port()
            if pid:
                print(f"[INFO] PID trouve via port {self.port}: {pid}")
                # Nettoyer l'ancien fichier PID désynchronisé
                self.delete_pid_file()
            else:
                return False, "[WARN] Serveur actif mais PID introuvable. Verifiez manuellement avec 'netstat -ano | findstr :8000'"

        if not pid:
            self.delete_pid_file()  # Nettoyer fichier PID orphelin
            return True, "[INFO] Serveur ChromaDB n'est pas actif"

        try:
            # Tenter d'arrêter le processus
            if sys.platform == "win32":
                # Windows: utiliser taskkill avec /T pour tuer l'arbre de processus complet
                subprocess.run(
                    ["taskkill", "/F", "/T", "/PID", str(pid)],
                    capture_output=True
                )
            else:
                # Unix: SIGTERM puis SIGKILL
                os.kill(pid, signal.SIGTERM)
                time.sleep(2)
                try:
                    os.kill(pid, 0)  # Vérifier si encore vivant
                    os.kill(pid, signal.SIGKILL)  # Forcer
                except OSError:
                    pass  # Déjà mort

            # Attendre un peu
            time.sleep(1)

            # Vérifier que le serveur est vraiment arrêté
            if self.is_running():
                return False, "[ERREUR] Le serveur repond toujours apres tentative d'arret"

            self.delete_pid_file()
            return True, f"[OK] Serveur ChromaDB arrete (PID: {pid})"

        except Exception as e:
            return False, f"[ERREUR] Erreur lors de l'arret: {e}"

    def restart(self) -> Tuple[bool, str]:
        """
        Redémarre le serveur ChromaDB

        Returns:
            Tuple[bool, str]: (succès, message)
        """
        # Arrêter d'abord
        success_stop, msg_stop = self.stop()
        if not success_stop and "n'est pas actif" not in msg_stop:
            return False, f"[ERREUR] Echec de l'arret:\n{msg_stop}"

        # Attendre un peu
        time.sleep(1)

        # Démarrer
        success_start, msg_start = self.start()
        if success_start:
            return True, f"[OK] Serveur ChromaDB redemarre\n{msg_start}"
        else:
            return False, f"[ERREUR] Echec du demarrage:\n{msg_start}"

    def get_status(self) -> str:
        """
        Obtient le statut détaillé du serveur

        Returns:
            str: Message de statut formaté
        """
        is_running = self.is_running()
        pid = self.get_pid()

        output = "=" * 70 + "\n"
        output += "STATUS DU SERVEUR CHROMADB\n"
        output += "=" * 70 + "\n\n"

        if is_running:
            output += "[OK] Serveur actif\n"
            output += f"  URL: {self.base_url}\n"
            if pid:
                output += f"  PID: {pid}\n"
            output += f"\n  Testez avec: curl {self.heartbeat_url}\n"
        else:
            output += "[INACTIF] Serveur inactif\n\n"
            if pid:
                output += f"  [WARN] Fichier PID trouve ({pid}) mais serveur ne repond pas\n"
                output += f"  -> Lancez 'ragdoc server stop' pour nettoyer\n"
            else:
                output += f"  -> Lancez 'ragdoc server start' pour demarrer\n"

        output += "\n" + "=" * 70

        return output


# Instance globale
server_manager = ChromaDBServerManager()


def main():
    """Test du gestionnaire"""
    import sys

    if len(sys.argv) < 2:
        print("Usage: python chromadb_server_manager.py [start|stop|status|restart]")
        sys.exit(1)

    command = sys.argv[1]
    manager = ChromaDBServerManager()

    if command == "start":
        success, msg = manager.start()
        print(msg)
        sys.exit(0 if success else 1)
    elif command == "stop":
        success, msg = manager.stop()
        print(msg)
        sys.exit(0 if success else 1)
    elif command == "restart":
        success, msg = manager.restart()
        print(msg)
        sys.exit(0 if success else 1)
    elif command == "status":
        print(manager.get_status())
        sys.exit(0)
    else:
        print(f"Commande inconnue: {command}")
        sys.exit(1)


if __name__ == "__main__":
    main()
