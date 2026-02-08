import os
import sys
import subprocess
import time
import socket
from pathlib import Path

# Determine paths
CURRENT_DIR = Path(__file__).parent.resolve()
PID_FILE = CURRENT_DIR / ".chroma_server.pid"
LOG_FILE = CURRENT_DIR / "chroma_server.log"

# Try to import config for DB path
try:
    sys.path.insert(0, str(CURRENT_DIR / "src"))
    from config import ACTIVE_DB_PATH
except ImportError:
    # Fallback if config cannot be imported
    ACTIVE_DB_PATH = CURRENT_DIR / "chroma_db_new"

class ChromaDBServerManager:
    def __init__(self):
        self.pid_file = PID_FILE
        self.log_file = LOG_FILE
        self.host = "localhost"
        self.port = 8000
        self.db_path = ACTIVE_DB_PATH

    def _get_python_executable(self):
        # Use current python executable
        return sys.executable

    def _get_chroma_command(self):
        """Find the chroma executable or module command"""
        python_exe = Path(self._get_python_executable())

        if os.name == 'nt':
            # Windows: Look for Scripts/chroma.exe
            # This is critical for conda/venv environments on Windows
            chroma_exe = python_exe.parent / "Scripts" / "chroma.exe"
            if chroma_exe.exists():
                return [str(chroma_exe)]

        # Linux/Mac or fallback
        bin_chroma = python_exe.parent / "bin" / "chroma"
        if bin_chroma.exists():
            return [str(bin_chroma)]

        # Last resort fallback
        return [str(python_exe), "-m", "chromadb.cli.cli"]

    def _is_port_open(self):
        """Check if port is listening (IPv4 and IPv6)"""
        # Try IPv4 127.0.0.1
        try:
            with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
                s.settimeout(1)
                if s.connect_ex(("127.0.0.1", self.port)) == 0:
                    return True
        except Exception:
            pass

        # Try IPv6 ::1 (loopback)
        try:
            with socket.socket(socket.AF_INET6, socket.SOCK_STREAM) as s6:
                s6.settimeout(1)
                if s6.connect_ex(("::1", self.port, 0, 0)) == 0:
                    return True
        except Exception:
            pass

        return False

    def _get_processes_on_port(self):
        """Return list of PIDs currently using the configured port"""
        pids = set()

        if os.name == 'nt':
            try:
                output = subprocess.check_output(["netstat", "-ano"], encoding="utf-8", errors="ignore")
                for line in output.splitlines():
                    if f":{self.port}" not in line:
                        continue
                    parts = line.split()
                    if len(parts) < 5:
                        continue
                    local_addr = parts[1]
                    pid_str = parts[-1]
                    if (local_addr.endswith(f":{self.port}") or local_addr.endswith(f".{self.port}")) and pid_str.isdigit():
                        pid = int(pid_str)
                        if pid != 0:
                            pids.add(pid)
            except Exception:
                return []
        else:
            try:
                output = subprocess.check_output(["lsof", "-ti", f":{self.port}"], text=True)
                for pid_str in output.splitlines():
                    pid_str = pid_str.strip()
                    if pid_str.isdigit():
                        pids.add(int(pid_str))
            except Exception:
                return []

        return sorted(pids)

    def _is_process_running(self, pid):
        """Check if process with PID exists"""
        if os.name == 'nt':
            # Windows
            try:
                # tasklist /FI "PID eq 1234"
                cmd = ["tasklist", "/FI", f"PID eq {pid}", "/NH"]
                # Create no window
                startupinfo = subprocess.STARTUPINFO()
                startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                
                output = subprocess.check_output(
                    cmd, 
                    stderr=subprocess.STDOUT,
                    startupinfo=startupinfo,
                    encoding='oem' # Windows default
                )
                return str(pid) in output
            except Exception:
                return False
        else:
            # POSIX
            try:
                os.kill(pid, 0)
                return True
            except OSError:
                return False

    def start(self):
        """Start the ChromaDB server"""
        # 1. Check PID file
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                if self._is_process_running(pid):
                    return False, f"Serveur déjà en cours (PID {pid})"
                else:
                    # Stale PID file
                    self.pid_file.unlink()
            except Exception:
                if self.pid_file.exists(): self.pid_file.unlink()

        # 2. Check Port
        if self._is_port_open():
            pids = self._get_processes_on_port()
            if pids:
                pid_list = ", ".join(str(pid) for pid in pids)
                return False, f"Port {self.port} occupé par PID(s) {pid_list}. Utilisez 'Arrêter Serveur' ou 'Forcer arrêt'."
            return False, f"Port {self.port} déjà utilisé (autre instance ?)"

        # 3. Prepare command
        base_cmd = self._get_chroma_command()
        # Ensure db path is absolute string
        db_path_str = str(self.db_path.resolve()) if self.db_path.exists() else str(self.db_path)
        
        cmd = base_cmd + ["run", "--path", db_path_str, "--port", str(self.port)]
        
        # 4. Run process
        try:
            with open(self.log_file, "w") as log:
                # Windows specific setup to hide console window
                startupinfo = None
                creationflags = 0

                if os.name == 'nt':
                    # Use STARTUPINFO to hide the window instead of CREATE_NO_WINDOW
                    # This allows the process to have stdout/stderr but hides the window
                    startupinfo = subprocess.STARTUPINFO()
                    startupinfo.dwFlags |= subprocess.STARTF_USESHOWWINDOW
                    startupinfo.wShowWindow = subprocess.SW_HIDE

                    # Flags to detach the process
                    DETACHED_PROCESS = 0x00000008
                    CREATE_NEW_PROCESS_GROUP = 0x00000200
                    CREATE_BREAKAWAY_FROM_JOB = 0x01000000
                    creationflags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP | CREATE_BREAKAWAY_FROM_JOB

                try:
                    process = subprocess.Popen(
                        cmd,
                        stdin=subprocess.DEVNULL,
                        stdout=log,
                        stderr=log,
                        creationflags=creationflags,
                        startupinfo=startupinfo,
                        close_fds=True if os.name != 'nt' else False
                    )
                except OSError as e:
                    # If breakaway is not allowed (error 1314), retry without the flag
                    if os.name == 'nt' and getattr(e, "winerror", None) == 1314:
                        DETACHED_PROCESS = 0x00000008
                        CREATE_NEW_PROCESS_GROUP = 0x00000200
                        fallback_flags = DETACHED_PROCESS | CREATE_NEW_PROCESS_GROUP
                        process = subprocess.Popen(
                            cmd,
                            stdin=subprocess.DEVNULL,
                            stdout=log,
                            stderr=log,
                            creationflags=fallback_flags,
                            startupinfo=startupinfo,
                            close_fds=True if os.name != 'nt' else False
                        )
                    else:
                        raise
            
            self.pid_file.write_text(str(process.pid))
            
            # Wait for server to be ready (up to 15 seconds)
            max_retries = 30
            for i in range(max_retries):
                if process.poll() is not None:
                    return False, "Le serveur s'est arrêté immédiatement. Voir chroma_server.log."
                
                if self._is_port_open():
                    return True, f"Serveur démarré avec succès (PID {process.pid})"
                
                time.sleep(0.5)
            
            # If we get here, process is running but port not yet open. 
            # It might be initializing or slow.
            return True, f"Processus lancé (PID {process.pid}), mais le port n'est pas encore accessible. Vérifiez le statut dans quelques secondes."
        except Exception as e:
            return False, f"Erreur de démarrage: {str(e)}"

    def stop(self):
        """Stop the ChromaDB server"""
        if not self.pid_file.exists():
            return False, "Aucun fichier PID trouvé"

        try:
            pid = int(self.pid_file.read_text().strip())
        except Exception:
            return False, "Fichier PID invalide"

        try:
            if os.name == 'nt':
                # Kill process tree on Windows
                subprocess.run(
                    ["taskkill", "/F", "/PID", str(pid), "/T"], 
                    capture_output=True,
                    creationflags=subprocess.CREATE_NO_WINDOW
                )
            else:
                os.kill(pid, 15) # SIGTERM
                time.sleep(1)
                try:
                    os.kill(pid, 9) # SIGKILL
                except OSError:
                    pass
            
            if self.pid_file.exists():
                self.pid_file.unlink()
            return True, "Serveur arrêté"
        except Exception as e:
            return False, f"Erreur lors de l'arrêt: {str(e)}"

    def force_kill(self):
        """Force kill any process listening on the configured port"""
        pids = self._get_processes_on_port()
        if not pids:
            return False, f"Aucun processus détecté sur le port {self.port}."

        errors = []
        for pid in pids:
            try:
                if os.name == 'nt':
                    subprocess.run(
                        ["taskkill", "/F", "/PID", str(pid)],
                        capture_output=True,
                        creationflags=subprocess.CREATE_NO_WINDOW
                    )
                else:
                    os.kill(pid, 15)
            except Exception as e:
                errors.append(f"PID {pid}: {e}")

        if self.pid_file.exists():
            try:
                self.pid_file.unlink()
            except Exception:
                pass

        if errors:
            return False, " ; ".join(errors)
        return True, f"Processus sur le port {self.port} terminés ({', '.join(str(pid) for pid in pids)})"

    def get_status(self):
        """Get formatted status string"""
        status_lines = []
        
        # Check PID
        pid_status = "[red]❌ PID File absent[/red]"
        if self.pid_file.exists():
            try:
                pid = int(self.pid_file.read_text().strip())
                if self._is_process_running(pid):
                    pid_status = f"[green]✅ Processus actif (PID {pid})[/green]"
                else:
                    pid_status = f"[yellow]⚠️ Fichier PID présent mais processus inactif (PID {pid})[/yellow]"
            except Exception:
                pid_status = "[yellow]⚠️ Fichier PID invalide[/yellow]"
        status_lines.append(pid_status)
        
        # Check Port
        if self._is_port_open():
             status_lines.append(f"[green]✅ Port {self.port} ouvert (Ecoute)[/green]")
        else:
             status_lines.append(f"[red]❌ Port {self.port} fermé[/red]")
             
        return "\n".join(status_lines)

# Export singleton
server_manager = ChromaDBServerManager()

