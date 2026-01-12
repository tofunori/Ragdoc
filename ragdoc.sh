#!/bin/bash
# Script shell pour lancer RAGDOC (Menu ou CLI) sous Linux
# Usage: ./ragdoc.sh           -> Lance le menu interactif
#        ./ragdoc.sh status    -> Lance la CLI avec arguments

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
CONDA_BASE="$HOME/miniforge3"
CONDA_ENV="ragdoc"

# Utiliser l'environnement conda ragdoc
PYTHON_EXE="$CONDA_BASE/envs/$CONDA_ENV/bin/python"

# Verifier que Python existe
if [ ! -f "$PYTHON_EXE" ]; then
    echo "Erreur: Environnement conda '$CONDA_ENV' non trouve."
    echo "Creez-le avec: conda env create -f $SCRIPT_DIR/environment.yml"
    exit 1
fi

# Si aucun argument, lancer le menu
if [ $# -eq 0 ]; then
    "$PYTHON_EXE" "$SCRIPT_DIR/ragdoc-menu.py"
else
    # Sinon utiliser la CLI avec les arguments
    "$PYTHON_EXE" "$SCRIPT_DIR/ragdoc-cli.py" "$@"
fi
