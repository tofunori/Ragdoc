@echo off
REM Script batch pour lancer RAGDOC (Menu ou CLI)
REM A copier dans C:\Users\thier\bin\ ou un dossier du PATH
REM Utilise l'environnement conda ragdoc-env

REM Si aucun argument, lancer le menu
if "%1"=="" (
    conda run -n ragdoc-env python "D:\Claude Code\ragdoc-mcp\ragdoc-menu.py"
) else (
    REM Sinon utiliser la CLI avec les arguments
    conda run -n ragdoc-env python "D:\Claude Code\ragdoc-mcp\ragdoc-cli.py" %*
)
