@echo off
REM Script batch pour lancer RAGDOC (Menu ou CLI)
REM Historical launcher; adjust the paths for your own checkout.
REM Utilise l'environnement conda ragdoc-env

REM Si aucun argument, lancer le menu
if "%1"=="" (
    conda run -n ragdoc-env python "C:\path\to\Ragdoc\ragdoc-menu.py"
) else (
    REM Sinon utiliser la CLI avec les arguments
    conda run -n ragdoc-env python "C:\path\to\Ragdoc\ragdoc-cli.py" %*
)
