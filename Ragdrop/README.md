# Ragdrop

Ragdrop est une petite application macOS SwiftUI qui ajoute des PDF locaux à la
base Ragdoc canonique. Elle appelle Mistral OCR, transfère le
Markdown vers le NAS, lance une seule indexation incrémentale pour le lot, puis
vérifie que chaque document est lisible dans Chroma.

## Utilisation

1. Déposer un ou plusieurs PDF dans la fenêtre; les lots sont pris en charge.
2. Cliquer sur **Ajouter à Ragdoc**.
3. Attendre la confirmation et le nombre de passages indexés.

Mistral conserve les formules, les tableaux, les figures et les numéros de page
dans le Markdown remis à Ragdoc. Le bouton corbeille retire un élément de la file
et nettoie seulement ses fichiers temporaires.

La clé Mistral doit être ajoutée dans **Ragdrop > Réglages > Mistral OCR**. Elle
est enregistrée dans le trousseau macOS. MinerU demeure sélectionnable comme
solution de secours et utilise toujours `~/.mineru_token`. L’alias SSH `rorqual`
doit donner accès à `/volume1/Services/mcp/ragdoc`.

## Développement

```bash
./script/build_and_run.sh --verify
```

L’application construite se trouve dans `dist/Ragdrop.app`.
