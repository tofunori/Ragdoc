#!/usr/bin/env python3
"""
Script pour ajouter automatiquement un frontmatter YAML aux fichiers markdown
convertis depuis PDF (Docling/LlamaParse).

Usage:
    python scripts/add_metadata_to_markdown.py article.md
    python scripts/add_metadata_to_markdown.py articles_markdown/*.md
    python scripts/add_metadata_to_markdown.py articles_markdown/ --all
"""

import sys
import argparse
from pathlib import Path
from metadata_extractor import extract_metadata


def has_yaml_frontmatter(content: str) -> bool:
    """Vérifie si le fichier a déjà un frontmatter YAML."""
    return content.strip().startswith('---\n')


def add_frontmatter(file_path: Path, dry_run: bool = False) -> bool:
    """
    Ajoute un frontmatter YAML à un fichier markdown.

    Args:
        file_path: Chemin vers le fichier markdown
        dry_run: Si True, affiche seulement ce qui serait fait

    Returns:
        bool: True si modifications effectuées
    """
    try:
        # Lire le fichier
        content = file_path.read_text(encoding='utf-8')

        # Vérifier s'il a déjà un frontmatter
        if has_yaml_frontmatter(content):
            print(f"[SKIP] {file_path.name} - A déjà un frontmatter YAML")
            return False

        # Extraire métadonnées
        metadata = extract_metadata(content, file_path.name)

        # Si aucune métadonnée trouvée, skip
        if not any([metadata.get('title'), metadata.get('author'), metadata.get('date')]):
            print(f"[SKIP] {file_path.name} - Aucune métadonnée détectée")
            return False

        # Construire le frontmatter
        frontmatter_lines = ['---']

        if metadata.get('title'):
            # Échapper les guillemets dans le titre
            title = metadata['title'].replace('"', '\\"')
            frontmatter_lines.append(f'title: "{title}"')

        if metadata.get('author'):
            author = metadata['author'].replace('"', '\\"')
            frontmatter_lines.append(f'author: "{author}"')

        if metadata.get('date'):
            frontmatter_lines.append(f'date: "{metadata["date"]}"')

        frontmatter_lines.append('---')
        frontmatter_lines.append('')  # Ligne vide après frontmatter

        frontmatter = '\n'.join(frontmatter_lines)

        # Afficher les métadonnées trouvées
        print(f"\n[FOUND] {file_path.name}")
        if metadata.get('title'):
            print(f"  Title:  {metadata['title'][:60]}...")
        if metadata.get('author'):
            print(f"  Author: {metadata['author']}")
        if metadata.get('date'):
            print(f"  Date:   {metadata['date']}")

        if dry_run:
            print(f"  [DRY-RUN] Pas de modification (utilisez --apply pour appliquer)")
            return True

        # Ajouter le frontmatter au début du fichier
        new_content = frontmatter + '\n' + content

        # Sauvegarder
        file_path.write_text(new_content, encoding='utf-8')
        print(f"  [SAVED] Frontmatter YAML ajouté ✓")

        return True

    except Exception as e:
        print(f"[ERROR] {file_path.name}: {e}")
        return False


def process_directory(directory: Path, dry_run: bool = False) -> tuple:
    """
    Traite tous les fichiers .md dans un répertoire.

    Returns:
        tuple: (nb_modifiés, nb_skippés)
    """
    md_files = list(directory.glob('*.md'))

    if not md_files:
        print(f"[INFO] Aucun fichier .md trouvé dans {directory}")
        return 0, 0

    print(f"\n[INFO] Trouvé {len(md_files)} fichiers markdown dans {directory}")
    print("=" * 70)

    modified = 0
    skipped = 0

    for md_file in md_files:
        if add_frontmatter(md_file, dry_run):
            modified += 1
        else:
            skipped += 1

    return modified, skipped


def main():
    parser = argparse.ArgumentParser(
        description="Ajouter automatiquement un frontmatter YAML aux fichiers markdown",
        epilog="Exemples:\n"
               "  python add_metadata_to_markdown.py article.md\n"
               "  python add_metadata_to_markdown.py article.md --apply\n"
               "  python add_metadata_to_markdown.py articles_markdown/ --all --apply\n"
               "  python add_metadata_to_markdown.py articles_markdown/*.md --apply",
        formatter_class=argparse.RawDescriptionHelpFormatter
    )

    parser.add_argument(
        'files',
        nargs='+',
        help='Fichier(s) markdown ou répertoire à traiter'
    )

    parser.add_argument(
        '--apply',
        action='store_true',
        help='Appliquer les modifications (par défaut: dry-run, affiche seulement)'
    )

    parser.add_argument(
        '--all',
        action='store_true',
        help='Traiter tous les .md dans les répertoires spécifiés'
    )

    args = parser.parse_args()

    dry_run = not args.apply

    if dry_run:
        print("\n" + "=" * 70)
        print("MODE DRY-RUN - Aucune modification ne sera appliquée")
        print("Utilisez --apply pour effectuer les modifications")
        print("=" * 70)

    total_modified = 0
    total_skipped = 0

    for file_path_str in args.files:
        file_path = Path(file_path_str)

        if file_path.is_dir():
            if not args.all:
                print(f"\n[INFO] {file_path} est un répertoire. Utilisez --all pour traiter tous les fichiers.")
                continue

            modified, skipped = process_directory(file_path, dry_run)
            total_modified += modified
            total_skipped += skipped

        elif file_path.is_file():
            if file_path.suffix.lower() != '.md':
                print(f"[SKIP] {file_path.name} - Pas un fichier markdown")
                continue

            if add_frontmatter(file_path, dry_run):
                total_modified += 1
            else:
                total_skipped += 1

        else:
            print(f"[ERROR] Fichier non trouvé: {file_path}")

    # Résumé
    print("\n" + "=" * 70)
    print("RÉSUMÉ")
    print("=" * 70)
    print(f"Fichiers avec métadonnées détectées: {total_modified}")
    print(f"Fichiers skippés:                    {total_skipped}")

    if dry_run and total_modified > 0:
        print("\n⚠️  MODE DRY-RUN - Utilisez --apply pour appliquer les modifications")
    elif total_modified > 0:
        print("\n✓ Modifications appliquées avec succès!")
        print("\nProchaine étape: Réindexez vos documents")
        print("  python scripts/index_contextualized_incremental.py")

    return 0 if total_modified > 0 or total_skipped > 0 else 1


if __name__ == "__main__":
    sys.exit(main())
