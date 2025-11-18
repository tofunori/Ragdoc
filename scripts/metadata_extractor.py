#!/usr/bin/env python3
"""
Module d'extraction de métadonnées depuis fichiers Markdown
Supporte YAML frontmatter et extraction depuis le contenu
"""

import re
from pathlib import Path
from typing import Dict, Optional
from datetime import datetime
import yaml


class MetadataExtractor:
    """Extrait auteur, date et autres métadonnées depuis documents markdown"""

    def __init__(self):
        # Patterns pour extraction depuis le texte
        self.author_patterns = [
            # YAML frontmatter
            r'(?:Author|Auteur)s?:\s*(.+?)(?:\n|$)',
            r'(?:By|Par)\s+(.+?)(?:\n|$)',

            # Patterns académiques (articles scientifiques)
            # Format: "Name1, Name2, and Name3" ou "Name1 & Name2"
            r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+(?:\s*,\s*[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)*(?:\s+(?:and|&|et)\s+[A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)?)\s*$',

            # Format: "Last, F., Last2, F., & Last3, F."
            r'^([A-Z][a-z]+,\s+[A-Z]\.\s*(?:,\s*[A-Z][a-z]+,\s+[A-Z]\.)*(?:,?\s*(?:&|and|et)\s+[A-Z][a-z]+,\s+[A-Z]\.)?)',

            # Format: "Author et al."
            r'^([A-Z][a-z]+(?:\s+[A-Z]\.?)?\s+[A-Z][a-z]+)\s+et\s+al\.',

            # Après un titre, ligne simple avec des noms
            r'^\*\*Authors?:\*\*\s*(.+?)(?:\n|$)',
        ]

        self.date_patterns = [
            # Formats explicites
            r'(?:Date|Published|Publication):\s*(\d{4}[-/]\d{1,2}[-/]\d{1,2})',
            r'(?:Date|Published|Publication):\s*(\d{1,2}[-/]\d{1,2}[-/]\d{4})',
            r'(?:Date|Published|Publication):\s*([A-Za-z]+\s+\d{1,2},?\s+\d{4})',
            r'(?:Date|Published|Publication):\s*(\d{4})',

            # Patterns académiques (journaux scientifiques)
            # Format: "Journal Name, Vol. 123, 2023"
            r',\s+(?:Vol\.|Volume)\s+\d+,?\s+(\d{4})',

            # Format: "Journal Name (2023)"
            r'\((\d{4})\)',

            # Format: "Journal, 2023"
            r',\s+(\d{4})\s*(?:\.|$)',

            # Année simple (1900-2099) - en dernier car moins spécifique
            r'\b((?:19|20)\d{2})\b',
        ]

    def extract_yaml_frontmatter(self, content: str) -> Optional[Dict]:
        """
        Extrait le frontmatter YAML s'il existe

        Format attendu:
        ---
        title: "Article Title"
        author: "John Doe"
        date: "2024-01-15"
        ---
        """
        yaml_pattern = r'^---\s*\n(.*?)\n---\s*\n'
        match = re.match(yaml_pattern, content, re.DOTALL)

        if match:
            try:
                yaml_content = match.group(1)
                metadata = yaml.safe_load(yaml_content)
                return metadata if isinstance(metadata, dict) else None
            except yaml.YAMLError:
                return None
        return None

    def extract_author_from_text(self, content: str, max_lines: int = 50) -> Optional[str]:
        """
        Extrait l'auteur depuis le début du texte

        Args:
            content: Contenu du document
            max_lines: Nombre de lignes à analyser (début du document)
        """
        # Analyser seulement les premières lignes
        lines = content.split('\n')[:max_lines]
        text_start = '\n'.join(lines)

        for pattern in self.author_patterns:
            match = re.search(pattern, text_start, re.MULTILINE | re.IGNORECASE)
            if match:
                author = match.group(1).strip()
                # Nettoyer (enlever markdown, etc.)
                author = re.sub(r'[#*_\[\]]', '', author)
                author = author.strip()
                if len(author) > 3 and len(author) < 200:  # Validation basique
                    return author

        return None

    def extract_date_from_text(self, content: str, max_lines: int = 50) -> Optional[str]:
        """
        Extrait la date depuis le début du texte

        Args:
            content: Contenu du document
            max_lines: Nombre de lignes à analyser

        Returns:
            Date au format ISO (YYYY-MM-DD) ou YYYY si seule l'année est trouvée
        """
        lines = content.split('\n')[:max_lines]
        text_start = '\n'.join(lines)

        for pattern in self.date_patterns:
            match = re.search(pattern, text_start, re.MULTILINE | re.IGNORECASE)
            if match:
                date_str = match.group(1).strip()

                # Tenter de normaliser au format ISO
                normalized = self._normalize_date(date_str)
                if normalized:
                    return normalized

        return None

    def extract_date_from_filename(self, filename: str) -> Optional[str]:
        """
        Extrait la date depuis le nom du fichier

        Examples:
            "2024_paper.md" -> "2024"
            "Warren_1982.md" -> "1982"
            "2024-01-15_article.md" -> "2024-01-15"
        """
        # Pattern pour date dans filename
        date_patterns = [
            r'(\d{4}[-_]\d{2}[-_]\d{2})',  # YYYY-MM-DD ou YYYY_MM_DD
            r'\b((?:19|20)\d{2})\b',        # Année seule
        ]

        for pattern in date_patterns:
            match = re.search(pattern, filename)
            if match:
                date_str = match.group(1).replace('_', '-')
                normalized = self._normalize_date(date_str)
                if normalized:
                    return normalized

        return None

    def _normalize_date(self, date_str: str) -> Optional[str]:
        """
        Normalise une date au format ISO (YYYY-MM-DD) ou YYYY

        Supporte:
        - YYYY-MM-DD, YYYY/MM/DD
        - DD-MM-YYYY, DD/MM/YYYY
        - "January 15, 2024", "Jan 15, 2024"
        - YYYY seul
        """
        date_str = date_str.strip()

        # Année seule (4 chiffres)
        if re.match(r'^\d{4}$', date_str):
            year = int(date_str)
            if 1900 <= year <= 2100:
                return date_str

        # Format ISO: YYYY-MM-DD ou YYYY/MM/DD
        match = re.match(r'^(\d{4})[-/](\d{1,2})[-/](\d{1,2})$', date_str)
        if match:
            year, month, day = match.groups()
            try:
                # Validation
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Format: DD-MM-YYYY ou DD/MM/YYYY
        match = re.match(r'^(\d{1,2})[-/](\d{1,2})[-/](\d{4})$', date_str)
        if match:
            day, month, year = match.groups()
            try:
                dt = datetime(int(year), int(month), int(day))
                return dt.strftime('%Y-%m-%d')
            except ValueError:
                pass

        # Format texte: "January 15, 2024", "Jan 15, 2024"
        months = {
            'january': 1, 'jan': 1, 'février': 2, 'february': 2, 'feb': 2,
            'march': 3, 'mar': 3, 'mars': 3, 'april': 4, 'apr': 4, 'avril': 4,
            'may': 5, 'mai': 5, 'june': 6, 'jun': 6, 'juin': 6,
            'july': 7, 'jul': 7, 'juillet': 7, 'august': 8, 'aug': 8, 'août': 8,
            'september': 9, 'sep': 9, 'sept': 9, 'septembre': 9,
            'october': 10, 'oct': 10, 'octobre': 10, 'november': 11, 'nov': 11, 'novembre': 11,
            'december': 12, 'dec': 12, 'décembre': 12
        }

        match = re.match(r'^([A-Za-zéû]+)\s+(\d{1,2}),?\s+(\d{4})$', date_str, re.IGNORECASE)
        if match:
            month_name, day, year = match.groups()
            month_num = months.get(month_name.lower())
            if month_num:
                try:
                    dt = datetime(int(year), month_num, int(day))
                    return dt.strftime('%Y-%m-%d')
                except ValueError:
                    pass

        return None

    def extract_title(self, content: str) -> Optional[str]:
        """
        Extrait le titre du document

        Cherche:
        1. Frontmatter YAML (title field)
        2. Premier titre markdown (#)
        3. Première ligne non-vide
        """
        # Essayer frontmatter
        frontmatter = self.extract_yaml_frontmatter(content)
        if frontmatter and 'title' in frontmatter:
            return str(frontmatter['title']).strip()

        # Chercher premier titre markdown
        lines = content.split('\n')
        for line in lines[:20]:  # Analyser premières lignes
            line = line.strip()
            if line.startswith('# '):
                title = line.lstrip('#').strip()
                return title

        # Fallback: première ligne non-vide
        for line in lines[:10]:
            line = line.strip()
            if line and len(line) > 5:
                # Nettoyer
                title = re.sub(r'[#*_]', '', line).strip()
                if len(title) < 200:
                    return title

        return None

    def extract_all_metadata(
        self,
        content: str,
        filename: str,
        fallback_title: Optional[str] = None
    ) -> Dict[str, Optional[str]]:
        """
        Extrait toutes les métadonnées disponibles

        Args:
            content: Contenu du document
            filename: Nom du fichier (pour extraction date)
            fallback_title: Titre par défaut si aucun trouvé

        Returns:
            Dict avec 'author', 'date', 'title'
        """
        metadata = {}

        # 1. Essayer frontmatter YAML d'abord (priorité)
        frontmatter = self.extract_yaml_frontmatter(content)
        if frontmatter:
            metadata['author'] = frontmatter.get('author') or frontmatter.get('authors')
            metadata['date'] = frontmatter.get('date') or frontmatter.get('published')
            metadata['title'] = frontmatter.get('title')
        else:
            metadata['author'] = None
            metadata['date'] = None
            metadata['title'] = None

        # 2. Si pas trouvé dans frontmatter, chercher dans le texte
        if not metadata['author']:
            metadata['author'] = self.extract_author_from_text(content)

        if not metadata['date']:
            # Essayer d'abord le contenu, puis le nom de fichier
            metadata['date'] = self.extract_date_from_text(content)
            if not metadata['date']:
                metadata['date'] = self.extract_date_from_filename(filename)

        if not metadata['title']:
            metadata['title'] = self.extract_title(content)

        # 3. Normaliser la date si présente
        if metadata['date']:
            normalized = self._normalize_date(str(metadata['date']))
            metadata['date'] = normalized

        # 4. Fallback pour le titre
        if not metadata['title'] and fallback_title:
            metadata['title'] = fallback_title

        # 5. Nettoyer les chaînes
        for key in ['author', 'title']:
            if metadata.get(key):
                # Convertir en string et nettoyer
                metadata[key] = str(metadata[key]).strip()
                # Limiter la longueur
                if len(metadata[key]) > 500:
                    metadata[key] = metadata[key][:500] + '...'

        return metadata


# Fonction utilitaire pour usage simple
def extract_metadata(content: str, filename: str) -> Dict[str, Optional[str]]:
    """
    Fonction helper pour extraction simple

    Usage:
        metadata = extract_metadata(file_content, "article.md")
        print(metadata['author'])  # "John Doe"
        print(metadata['date'])    # "2024-01-15"
        print(metadata['title'])   # "My Article"
    """
    extractor = MetadataExtractor()
    return extractor.extract_all_metadata(content, filename)


if __name__ == "__main__":
    # Tests
    print("Testing MetadataExtractor...")

    # Test 1: YAML frontmatter
    test_yaml = """---
title: "Glacier Dynamics Study"
author: "Dr. Jane Smith"
date: "2024-01-15"
---

# Introduction

This is a test document.
"""

    metadata = extract_metadata(test_yaml, "test.md")
    print("\nTest 1 (YAML frontmatter):")
    print(f"  Author: {metadata['author']}")
    print(f"  Date: {metadata['date']}")
    print(f"  Title: {metadata['title']}")

    # Test 2: Extraction depuis texte
    test_text = """# Impact of Black Carbon on Glacier Albedo

Author: Warren et al.
Date: 1982

This study examines...
"""

    metadata = extract_metadata(test_text, "Warren_1982.md")
    print("\nTest 2 (Text extraction):")
    print(f"  Author: {metadata['author']}")
    print(f"  Date: {metadata['date']}")
    print(f"  Title: {metadata['title']}")

    # Test 3: Extraction depuis filename
    test_simple = """# Some Article

This is content without metadata.
"""

    metadata = extract_metadata(test_simple, "2009_RSE_Painter.md")
    print("\nTest 3 (Filename extraction):")
    print(f"  Author: {metadata['author']}")
    print(f"  Date: {metadata['date']}")
    print(f"  Title: {metadata['title']}")

    print("\n✓ Tests completed!")
