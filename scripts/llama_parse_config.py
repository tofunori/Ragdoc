"""
Configuration pour LlamaParse - Instructions de parsing et formatage.
Permet de garder les settings organises et faciles a modifier.
"""

# Instructions pour l'extraction du contenu
PARSING_INSTRUCTIONS = {
    "scientific": """Extract all content with focus on scientific accuracy.
For mathematical equations:
- Extract in LaTeX format with clean delimiters
- Preserve equation structure and notation
- Keep equation numbers with their equations

For tables:
- Preserve column structure and merged cells
- Extract table captions and notes

For figures:
- Extract figure captions and descriptions
""",

    "equations_only": """Focus on extracting mathematical equations and formulas.
- Use LaTeX format for all equations
- Remove extra whitespace around delimiters
- Keep equation numbers inline with equations
""",
}

# Instructions pour le formatage de la sortie
FORMATTING_INSTRUCTIONS = {
    "scientific": """Format output as clean markdown:

Equations:
- Inline math: $equation$
- Display math: $$equation$$
- Equation numbers: place immediately after $$ on same line, format as (n)
- Remove excessive whitespace between equation and number

Tables:
- Use HTML format for complex tables with merged cells
- Use markdown format for simple tables

Sections:
- Use proper markdown headers (#, ##, ###)
- Preserve section numbering from original
""",

    "compact": """Use compact formatting:
- Minimal whitespace
- Equation numbers on same line: $$equation$$ (n)
- Preserve LaTeX but remove extra spacing
""",
}

# Configurations prédéfinies par cas d'usage
PARSE_CONFIGS = {
    # Mode PREMIUM (cher mais précis) - parse_page_with_agent
    "premium": {
        "parse_mode": "parse_page_with_agent",
        "high_res_ocr": True,
        "adaptive_long_table": True,
        "outlined_table_extraction": True,
        "output_tables_as_HTML": True,
        "parsing_instruction": PARSING_INSTRUCTIONS["scientific"],
        "formatting_instruction": FORMATTING_INSTRUCTIONS["scientific"],
        "preset": "scientific",
        "description": "Qualité maximale (agentic) - ~10-30 crédits/page",
        "cost_level": "premium"
    },

    # Mode STANDARD (équilibré) - parse_page_with_llm
    "standard": {
        "parse_mode": "parse_page_with_llm",
        "high_res_ocr": True,
        "adaptive_long_table": True,
        "outlined_table_extraction": True,
        "output_tables_as_HTML": True,
        "parsing_instruction": PARSING_INSTRUCTIONS["scientific"],
        "formatting_instruction": FORMATTING_INSTRUCTIONS["compact"],
        "preset": "scientific",
        "description": "Qualité standard - 3 crédits/page",
        "cost_level": "standard"
    },

    # Mode ÉCONOMIQUE (rapide et pas cher)
    "economy": {
        "parse_mode": "parse_page_with_llm",
        "high_res_ocr": False,  # Désactiver OCR haute-res
        "adaptive_long_table": False,
        "outlined_table_extraction": True,
        "output_tables_as_HTML": False,  # Markdown simple
        "parsing_instruction": PARSING_INSTRUCTIONS["equations_only"],
        "formatting_instruction": FORMATTING_INSTRUCTIONS["compact"],
        "fast_mode": True,
        "description": "Mode économique rapide - ~1-2 crédits/page",
        "cost_level": "economy"
    },

    # Mode TEST (pour tester sans se ruiner)
    "test": {
        "parse_mode": "parse_page_with_llm",
        "high_res_ocr": False,
        "adaptive_long_table": False,
        "outlined_table_extraction": False,
        "output_tables_as_HTML": False,
        "max_pages": 3,  # Limiter à 3 pages
        "description": "Mode test (3 premières pages seulement) - minimal cost",
        "cost_level": "test"
    },
}


def get_config(mode: str = "standard") -> dict:
    """
    Récupérer une configuration prédéfinie.

    Args:
        mode: 'premium', 'standard', 'economy', ou 'test'

    Returns:
        dict: Configuration pour LlamaParse
    """
    if mode not in PARSE_CONFIGS:
        print(f"[WARNING] Mode '{mode}' inconnu, utilisation de 'standard'")
        mode = "standard"

    config = PARSE_CONFIGS[mode].copy()
    # Retirer description et cost_level (pas des params LlamaParse)
    config.pop("description", None)
    config.pop("cost_level", None)
    return config


def print_available_modes():
    """Afficher les modes disponibles avec leurs caractéristiques."""
    print("\n" + "="*70)
    print("MODES DE PARSING DISPONIBLES")
    print("="*70)
    for mode_name, config in PARSE_CONFIGS.items():
        print(f"\n[{mode_name.upper()}]")
        print(f"  Description: {config['description']}")
        print(f"  Parse mode:  {config.get('parse_mode', 'N/A')}")
        print(f"  Coût:        {config.get('cost_level', 'N/A')}")
    print()
