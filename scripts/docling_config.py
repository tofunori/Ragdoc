"""
Configuration pour Docling - Parsing PDF local et gratuit avec support LaTeX.
Alternative open-source a LlamaParse sans cout API.
"""

from docling.datamodel.pipeline_options import PdfPipelineOptions, TableFormerMode
from docling.datamodel.accelerator_options import AcceleratorOptions, AcceleratorDevice

# Configurations predefinies par mode
DOCLING_CONFIGS = {
    # Mode RAPIDE (sans OCR, pour PDFs natifs)
    "fast": {
        "description": "Mode rapide sans OCR - PDFs natifs seulement",
        "pipeline_options": {
            "do_ocr": False,
            "do_table_structure": True,
            "do_formula_enrichment": False,  # Pas de formules en mode rapide
            "images_scale": 1.0,
            "generate_page_images": False,
            "generate_picture_images": False,
        },
        "cost_level": "minimal",
    },

    # Mode ECONOMY (OCR + tables, SANS formules - similaire LlamaParse)
    "economy": {
        "description": "Mode economique OCR+tables sans formules (rapide, similaire LlamaParse)",
        "pipeline_options": {
            "do_ocr": True,
            "do_table_structure": True,
            "do_formula_enrichment": False,  # ❌ DESACTIVER formules pour rapidite
            "do_code_enrichment": False,
            "images_scale": 2.0,
            "generate_page_images": False,
            "generate_picture_images": True,
            # Configuration tables
            "table_structure_options": {
                "mode": TableFormerMode.FAST,
                "do_cell_matching": True,
            },
            # GPU acceleration
            "accelerator_options": {
                "num_threads": 8,
                "device": AcceleratorDevice.AUTO,  # GPU pour OCR rapide
            },
        },
        "cost_level": "economy",
    },

    # Mode STANDARD (OCR + formules LaTeX + GPU)
    "standard": {
        "description": "Mode standard avec OCR et formules LaTeX (GPU accelere, LENT)",
        "pipeline_options": {
            "do_ocr": True,
            "do_table_structure": True,
            "do_formula_enrichment": True,  # ✨ ACTIVER extraction LaTeX (TRES LENT)
            "do_code_enrichment": False,
            "images_scale": 2.0,
            "generate_page_images": False,
            "generate_picture_images": True,
            # Configuration tables
            "table_structure_options": {
                "mode": TableFormerMode.FAST,
                "do_cell_matching": True,
            },
            # GPU acceleration
            "accelerator_options": {
                "num_threads": 8,
                "device": AcceleratorDevice.AUTO,  # Detecte GPU automatiquement
            },
        },
        "cost_level": "standard",
    },

    # Mode PREMIUM (VLM + qualite max + GPU)
    "premium": {
        "description": "Mode premium avec VLM (GPU recommande)",
        "pipeline_options": {
            "do_ocr": True,
            "do_table_structure": True,
            "do_formula_enrichment": True,  # ✨ Formules LaTeX
            "do_code_enrichment": True,
            "images_scale": 3.0,
            "generate_page_images": True,
            "generate_picture_images": True,
            # Configuration tables haute precision
            "table_structure_options": {
                "mode": TableFormerMode.ACCURATE,  # Mode precis
                "do_cell_matching": True,
            },
            # GPU acceleration
            "accelerator_options": {
                "num_threads": 8,
                "device": AcceleratorDevice.AUTO,  # Detecte GPU automatiquement
            },
        },
        "cost_level": "premium",
    },

    # Mode TEST (3 pages seulement)
    "test": {
        "description": "Mode test - 3 premieres pages avec formules",
        "pipeline_options": {
            "do_ocr": True,
            "do_table_structure": True,
            "do_formula_enrichment": True,  # ✨ Formules LaTeX
            "images_scale": 2.0,
            "generate_page_images": False,
            "generate_picture_images": False,
            # GPU acceleration
            "accelerator_options": {
                "num_threads": 8,
                "device": AcceleratorDevice.AUTO,  # Detecte GPU automatiquement
            },
        },
        "max_pages": 3,  # Limiter a 3 pages
        "cost_level": "test",
    },
}


def get_config(mode: str = "standard") -> dict:
    """
    Recuperer une configuration predéfinie pour Docling.

    Args:
        mode: 'fast', 'standard', 'premium', ou 'test'

    Returns:
        dict: Configuration pour Docling avec PdfPipelineOptions
    """
    if mode not in DOCLING_CONFIGS:
        print(f"[WARNING] Mode '{mode}' inconnu, utilisation de 'standard'")
        mode = "standard"

    config = DOCLING_CONFIGS[mode].copy()

    # Creer PdfPipelineOptions a partir du dict
    pipeline_dict = config.pop("pipeline_options", {})
    pipeline_options = PdfPipelineOptions()

    # Appliquer les options
    for key, value in pipeline_dict.items():
        if key == "table_structure_options":
            # Options speciales pour les tableaux
            pipeline_options.do_table_structure = True
            if "mode" in value:
                pipeline_options.table_structure_options.mode = value["mode"]
            if "do_cell_matching" in value:
                pipeline_options.table_structure_options.do_cell_matching = value["do_cell_matching"]
        elif key == "accelerator_options":
            # Options speciales pour GPU acceleration
            accel_opts = AcceleratorOptions()
            if "num_threads" in value:
                accel_opts.num_threads = value["num_threads"]
            if "device" in value:
                accel_opts.device = value["device"]
            pipeline_options.accelerator_options = accel_opts
        else:
            # Options standard
            setattr(pipeline_options, key, value)

    # Retourner config avec pipeline_options configure
    result = {
        "pipeline_options": pipeline_options,
        "description": config.get("description", ""),
        "cost_level": config.get("cost_level", "standard"),
    }

    # Ajouter max_pages si present
    if "max_pages" in config:
        result["max_pages"] = config["max_pages"]

    return result


def print_available_modes():
    """Afficher les modes disponibles avec leurs caracteristiques."""
    print("\n" + "="*70)
    print("MODES DOCLING DISPONIBLES (100% GRATUIT)")
    print("="*70)
    for mode_name, config in DOCLING_CONFIGS.items():
        print(f"\n[{mode_name.upper()}]")
        print(f"  Description: {config['description']}")
        print(f"  Formules:    {config['pipeline_options'].get('do_formula_enrichment', False)}")
        print(f"  OCR:         {config['pipeline_options'].get('do_ocr', False)}")
        print(f"  Tables:      {config['pipeline_options'].get('do_table_structure', False)}")
        print(f"  Cout:        Gratuit (local)")
        if "max_pages" in config:
            print(f"  Pages:       {config['max_pages']} max")
    print("\nNote: Tous les modes sont 100% gratuits et locaux (pas d'API key)")
    print()
