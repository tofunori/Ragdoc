"""
Scientific domain-specific terms and stopwords configuration.

This module contains curated lists of:
- Multi-word scientific terms to preserve as single tokens
- Stopwords that should be kept (scientifically important)
"""

# Multi-word scientific terms (preserved as single tokens)
# Example: "black carbon" -> "black_carbon"
SCIENTIFIC_COMPOUNDS = {
    # Glaciology & Cryosphere
    'black carbon',
    'mass balance',
    'ice sheet',
    'ice cap',
    'snow cover',
    'glacier albedo',
    'spectral albedo',
    'broadband albedo',
    'snow albedo',

    # Optical Properties
    'specific surface area',
    'absorption coefficient',
    'scattering coefficient',
    'extinction coefficient',
    'radiative forcing',
    'radiative transfer',
    'optical depth',
    'grain size',
    'effective radius',

    # Climate & Atmospheric
    'climate change',
    'global warming',
    'greenhouse gas',
    'greenhouse effect',
    'carbon dioxide',
    'solar radiation',
    'longwave radiation',
    'shortwave radiation',
    'radiation budget',

    # Measurement & Methods
    'remote sensing',
    'ground truth',
    'satellite imagery',
    'field measurements',
    'in situ',
    'ex situ',

    # Add more domain-specific terms as needed
}

# Stopwords to KEEP (scientifically important)
# These are normally removed but have meaning in scientific context
KEEP_STOPWORDS = {
    # Negation (critical for meaning)
    'not',
    'no',
    'nor',
    'neither',

    # Position/Direction
    'above',
    'below',
    'over',
    'under',
    'between',

    # Comparison
    'more',
    'most',
    'less',
    'least',
    'fewer',

    # Intensity
    'very',
    'much',
    'only',
    'just',

    # Modality (important for scientific claims)
    'may',
    'might',
    'must',
    'should',
    'would',
    'could',
}
