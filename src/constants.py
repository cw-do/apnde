"""
Physical constants and SLD reference data for polymer nanocomposite systems.
"""

# Scattering Length Densities (SLD) in Å⁻²
SLD_MAP: dict[str, float] = {
    # Polymers
    "h-PS": 1.41e-6,
    "PS": 1.41e-6,
    "polystyrene": 1.41e-6,
    "d-PS": 6.47e-6,
    "dPS": 6.47e-6,
    "h-P2VP": 1.95e-6,
    "P2VP": 1.95e-6,
    "poly(2-vinylpyridine)": 1.95e-6,
    "PEO": 0.64e-6,
    "polyethylene oxide": 0.64e-6,
    "d-PEO": 7.14e-6,
    "dPEO": 7.14e-6,
    "PMMA": 1.07e-6,
    "poly(methyl methacrylate)": 1.07e-6,
    "PVME": 0.35e-6,
    "PEP": -0.32e-6,
    "PI": 0.27e-6,
    "polyisoprene": 0.27e-6,
    "PDMS": 0.06e-6,
    # Fillers
    "SiO2": 3.48e-6,
    "silica": 3.48e-6,
    "Au": 4.50e-6,
    "gold": 4.50e-6,
    "CNT": 5.71e-6,
    "SWCNT": 5.71e-6,
    "MWCNT": 5.71e-6,
    "carbon nanotube": 5.71e-6,
    "graphene oxide": 4.09e-6,
    "GO": 4.09e-6,
    "POSS": 1.18e-6,
}

# Material densities in g/cm³
DENSITY_MAP: dict[str, float] = {
    "h-PS": 1.05,
    "PS": 1.05,
    "polystyrene": 1.05,
    "h-P2VP": 1.12,
    "P2VP": 1.12,
    "PEO": 1.21,
    "PMMA": 1.18,
    "SiO2": 2.20,
    "silica": 2.20,
    "Au": 19.32,
    "gold": 19.32,
    "CNT": 1.40,
    "SWCNT": 1.40,
    "MWCNT": 1.80,
    "graphene oxide": 1.80,
    "GO": 1.80,
    "POSS": 1.13,
}


def lookup_sld(name: str | None) -> float | None:
    """Look up SLD for a material by name or abbreviation (case-insensitive)."""
    if not name:
        return None
    key = name.strip()
    if key in SLD_MAP:
        return SLD_MAP[key]
    key_lower = key.lower()
    for k, v in SLD_MAP.items():
        if k.lower() == key_lower:
            return v
    return None


def lookup_density(name: str | None) -> float | None:
    """Look up density for a material by name or abbreviation (case-insensitive)."""
    if not name:
        return None
    key = name.strip()
    if key in DENSITY_MAP:
        return DENSITY_MAP[key]
    key_lower = key.lower()
    for k, v in DENSITY_MAP.items():
        if k.lower() == key_lower:
            return v
    return None
