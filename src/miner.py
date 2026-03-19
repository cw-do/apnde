"""
Module A: The Miner — NLP & Metadata Extraction

Extracts polymer/filler compositions, molecular weights, particle sizes,
Tg values, and other PNC-relevant properties from scientific PDFs using
PyMuPDF for text extraction and GPT-4o via OpenRouter for structured
data extraction.
"""

import json
import logging
import os
import re
from pathlib import Path

import pymupdf
from openai import OpenAI

logger = logging.getLogger(__name__)

EXTRACTION_SYSTEM_PROMPT = """You are a materials science data extraction expert specializing in polymer nanocomposites (PNCs) and small-angle scattering.

Your task is to extract ALL polymer nanocomposite samples, their compositions, AND all physical/structural properties reported in a scientific paper. Be exhaustive — extract every quantitative value reported for each sample.

For each distinct sample or composition found in the paper, extract these fields:

=== COMPOSITION ===
1. **matrix_name**: Full chemical name of the polymer matrix (e.g., "polystyrene")
2. **matrix_abbreviation**: Common abbreviation (e.g., "PS", "P2VP", "PEO", "dPS" for deuterated)
3. **filler_name**: Full chemical name of the filler (e.g., "silicon dioxide", "single-wall carbon nanotubes")
4. **filler_abbreviation**: Common abbreviation (e.g., "SiO2", "SWCNT", "MWCNT", "GO")
5. **filler_loading_wt_percent**: Weight percent loading as a number, or null
6. **filler_loading_vol_fraction**: Volume fraction (0-1), or null

=== MOLECULAR PROPERTIES ===
7. **mw_kDa**: Weight-average molecular weight Mw in kDa, or null
8. **mn_kDa**: Number-average molecular weight Mn in kDa, or null
9. **pdi**: Polydispersity index Mw/Mn, or null
10. **particle_size_nm**: Filler particle diameter in nm, or null

=== POLYMER CHAIN CONFORMATION (critical for SANS) ===
11. **rg_angstrom**: Radius of gyration Rg in Angstroms. Look in Guinier fits, Debye model fits, Zimm plots, Kratky plots, and tables. Convert nm to Å (multiply by 10). This is a key parameter — search carefully.
12. **rg_bulk_angstrom**: Rg of the neat/bulk polymer (no filler) in Angstroms, if reported as reference
13. **rg_over_rg_bulk**: Ratio Rg/Rg_bulk or Rg/Rg_0 (dimensionless), if reported
14. **rg_over_r_particle**: Ratio Rg/R where R is the filler particle radius, if reported
15. **chain_expansion_or_contraction**: "expanded", "contracted", "unchanged", or null — relative to bulk

=== SCATTERING-DERIVED STRUCTURAL PARAMETERS ===
16. **correlation_length_angstrom**: Correlation length ξ (xi) from Ornstein-Zernike or similar fits, in Å
17. **d_spacing_angstrom**: d-spacing or characteristic length from peak position (d = 2π/q*), in Å
18. **interparticle_distance_nm**: Center-to-center distance between filler particles, in nm
19. **porod_exponent**: Power-law exponent from Porod region (high-q slope on log-log plot)
20. **guinier_rg_angstrom**: Rg specifically from Guinier analysis if reported separately from other Rg values
21. **forward_scattering_I0**: I(q→0) or I(0) value in cm⁻¹

=== SURFACE / INTERFACE PROPERTIES ===
22. **grafting_density_chains_per_nm2**: Grafting density σ in chains/nm², for grafted systems
23. **brush_height_nm**: Polymer brush height/thickness in nm
24. **bound_layer_thickness_nm**: Bound polymer layer thickness around particles, in nm

=== THERMAL & MECHANICAL PROPERTIES ===
25. **tg_celsius**: Glass transition temperature in °C
26. **elastic_modulus_mpa**: Young's modulus or elastic modulus in MPa
27. **viscosity_pa_s**: Viscosity in Pa·s (at specified shear rate if available)
28. **viscosity_ratio**: η/η_0 ratio (nanocomposite viscosity / neat polymer viscosity)

=== CATCH-ALL ===
29. **additional_properties**: Dict of any OTHER reported quantitative properties not covered above. Include: specific surface area, chi parameter, Flory-Huggins interaction, tube diameter, entanglement Mw, diffusion coefficients, relaxation times, etc. Use descriptive keys with units in the key name.

=== EXTRACTION RULES ===
- Create one sample entry per distinct composition/loading. If same matrix+filler has 3 different loadings → 3 entries.
- For deuterated polymers: use "dPS", "d-P2VP" etc. in the abbreviation.
- If the paper is theoretical/computational, extract model parameters as-is.
- Convert ALL units to the specified standard units above.
- Use null for values not reported. Do NOT guess or estimate values.
- If a property has a range, report as {"value": midpoint, "range": [low, high]}.
- Extract from text, tables, figure captions, and supplementary information.
- For Rg: check Guinier plots, Debye fits, Kratky plots, Zimm analysis, RPA fits, and any table that lists Rg values. Often reported as Rg(Å) or Rg(nm) — always convert to Angstroms.
- For scattering parameters: look for fit results, model parameters, and values quoted in text.

Return ONLY a valid JSON object with this structure:
{
  "paper_title": "...",
  "doi": "..." or null,
  "paper_type": "experimental" | "theoretical" | "review",
  "samples": [
    {
      "matrix_name": "...",
      "matrix_abbreviation": "...",
      "filler_name": "...",
      "filler_abbreviation": "...",
      "filler_loading_wt_percent": number or null,
      "filler_loading_vol_fraction": number or null,
      "mw_kDa": number or null,
      "mn_kDa": number or null,
      "pdi": number or null,
      "particle_size_nm": number or null,
      "rg_angstrom": number or null,
      "rg_bulk_angstrom": number or null,
      "rg_over_rg_bulk": number or null,
      "rg_over_r_particle": number or null,
      "chain_expansion_or_contraction": "expanded" | "contracted" | "unchanged" | null,
      "correlation_length_angstrom": number or null,
      "d_spacing_angstrom": number or null,
      "interparticle_distance_nm": number or null,
      "porod_exponent": number or null,
      "guinier_rg_angstrom": number or null,
      "forward_scattering_I0": number or null,
      "grafting_density_chains_per_nm2": number or null,
      "brush_height_nm": number or null,
      "bound_layer_thickness_nm": number or null,
      "tg_celsius": number or null,
      "elastic_modulus_mpa": number or null,
      "viscosity_pa_s": number or null,
      "viscosity_ratio": number or null,
      "additional_properties": {}
    }
  ],
  "scattering_techniques": ["SANS", "SAXS", ...] or [],
  "instruments": ["list of instruments/facilities mentioned"] or [],
  "q_range_inv_angstrom": [q_min, q_max] or null,
  "scattering_models_used": ["Debye", "Guinier", "Ornstein-Zernike", ...] or [],
  "notes": "any important context about the measurements or analysis"
}"""


def extract_text_from_pdf(pdf_path: str) -> str:
    """Extract full text from a PDF using PyMuPDF."""
    doc = pymupdf.open(pdf_path)
    text_parts: list[str] = []
    for page in doc:
        page_text = page.get_text("text")
        if isinstance(page_text, str):
            text_parts.append(page_text)
    doc.close()
    full_text = "\n".join(text_parts)
    logger.info("Extracted %d characters from %s", len(full_text), pdf_path)
    return full_text


def _clean_json_response(content: str) -> str:
    """Strip markdown fences and trailing commas from LLM JSON response."""
    text = content.strip()
    if text.startswith("```"):
        text = re.sub(r"^```(?:json)?\s*", "", text)
        text = re.sub(r"\s*```$", "", text)
    # Remove trailing commas before } or ]
    text = re.sub(r",\s*([}\]])", r"\1", text)
    return text


def extract_pnc_data(paper_text: str, client: OpenAI, model: str) -> dict:
    """
    Use GPT-4o to extract structured PNC data from paper text.

    Args:
        paper_text: Full text of the scientific paper.
        client: OpenAI-compatible client configured for OpenRouter.
        model: Model identifier (e.g. "openai/gpt-4o").

    Returns:
        Parsed dict of extracted PNC data.
    """
    # Truncate if extremely long (GPT-4o context is 128k tokens)
    max_chars = 120_000
    if len(paper_text) > max_chars:
        logger.warning("Paper text truncated from %d to %d chars", len(paper_text), max_chars)
        paper_text = paper_text[:max_chars]

    response = client.chat.completions.create(
        model=model,
        messages=[
            {"role": "system", "content": EXTRACTION_SYSTEM_PROMPT},
            {"role": "user", "content": f"Extract all polymer nanocomposite data from this paper:\n\n{paper_text}"},
        ],
        temperature=0.15,
        max_tokens=4096,
    )

    raw = response.choices[0].message.content or ""
    cleaned = _clean_json_response(raw)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError as exc:
        logger.error("Failed to parse LLM JSON response: %s\nRaw: %s", exc, raw[:500])
        return {
            "paper_title": "PARSE_ERROR",
            "samples": [],
            "scattering_techniques": [],
            "instruments": [],
            "notes": f"JSON parse error: {exc}. Raw response saved.",
            "_raw_response": raw,
        }


def mine_pdf(pdf_path: str, client: OpenAI, model: str) -> dict:
    """
    Full Miner pipeline for a single PDF.

    1. Extract text from PDF
    2. Send to LLM for structured extraction
    3. Return parsed data dict

    Args:
        pdf_path: Path to the PDF file.
        client: OpenAI-compatible client.
        model: Model name.

    Returns:
        Dict with extracted PNC data.
    """
    logger.info("Mining PDF: %s", pdf_path)
    text = extract_text_from_pdf(pdf_path)

    if len(text.strip()) < 100:
        logger.warning("Very short text extracted from %s — PDF may be image-only", pdf_path)
        return {
            "paper_title": Path(pdf_path).stem,
            "samples": [],
            "scattering_techniques": [],
            "instruments": [],
            "notes": "Insufficient text extracted from PDF.",
        }

    data = extract_pnc_data(text, client, model)
    data["source_pdf"] = Path(pdf_path).name
    return data
