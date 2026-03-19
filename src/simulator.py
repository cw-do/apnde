"""
Module C: The Simulator — Physics Link

Computes neutron scattering contrast, generates pymatgen-compatible
composition metadata, and produces SasModels simulation-ready parameter files.
"""

import json
import logging
import math
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

from .constants import DENSITY_MAP, SLD_MAP, lookup_density, lookup_sld

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Neutron contrast calculation
# ---------------------------------------------------------------------------


def compute_contrast(matrix_sld: float, filler_sld: float) -> dict:
    """
    Compute neutron scattering contrast between matrix and filler.

    Args:
        matrix_sld: SLD of the polymer matrix in Å⁻².
        filler_sld: SLD of the filler in Å⁻².

    Returns:
        Dict with delta_rho, delta_rho_squared.
    """
    delta_rho = filler_sld - matrix_sld
    return {
        "delta_rho": delta_rho,
        "delta_rho_squared": delta_rho ** 2,
        "matrix_sld": matrix_sld,
        "filler_sld": filler_sld,
    }


# ---------------------------------------------------------------------------
# Pymatgen-compatible structure metadata
# ---------------------------------------------------------------------------


def generate_structure_metadata(sample: dict) -> dict:
    """
    Generate a pymatgen-compatible composition/metadata object for a PNC sample.

    Since polymer nanocomposites are amorphous (no crystal lattice), we generate
    a metadata record that captures composition, density, and scattering properties
    rather than an atomic-coordinate Structure. This is the standard approach for
    amorphous soft-matter systems in pymatgen workflows.

    Args:
        sample: A sample dict from the Miner output.

    Returns:
        Dict with composition metadata compatible with pymatgen serialization.
    """
    matrix_abbr = sample.get("matrix_abbreviation", "")
    filler_abbr = sample.get("filler_abbreviation", "")

    matrix_sld = lookup_sld(matrix_abbr)
    filler_sld = lookup_sld(filler_abbr)
    matrix_density = lookup_density(matrix_abbr)
    filler_density = lookup_density(filler_abbr)

    vol_frac = sample.get("filler_loading_vol_fraction")
    wt_pct = sample.get("filler_loading_wt_percent")

    # Convert wt% to volume fraction if needed
    if vol_frac is None and wt_pct is not None and matrix_density and filler_density:
        wt_frac = wt_pct / 100.0
        if wt_frac > 0 and wt_frac < 1:
            vol_frac = (wt_frac / filler_density) / (
                wt_frac / filler_density + (1 - wt_frac) / matrix_density
            )

    contrast = None
    if matrix_sld is not None and filler_sld is not None:
        contrast = compute_contrast(matrix_sld, filler_sld)

    structure = {
        "@module": "apnde.structure",
        "@class": "PNCComposition",
        "matrix": {
            "name": sample.get("matrix_name"),
            "abbreviation": matrix_abbr,
            "sld_per_angstrom_sq": matrix_sld,
            "density_g_per_cm3": matrix_density,
            "mw_kDa": sample.get("mw_kDa") or sample.get("molecular_weight_kDa"),
            "mn_kDa": sample.get("mn_kDa"),
            "pdi": sample.get("pdi"),
            "rg_angstrom": sample.get("rg_angstrom"),
        },
        "filler": {
            "name": sample.get("filler_name"),
            "abbreviation": filler_abbr,
            "sld_per_angstrom_sq": filler_sld,
            "density_g_per_cm3": filler_density,
            "particle_size_nm": sample.get("particle_size_nm"),
        },
        "loading": {
            "weight_percent": wt_pct,
            "volume_fraction": vol_frac,
        },
        "thermal": {
            "tg_celsius": sample.get("tg_celsius"),
        },
        "conformation": {
            "rg_angstrom": sample.get("rg_angstrom"),
            "rg_bulk_angstrom": sample.get("rg_bulk_angstrom"),
            "rg_over_rg_bulk": sample.get("rg_over_rg_bulk"),
            "chain_state": sample.get("chain_expansion_or_contraction"),
        },
        "neutron_contrast": contrast,
        "additional_properties": sample.get("additional_properties", {}),
    }
    return structure


# ---------------------------------------------------------------------------
# SasModels simulation parameters
# ---------------------------------------------------------------------------


def _select_sasmodel(sample: dict) -> tuple[str, dict]:
    """
    Select the appropriate SasModels model and default parameters based on
    filler geometry and available data.

    Returns:
        (model_name, parameter_dict)
    """
    filler_abbr = (sample.get("filler_abbreviation") or "").upper()
    particle_size_nm = sample.get("particle_size_nm")

    # Carbon nanotubes -> cylinder model
    if filler_abbr in ("CNT", "SWCNT", "MWCNT", "CARBON NANOTUBE"):
        radius_ang = (particle_size_nm or 5.0) * 10.0 / 2.0  # diameter -> radius in Å
        length_ang = 3000.0  # default 300 nm length
        props = sample.get("additional_properties", {})
        if "length_nm" in props:
            length_ang = float(props["length_nm"]) * 10.0
        return "cylinder", {
            "radius": radius_ang,
            "length": length_ang,
        }

    # Default: sphere model
    radius_ang = (particle_size_nm or 10.0) * 10.0 / 2.0  # nm diameter -> Å radius
    return "sphere", {"radius": radius_ang}


def generate_simulation_params(sample: dict, structure_meta: dict) -> dict:
    """
    Generate a SasModels-compatible simulation parameter file.

    Args:
        sample: Sample dict from the Miner.
        structure_meta: Structure metadata from generate_structure_metadata().

    Returns:
        Dict with model type, parameters, q-range, ready for JSON serialization.
    """
    contrast = structure_meta.get("neutron_contrast") or {}
    matrix_sld = contrast.get("matrix_sld", 1.41e-6)
    filler_sld = contrast.get("filler_sld", 3.48e-6)

    vol_frac = structure_meta.get("loading", {}).get("volume_fraction")
    if vol_frac is None:
        vol_frac = 0.01  # default dilute

    model_name, model_params = _select_sasmodel(sample)

    # SasModels uses SLD in 10⁻⁶ Å⁻² units (i.e. the value times 1e6)
    sld_filler = filler_sld * 1e6 if filler_sld < 1 else filler_sld
    sld_matrix = matrix_sld * 1e6 if matrix_sld < 1 else matrix_sld

    params = {
        "model": model_name,
        "parameters": {
            "scale": vol_frac,
            "background": 0.001,
            "sld": sld_filler,
            "sld_solvent": sld_matrix,
            **model_params,
        },
        "q_range": {
            "q_min_inv_angstrom": 0.001,
            "q_max_inv_angstrom": 0.5,
            "num_points": 200,
        },
        "metadata": {
            "matrix": sample.get("matrix_abbreviation"),
            "filler": sample.get("filler_abbreviation"),
            "volume_fraction": vol_frac,
            "delta_rho_squared": contrast.get("delta_rho_squared"),
        },
    }
    return params


def run_sasmodels_simulation(sim_params: dict) -> dict | None:
    """
    Execute a SasModels simulation and return computed I(q) data.

    Args:
        sim_params: Output from generate_simulation_params().

    Returns:
        Dict with q and I_q arrays, or None if sasmodels is unavailable.
    """
    try:
        from sasmodels.core import load_model
        from sasmodels.data import empty_data1D
        from sasmodels.direct_model import DirectModel
    except ImportError:
        logger.warning("sasmodels not available — skipping simulation")
        return None

    model_name = sim_params["model"]
    params = sim_params["parameters"]
    qr = sim_params["q_range"]

    q = np.logspace(
        np.log10(qr["q_min_inv_angstrom"]),
        np.log10(qr["q_max_inv_angstrom"]),
        qr["num_points"],
    )
    data = empty_data1D(q)

    model = load_model(model_name)
    calculator = DirectModel(data, model)

    I_q = calculator(**params)

    return {
        "q_inv_angstrom": q.tolist(),
        "I_q_per_cm": I_q.tolist(),
        "model": model_name,
    }


def generate_iq_plot(
    sim_result: dict,
    sample: dict,
    output_path: str,
) -> str | None:
    """
    Generate a publication-quality log-log I(q) vs q plot as a PNG image.

    Args:
        sim_result: Output from run_sasmodels_simulation() with q and I_q arrays.
        sample: Sample dict from the Miner (for labels).
        output_path: Full path for the output PNG file.

    Returns:
        Path to the generated PNG, or None on failure.
    """
    q = sim_result.get("q_inv_angstrom", [])
    iq = sim_result.get("I_q_per_cm", [])
    if not q or not iq:
        return None

    q_arr = np.array(q)
    iq_arr = np.array(iq)

    # Filter out zeros/negatives for log scale
    mask = (q_arr > 0) & (iq_arr > 0)
    q_arr = q_arr[mask]
    iq_arr = iq_arr[mask]
    if len(q_arr) < 2:
        return None

    matrix = sample.get("matrix_abbreviation") or sample.get("matrix_name") or "Matrix"
    filler = sample.get("filler_abbreviation") or sample.get("filler_name") or "Filler"
    model_name = sim_result.get("model", "sphere")

    fig, ax = plt.subplots(figsize=(5, 3.8), dpi=130)
    ax.loglog(q_arr, iq_arr, color="#60a5fa", linewidth=1.6, solid_capstyle="round")
    ax.set_xlabel(r"$q\;(\mathrm{\AA}^{-1})$", fontsize=11)
    ax.set_ylabel(r"$I(q)\;(\mathrm{cm}^{-1})$", fontsize=11)
    ax.set_title(f"{matrix} / {filler}  ({model_name})", fontsize=10, color="#e0e0e0")
    ax.tick_params(labelsize=9, colors="#b0b0b0")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4, color="#555")

    # Dark theme
    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#13151e")
    ax.spines["bottom"].set_color("#444")
    ax.spines["left"].set_color("#444")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.xaxis.label.set_color("#c0c0c0")
    ax.yaxis.label.set_color("#c0c0c0")

    fig.tight_layout(pad=1.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    logger.info("Generated I(q) plot: %s", output_path)
    return output_path


# ---------------------------------------------------------------------------
# Top-level simulate for a paper
# ---------------------------------------------------------------------------


def simulate_paper(
    miner_output: dict,
    output_dir: str,
    paper_stem: str,
) -> list[dict]:
    """
    Full Simulator pipeline for one paper.

    For each extracted sample:
      1. Generate pymatgen-compatible structure metadata
      2. Generate SasModels simulation parameters
      3. Run SasModels simulation
      4. Save all outputs

    Args:
        miner_output: Output dict from the Miner module.
        output_dir: Base output directory.
        paper_stem: Stem name for the paper (used in filenames).

    Returns:
        List of result dicts, one per sample.
    """
    samples = miner_output.get("samples", [])
    if not samples:
        logger.warning("No samples found for %s — skipping simulation", paper_stem)
        return []

    struct_dir = Path(output_dir) / "structures"
    sim_dir = Path(output_dir) / "simulations"
    plots_dir = Path(output_dir) / "plots"
    struct_dir.mkdir(parents=True, exist_ok=True)
    sim_dir.mkdir(parents=True, exist_ok=True)
    plots_dir.mkdir(parents=True, exist_ok=True)

    results: list[dict] = []

    for idx, sample in enumerate(samples):
        sample_id = f"{paper_stem}_sample{idx + 1}"
        logger.info("Simulating sample %s: %s + %s",
                     sample_id,
                     sample.get("matrix_abbreviation", "?"),
                     sample.get("filler_abbreviation", "?"))

        # 1. Structure metadata
        struct_meta = generate_structure_metadata(sample)
        struct_path = struct_dir / f"{sample_id}_structure.json"
        with open(struct_path, "w") as f:
            json.dump(struct_meta, f, indent=2, default=str)

        # 2. Simulation parameters
        sim_params = generate_simulation_params(sample, struct_meta)
        sim_path = sim_dir / f"{sample_id}_simulation_params.json"
        with open(sim_path, "w") as f:
            json.dump(sim_params, f, indent=2, default=str)

        # 3. Run SasModels
        sim_result = run_sasmodels_simulation(sim_params)
        plot_path: str | None = None
        if sim_result:
            sim_result_path = sim_dir / f"{sample_id}_simulated_Iq.json"
            with open(sim_result_path, "w") as f:
                json.dump(sim_result, f, indent=2)
            logger.info("Simulation complete: %s", sim_result_path)

            plot_png = str(plots_dir / f"{sample_id}_Iq.png")
            plot_path = generate_iq_plot(sim_result, sample, plot_png)

        results.append({
            "sample_id": sample_id,
            "sample": sample,
            "structure_metadata": struct_meta,
            "simulation_params": sim_params,
            "simulation_result": sim_result,
            "files": {
                "structure": str(struct_path),
                "simulation_params": str(sim_path),
                "simulated_Iq": str(sim_dir / f"{sample_id}_simulated_Iq.json") if sim_result else None,
                "iq_plot": plot_path,
            },
        })

    return results
