"""
Main Pipeline Orchestrator — APNDE

Processes all PDFs in the input directory through:
  Module A (Miner)      -> Structured PNC data extraction
  Module B (Visualizer) -> SANS/SAXS plot digitization
  Module C (Simulator)  -> Neutron contrast + SasModels simulation
Then generates a summary index.html.
"""

import json
import logging
import os
import sys
import time
from pathlib import Path

import pandas as pd
from dotenv import load_dotenv
from openai import OpenAI

from .miner import mine_pdf
from .visualizer import visualize_pdf
from .simulator import simulate_paper

logger = logging.getLogger("apnde")


def setup_logging() -> None:
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(name)s] %(levelname)s: %(message)s",
        handlers=[logging.StreamHandler(sys.stdout)],
    )


def create_client() -> tuple[OpenAI, str]:
    """Create OpenAI-compatible client from .env configuration."""
    load_dotenv()
    api_key = os.getenv("OPENROUTER_API_KEY")
    base_url = os.getenv("OPENROUTER_BASE_URL", "https://openrouter.ai/api/v1")
    model = os.getenv("MODEL_NAME", "openai/gpt-4o")

    if not api_key:
        raise RuntimeError("OPENROUTER_API_KEY not set in .env")

    client = OpenAI(api_key=api_key, base_url=base_url)
    logger.info("OpenRouter client created (model=%s)", model)
    return client, model


def save_digitized_csvs(digitized_plots: list[dict], data_dir: str, paper_stem: str) -> list[str]:
    """Save digitized scattering data as CSV files."""
    csv_paths: list[str] = []
    data_path = Path(data_dir)
    data_path.mkdir(parents=True, exist_ok=True)

    for plot_idx, plot in enumerate(digitized_plots):
        curves = plot.get("curves", [])
        for curve_idx, curve in enumerate(curves):
            data_points = curve.get("data", [])
            if not data_points:
                continue

            label = curve.get("label", f"curve{curve_idx + 1}")
            # Sanitize label for filename
            safe_label = "".join(c if c.isalnum() or c in "-_" else "_" for c in str(label))
            fname = f"{paper_stem}_plot{plot_idx + 1}_{safe_label}.csv"
            fpath = data_path / fname

            df = pd.DataFrame(data_points)
            df.to_csv(fpath, index=False)
            csv_paths.append(str(fpath))
            logger.info("Saved %d data points to %s", len(data_points), fpath)

    return csv_paths


def process_single_pdf(
    pdf_path: str,
    output_dir: str,
    client: OpenAI,
    model: str,
) -> dict:
    """
    Process a single PDF through the full APNDE pipeline.

    Returns:
        Summary dict with all results from all modules.
    """
    paper_stem = Path(pdf_path).stem
    logger.info("=" * 70)
    logger.info("PROCESSING: %s", paper_stem)
    logger.info("=" * 70)

    result: dict = {
        "source_pdf": Path(pdf_path).name,
        "paper_stem": paper_stem,
    }

    # --- Module A: Miner ---
    t0 = time.time()
    logger.info("[Module A] Mining PNC data...")
    miner_output = mine_pdf(pdf_path, client, model)
    result["miner"] = miner_output
    result["miner_time_sec"] = round(time.time() - t0, 1)
    logger.info(
        "[Module A] Extracted %d samples in %.1fs",
        len(miner_output.get("samples", [])),
        result["miner_time_sec"],
    )

    # Save miner output
    miner_path = Path(output_dir) / f"{paper_stem}_miner.json"
    with open(miner_path, "w") as f:
        json.dump(miner_output, f, indent=2, default=str)

    # --- Module B: Visualizer ---
    t0 = time.time()
    logger.info("[Module B] Extracting and digitizing scattering plots...")
    figures_dir = str(Path(output_dir) / "figures" / paper_stem)
    digitized_plots = visualize_pdf(pdf_path, figures_dir, client, model)
    result["visualizer"] = {
        "num_plots_found": len(digitized_plots),
        "plots": digitized_plots,
    }
    result["visualizer_time_sec"] = round(time.time() - t0, 1)
    logger.info(
        "[Module B] Found %d scattering plots in %.1fs",
        len(digitized_plots),
        result["visualizer_time_sec"],
    )

    # Save digitized data as CSV
    csv_paths = save_digitized_csvs(
        digitized_plots, str(Path(output_dir) / "data"), paper_stem
    )
    result["csv_files"] = csv_paths

    # Save digitized plot JSON
    if digitized_plots:
        vis_path = Path(output_dir) / f"{paper_stem}_digitized.json"
        with open(vis_path, "w") as f:
            json.dump(digitized_plots, f, indent=2, default=str)

    # --- Module C: Simulator ---
    t0 = time.time()
    logger.info("[Module C] Running simulations...")
    sim_results = simulate_paper(miner_output, output_dir, paper_stem)
    result["simulator"] = {
        "num_samples_simulated": len(sim_results),
        "results": sim_results,
    }
    result["simulator_time_sec"] = round(time.time() - t0, 1)
    logger.info(
        "[Module C] Simulated %d samples in %.1fs",
        len(sim_results),
        result["simulator_time_sec"],
    )

    # Save full result
    full_path = Path(output_dir) / f"{paper_stem}_full_result.json"
    with open(full_path, "w") as f:
        json.dump(result, f, indent=2, default=str)

    return result


def _build_digitized_plots_section(all_results: list[dict]) -> str:
    items: list[str] = []
    for r in all_results:
        title = r.get("miner", {}).get("paper_title", r.get("paper_stem", ""))
        plots = r.get("visualizer", {}).get("plots", [])
        for pidx, plot in enumerate(plots):
            dig_img = plot.get("digitized_plot_image", "")
            if not dig_img:
                continue
            ident = plot.get("identification", {})
            ptype = ident.get("plot_type", "scattering")
            n_curves = len([c for c in plot.get("curves", []) if c.get("data")])
            n_pts = sum(len(c.get("data", [])) for c in plot.get("curves", []))
            safe_title = title.replace("'", "\\'")
            caption = f"{ptype} &middot; {n_curves} curves &middot; {n_pts} pts"
            items.append(
                f'<div class="gallery-item">'
                f'<img src="{dig_img}" onclick="showModal(this.src, \'{safe_title}\')" '
                f'alt="{ptype} plot">'
                f'<div class="gallery-caption">{caption}</div></div>'
            )
    if not items:
        return '<p style="color:#9ca3af;padding:1rem;">No digitized data with extractable points was found.</p>'
    return '<div class="gallery">' + "".join(items) + "</div>"


def _build_simulations_section(all_results: list[dict]) -> str:
    rows: list[str] = []
    for r in all_results:
        for sim in r.get("simulator", {}).get("results", []):
            sid = sim.get("sample_id", "?")
            sp = sim.get("simulation_params", {})
            model = sp.get("model", "?")
            params = sp.get("parameters", {})
            contrast = sim.get("structure_metadata", {}).get("neutron_contrast") or {}
            dr2 = contrast.get("delta_rho_squared")
            dr2_str = f"{dr2:.2e}" if dr2 else "-"
            radius_val = params.get("radius", params.get("length", "?"))
            radius_str = f"{float(radius_val):.0f}" if isinstance(radius_val, (int, float)) else str(radius_val)
            plot_file = sim.get("files", {}).get("iq_plot")
            plot_cell = ""
            if plot_file:
                rel = Path(plot_file).name
                plot_cell = (
                    f'<img src="plots/{rel}" class="iq-thumb" '
                    f'onclick="showModal(this.src,\'{sid}\')">'
                )
            rows.append(
                f"<tr><td>{sid}</td><td>{model}</td>"
                f"<td>{radius_str} &Aring;</td>"
                f"<td>{params.get('scale', 0):.4f}</td>"
                f"<td>{dr2_str}</td>"
                f"<td class='plot-cell'>{plot_cell}</td></tr>"
            )
    if not rows:
        return '<p style="color:#9ca3af;padding:1rem;">No simulations were generated.</p>'
    return (
        '<table><thead><tr>'
        '<th>Sample</th><th>Model</th><th>Radius</th><th>Vol. Frac.</th>'
        '<th>(&Delta;&rho;)&sup2;</th><th>Simulated I(q)</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


def _build_papers_section(all_results: list[dict]) -> str:
    rows: list[str] = []
    for r in all_results:
        paper = r.get("miner", {})
        title = paper.get("paper_title", r.get("paper_stem", "?"))
        n_s = len(paper.get("samples", []))
        n_p = r.get("visualizer", {}).get("num_plots_found", 0)
        tech = ", ".join(paper.get("scattering_techniques", [])) or "none"
        err = r.get("error")
        status = f'<span style="color:#f87171">{err}</span>' if err else '<span style="color:#4ade80">OK</span>'
        rows.append(
            f"<tr><td>{title}</td><td>{n_s}</td><td>{n_p}</td><td>{tech}</td><td>{status}</td></tr>"
        )
    return (
        '<table><thead><tr>'
        '<th>Title</th><th>Samples</th><th>Plots</th><th>Techniques</th><th>Status</th>'
        '</tr></thead><tbody>' + "".join(rows) + '</tbody></table>'
    )


_EXTRA_FIELDS = [
    ("tg_celsius", "Tg", "°C"),
    ("correlation_length_angstrom", "ξ", "Å"),
    ("d_spacing_angstrom", "d", "Å"),
    ("interparticle_distance_nm", "d_pp", "nm"),
    ("porod_exponent", "Porod n", ""),
    ("forward_scattering_I0", "I(0)", "cm⁻¹"),
    ("grafting_density_chains_per_nm2", "σ", "ch/nm²"),
    ("brush_height_nm", "h_brush", "nm"),
    ("bound_layer_thickness_nm", "t_bound", "nm"),
    ("elastic_modulus_mpa", "E", "MPa"),
    ("viscosity_ratio", "η/η₀", ""),
    ("mn_kDa", "Mn", "kDa"),
    ("pdi", "PDI", ""),
]


def _build_extra_props_cell(sample: dict) -> str:
    tags: list[str] = []
    for key, label, unit in _EXTRA_FIELDS:
        val = sample.get(key)
        if val is None:
            continue
        if isinstance(val, dict):
            val = val.get("value", val)
        unit_str = f" {unit}" if unit else ""
        tags.append(f'<span class="prop-tag">{label}: {val}{unit_str}</span>')

    ap = sample.get("additional_properties", {})
    for k, v in ap.items():
        if k in ("Rg", "rg_angstrom", "Rg_angstrom"):
            continue
        if isinstance(v, (int, float)):
            tags.append(f'<span class="prop-tag">{k}: {v}</span>')
        elif isinstance(v, str) and len(v) < 40:
            tags.append(f'<span class="prop-tag">{k}: {v}</span>')

    return " ".join(tags) if tags else "-"


def generate_index_html(all_results: list[dict], output_dir: str) -> str:
    html_path = Path(output_dir) / "index.html"

    total_samples = sum(len(r.get("miner", {}).get("samples", [])) for r in all_results)
    total_plots = sum(r.get("visualizer", {}).get("num_plots_found", 0) for r in all_results)
    total_sims = sum(r.get("simulator", {}).get("num_samples_simulated", 0) for r in all_results)
    total_csvs = sum(len(r.get("csv_files", [])) for r in all_results)

    papers_detail = _build_papers_section(all_results)
    plots_detail = _build_digitized_plots_section(all_results)
    sims_detail = _build_simulations_section(all_results)

    paper_cards: list[str] = []
    for r in all_results:
        paper = r.get("miner", {})
        title = paper.get("paper_title", r.get("paper_stem", "Unknown"))
        n_samples = len(paper.get("samples", []))
        techniques = ", ".join(paper.get("scattering_techniques", []))
        n_plots = r.get("visualizer", {}).get("num_plots_found", 0)

        digitized = r.get("visualizer", {}).get("plots", [])

        samples_html = ""
        for idx, s in enumerate(paper.get("samples", []), 1):
            matrix = s.get("matrix_abbreviation") or s.get("matrix_name") or "?"
            filler = s.get("filler_abbreviation") or s.get("filler_name") or "?"
            loading = ""
            if s.get("filler_loading_wt_percent") is not None:
                loading = f"{s['filler_loading_wt_percent']} wt%"
            elif s.get("filler_loading_vol_fraction") is not None:
                loading = f"{s['filler_loading_vol_fraction']*100:.1f} vol%"
            mw_val = s.get("mw_kDa") or s.get("molecular_weight_kDa")
            mw = f"{mw_val} kDa" if mw_val else "-"
            size = f"{s['particle_size_nm']} nm" if s.get("particle_size_nm") else "-"

            rg_val = s.get("rg_angstrom") or s.get("guinier_rg_angstrom")
            if not rg_val:
                ap = s.get("additional_properties", {})
                rg_val = ap.get("Rg") or ap.get("rg_angstrom") or ap.get("Rg_angstrom")
            rg = f"{rg_val} &Aring;" if rg_val else "-"

            conformation = s.get("chain_expansion_or_contraction") or ""
            conf_badge = ""
            if conformation == "expanded":
                conf_badge = '<span class="conf-badge conf-exp">expanded</span>'
            elif conformation == "contracted":
                conf_badge = '<span class="conf-badge conf-con">contracted</span>'
            elif conformation == "unchanged":
                conf_badge = '<span class="conf-badge conf-unc">unchanged</span>'

            rg_ratio = s.get("rg_over_rg_bulk")
            rg_ratio_str = f"{rg_ratio:.3f}" if rg_ratio else "-"

            contrast_str = "-"
            sim_results = r.get("simulator", {}).get("results", [])
            if idx - 1 < len(sim_results):
                contrast_data = sim_results[idx - 1].get("structure_metadata", {}).get("neutron_contrast")
                if contrast_data and contrast_data.get("delta_rho_squared"):
                    contrast_str = f"{contrast_data['delta_rho_squared']:.2e}"

            extra_cells = _build_extra_props_cell(s)

            samples_html += (
                f"<tr><td>{idx}</td><td>{matrix}</td><td>{filler}</td>"
                f"<td>{loading or '-'}</td><td>{mw}</td><td>{size}</td>"
                f"<td>{rg}</td><td>{rg_ratio_str} {conf_badge}</td>"
                f"<td>{contrast_str}</td><td class='extra-cell'>{extra_cells}</td></tr>"
            )

        exp_plots_html = ""
        if digitized:
            for pidx, plot in enumerate(digitized):
                dig_img = plot.get("digitized_plot_image", "")
                ident = plot.get("identification", {})
                ptype = ident.get("plot_type", "scattering")
                curves = plot.get("curves", [])
                n_pts = sum(len(c.get("data", [])) for c in curves)
                page = plot.get("source_page", "?")

                if not dig_img and n_pts == 0:
                    continue

                safe_title = title.replace("'", "\\'")
                img_tag = ""
                if dig_img:
                    img_tag = (
                        f'<img src="{dig_img}" class="exp-thumb" '
                        f'onclick="showModal(this.src,\'{safe_title} — page {page}\')" '
                        f'alt="{ptype}">'
                    )

                curve_tags: list[str] = []
                for c in curves:
                    cdata = c.get("data", [])
                    if not cdata:
                        continue
                    clabel = c.get("label", "data")
                    q_vals = [str(p.get("q", "")) for p in cdata[:6]]
                    iq_vals = [str(p.get("I_q", "")) for p in cdata[:6]]
                    q_preview = ", ".join(q_vals) + ("…" if len(cdata) > 6 else "")
                    iq_preview = ", ".join(iq_vals) + ("…" if len(cdata) > 6 else "")
                    curve_tags.append(
                        f'<div class="curve-block">'
                        f'<span class="curve-label">{clabel}</span> '
                        f'<span class="curve-pts">({len(cdata)} pts)</span>'
                        f'<div class="curve-preview">q: {q_preview}<br>I(q): {iq_preview}</div>'
                        f'</div>'
                    )

                curves_html = "".join(curve_tags) if curve_tags else '<span class="no-data">digitized — no data extracted</span>'

                exp_plots_html += (
                    f'<div class="exp-plot-row">'
                    f'<div class="exp-plot-img">{img_tag}</div>'
                    f'<div class="exp-plot-info">'
                    f'<div class="exp-plot-meta">{ptype} &middot; page {page} &middot; {len(curves)} curve(s) &middot; {n_pts} pts</div>'
                    f'{curves_html}'
                    f'</div></div>'
                )

        exp_section = ""
        if exp_plots_html:
            exp_section = (
                f'<div class="exp-section">'
                f'<div class="exp-section-title">Experimental Scattering Data ({n_plots} plots identified)</div>'
                f'{exp_plots_html}</div>'
            )
        elif techniques:
            exp_section = (
                f'<div class="exp-section">'
                f'<div class="exp-section-title">Scattering Data</div>'
                f'<p class="no-data">Scattering plots identified but digitization failed for all images.</p>'
                f'</div>'
            )

        csv_links = ""
        for csv_path in r.get("csv_files", []):
            fname = Path(csv_path).name
            csv_links += f'<a href="data/{fname}" class="csv-link">{fname}</a> '

        paper_cards.append(f"""
    <div class="paper-card">
      <h2>{title}</h2>
      <div class="meta">
        <span class="badge">{r.get('miner', {}).get('paper_type', 'unknown')}</span>
        <span class="badge">{n_samples} samples</span>
        <span class="badge">{n_plots} scattering plots</span>
        <span class="badge">{techniques or 'no scattering'}</span>
      </div>
      <table>
        <thead><tr>
          <th>#</th><th>Matrix</th><th>Filler</th><th>Loading</th>
          <th>Mw</th><th>Size</th><th>Rg</th><th>Rg/Rg<sub>0</sub></th>
          <th>(&Delta;&rho;)&sup2;</th><th>Properties</th>
        </tr></thead>
        <tbody>{samples_html}</tbody>
      </table>
      {exp_section}
      <div class="files">
        <strong>CSV data:</strong> {csv_links or 'None'}
        <br>
        <a href="{r.get('paper_stem', '')}_miner.json">Miner JSON</a> |
        <a href="{r.get('paper_stem', '')}_full_result.json">Full Result</a>
      </div>
    </div>""")

    html = f"""<!DOCTYPE html>
<html lang="en">
<head>
<meta charset="UTF-8">
<meta name="viewport" content="width=device-width, initial-scale=1.0">
<title>APNDE — Polymer Nanocomposite Discovery Results</title>
<style>
  *{{margin:0;padding:0;box-sizing:border-box}}
  body{{font-family:'Segoe UI',system-ui,sans-serif;background:#0f1117;color:#e0e0e0;padding:2rem}}
  h1{{color:#60a5fa;margin-bottom:.5rem;font-size:1.8rem}}
  .subtitle{{color:#9ca3af;margin-bottom:2rem}}

  .stats{{display:flex;gap:1.2rem;margin-bottom:.5rem;flex-wrap:wrap}}
  .stat-card{{background:#1e2130;border:1px solid #2d3148;border-radius:8px;padding:1rem 1.5rem;
    cursor:pointer;transition:border-color .15s,box-shadow .15s;user-select:none}}
  .stat-card:hover,.stat-card.active{{border-color:#60a5fa;box-shadow:0 0 12px rgba(96,165,250,.15)}}
  .stat-card .num{{font-size:2rem;font-weight:bold;color:#60a5fa}}
  .stat-card .label{{color:#9ca3af;font-size:.85rem}}
  .stat-card .label::after{{content:' ▸';font-size:.7rem}}
  .stat-card.active .label::after{{content:' ▾'}}

  .detail-panel{{display:none;background:#181b28;border:1px solid #2d3148;border-radius:8px;
    padding:1.2rem;margin-bottom:1.5rem;max-height:70vh;overflow-y:auto}}
  .detail-panel.active{{display:block}}
  .detail-panel table{{width:100%;font-size:.85rem}}

  .paper-card{{background:#1e2130;border:1px solid #2d3148;border-radius:8px;padding:1.5rem;margin-bottom:1.5rem}}
  .paper-card h2{{color:#f0f0f0;font-size:1.15rem;margin-bottom:.75rem}}
  .meta{{display:flex;gap:.5rem;margin-bottom:1rem;flex-wrap:wrap}}
  .badge{{background:#2d3148;color:#a5b4fc;padding:.2rem .6rem;border-radius:4px;font-size:.8rem}}
  table{{width:100%;border-collapse:collapse;margin-bottom:1rem;font-size:.88rem}}
  th{{background:#262a3d;color:#a5b4fc;padding:.5rem;text-align:left;border-bottom:2px solid #3b4260}}
  td{{padding:.45rem .5rem;border-bottom:1px solid #2d3148}}
  tr:hover td{{background:#262a3d}}
  .files{{font-size:.85rem;color:#9ca3af}}
  .files a,.csv-link{{color:#60a5fa;text-decoration:none}}
  .files a:hover,.csv-link:hover{{text-decoration:underline}}

  .exp-section{{background:#161926;border:1px solid #252840;border-radius:6px;padding:1rem;margin-bottom:1rem}}
  .exp-section-title{{color:#a5b4fc;font-size:.9rem;font-weight:600;margin-bottom:.75rem}}
  .exp-plot-row{{display:flex;gap:1rem;margin-bottom:1rem;padding-bottom:.75rem;border-bottom:1px solid #1e2130}}
  .exp-plot-row:last-child{{border-bottom:none;margin-bottom:0;padding-bottom:0}}
  .exp-plot-img{{flex:0 0 140px}}
  .exp-thumb{{width:140px;height:105px;object-fit:cover;border-radius:4px;border:1px solid #3b4260;
    cursor:pointer;transition:transform .15s,border-color .15s}}
  .exp-thumb:hover{{transform:scale(1.04);border-color:#60a5fa}}
  .exp-plot-info{{flex:1;min-width:0}}
  .exp-plot-meta{{color:#9ca3af;font-size:.8rem;margin-bottom:.4rem}}
  .curve-block{{background:#1e2130;border-radius:4px;padding:.4rem .6rem;margin-bottom:.35rem;font-size:.8rem}}
  .curve-label{{color:#a5b4fc;font-weight:600}}
  .curve-pts{{color:#6b7280}}
  .curve-preview{{color:#9ca3af;font-size:.75rem;font-family:'Courier New',monospace;margin-top:.2rem;
    white-space:nowrap;overflow:hidden;text-overflow:ellipsis}}
  .no-data{{color:#6b7280;font-size:.8rem;font-style:italic}}

  .gallery{{display:flex;flex-wrap:wrap;gap:.8rem}}
  .gallery-item{{width:160px}}
  .gallery-item img{{width:160px;height:120px;object-fit:cover;border-radius:4px;border:1px solid #3b4260;
    cursor:pointer;transition:transform .15s,border-color .15s}}
  .gallery-item img:hover{{transform:scale(1.04);border-color:#60a5fa}}
  .gallery-caption{{color:#9ca3af;font-size:.72rem;margin-top:.25rem;text-align:center}}

  .plot-cell{{text-align:center;padding:.2rem}}
  .iq-thumb{{width:80px;height:60px;object-fit:cover;border-radius:4px;
    border:1px solid #3b4260;cursor:pointer;transition:transform .15s,border-color .15s}}
  .iq-thumb:hover{{transform:scale(1.08);border-color:#60a5fa}}

  .modal-overlay{{display:none;position:fixed;inset:0;background:rgba(0,0,0,.82);
    z-index:1000;align-items:center;justify-content:center}}
  .modal-overlay.active{{display:flex}}
  .modal-box{{position:relative;max-width:860px;width:92vw;background:#1e2130;
    border:1px solid #3b4260;border-radius:10px;padding:1rem;box-shadow:0 12px 40px rgba(0,0,0,.6)}}
  .modal-box img{{width:100%;border-radius:6px}}
  .modal-title{{color:#a5b4fc;font-size:.95rem;margin-bottom:.6rem;text-align:center}}
  .modal-close{{position:absolute;top:.5rem;right:.8rem;background:none;border:none;
    color:#9ca3af;font-size:1.5rem;cursor:pointer;line-height:1}}
  .modal-close:hover{{color:#f0f0f0}}
  .extra-cell{{max-width:280px}}
  .prop-tag{{display:inline-block;background:#262a3d;color:#d4d4d8;padding:.15rem .45rem;
    border-radius:3px;font-size:.72rem;margin:.1rem .15rem .1rem 0;white-space:nowrap}}
  .conf-badge{{display:inline-block;padding:.1rem .4rem;border-radius:3px;font-size:.7rem;font-weight:600;vertical-align:middle}}
  .conf-exp{{background:#064e3b;color:#6ee7b7}}
  .conf-con{{background:#7f1d1d;color:#fca5a5}}
  .conf-unc{{background:#3b3b1f;color:#fde68a}}
  footer{{margin-top:3rem;color:#6b7280;font-size:.8rem;text-align:center}}
</style>
</head>
<body>
<div style="display:flex;justify-content:space-between;align-items:center;flex-wrap:wrap;gap:.5rem">
  <div>
    <h1>APNDE &mdash; Autonomous Polymer Nanocomposite Discovery Engine</h1>
    <p class="subtitle">Automated extraction of PNC data from scientific literature</p>
  </div>
  <a href="framework.html" style="background:#262a3d;color:#a5b4fc;padding:.5rem 1rem;border-radius:6px;
    font-size:.85rem;text-decoration:none;border:1px solid #3b4260;white-space:nowrap;
    transition:border-color .15s,background .15s"
    onmouseover="this.style.borderColor='#60a5fa';this.style.background='#2d3148'"
    onmouseout="this.style.borderColor='#3b4260';this.style.background='#262a3d'">
    Framework &amp; Architecture &rarr;</a>
</div>

<div class="stats">
  <div class="stat-card" onclick="togglePanel('panelPapers',this)">
    <div class="num">{len(all_results)}</div><div class="label">Papers Processed</div></div>
  <div class="stat-card" onclick="togglePanel('panelSamples',this)">
    <div class="num">{total_samples}</div><div class="label">Samples Extracted</div></div>
  <div class="stat-card" onclick="togglePanel('panelPlots',this)">
    <div class="num">{total_plots}</div><div class="label">Scattering Plots Digitized</div></div>
  <div class="stat-card" onclick="togglePanel('panelSims',this)">
    <div class="num">{total_sims}</div><div class="label">Simulations Generated</div></div>
  <div class="stat-card" style="cursor:default">
    <div class="num">{total_csvs}</div><div class="label" style="cursor:default">CSV Files Saved</div></div>
</div>

<div class="detail-panel" id="panelPapers">{papers_detail}</div>
<div class="detail-panel" id="panelSamples">
  <p style="color:#9ca3af;margin-bottom:.5rem">Scroll down to see samples per paper.</p></div>
<div class="detail-panel" id="panelPlots">{plots_detail}</div>
<div class="detail-panel" id="panelSims">{sims_detail}</div>

{''.join(paper_cards)}

<div class="modal-overlay" id="imgModal" onclick="closeModal(event)">
  <div class="modal-box" onclick="event.stopPropagation()">
    <button class="modal-close" onclick="closeModal()">&times;</button>
    <div class="modal-title" id="modalTitle"></div>
    <img id="modalImg" src="" alt="plot">
  </div>
</div>

<footer>
  Generated by APNDE | <a href="framework.html">Framework &amp; Architecture</a> | FOA DE-FOA-0003612 Focus Area 12-A
</footer>

<script>
function togglePanel(id, card) {{
  var p = document.getElementById(id);
  var wasActive = p.classList.contains('active');
  document.querySelectorAll('.detail-panel').forEach(function(el){{el.classList.remove('active')}});
  document.querySelectorAll('.stat-card').forEach(function(el){{el.classList.remove('active')}});
  if (!wasActive) {{ p.classList.add('active'); card.classList.add('active'); }}
}}
function showModal(src, title) {{
  document.getElementById('modalImg').src = src;
  document.getElementById('modalTitle').textContent = title;
  document.getElementById('imgModal').classList.add('active');
}}
function closeModal(e) {{
  if (!e || e.target === document.getElementById('imgModal'))
    document.getElementById('imgModal').classList.remove('active');
}}
document.addEventListener('keydown', function(e) {{
  if (e.key === 'Escape') document.getElementById('imgModal').classList.remove('active');
}});
</script>
</body>
</html>"""

    with open(html_path, "w") as f:
        f.write(html)

    logger.info("Generated index.html at %s", html_path)
    return str(html_path)


def run_pipeline(pdf_dir: str = "./pdf", output_dir: str = "./docs") -> None:
    """
    Run the full APNDE pipeline on all PDFs in the input directory.
    """
    setup_logging()

    pdf_path = Path(pdf_dir)
    out_path = Path(output_dir)
    out_path.mkdir(parents=True, exist_ok=True)
    (out_path / "data").mkdir(exist_ok=True)
    (out_path / "figures").mkdir(exist_ok=True)
    (out_path / "structures").mkdir(exist_ok=True)
    (out_path / "simulations").mkdir(exist_ok=True)
    (out_path / "plots").mkdir(exist_ok=True)

    pdfs = sorted(pdf_path.glob("*.pdf"))
    if not pdfs:
        logger.error("No PDF files found in %s", pdf_dir)
        return

    logger.info("Found %d PDFs to process", len(pdfs))

    client, model = create_client()

    all_results: list[dict] = []
    for pdf in pdfs:
        try:
            result = process_single_pdf(str(pdf), str(out_path), client, model)
            all_results.append(result)
        except Exception as exc:
            logger.error("Failed to process %s: %s", pdf.name, exc, exc_info=True)
            all_results.append({
                "source_pdf": pdf.name,
                "paper_stem": pdf.stem,
                "error": str(exc),
                "miner": {"paper_title": pdf.stem, "samples": []},
                "visualizer": {"num_plots_found": 0},
                "simulator": {"num_samples_simulated": 0},
            })

    # Generate summary page
    generate_index_html(all_results, str(out_path))

    # Save master summary
    summary_path = out_path / "pipeline_summary.json"
    summary = {
        "total_papers": len(all_results),
        "total_samples": sum(len(r.get("miner", {}).get("samples", [])) for r in all_results),
        "total_plots_digitized": sum(r.get("visualizer", {}).get("num_plots_found", 0) for r in all_results),
        "total_simulations": sum(r.get("simulator", {}).get("num_samples_simulated", 0) for r in all_results),
        "papers": [
            {
                "pdf": r.get("source_pdf"),
                "title": r.get("miner", {}).get("paper_title"),
                "samples": len(r.get("miner", {}).get("samples", [])),
                "plots": r.get("visualizer", {}).get("num_plots_found", 0),
                "simulations": r.get("simulator", {}).get("num_samples_simulated", 0),
                "error": r.get("error"),
            }
            for r in all_results
        ],
    }
    with open(summary_path, "w") as f:
        json.dump(summary, f, indent=2)

    logger.info("=" * 70)
    logger.info("PIPELINE COMPLETE")
    logger.info("  Papers: %d", summary["total_papers"])
    logger.info("  Samples: %d", summary["total_samples"])
    logger.info("  Plots digitized: %d", summary["total_plots_digitized"])
    logger.info("  Simulations: %d", summary["total_simulations"])
    logger.info("  Output: %s", out_path)
    logger.info("=" * 70)


if __name__ == "__main__":
    run_pipeline()
