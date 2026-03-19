"""
Module B: The Visualizer — SANS/SAXS Plot Digitization

Extracts figures from PDFs, identifies SANS/SAXS scattering plots using
GPT-4o Vision, and digitizes them into numerical I(q) vs q data.
"""

import base64
import json
import logging
import re
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pymupdf
from openai import OpenAI

logger = logging.getLogger(__name__)

# Minimum image dimensions to consider (skip tiny icons/logos)
MIN_IMAGE_WIDTH = 150
MIN_IMAGE_HEIGHT = 150


def extract_figures_from_pdf(pdf_path: str, output_dir: str) -> list[dict]:
    """
    Extract all meaningful images from a PDF file.

    Args:
        pdf_path: Path to the PDF.
        output_dir: Directory to save extracted images.

    Returns:
        List of dicts with image metadata (path, page, dimensions).
    """
    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    doc = pymupdf.open(pdf_path)
    extracted: list[dict] = []
    stem = Path(pdf_path).stem

    for page_num in range(len(doc)):
        page = doc[page_num]
        image_list = page.get_images(full=True)

        for img_idx, img_info in enumerate(image_list):
            xref = img_info[0]
            try:
                base_image = doc.extract_image(xref)
            except Exception:
                logger.debug("Could not extract image xref=%d on page %d", xref, page_num + 1)
                continue

            width = base_image["width"]
            height = base_image["height"]

            # Skip small images (icons, logos, decorations)
            if width < MIN_IMAGE_WIDTH or height < MIN_IMAGE_HEIGHT:
                continue

            ext = base_image["ext"]
            image_bytes = base_image["image"]
            fname = f"{stem}_p{page_num + 1}_img{img_idx + 1}.{ext}"
            fpath = out / fname

            with open(fpath, "wb") as f:
                f.write(image_bytes)

            extracted.append({
                "path": str(fpath),
                "page": page_num + 1,
                "width": width,
                "height": height,
                "format": ext,
            })

    doc.close()

    # Fallback: if no images extracted, render pages as images
    if not extracted:
        logger.info("No embedded images found in %s — rendering pages as images", pdf_path)
        doc = pymupdf.open(pdf_path)
        for page_num in range(len(doc)):
            page = doc[page_num]
            pix = page.get_pixmap(dpi=200)
            fname = f"{stem}_p{page_num + 1}_render.png"
            fpath = out / fname
            pix.save(str(fpath))
            extracted.append({
                "path": str(fpath),
                "page": page_num + 1,
                "width": pix.width,
                "height": pix.height,
                "format": "png",
                "rendered": True,
            })
        doc.close()

    logger.info("Extracted %d images from %s", len(extracted), pdf_path)
    return extracted


def _encode_image(image_path: str) -> str:
    """Read an image file and return its base64-encoded string."""
    with open(image_path, "rb") as f:
        return base64.b64encode(f.read()).decode("utf-8")


def _image_media_type(fmt: str) -> str:
    """Map image format extension to MIME type."""
    mapping = {"png": "image/png", "jpg": "image/jpeg", "jpeg": "image/jpeg", "tiff": "image/tiff"}
    return mapping.get(fmt.lower(), "image/png")


def identify_scattering_plot(image_path: str, image_format: str, client: OpenAI, model: str) -> dict:
    """
    Use GPT-4o Vision to determine if an image is a SANS/SAXS scattering plot.

    Returns:
        Dict with is_scattering_plot (bool), plot_type, confidence, axis info.
    """
    b64 = _encode_image(image_path)
    media = _image_media_type(image_format)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Analyze this image from a scientific paper. "
                            "Determine if it is a small-angle scattering plot (SANS or SAXS).\n\n"
                            "Look for:\n"
                            "- X-axis: q or Q (scattering vector) in Å⁻¹ or nm⁻¹\n"
                            "- Y-axis: I(q), Intensity, S(q), P(q), or scattering cross-section in cm⁻¹\n"
                            "- Log-log or semi-log scales\n"
                            "- Scattering curves (power-law decay, Guinier region, etc.)\n\n"
                            "Also accept: Kratky plots (I*q² vs q), Porod plots, "
                            "structure factor S(q) plots, form factor P(q) plots.\n\n"
                            "Respond ONLY with valid JSON:\n"
                            "{\n"
                            '  "is_scattering_plot": true/false,\n'
                            '  "confidence": "high"/"medium"/"low",\n'
                            '  "plot_type": "SANS"/"SAXS"/"structure_factor"/"form_factor"/"Kratky"/"other"/"none",\n'
                            '  "x_label": "detected x-axis label or null",\n'
                            '  "y_label": "detected y-axis label or null",\n'
                            '  "x_unit": "detected unit or null",\n'
                            '  "y_unit": "detected unit or null",\n'
                            '  "num_curves": estimated number of data curves,\n'
                            '  "reasoning": "brief explanation"\n'
                            "}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=600,
    )

    raw = response.choices[0].message.content or ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse scattering-plot identification: %s", raw[:300])
        return {"is_scattering_plot": False, "confidence": "low", "reasoning": "parse error"}


def digitize_scattering_plot(image_path: str, image_format: str, client: OpenAI, model: str) -> dict:
    """
    Use GPT-4o Vision to extract numerical I(q) vs q data from a scattering plot.

    Returns:
        Dict with axes metadata and list of curves, each containing data points.
    """
    b64 = _encode_image(image_path)
    media = _image_media_type(image_format)

    response = client.chat.completions.create(
        model=model,
        messages=[
            {
                "role": "user",
                "content": [
                    {
                        "type": "text",
                        "text": (
                            "Extract numerical data from this small-angle scattering plot.\n\n"
                            "INSTRUCTIONS:\n"
                            "1. Identify axis scales (log10, linear, ln).\n"
                            "2. Read axis ranges carefully.\n"
                            "3. For EACH curve/dataset visible in the plot, extract q and I(q) pairs.\n"
                            "4. Report values in REAL scale (not log). "
                            "If axes are log-scale, convert back to linear values.\n"
                            "5. Extract at least 15-30 data points per curve, sampling key features "
                            "(high-q, low-q, inflection points, peaks).\n\n"
                            "UNIT CONVENTIONS:\n"
                            "- q in Å⁻¹ (if nm⁻¹, multiply by 0.1 to convert)\n"
                            "- I(q) in cm⁻¹ (report as-is from plot if units unclear)\n\n"
                            "Respond ONLY with valid JSON:\n"
                            "{\n"
                            '  "axes": {\n'
                            '    "x_label": "q",\n'
                            '    "x_unit": "Å⁻¹",\n'
                            '    "y_label": "I(q)",\n'
                            '    "y_unit": "cm⁻¹",\n'
                            '    "x_scale": "log10" or "linear",\n'
                            '    "y_scale": "log10" or "linear"\n'
                            "  },\n"
                            '  "curves": [\n'
                            "    {\n"
                            '      "label": "curve name/legend if visible",\n'
                            '      "data": [\n'
                            '        {"q": 0.005, "I_q": 1200.0},\n'
                            '        {"q": 0.01, "I_q": 450.0}\n'
                            "      ]\n"
                            "    }\n"
                            "  ],\n"
                            '  "notes": "observations about data quality, scale, etc."\n'
                            "}"
                        ),
                    },
                    {
                        "type": "image_url",
                        "image_url": {"url": f"data:{media};base64,{b64}"},
                    },
                ],
            }
        ],
        temperature=0.1,
        max_tokens=8000,
    )

    raw = response.choices[0].message.content or ""
    cleaned = re.sub(r"^```(?:json)?\s*", "", raw.strip())
    cleaned = re.sub(r"\s*```$", "", cleaned)
    cleaned = re.sub(r",\s*([}\]])", r"\1", cleaned)

    try:
        return json.loads(cleaned)
    except json.JSONDecodeError:
        logger.warning("Could not parse digitized data: %s", raw[:500])
        return {"axes": {}, "curves": [], "notes": f"Parse error. Raw: {raw[:1000]}"}


def visualize_pdf(
    pdf_path: str,
    figures_dir: str,
    client: OpenAI,
    model: str,
) -> list[dict]:
    """
    Full Visualizer pipeline for a single PDF.

    1. Extract figures from PDF
    2. Identify scattering plots via vision LLM
    3. Digitize identified plots

    Returns:
        List of digitized plot dicts (one per identified scattering plot).
    """
    logger.info("Visualizing PDF: %s", pdf_path)

    images = extract_figures_from_pdf(pdf_path, figures_dir)
    if not images:
        logger.warning("No images extracted from %s", pdf_path)
        return []

    results: list[dict] = []

    for img in images:
        # Step 1: Identify
        ident = identify_scattering_plot(img["path"], img["format"], client, model)
        if not ident.get("is_scattering_plot"):
            continue
        if ident.get("confidence") == "low":
            continue

        logger.info(
            "Found %s plot (confidence=%s): %s",
            ident.get("plot_type", "unknown"),
            ident.get("confidence", "?"),
            img["path"],
        )

        # Step 2: Digitize
        digitized = digitize_scattering_plot(img["path"], img["format"], client, model)
        digitized["source_image"] = img["path"]
        digitized["source_page"] = img["page"]
        digitized["identification"] = ident
        results.append(digitized)

    logger.info("Digitized %d scattering plots from %s", len(results), pdf_path)
    return results


CURVE_COLORS = [
    "#60a5fa", "#f472b6", "#34d399", "#fbbf24", "#a78bfa",
    "#fb923c", "#22d3ee", "#e879f9", "#84cc16", "#f87171",
]


def generate_digitized_plot(
    plot_data: dict,
    output_path: str,
    paper_title: str = "",
) -> str | None:
    """Generate a log-log PNG from digitized I(q) vs q curve data.

    Args:
        plot_data: A single plot dict from the visualizer with 'curves' list.
        output_path: Where to save the PNG.
        paper_title: Short title for the plot header.

    Returns:
        Path to the PNG or None if no plottable data.
    """
    curves = plot_data.get("curves", [])
    plottable = [c for c in curves if c.get("data")]
    if not plottable:
        return None

    ident = plot_data.get("identification", {})
    plot_type = ident.get("plot_type", "scattering")
    axes = plot_data.get("axes", {})
    x_unit = axes.get("x_unit", "Å⁻¹")
    y_unit = axes.get("y_unit", "cm⁻¹")
    y_label = axes.get("y_label", "I(q)")

    fig, ax = plt.subplots(figsize=(5.5, 4), dpi=130)

    for cidx, curve in enumerate(plottable):
        data = curve["data"]
        q_vals = np.array([float(p.get("q", 0)) for p in data])
        iq_vals = np.array([float(p.get("I_q", 0)) for p in data])

        mask = (q_vals > 0) & (iq_vals > 0)
        q_vals, iq_vals = q_vals[mask], iq_vals[mask]
        if len(q_vals) < 2:
            continue

        color = CURVE_COLORS[cidx % len(CURVE_COLORS)]
        label = curve.get("label", f"curve {cidx + 1}")
        ax.loglog(q_vals, iq_vals, "o-", color=color, linewidth=1.4,
                  markersize=3.5, label=label, solid_capstyle="round")

    if not ax.get_lines():
        plt.close(fig)
        return None

    ax.set_xlabel(f"q ({x_unit})", fontsize=10)
    ax.set_ylabel(f"{y_label} ({y_unit})", fontsize=10)

    title_text = f"{plot_type}"
    if paper_title:
        short = paper_title[:55] + ("…" if len(paper_title) > 55 else "")
        title_text = f"{short}\n{plot_type} — page {plot_data.get('source_page', '?')}"
    ax.set_title(title_text, fontsize=8.5, color="#c0c0c0", pad=8)

    ax.tick_params(labelsize=8, colors="#b0b0b0")
    ax.grid(True, which="both", linewidth=0.3, alpha=0.4, color="#555")
    ax.legend(fontsize=7, loc="best", framealpha=0.6,
              facecolor="#1e2130", edgecolor="#3b4260", labelcolor="#d0d0d0")

    fig.patch.set_facecolor("#1e2130")
    ax.set_facecolor("#13151e")
    for spine in ("bottom", "left"):
        ax.spines[spine].set_color("#444")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.xaxis.label.set_color("#c0c0c0")
    ax.yaxis.label.set_color("#c0c0c0")

    fig.tight_layout(pad=1.2)
    Path(output_path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output_path, facecolor=fig.get_facecolor(), edgecolor="none")
    plt.close(fig)

    logger.info("Generated digitized plot: %s (%d curves)", output_path, len(plottable))
    return output_path
