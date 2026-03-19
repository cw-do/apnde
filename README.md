# APNDE — Autonomous Polymer Nanocomposite Discovery Engine

AI-driven pipeline that automates the transition from "dark data" in polymer nanocomposite (PNC) research literature to predictive material models. Ingests PDF papers, extracts structured chemical data, digitizes SANS/SAXS plots, computes neutron scattering contrast, and generates simulation-ready inputs.

**FOA Alignment:** DE-FOA-0003612 (The Genesis Mission) — Focus Area 12-A (Functional Materials)

## What It Does

```
PDF Papers ──► Text Extraction ──► LLM Structured Extraction ──► Structured JSON
                                                                      │
PDF Figures ──► Image Extraction ──► Vision LLM ──► Digitized I(q) CSVs + Plots
                                                                      │
Composition Data ──► SLD Lookup ──► Neutron Contrast ──► SasModels Simulation
                                                                      │
All Results ──────────────────────────────────────────► Interactive HTML Dashboard
```

### Three Modules

| Module | Role | Implementation |
|--------|------|----------------|
| **Miner** | Extract polymer/filler compositions, Rg, grafting density, chain conformation, and 29 structured fields from paper text | GPT-4o via OpenRouter + PyMuPDF |
| **Visualizer** | Identify SANS/SAXS figures, digitize I(q) vs q data into CSV, generate log-log plots | GPT-4o Vision via OpenRouter |
| **Simulator** | Compute neutron SLD contrast (Δρ)², generate pymatgen-compatible metadata, run SasModels simulations | SasModels + pymatgen + matplotlib |

### Extracted Fields (29 per sample)

The Miner extracts structured data including:

- **Composition**: matrix/filler names, abbreviations, loading (wt%, vol fraction)
- **Molecular**: Mw, Mn, PDI, particle size
- **Chain conformation**: Rg, Rg_bulk, Rg/Rg₀ ratio, expansion/contraction state
- **Scattering-derived**: correlation length ξ, d-spacing, interparticle distance, Porod exponent, I(0)
- **Surface/interface**: grafting density σ, brush height, bound layer thickness
- **Thermal/mechanical**: Tg, elastic modulus, viscosity ratio η/η₀

## Example Output

The `docs/` directory contains results from processing 5 PNC research papers. It is also served as a [GitHub Pages site](https://cw-do.github.io/apnde/).

| Paper | Samples | Scattering Plots | Simulations |
|-------|---------|-------------------|-------------|
| Silica-PS grafted layers | 5 | 7 | 5 |
| CNT/PS SANS conformations | 4 | 5 | 4 |
| Viscosity reduction in PNCs | 3 | 3 | 3 |
| Model PNC scattering (theory) | 3 | 6 | 3 |
| GO-PS nanocomposites | 5 | 0 | 5 |

### Output Structure

```
docs/
├── index.html                          # Interactive dashboard (GitHub Pages root)
├── pipeline_summary.json               # Aggregate statistics
├── data/                               # Digitized I(q) vs q CSVs (28 files)
├── figures/                            # Extracted PDF figures
├── plots/                              # Generated log-log I(q) PNGs (50 files)
│   ├── *_digitized_plot*.png           # From experimental data
│   └── *_simulated_Iq*.png            # From SasModels
├── structures/                         # Pymatgen-compatible composition JSON (44 files)
├── simulations/                        # SasModels params + simulated I(q) (88 files)
├── *_miner.json                        # Per-paper extraction results
├── *_digitized.json                    # Per-paper digitized plot data
└── *_full_result.json                  # Complete pipeline output per paper
```

### Dashboard Features

Open `docs/index.html` in a browser, or visit the [live dashboard](https://cw-do.github.io/apnde/):

- **Clickable stat cards** expand to show papers list, scattering plot gallery, or simulation table
- **Per-paper sample tables** with Rg, Rg/Rg₀, conformation badges (expanded/contracted/unchanged), and property tags
- **Experimental scattering data** sections with clickable plot thumbnails, curve labels, and inline data previews
- **Modal lightbox** for full-size plot viewing (click any thumbnail)

## Setup

### Requirements

- Python 3.12+
- OpenRouter API key (for GPT-4o access)

### Install

```bash
python3.12 -m venv .venv
source .venv/bin/activate
pip install python-dotenv pymatgen sasmodels pymupdf opencv-python-headless openai pandas numpy Pillow matplotlib
```

### Configure

Create `.env` in the project root:

```env
OPENROUTER_API_KEY=your_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o
```

## Usage

```bash
source .venv/bin/activate
python run.py --pdf-dir ./pdf --output-dir ./docs
```

The pipeline processes each PDF through all three modules sequentially, then generates `docs/index.html`.

To regenerate the HTML dashboard from existing results without re-running extraction:

```python
import json, glob, sys
sys.path.insert(0, '.')
from src.pipeline import generate_index_html

results = []
for f in sorted(glob.glob('docs/*_full_result.json')):
    with open(f) as fh:
        results.append(json.load(fh))

generate_index_html(results, 'docs')
```

## Reference Physical Constants

| Component | SLD (Å⁻²) | Density (g/cm³) |
|-----------|-----------|-----------------|
| h-PS | 1.41×10⁻⁶ | 1.05 |
| d-PS | 6.47×10⁻⁶ | 1.05 |
| h-P2VP | 1.95×10⁻⁶ | 1.12 |
| SiO₂ | 3.48×10⁻⁶ | 2.20 |
| SWCNT | 5.71×10⁻⁶ | 1.40 |
| Au | 4.50×10⁻⁶ | 19.32 |
| GO | 4.09×10⁻⁶ | 1.80 |

Full SLD/density lookup table in `src/constants.py`.
