# AGENTS.md: Autonomous Polymer Nanocomposite Discovery Engine (APNDE)

## 1. Project Overview & Strategic Mission
**Objective:** To establish an AI-driven pipeline that automates the transition from "dark data" in research literature to predictive material models. This system ingests PDF papers, extracts structured chemical data, digitizes SANS plots, and generates simulation-ready inputs.

**Focus:** Polymer Nanocomposites (PNCs), specifically P2VP and PS systems with inorganic fillers.

**FOA Alignment:** DE-FOA-0003612 (The Genesis Mission) - Focus Area 12-A (Functional Materials).

---

## 2. Infrastructure & Environment Setup

### 2.1 Virtual Environment (.venv)

All operations must be performed within a local virtual environment to ensure dependency isolation.

```bash
# 1. Create the virtual environment
python -m venv .venv

# 2. Activate the environment
# macOS/Linux:
source .venv/bin/activate
# Windows:
.venv\Scripts\activate

# 3. Upgrade pip
pip install --upgrade pip

# 4. Install dependencies
pip install python-dotenv pncextract pymatgen sasmodels opencv-python grobid-client langchain-openai
```

### 2.2 Configuration (.env)

Create a `.env` file in the project root:

```env
OPENROUTER_API_KEY=your_openrouter_api_key_here
OPENROUTER_BASE_URL=https://openrouter.ai/api/v1
MODEL_NAME=openai/gpt-4o
PDF_INPUT_DIR=./pdf
OUTPUT_DIR=./output
```

---

## 3. Agentic Workflow Modules

### Module A: The "Miner" (NLP & Metadata Extraction)

**Core Tool:** PNCExtract

**Function:** Extract N-ary relations such as Matrix–Filler–Property triplets from scientific text.

**Agent Logic:**
- Iterate through all files in `./pdf/`
- Use PNCExtract to identify:
  - Polymer matrix
  - Filler material
  - Loading (wt% or volume fraction)
- Use GPT-4o to resolve:
  - Complex tables
  - Implicit units
  - Nested relationships (Mw, Tg, etc.)

**SLD Mapping (Å⁻²):**
- h-PS: 1.41e-6
- h-P2VP: 1.95e-6
- Silica (SiO2): 3.48e-6

---

### Module B: The "Visualizer" (SANS Plot Digitization)

**Core Tools:** DePlot / GPT-4o Vision

**Function:** Extract numerical scattering data (I(q) vs q) from figures.

**Agent Logic:**
- Identify figures containing:
  - "SANS"
  - "SAXS"
  - "I(q)"
  - "Intensity"
- Extract plot data using vision-language models
- Convert extracted data into CSV format

**Normalization Rules:**
- Q → Å⁻¹
- I(q) → cm⁻¹

**Output Example:**
```
q, I(q)
0.001, 120
0.002, 98
...
```

---

### Module C: The "Simulator" (Physics Link)

**Core Tools:** Pymatgen, SasModels

**Function:** Convert extracted data into simulation-ready representations.

**Agent Logic:**
- Use:
  - Volume fraction (φ)
  - Molecular weight (Mw)
  - Particle size (if available)
- Generate structure using Pymatgen
- Create:
  - `simulation_params.json`
  - SasModels-compatible script for I(q)

---

## 4. Operational Roadmap

### Step 1: Initialization
- Activate `.venv`
- Load `.env`
- Verify OpenRouter connectivity

### Step 2: Ingestion
- Scan `./pdf` directory
- Use Grobid to convert PDFs → structured XML/TEI

### Step 3: Extraction
- Run PNCExtract on parsed content
- Use GPT-4o for:
  - Table interpretation
  - Context disambiguation

### Step 4: Digitization
- Locate scattering figures
- Apply DePlot / vision extraction
- Save CSV to:
  - `./output/data/<paper_name>_sans.csv`

### Step 5: Synthesis
- Compute neutron contrast:

```
Δρ = ρ_filler - ρ_matrix
(Δρ)^2 used for scattering intensity scaling
```

- Identify:
  - Interphase effects
  - Deviations from models

### Step 6: Output
Save results to `./output/`:
- Structured JSON summary
- Digitized CSV data
- Pymatgen structure file
- Simulation parameters JSON
- Create a webpage index.html that summarizes the output

---

## 5. Reference Physical Constants

| Component | SLD (Å⁻²) | Density (g/cm³) |
|----------|----------|----------------|
| h-PS     | 1.41e-6  | 1.05           |
| h-P2VP   | 1.95e-6  | 1.12           |
| Silica   | 3.48e-6  | 2.20           |

---

## 6. Execution Command for Agent

```
Activate the .venv environment.
Load OPENROUTER_API_KEY from .env.

Process all PDFs in ./pdf.

For each document:
- Extract polymer/filler compositions using PNCExtract
- Identify and digitize SANS plots using vision models
- Normalize and store I(q) data as CSV
- Calculate SLD contrast (Δρ)^2
- Generate:
    - JSON summary
    - Pymatgen structure
    - simulation_params.json

Save all outputs to ./output.
```