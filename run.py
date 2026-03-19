#!/usr/bin/env python3
"""
APNDE Runner — Autonomous Polymer Nanocomposite Discovery Engine

Usage:
    python run.py [--pdf-dir ./pdf] [--output-dir ./output]
"""
import argparse
import sys
from pathlib import Path

# Add project root to path
sys.path.insert(0, str(Path(__file__).parent))

from src.pipeline import run_pipeline


def main() -> None:
    parser = argparse.ArgumentParser(description="APNDE Pipeline Runner")
    parser.add_argument("--pdf-dir", default="./pdf", help="Directory containing input PDFs")
    parser.add_argument("--output-dir", default="./output", help="Output directory")
    args = parser.parse_args()

    run_pipeline(pdf_dir=args.pdf_dir, output_dir=args.output_dir)


if __name__ == "__main__":
    main()
