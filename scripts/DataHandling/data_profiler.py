"""
Data Profiler
=============
Scans yelp_academic_dataset_business.json and extracts all unique values for
the ``categories`` field and every key under ``attributes``.

Outputs two PDFs to data/profiles/:
    - categories_profile.pdf
    - attributes_profile.pdf

Usage
-----
    python scripts/DataHandling/data_profiler.py \
        --input_path data/raw/yelp_dataset/yelp_academic_dataset_business.json
"""
from __future__ import annotations

import argparse
import ast
import json
import os
from collections import defaultdict
from pathlib import Path

from fpdf import FPDF
from tqdm import tqdm


# ---------------------------------------------------------------------------
# Profiling
# ---------------------------------------------------------------------------

def profile_categories(file_path: Path) -> set[str]:
    """Return a sorted set of every unique category string."""
    categories: set[str] = set()
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Profiling categories"):
            business = json.loads(line)
            raw = business.get("categories")
            if raw:
                for cat in raw.split(","):
                    cat = cat.strip()
                    if cat:
                        categories.add(cat)
    return categories


def profile_attributes(file_path: Path) -> dict[str, set[str]]:
    """Return a mapping of attribute key -> set of unique values.

    Nested dict-like attributes (e.g. Ambience, BusinessParking) are expanded
    into separate keys like ``Ambience.romantic``, ``Ambience.casual``, etc.
    """
    attribute_map: dict[str, set[str]] = defaultdict(set)
    with open(file_path, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Profiling attributes"):
            business = json.loads(line)
            attrs = business.get("attributes")
            if attrs:
                for key, value in attrs.items():
                    clean_val = str(value).replace("u'", "").replace("'", "").strip()
                    # Try to parse nested dicts (e.g. Ambience, BusinessParking)
                    if clean_val.startswith("{"):
                        try:
                            nested = ast.literal_eval(value)
                            for sub_key, sub_val in nested.items():
                                attribute_map[f"{key}.{sub_key}"].add(str(sub_val))
                        except (ValueError, SyntaxError):
                            attribute_map[key].add(clean_val)
                    else:
                        attribute_map[key].add(clean_val)
    return attribute_map


# ---------------------------------------------------------------------------
# PDF generation
# ---------------------------------------------------------------------------

def _write_categories_pdf(categories: set[str], output_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Categories Profile", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Total unique categories: {len(categories)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for cat in sorted(categories):
        pdf.cell(0, 6, f"  - {cat}", new_x="LMARGIN", new_y="NEXT")

    pdf.output(str(output_path))


def _write_attributes_pdf(attribute_map: dict[str, set[str]], output_path: Path) -> None:
    pdf = FPDF()
    pdf.set_auto_page_break(auto=True, margin=15)
    pdf.add_page()
    pdf.set_font("Helvetica", "B", 16)
    pdf.cell(0, 10, "Attributes Profile", new_x="LMARGIN", new_y="NEXT")
    pdf.set_font("Helvetica", "", 10)
    pdf.cell(0, 8, f"Total unique attribute keys: {len(attribute_map)}", new_x="LMARGIN", new_y="NEXT")
    pdf.ln(4)

    for key in sorted(attribute_map):
        values = sorted(attribute_map[key])
        pdf.set_font("Helvetica", "B", 11)
        pdf.cell(0, 7, key, new_x="LMARGIN", new_y="NEXT")
        pdf.set_font("Helvetica", "", 9)
        for val in values:
            # Truncate very long values (e.g. nested dicts) to fit the page
            display = val if len(val) <= 120 else val[:120] + "..."
            pdf.cell(0, 5, f"    {display}", new_x="LMARGIN", new_y="NEXT")
        pdf.ln(2)

    pdf.output(str(output_path))


# ---------------------------------------------------------------------------
# CLI
# ---------------------------------------------------------------------------

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Profile categories and attributes from the Yelp business dataset.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )
    p.add_argument(
        "--input_path",
        type=Path,
        required=True,
        help="Path to yelp_academic_dataset_business.json",
    )
    p.add_argument(
        "--output_dir",
        type=Path,
        default=Path("data/profiles"),
        help="Directory for output PDFs",
    )
    return p.parse_args()


if __name__ == "__main__":
    args = _parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    categories = profile_categories(args.input_path)
    attribute_map = profile_attributes(args.input_path)

    cat_pdf = args.output_dir / "categories_profile.pdf"
    attr_pdf = args.output_dir / "attributes_profile.pdf"

    _write_categories_pdf(categories, cat_pdf)
    _write_attributes_pdf(attribute_map, attr_pdf)

    print(f"Categories PDF: {cat_pdf}  ({len(categories)} unique)")
    print(f"Attributes PDF: {attr_pdf}  ({len(attribute_map)} keys)")
