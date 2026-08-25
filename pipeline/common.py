"""Shared constants and helpers for the labelling pipeline."""

import os
import re
from pathlib import Path

ROOT = Path(os.environ.get("CTSV_ROOT", Path(__file__).resolve().parents[1]))
DATA = ROOT / "data"
WORK = ROOT / "work"

PROMPT = "Non-contrast chest CT. Provide the radiology findings."

CLASSES = [
    "Medical material", "Arterial wall calcification", "Cardiomegaly",
    "Pericardial effusion", "Coronary artery wall calcification", "Hiatal hernia",
    "Lymphadenopathy", "Emphysema", "Atelectasis", "Lung nodule", "Lung opacity",
    "Pulmonary fibrotic sequela", "Pleural effusion", "Mosaic attenuation pattern",
    "Peribronchial thickening", "Consolidation", "Bronchiectasis",
    "Interlobular septal thickening",
]

# --- deterministic radiology sentence segmentation -------------------------
_HEADERS = {"findings:", "impression:", "findings", "impression"}
_ABBREVIATIONS = ("RUL.", "RML.", "RLL.", "LUL.", "LLL.", "Dr.", "vs.")
_DEC = "<DECIMAL_DOT>"
_ABBR = "<ABBR_DOT>"


def split_report(report: str) -> list[str]:
    """Split a report into sentences, preserving decimals and abbreviations."""
    lines = []
    for line in report.replace("\r\n", "\n").replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line or line.lower() in _HEADERS:
            continue
        lines.append(re.sub(r"^\d+[.)]\s*", "", line))
    text = " ".join(lines)
    text = re.sub(r"(?<=\d)\.(?=\d)", _DEC, text)
    for abbr in _ABBREVIATIONS:
        text = text.replace(abbr, abbr[:-1] + _ABBR)
    out = []
    for piece in re.split(r"(?<=[.!?])\s+", text):
        piece = piece.replace(_DEC, ".").replace(_ABBR, ".").strip()
        if piece:
            out.append(piece)
    return out
