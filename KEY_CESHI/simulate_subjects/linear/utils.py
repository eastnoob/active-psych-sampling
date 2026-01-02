"""Utility helpers for simulate_subjects linear cluster usage."""
from pathlib import Path
import json
from subject_simulator_v2 import LinearSubject


def load_subject_spec(path: str) -> LinearSubject:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        spec = json.load(f)
    sub = LinearSubject.from_dict(spec)
    return sub


def load_cluster_summary(path: str) -> dict:
    p = Path(path)
    with p.open('r', encoding='utf-8') as f:
        return json.load(f)
