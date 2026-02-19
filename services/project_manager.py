"""
services/project_manager.py
All file-system operations for projects (create, load, save, restore).
No Streamlit imports – pure service layer tested independently.
"""
from __future__ import annotations

import json
import logging
import os
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import config.settings as cfg
from core.exceptions import ProjectError

logger = logging.getLogger(__name__)


# ── Helpers ───────────────────────────────────────────────────────────────────

def _user_dir(user_id: str) -> Path:
    return cfg.PROJECTS_DIR / user_id


def _metadata_path(project_path: Path) -> Path:
    return project_path / "metadata.json"


def _read_metadata(project_path: Path) -> Dict[str, Any]:
    mp = _metadata_path(project_path)
    if not mp.exists():
        raise ProjectError(f"metadata.json not found in '{project_path}'")
    with mp.open("r", encoding="utf-8") as fh:
        return json.load(fh)


def _write_metadata(project_path: Path, data: Dict[str, Any]) -> None:
    with _metadata_path(project_path).open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)


# ── Public API ────────────────────────────────────────────────────────────────

def list_projects(user_id: str = "default_user") -> List[Dict[str, Any]]:
    """Return list of project dicts sorted by creation date (newest first)."""
    user_dir = _user_dir(user_id)
    if not user_dir.exists():
        return []

    projects = []
    for entry in user_dir.iterdir():
        if not entry.is_dir():
            continue
        try:
            meta = _read_metadata(entry)
            projects.append({"name": entry.name, "path": str(entry), "metadata": meta})
        except ProjectError:
            logger.warning("Skipping project dir without metadata: %s", entry)

    return sorted(projects, key=lambda p: p["metadata"].get("created", ""), reverse=True)


def create_project(
    project_name: str,
    patent_type: str,
    user_id: str = "default_user",
) -> tuple[Path, Dict[str, Any]]:
    """Create folder structure + initial metadata. Returns (path, metadata)."""
    project_path = _user_dir(user_id) / project_name
    project_path.mkdir(parents=True, exist_ok=True)

    metadata: Dict[str, Any] = {
        "project_name": project_name,
        "patent_type": patent_type,
        "created": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
        "user_id": user_id,
        "draft1_uploaded": False,
        "questions_generated": False,
        "qa_uploaded": False,
        "draft2_generated": False,
    }
    _write_metadata(project_path, metadata)
    logger.info("Created project '%s' at '%s'", project_name, project_path)
    return project_path, metadata


def load_project(project_path: str | Path) -> Optional[Dict[str, Any]]:
    """Load metadata (+ cached questions). Returns None if corrupt."""
    project_path = Path(project_path)
    try:
        meta = _read_metadata(project_path)
    except ProjectError:
        return None

    qf = project_path / "scrutiny_questions.json"
    if qf.exists():
        try:
            with qf.open("r", encoding="utf-8") as fh:
                qdata = json.load(fh)
            meta["questions"] = qdata.get("questions", "")
            meta["questions_generated"] = True
        except Exception:
            pass

    return meta


def update_metadata(project_path: str | Path, updates: Dict[str, Any]) -> None:
    project_path = Path(project_path)
    try:
        meta = _read_metadata(project_path)
    except ProjectError:
        meta = {}
    meta.update(updates)
    _write_metadata(project_path, meta)


def save_questions(project_path: str | Path, questions_text: str) -> None:
    project_path = Path(project_path)
    data = {
        "questions": questions_text,
        "generated_at": datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    }
    with (project_path / "scrutiny_questions.json").open("w", encoding="utf-8") as fh:
        json.dump(data, fh, indent=2)
    update_metadata(project_path, {"questions_generated": True})


def get_file_path(project_path: str | Path, doc_type: str) -> Path:
    """
    Return the path of a project document.
    Searches for any existing file with the right prefix (preserves .pdf/.docx).
    Falls back to default names.
    """
    project_path = Path(project_path)
    defaults = {
        "draft1":    "draft1.pdf",
        "qa":        "draft1_qa.pdf",
        "draft2":    "draft2.md",
        "questions": "scrutiny_questions.json",
        "metadata":  "metadata.json",
    }
    prefixes = {"draft1": "draft1", "qa": "draft1_qa", "draft2": "draft2"}

    if doc_type in prefixes:
        prefix = prefixes[doc_type]
        for fname in project_path.iterdir():
            if fname.stem.lower() == prefix.lower():
                return fname

    return project_path / defaults.get(doc_type, doc_type)


def save_document(
    project_path: str | Path,
    file_bytes: bytes,
    original_filename: str,
    doc_type: str,
) -> Path:
    """Save uploaded bytes to the project folder with a canonical name."""
    project_path = Path(project_path)
    ext = Path(original_filename).suffix.lower()
    prefix_map = {"draft1": "draft1", "qa": "draft1_qa"}
    prefix = prefix_map.get(doc_type, doc_type)
    dest = project_path / f"{prefix}{ext}"
    dest.write_bytes(file_bytes)
    logger.info("Saved %s → %s", doc_type, dest)
    return dest


def delete_project(project_path: str | Path) -> None:
    shutil.rmtree(str(project_path), ignore_errors=True)
    logger.info("Deleted project at '%s'", project_path)
