"""
ui/session_state.py
Single source of truth for Streamlit session state keys and their defaults.
Call init() once at the top of app.py.
"""
from __future__ import annotations
from typing import Any, Dict

# Key → default value
_DEFAULTS: Dict[str, Any] = {
    # Project
    "current_project":           None,
    "project_metadata":          {},
    # Document processing
    "draft1_chunks":             [],
    "draft1_collection":         None,
    "draft1_file":               None,
    "draft1_processed":          False,
    "draft1_docx":               None,      # python-docx Document (for structure export)
    "draft1_docx_path":          None,
    "qa_chunks":                 [],
    "qa_collection":             None,
    "qa_file":                   None,
    "qa_content":                "",        # flattened Q&A text
    "qa_processed":              False,
    # Workflow flags
    "processed":                 False,     # enables workflow tabs
    # Classification
    "classification_result":     None,      # ClassificationResult.to_dict()
    "classification_done":       False,
    "classification_in_progress": False,
    "user_override":             False,
    "patent_type":               "Electronics",
    # Scrutiny
    "patent_questions":          None,
    "field_of_invention":         "",        # extracted verbatim from document
    "mechanism":                  "",        # enabling technical feature
    "readiness_report":           None,      # ReadinessReport from question_rater
    "agent_log":                 "",
    # Consolidation
    "final_draft":               None,
    "final_draft_audit":         None,
    "final_draft_valid":         True,
    "final_draft_missing":       [],
    "draft2_saved":              False,
    # Custom agent persona
    "custom_role":               None,
    "custom_backstory":          None,
    "persona_tuned":             False,
}


def init() -> None:
    """Initialise all session state keys with defaults if not already set."""
    import streamlit as st
    for key, default in _DEFAULTS.items():
        if key not in st.session_state:
            st.session_state[key] = default


def reset_workflow(keep_project: bool = True) -> None:
    """
    Clear workflow state.
    If keep_project=True, the current_project and project_metadata are preserved.
    """
    import streamlit as st
    preserved = {}
    if keep_project:
        for k in ("current_project", "project_metadata"):
            preserved[k] = st.session_state.get(k)

    st.session_state.clear()
    for key, default in _DEFAULTS.items():
        st.session_state[key] = preserved.get(key, default)


def reset_classification() -> None:
    import streamlit as st
    for k in (
        "classification_result",
        "classification_done",
        "classification_in_progress",
        "user_override",
    ):
        st.session_state[k] = _DEFAULTS[k]
