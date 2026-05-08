"""
services/ollama_service.py
Manages Ollama process lifecycle and model resolution.
No UI imports – pure service layer.
"""
from __future__ import annotations

import logging
import platform
import subprocess
import time
from typing import Optional, Tuple

import requests

import config.settings as cfg
from core.exceptions import NoModelsFoundError, OllamaNotAvailableError

logger = logging.getLogger(__name__)


class OllamaService:
    """Single responsibility: keep Ollama running and resolve which model to use."""

    _instance: Optional["OllamaService"] = None  # lightweight singleton

    def __new__(cls) -> "OllamaService":
        if cls._instance is None:
            cls._instance = super().__new__(cls)
            cls._instance._process = None
        return cls._instance

    # ── Status ──────────────────────────────────────────────────────────────

    def is_running(self) -> bool:
        try:
            r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=cfg.OLLAMA_TIMEOUT)
            return r.status_code == 200
        except requests.RequestException:
            return False

    # ── Lifecycle ────────────────────────────────────────────────────────────

    def ensure_running(self) -> None:
        """Start Ollama if not already running. Raises OllamaNotAvailableError on failure."""
        if self.is_running():
            return

        logger.info("Starting Ollama server…")
        try:
            self._process = subprocess.Popen(
                ["ollama", "serve"],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
            )
        except FileNotFoundError as exc:
            raise OllamaNotAvailableError(
                "ollama binary not found – is Ollama installed and on PATH?"
            ) from exc

        for tick in range(cfg.OLLAMA_STARTUP_WAIT):
            time.sleep(1)
            if self.is_running():
                logger.info("Ollama ready after %d s.", tick + 1)
                return
            logger.debug("Waiting for Ollama (%d/%d)…", tick + 1, cfg.OLLAMA_STARTUP_WAIT)

        raise OllamaNotAvailableError(
            f"Ollama did not become ready within {cfg.OLLAMA_STARTUP_WAIT} s."
        )

    def stop(self) -> None:
        """Stop Ollama to free RAM between heavy phases."""
        logger.info("Stopping Ollama to reclaim RAM…")
        
        # Use platform-appropriate command to kill Ollama
        if platform.system() == "Windows":
            subprocess.run(["taskkill", "/F", "/IM", "ollama.exe"], capture_output=True, check=False)
        else:
            # Linux, WSL, macOS
            subprocess.run(["pkill", "-f", "ollama"], capture_output=True, check=False)

        if self._process:
            self._process.terminate()
            self._process = None

        time.sleep(2)
        logger.info("Ollama stopped.")

    # ── Model resolution ─────────────────────────────────────────────────────

    def resolve_model(self) -> str:
        """
        Return CrewAI-compatible model string (e.g. ``ollama/mistral:7b``).

        Priority
        --------
        1. OLLAMA_MODEL env var  (explicit pin – strongly recommended for prod)
        2. MISTRAL_MODEL env var (legacy alias)
        3. First Mistral-family model found in running Ollama instance
        4. First available model of any kind
        """
        for env_val in (cfg.OLLAMA_MODEL, cfg.MISTRAL_MODEL):
            if env_val and env_val.strip():
                model = env_val.strip()
                return model if model.startswith("ollama/") else f"ollama/{model}"

        self.ensure_running()

        try:
            r = requests.get(f"{cfg.OLLAMA_BASE_URL}/api/tags", timeout=cfg.OLLAMA_TIMEOUT)
            r.raise_for_status()
            names: list[str] = [m.get("name", "") for m in r.json().get("models", [])]
        except Exception as exc:
            raise NoModelsFoundError(f"Could not query Ollama model list: {exc}") from exc

        if not names:
            raise NoModelsFoundError("No models installed in Ollama. Run: ollama pull mistral")

        preferred = [n for n in names if "mistral" in n.lower()]
        chosen = (preferred or names)[0]
        logger.info("Auto-selected model: %s", chosen)
        return f"ollama/{chosen}"

    # ── Health probe ─────────────────────────────────────────────────────────

    def test_generation(self, model_name: str) -> Tuple[bool, str]:
        """Send a minimal generation request to verify the model can respond."""
        api_model = model_name.replace("ollama/", "")
        try:
            r = requests.post(
                f"{cfg.OLLAMA_BASE_URL}/api/generate",
                json={
                    "model": api_model,
                    "prompt": "Reply with YES only.",
                    "stream": False,
                    "options": {"num_predict": 1},  # 1 token — fast even on cold start
                },
                timeout=cfg.OLLAMA_GEN_TIMEOUT,
            )
            if r.status_code == 200:
                return True, r.json().get("response", "")
            return False, f"HTTP {r.status_code}"
        except Exception as exc:
            return False, str(exc)

    def diagnose_llm_stack(self) -> Tuple[bool, str]:
        """
        Pre-flight check for the LLM stack before building agents.
        Returns (ok: bool, message: str).

        Catches the most common failure: litellm not installed for crewai >= 0.80.
        """
        try:
            import litellm  # noqa: F401
        except ImportError:
            return False, (
                "litellm is not installed – required by crewai >= 0.80.\n"
                "Fix: pip install litellm"
            )

        try:
            from crewai import LLM
            model = self.resolve_model()
            LLM(model=model, base_url=cfg.OLLAMA_BASE_URL)
        except ImportError:
            pass  # crewai < 0.80, plain-string LLM is fine
        except Exception as exc:
            return False, f"crewai.LLM construction failed: {exc}"

        return True, "LLM stack OK"


# ── Provider-agnostic model resolver ─────────────────────────────────────────

def get_model_string() -> str:
    """
    Return the CrewAI/litellm-compatible model string for the configured provider.

    - Mistral API mode  →  ``mistral/mistral-large-latest``  (litellm prefix)
    - Ollama mode       →  ``ollama/<model>``                (existing resolver)
    """
    if cfg.LLM_PROVIDER == "mistral_api":
        return f"mistral/{cfg.MISTRAL_ORCHESTRATOR_MODEL}"
    return OllamaService().resolve_model()
