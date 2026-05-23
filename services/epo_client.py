# services/epo_client.py
# Copied from patent-search-automation/models/epo_client.py
# Extended with fetch_claims() and fetch_description() for gap analysis.

import logging
import os
import time
import requests
from requests.auth import HTTPBasicAuth
from lxml import etree

log = logging.getLogger(__name__)

class _Settings:
    @property
    def epo_consumer_key(self):    return os.environ.get("EPO_CLIENT_ID", "")
    @property
    def epo_consumer_secret(self): return os.environ.get("EPO_CLIENT_SECRET", "")

settings = _Settings()

NAMESPACES = {
    "ops":      "http://ops.epo.org",
    "exchange": "http://www.epo.org/exchange",
}


class EPOClient:
    """
    Thin wrapper around EPO OPS 3.2.
    Handles auth token lifecycle — token is cached and reused
    until it expires (EPO tokens last 20 minutes).
    """
    BASE = "https://ops.epo.org/3.2/rest-services"

    def __init__(self):
        self._token: str | None = None
        self._token_expiry: float = 0.0   # epoch seconds

    # ------------------------------------------------------------------
    # Auth
    # ------------------------------------------------------------------

    def _get_token(self) -> str:
        """Return cached token, or fetch a new one if expired."""
        if self._token and time.time() < self._token_expiry - 30:
            return self._token

        log.info("EPO: requesting new access token")
        resp = requests.post(
            "https://ops.epo.org/3.2/auth/accesstoken",
            auth=HTTPBasicAuth(
                settings.epo_consumer_key,
                settings.epo_consumer_secret,
            ),
            data={"grant_type": "client_credentials"},
            timeout=10,
        )
        if resp.status_code != 200:
            raise RuntimeError(
                f"EPO auth failed: {resp.status_code} — {resp.text[:200]}"
            )
        data = resp.json()
        self._token = data["access_token"]
        # EPO tokens expire in 1200s (20 min); cache with 30s safety margin
        self._token_expiry = time.time() + int(data.get("expires_in", 1200))
        log.info("EPO: token acquired, valid for ~20 min")
        return self._token

    def _headers(self) -> dict:
        return {
            "Authorization": f"Bearer {self._get_token()}",
            "Accept": "application/xml",
        }

    # ------------------------------------------------------------------
    # Step 2 — Search: returns list of epodoc IDs
    # ------------------------------------------------------------------

    def search(self, cql_query: str, max_results: int = 25) -> list[str]:
        """
        Search EPO and return epodoc-format patent IDs.
        Returns: ["US.20260127435.A1", "DE.102024210701.A1", ...]
        """
        url = f"{self.BASE}/published-data/search/biblio"
        params = {
            "q": cql_query,
            "Range": f"1-{min(max_results, 100)}",
        }
        log.info(f"EPO search: {cql_query}")
        resp = requests.get(url, headers=self._headers(),
                            params=params, timeout=15)

        if resp.status_code == 404:
            log.warning(f"EPO search returned 404 (no results): {cql_query}")
            return []
        if resp.status_code != 200:
            log.error(f"EPO search failed: {resp.status_code} — {resp.text[:300]}")
            return []

        root = etree.fromstring(resp.content)
        docs = root.xpath("//exchange:exchange-document",
                          namespaces=NAMESPACES)

        ids = []
        for doc in docs:
            country    = doc.get("country", "")
            doc_number = doc.get("doc-number", "")
            kind       = doc.get("kind", "")
            if country and doc_number and kind:
                ids.append(f"{country}.{doc_number}.{kind}")

        log.info(f"EPO search: {len(ids)} patent IDs found")
        return ids

    # ------------------------------------------------------------------
    # Step 3 — Fetch full record: title + abstract per ID
    # ------------------------------------------------------------------

    def fetch_biblio(self, epodoc_id: str) -> dict | None:
        """
        Fetch title + abstract for one epodoc ID.
        Returns: {"patent_id": ..., "title": ..., "abstract": ...}
        or None on failure.

        Mirrors test_epo.py Steps 4-5 exactly.
        """
        url = (f"{self.BASE}/published-data/publication"
               f"/epodoc/{epodoc_id}/biblio")

        resp = requests.get(url, headers=self._headers(), timeout=10)

        if resp.status_code != 200:
            log.warning(f"EPO biblio fetch failed for {epodoc_id}: "
                        f"{resp.status_code}")
            return None

        try:
            root = etree.fromstring(resp.content)

            # Title — prefer English, fall back to first available
            title_nodes = root.xpath(
                "//exchange:invention-title",
                namespaces=NAMESPACES
            )
            title = "NO TITLE FOUND"
            for node in title_nodes:
                if node.get("lang", "").lower() == "en" and node.text:
                    title = node.text.strip()
                    break
            if title == "NO TITLE FOUND" and title_nodes:
                title = (title_nodes[0].text or "").strip() or "NO TITLE FOUND"

            # Abstract — mirrors test_epo.py exactly
            abstract_nodes = root.xpath(
                "//exchange:abstract//exchange:p",
                namespaces=NAMESPACES
            )
            abstract_parts = [
                n.text.strip() for n in abstract_nodes if n.text
            ]
            abstract = " ".join(abstract_parts) or "NO ABSTRACT FOUND"

            return {
                "patent_id": epodoc_id,
                "title":     title,
                "abstract":  abstract,
            }

        except Exception as e:
            log.error(f"EPO parse error for {epodoc_id}: {e}")
            return None

    def fetch_claims(self, epodoc_id: str) -> str:
        """Fetch claims text for one patent. Returns plain text or empty string."""
        url = f"{self.BASE}/published-data/publication/epodoc/{epodoc_id}/claims"
        resp = requests.get(url, headers=self._headers(), timeout=15)
        if resp.status_code not in (200,):
            log.warning(f"EPO claims {resp.status_code} for {epodoc_id}")
            return ""
        try:
            root = etree.fromstring(resp.content)
            texts = []
            # Try claim-text elements
            paras = root.xpath("//*[local-name()='claim-text'] | //*[local-name()='p']")
            for i, p in enumerate(paras):
                t = "".join(p.itertext()).strip()
                if t:
                    texts.append(t)
            if not texts:
                texts.append(" ".join(root.itertext()).strip())
            return "\n".join(texts)
        except Exception as e:
            log.error(f"EPO claims parse error {epodoc_id}: {e}")
            return ""

    def fetch_description(self, epodoc_id: str) -> str:
        """Fetch description text for one patent. Returns plain text or empty string."""
        url = f"{self.BASE}/published-data/publication/epodoc/{epodoc_id}/description"
        resp = requests.get(url, headers=self._headers(), timeout=20)
        if resp.status_code not in (200,):
            log.warning(f"EPO description {resp.status_code} for {epodoc_id}")
            return ""
        try:
            root = etree.fromstring(resp.content)
            texts = []
            paras = root.xpath("//*[local-name()='p']")
            for p in paras:
                t = "".join(p.itertext()).strip()
                if t:
                    texts.append(t)
            if not texts:
                texts.append(" ".join(root.itertext()).strip())
            return "\n".join(texts)
        except Exception as e:
            log.error(f"EPO description parse error {epodoc_id}: {e}")
            return ""
