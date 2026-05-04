"""HTTP client for OpenRouter image-output model catalog (subset of video_client.py).

Image generation itself goes through the standard `/api/v1/chat/completions`
endpoint per [image-generation.md line 7](.external/openrouter_docs/guides/overview/multimodal/image-generation.md);
this client is only responsible for fetching the model catalog filtered to
image-output models.
"""

from __future__ import annotations

from typing import Any

import aiohttp

from ..core.config import _OPENROUTER_CATEGORIES, _OPENROUTER_REFERER, _OPENROUTER_TITLE
from ..requests.debug import (
    _debug_print_error_response,
    _debug_print_request,
    _debug_print_response,
)


class OpenRouterImageClient:

    def __init__(
        self,
        session: aiohttp.ClientSession,
        *,
        base_url: str,
        api_key: str,
        logger: Any,
        http_referer: str | None = None,
    ) -> None:
        self._session = session
        self._base_url = (base_url or "https://openrouter.ai/api/v1").rstrip("/")
        self._api_key = api_key
        self._logger = logger
        self._http_referer = http_referer or _OPENROUTER_REFERER

    def _headers(self) -> dict[str, str]:
        if not self._api_key:
            raise RuntimeError("OpenRouter API key is required for image catalog fetch.")
        return {
            "Authorization": f"Bearer {self._api_key}",
            "X-OpenRouter-Title": _OPENROUTER_TITLE,
            "X-OpenRouter-Categories": _OPENROUTER_CATEGORIES,
            "HTTP-Referer": self._http_referer,
        }

    async def list_models(self) -> list[dict[str, Any]]:
        """Fetch image-output models via the standard models endpoint with the
        `output_modalities=image` query parameter (see image-generation.md
        lines 13-25). Returns models whose output_modalities include "image"
        (both pure-image-only and multimodal text+image)."""
        url = f"{self._base_url}/models?output_modalities=image"
        headers = self._headers()
        _debug_print_request(headers, {"method": "GET", "url": url}, logger=self._logger)
        async with self._session.get(url, headers=headers) as resp:
            if resp.status >= 400:
                await _debug_print_error_response(resp, logger=self._logger)
            resp.raise_for_status()
            payload = await resp.json()
        _debug_print_response(payload, logger=self._logger)
        data = payload.get("data") if isinstance(payload, dict) else None
        return [item for item in data if isinstance(item, dict)] if isinstance(data, list) else []
