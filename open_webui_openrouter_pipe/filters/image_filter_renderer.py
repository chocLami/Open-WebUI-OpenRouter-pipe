"""Image generation filter source code renderer.

Three filter variants — `generic`, `gemini`, `sourceful` — written to
`body.image_config` as top-level request fields per OpenRouter's
[image-generation.md](.external/openrouter_docs/guides/overview/multimodal/image-generation.md).
The pipe's orchestrator injects `body.modalities` separately based on the
registered model's `architecture.output_modalities` so filters do not need
runtime registry access.

Filter assignment rules (driven by `filter_manager.ensure_openrouter_image_filter_function_ids`):
- **All** models with `image` in `output_modalities` get the **generic** filter
- Models matching `^google/gemini-.*flash-image.*-preview$` ALSO get **gemini** filter (extended aspect ratios + 0.5K)
- Models matching `^sourceful/riverflow-v\\d+(\\.\\d+)?-(pro|fast)$` ALSO get **sourceful** filter (font_inputs + super_resolution_references)
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from ..core.config import _OPENROUTER_IMAGE_FILTER_MARKER

_IMAGE_FILTER_ID_RE = re.compile(r"[^a-zA-Z0-9_]+")


@dataclass(frozen=True, slots=True)
class ImageFilterSpec:
    """Metadata for an image-generation filter variant."""

    variant: str  # "generic" | "gemini" | "sourceful"
    function_id: str
    display_name: str
    marker: str


def sanitize_image_filter_id(variant: str) -> str:
    raw = (variant or "generic").strip().lower()
    cleaned = _IMAGE_FILTER_ID_RE.sub("_", raw).strip("_")
    if not cleaned:
        cleaned = "generic"
    if len(cleaned) > 30:
        suffix = hashlib.sha1(variant.encode("utf-8")).hexdigest()[:8]
        cleaned = f"{cleaned[:21].rstrip('_')}_{suffix}"
    return f"openrouter_image_filter_{cleaned}"


def build_generic_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="generic",
        function_id=sanitize_image_filter_id("generic"),
        display_name="OR Image Filter",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:generic",
    )


def build_gemini_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="gemini",
        function_id=sanitize_image_filter_id("gemini"),
        display_name="Gemini Options",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:gemini",
    )


def build_sourceful_image_filter_spec() -> ImageFilterSpec:
    return ImageFilterSpec(
        variant="sourceful",
        function_id=sanitize_image_filter_id("sourceful"),
        display_name="Sourceful Options",
        marker=f"{_OPENROUTER_IMAGE_FILTER_MARKER}:sourceful",
    )


def render_generic_image_filter_source() -> str:
    """Render the generic image filter — aspect_ratio (10 standard) + image_size (1K/2K/4K).

    Attached to ALL models with `image` in `output_modalities`. Inlet logic:
    1. Read user-valves
    2. Build image_config overrides dict
    3. Shallow-merge into body.image_config (overrides per-key; preserves any
       user-supplied keys not also set by this filter's UserValves)
    """
    spec = build_generic_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — generic."""

from __future__ import annotations

import json
import logging
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover - OWUI runtime only
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(
            default=0,
            description="Priority level for the filter operations.",
        )

    class UserValves(BaseModel):
        IMAGE_ASPECT_RATIO: Literal[
            "", "1:1", "2:3", "3:2", "3:4", "4:3", "4:5", "5:4", "9:16", "16:9", "21:9",
        ] = Field(
            default="",
            title="Image aspect ratio",
            description="Aspect ratio for generated images. Empty = model default.",
        )
        IMAGE_SIZE: Literal["", "1K", "2K", "4K"] = Field(
            default="",
            title="Image size",
            description="Image resolution tier. Empty = model default (1K).",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}
        aspect = (user_valves.IMAGE_ASPECT_RATIO or "").strip()
        if aspect:
            overrides["aspect_ratio"] = aspect
        size = (user_valves.IMAGE_SIZE or "").strip()
        if size:
            overrides["image_size"] = size

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''


def render_gemini_image_filter_source() -> str:
    """Render the Gemini-specific image filter — extended aspect ratios + 0.5K size.

    Attached only to models matching `^google/gemini-.*flash-image.*-preview$`.
    Shallow-merges into body.image_config alongside the generic filter (per-key
    overwrite; if both filters write the same key, the second one wins).
    """
    spec = build_gemini_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — Gemini extensions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"

_GEMINI_MODEL_PATTERN = re.compile(r"^google/gemini-.*flash-image.*-preview$")


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(default=0)

    class UserValves(BaseModel):
        IMAGE_ASPECT_RATIO_EXTENDED: Literal["", "1:4", "4:1", "1:8", "8:1"] = Field(
            default="",
            title="Image aspect ratio (Gemini extended)",
            description="Gemini-only extended aspect ratios. Overrides the generic aspect_ratio when set. Empty = use generic filter's value.",
        )
        IMAGE_SIZE_GEMINI: Literal["", "0.5K"] = Field(
            default="",
            title="Image size (Gemini-only 0.5K)",
            description="Gemini Flash Image only — 0.5K low-res tier. Empty = use generic filter's value.",
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        # Model gate: don't emit Gemini-specific knobs unless the model is a
        # Gemini Flash Image Preview variant. Defends against operator misconfiguration
        # (filter manually attached to non-Gemini model would otherwise send invalid params).
        model_id = body.get("model") or ""
        if not isinstance(model_id, str) or not _GEMINI_MODEL_PATTERN.match(model_id):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}
        ext_aspect = (user_valves.IMAGE_ASPECT_RATIO_EXTENDED or "").strip()
        if ext_aspect:
            overrides["aspect_ratio"] = ext_aspect
        gemini_size = (user_valves.IMAGE_SIZE_GEMINI or "").strip()
        if gemini_size:
            overrides["image_size"] = gemini_size

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''


def render_sourceful_image_filter_source() -> str:
    """Render the Sourceful-specific image filter — font_inputs + super_resolution_references.

    Attached only to models matching `^sourceful/riverflow-v\\d+(\\.\\d+)?-(pro|fast)$`.
    Pre-validates cardinality caps (max 2 font_inputs, max 4 super_resolution_references)
    and rejects invalid input BEFORE submission so users get clear errors instead of
    cryptic provider 400s.
    """
    spec = build_sourceful_image_filter_spec()
    return f'''"""OpenRouter image generation companion filter — Sourceful extensions."""

from __future__ import annotations

import json
import logging
import re
from typing import Any, Literal, Optional

from pydantic import BaseModel, Field

try:
    from open_webui.env import SRC_LOG_LEVELS
except Exception:  # pragma: no cover
    SRC_LOG_LEVELS = {{}}

OWUI_OPENROUTER_PIPE_MARKER = "{spec.marker}"
IMAGE_FILTER_VARIANT = "{spec.variant}"

_SOURCEFUL_MODEL_PATTERN = re.compile(r"^sourceful/riverflow-v\\d+(\\.\\d+)?-(pro|fast)$")
_MAX_FONT_INPUTS = 2
_MAX_SUPER_RESOLUTION_REFERENCES = 4


class ImageGenerationError(Exception):
    """Raised at inlet when Sourceful-specific limits or input validation fail
    (font_inputs cardinality > 2, super_resolution_references > 4, malformed
    JSON, missing required fields)."""


class Filter:
    toggle = True

    class Valves(BaseModel):
        priority: int = Field(default=0)

    class UserValves(BaseModel):
        IMAGE_FONT_INPUTS_JSON: str = Field(
            default="",
            title="Font inputs (JSON array)",
            description=(
                'Sourceful-only font rendering. JSON array of objects: '
                '[{{"font_url": "https://...", "text": "..."}}]. Max 2 entries, +$0.03 each. '
                'Empty = none.'
            ),
        )
        IMAGE_SUPER_RESOLUTION_REFERENCES_JSON: str = Field(
            default="",
            title="Super-resolution references (JSON array)",
            description=(
                'Sourceful-only image-to-image super-resolution. JSON array of URL strings. '
                'Max 4 entries, +$0.20 each. Image-to-image only (requires input images in messages). '
                'Empty = none.'
            ),
        )

    def __init__(self) -> None:
        self.log = logging.getLogger("openrouter.image.filter.{spec.variant}")
        self.log.setLevel(SRC_LOG_LEVELS.get("OPENAI", logging.INFO))
        self.toggle = True
        self.valves = self.Valves()

    def _parse_json_list(self, raw: str, field: str) -> list:
        cleaned = (raw or "").strip()
        if not cleaned:
            return []
        try:
            parsed = json.loads(cleaned)
        except json.JSONDecodeError as exc:
            raise ImageGenerationError(
                f"{{field}} is not valid JSON: {{exc}}"
            )
        if not isinstance(parsed, list):
            raise ImageGenerationError(
                f"{{field}} must be a JSON array, got {{type(parsed).__name__}}"
            )
        return parsed

    def inlet(
        self,
        body: dict,
        __metadata__: Optional[dict] = None,
        __user__: Optional[dict] = None,
    ) -> dict:
        if not isinstance(body, dict):
            return body
        # Model gate: only emit Sourceful-specific knobs for Riverflow Pro/Fast
        # variants. Defends against operator misconfiguration (filter manually
        # attached to non-Sourceful model would otherwise emit invalid params).
        model_id = body.get("model") or ""
        if not isinstance(model_id, str) or not _SOURCEFUL_MODEL_PATTERN.match(model_id):
            return body
        user_valves = None
        if isinstance(__user__, dict):
            uv_raw = __user__.get("valves")
            if uv_raw is not None and not isinstance(uv_raw, self.UserValves):
                try:
                    user_valves = self.UserValves.model_validate(
                        uv_raw if isinstance(uv_raw, dict) else uv_raw.model_dump()
                    )
                except Exception:
                    user_valves = self.UserValves()
            elif isinstance(uv_raw, self.UserValves):
                user_valves = uv_raw
        if user_valves is None:
            user_valves = self.UserValves()

        overrides: dict = {{}}

        font_inputs = self._parse_json_list(
            user_valves.IMAGE_FONT_INPUTS_JSON, "IMAGE_FONT_INPUTS_JSON"
        )
        if font_inputs:
            if len(font_inputs) > _MAX_FONT_INPUTS:
                raise ImageGenerationError(
                    f"font_inputs has {{len(font_inputs)}} entries; max is {{_MAX_FONT_INPUTS}}."
                )
            for idx, entry in enumerate(font_inputs):
                if not isinstance(entry, dict):
                    raise ImageGenerationError(
                        f"font_inputs[{{idx}}] must be an object with 'font_url' and 'text', "
                        f"got {{type(entry).__name__}}."
                    )
                if not entry.get("font_url") or not entry.get("text"):
                    raise ImageGenerationError(
                        f"font_inputs[{{idx}}] requires non-empty 'font_url' and 'text'."
                    )
            overrides["font_inputs"] = font_inputs

        super_refs = self._parse_json_list(
            user_valves.IMAGE_SUPER_RESOLUTION_REFERENCES_JSON,
            "IMAGE_SUPER_RESOLUTION_REFERENCES_JSON",
        )
        if super_refs:
            if len(super_refs) > _MAX_SUPER_RESOLUTION_REFERENCES:
                raise ImageGenerationError(
                    f"super_resolution_references has {{len(super_refs)}} entries; max is {{_MAX_SUPER_RESOLUTION_REFERENCES}}."
                )
            for idx, entry in enumerate(super_refs):
                if not isinstance(entry, str) or not entry.strip():
                    raise ImageGenerationError(
                        f"super_resolution_references[{{idx}}] must be a non-empty URL string."
                    )
            overrides["super_resolution_references"] = super_refs

        if overrides:
            existing = body.get("image_config")
            if not isinstance(existing, dict):
                existing = {{}}
            else:
                existing = dict(existing)
            existing.update(overrides)
            body["image_config"] = existing
        return body
'''
