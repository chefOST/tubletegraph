#!/usr/bin/env python3
"""Map TubeletGraph event descriptions to one canonical MOSCATO state each."""

from __future__ import annotations

import argparse
import base64
import copy
import hashlib
import json
import mimetypes
import os
from pathlib import Path
import re
from typing import Any

from openai import OpenAI


def load_json(path: Path) -> Any:
    with path.open(encoding="utf-8") as handle:
        return json.load(handle)


def canonical_states(
    state_dict: dict[str, Any], candidates: list[str] | None = None
) -> list[str]:
    all_states = {state.strip().lower() for state in state_dict["s2i"]}
    states = sorted(
        all_states
        if candidates is None
        else {state.strip().lower() for state in candidates}
    )
    if not states:
        raise ValueError("The MOSCATO candidate state list is empty")
    unknown = sorted(set(states) - all_states)
    if unknown:
        raise ValueError(f"Unknown MOSCATO candidate states: {unknown}")
    return states


def normalize_candidate(
    response: str, state_dict: dict[str, Any], allowed: list[str]
) -> str:
    candidate = response.strip().lower()
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", candidate).strip()
    if candidate.startswith("{"):
        try:
            payload = json.loads(candidate)
            candidate = str(payload["state"]).strip().lower()
        except (json.JSONDecodeError, KeyError, TypeError):
            pass
    candidate = candidate.strip("\"'` .")

    aliases = {
        key.strip().lower(): value.strip().lower()
        for key, value in state_dict.get("all2one", {}).items()
    }
    candidate = aliases.get(candidate, candidate)
    if candidate not in set(allowed):
        raise ValueError(f"Adapter returned non-canonical state {candidate!r}")
    return candidate


def semantic_evidence(raw_context: dict[str, Any]) -> list[str]:
    """Return only semantic text actually emitted by TubeletGraph."""
    evidence: list[str] = []
    semantic_keys = {"description", "prior_description", "action"}

    def visit(value: Any, key: str = "") -> None:
        if isinstance(value, dict):
            for child_key, child_value in value.items():
                visit(child_value, child_key)
        elif isinstance(value, list):
            for child in value:
                visit(child, key)
        elif key in semantic_keys and isinstance(value, str) and value.strip():
            evidence.append(value.strip())

    visit(raw_context)
    return evidence


def image_payload(path: Path) -> dict[str, Any]:
    """Return a low-detail vision payload without exposing a filename or path."""
    if not path.is_file():
        raise FileNotFoundError(path)
    mime_type = mimetypes.guess_type(path.name)[0] or "image/jpeg"
    encoded = base64.b64encode(path.read_bytes()).decode("ascii")
    return {
        "type": "image_url",
        "image_url": {
            "url": f"data:{mime_type};base64,{encoded}",
            "detail": "low",
        },
    }


DEFAULT_VISUAL_DEFINITIONS_PATH = (
    Path(__file__).resolve().parent.parent
    / "configs"
    / "moscato_state_visual_definitions.json"
)


def load_visual_definitions(
    path: Path = DEFAULT_VISUAL_DEFINITIONS_PATH,
) -> dict[str, str]:
    """Return per-state visual definitions, or {} when the config is absent."""
    if not path.is_file():
        return {}
    payload = load_json(path)
    return {
        str(key).strip().lower(): str(value).strip()
        for key, value in payload.get("definitions", {}).items()
    }


def frame_sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_event(
    client: OpenAI,
    *,
    model: str,
    target_object: str,
    raw_context: dict[str, Any],
    frame_path: Path,
    allowed: list[str],
    state_dict: dict[str, Any],
    visual_definitions: dict[str, str] | None = None,
) -> tuple[str | None, list[str], str | None]:
    """Return (state, raw_responses, observation).

    The prompt deliberately omits the MOSCATO alias lists: aliases are only used
    server-side in normalize_candidate. Showing them invited lexical matching
    between TubeletGraph words and alias words instead of looking at the frame.
    """
    evidence = semantic_evidence(raw_context)
    if not evidence:
        return None, [], None

    definitions = visual_definitions or {}
    state_guide = {state: definitions.get(state, state) for state in allowed}
    prompt = (
        "Map a TubeletGraph prediction to the MOSCATO state vocabulary. "
        "This is prediction post-processing. You are not given any ground-truth "
        "annotation timeline, video identifier, filename, or frame number. Use only "
        "the supplied image and raw TubeletGraph semantic text. The selected state "
        f"must describe the current visible state of the target object "
        f"{target_object!r}.\n\n"
        "Selection policy: first describe what is actually visible on the target "
        "object in the image. Then choose the one allowed state whose definition "
        "matches that observation. The TubeletGraph text is a hint about what "
        "changed; if it names a relation or attribute that is not visible in the "
        "image, ignore that part of the text. Do not choose a state because its "
        "name resembles a word in the text. Do not infer an unobserved action.\n\n"
        "Return a single JSON object with exactly two keys: "
        '"observation" (one sentence describing the visible target object) and '
        '"state" (one exact string copied from ALLOWED_STATES). No other text.\n\n'
        f"RAW_TUBELET_SEMANTIC_TEXT={json.dumps(evidence)}\n\n"
        f"ALLOWED_STATE_DEFINITIONS={json.dumps(state_guide, sort_keys=True)}\n\n"
        f"ALLOWED_STATES={json.dumps(allowed)}"
    )
    responses: list[str] = []
    for attempt in range(3):
        correction = ""
        if responses:
            correction = (
                "\n\nYour prior response was not an allowed state. Return the JSON "
                'object again with "state" copied exactly from ALLOWED_STATES.'
            )
        response = client.chat.completions.create(
            model=model,
            messages=[
                {
                    "role": "system",
                    "content": (
                        "You are a deterministic visual and semantic state-label "
                        "normalizer. "
                        "Follow the supplied closed vocabulary exactly."
                    ),
                },
                {
                    "role": "user",
                    "content": [
                        {"type": "text", "text": prompt + correction},
                        image_payload(frame_path),
                    ],
                },
            ],
            temperature=0,
        )
        raw_response = response.choices[0].message.content or ""
        responses.append(raw_response)
        try:
            state = normalize_candidate(raw_response, state_dict, allowed)
        except ValueError:
            if attempt == 2:
                raise
            continue
        return state, responses, extract_observation(raw_response)
    raise AssertionError("unreachable")


def extract_observation(response: str) -> str | None:
    """Return the adapter's free-text observation, if it returned JSON."""
    candidate = re.sub(r"^```(?:json)?\s*|\s*```$", "", response.strip()).strip()
    if not candidate.startswith("{"):
        return None
    try:
        payload = json.loads(candidate)
    except json.JSONDecodeError:
        return None
    observation = payload.get("observation") if isinstance(payload, dict) else None
    return str(observation).strip() if observation else None


def map_events(
    draft: dict[str, Any],
    state_dict: dict[str, Any],
    frames_dir: Path,
    client: OpenAI,
    model: str,
    allowed_states: list[str] | None = None,
    visual_definitions: dict[str, str] | None = None,
) -> dict[str, Any]:
    mapped = copy.deepcopy(draft)
    allowed = canonical_states(state_dict, allowed_states)
    if visual_definitions is None:
        visual_definitions = load_visual_definitions()
    target_object = mapped["object"]
    num_local_frames = int(mapped["num_local_frames"])
    initial_frame_path = frames_dir / "0000000.jpg"

    initial_state, initial_responses, initial_observation = classify_event(
        client,
        model=model,
        target_object=target_object,
        raw_context=mapped.get("initial_raw", {}),
        frame_path=initial_frame_path,
        allowed=allowed,
        state_dict=state_dict,
        visual_definitions=visual_definitions,
    )
    mapped["initial_state"] = initial_state
    mapped["initial_mapping"] = {
        "model": model,
        "raw_responses": initial_responses,
        "observation": initial_observation,
        "mapped_state": initial_state,
        "status": "mapped" if initial_state is not None else "abstained_no_tubelet_text",
        "semantic_evidence": semantic_evidence(mapped.get("initial_raw", {})),
        "input_frame_filename": initial_frame_path.name,
        "input_frame_sha256": (
            frame_sha256(initial_frame_path) if initial_responses else None
        ),
    }

    last_frame = -1
    for transition in mapped.get("transitions", []):
        local_frame = int(transition["local_frame"])
        if not 0 <= local_frame < num_local_frames:
            raise ValueError(
                f"Transition frame {local_frame} is outside [0, {num_local_frames})"
            )
        if local_frame <= last_frame:
            raise ValueError("Transition frames must be unique and strictly increasing")
        last_frame = local_frame
        frame_path = frames_dir / f"{local_frame:07d}.jpg"
        raw_context = {"raw_nodes": transition.get("raw_nodes", [])}
        state, responses, observation = classify_event(
            client,
            model=model,
            target_object=target_object,
            raw_context=raw_context,
            frame_path=frame_path,
            allowed=allowed,
            state_dict=state_dict,
            visual_definitions=visual_definitions,
        )
        transition["state"] = state
        transition["mapping"] = {
            "model": model,
            "raw_responses": responses,
            "observation": observation,
            "mapped_state": state,
            "status": "mapped" if state is not None else "abstained_no_tubelet_text",
            "semantic_evidence": semantic_evidence(raw_context),
            "input_frame_filename": frame_path.name,
            "input_frame_sha256": frame_sha256(frame_path) if responses else None,
        }

    vocabulary_digest = hashlib.sha256(
        json.dumps(allowed, separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    mapping_statuses = [mapped["initial_mapping"]["status"]] + [
        transition["mapping"]["status"]
        for transition in mapped.get("transitions", [])
    ]
    mapped_count = mapping_statuses.count("mapped")
    if mapped_count == len(mapping_statuses):
        overall_mapping_status = "mapped"
    elif mapped_count:
        overall_mapping_status = "partially_mapped"
    else:
        overall_mapping_status = "abstained_no_tubelet_text"
    mapped.setdefault("provenance", {}).update(
        {
            "state_mapping_status": overall_mapping_status,
            "state_mapping_model": model,
            "ground_truth_used_for_mapping": False,
            "video_frames_used_for_mapping": bool(mapped_count),
            "adapter_input_mode": "tubelet_semantic_text_plus_single_event_frame",
            "adapter_image_detail": "low",
            "adapter_prompt_version": 2,
            "adapter_prompt_shows_aliases": False,
            "adapter_output_format": "json_observation_then_state",
            "video_identifiers_used_in_prompt": False,
            "frame_numbers_used_in_prompt": False,
            "allowed_states_sha256": vocabulary_digest,
            "allowed_states": allowed,
        }
    )
    return mapped


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--draft", type=Path, required=True)
    parser.add_argument("--state-dict", type=Path, required=True)
    parser.add_argument("--frames", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--model", default="gpt-4.1")
    args = parser.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        raise RuntimeError(
            "OPENAI_API_KEY is missing. Use the Modal tubelet-openai secret; "
            "never put the key in a repository file."
        )
    mapped = map_events(
        load_json(args.draft),
        load_json(args.state_dict),
        args.frames,
        OpenAI(),
        args.model,
    )
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(mapped, indent=2) + "\n", encoding="utf-8")
    print(f"Mapped events: {args.out}")
    print(f"Initial state: {mapped['initial_state']}")
    print(f"Transitions: {len(mapped.get('transitions', []))}")


if __name__ == "__main__":
    main()
