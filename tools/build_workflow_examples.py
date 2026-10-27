from __future__ import annotations

import argparse
import copy
import json
from pathlib import Path
from typing import Any


EXACT_SOURCE = Path("r2v_lipsync_fast_test_20261026/workflows/hemsworth_exact_audio_lock_api.json")
T8_SOURCE = Path("comfyui-minimax-h3-audio-T8/examples/source_video_repaint_api.json")


def load(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def validate(graph: dict[str, Any]) -> None:
    for node_id, node in graph.items():
        if not isinstance(node, dict) or "class_type" not in node:
            raise ValueError(f"node {node_id!r} is not an API-format ComfyUI node")
        for name, value in node.get("inputs", {}).items():
            if (
                isinstance(value, list)
                and len(value) == 2
                and isinstance(value[0], str)
                and isinstance(value[1], int)
                and value[0] not in graph
            ):
                raise ValueError(f"node {node_id!r} input {name!r} references missing node {value[0]!r}")


def trace_node(model_input, source_input, target_input, label: str, tag: str) -> dict[str, Any]:
    return {
        "class_type": "MiniMaxH3AVInternalTrace",
        "inputs": {
            "model": model_input,
            "workflow_label": label,
            "run_tag": tag,
            "blocks": "auto",
            "steps": "auto",
            "capture_unconditional": False,
            "capture_cross_stream_attention": False,
            "attention_queries_per_stream": 16,
            "source_drift_thresholds": "0.02,0.05,0.10,0.20",
            "score_floor": 0.0001,
            "source_video_latent": source_input,
            "target_av_latent": target_input,
        },
        "_meta": {"title": "Output-neutral H3 internal trace"},
    }


def build_exact(source: dict[str, Any]) -> dict[str, Any]:
    graph = copy.deepcopy(source)
    graph["trace_source_video_latent"] = {
        "class_type": "VAEEncode",
        "inputs": {
            "pixels": ["r2v_input_video_components", 0],
            "vae": ["119", 0],
        },
        "_meta": {"title": "Trace-only source video encode"},
    }
    graph["trace_internal"] = trace_node(
        ["127", 0],
        ["trace_source_video_latent", 0],
        ["r2v_exact_audio_lock", 0],
        "exact_ref2va_good_sync",
        "exact_ref2va_good_sync",
    )
    graph["126"]["inputs"]["model"] = ["trace_internal", 0]
    validate(graph)
    return graph


def build_t8(source: dict[str, Any]) -> dict[str, Any]:
    graph = copy.deepcopy(source)
    graph["trace_internal"] = trace_node(
        ["12", 0],
        ["6", 0],
        ["8", 0],
        "t8_fl2va_source_preservation",
        "t8_fl2va_source_preservation",
    )
    graph["14"]["inputs"]["model"] = ["trace_internal", 0]
    validate(graph)
    return graph


def write(path: Path, graph: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(graph, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser(description="Build traced copies of the two H3 control workflows")
    parser.add_argument(
        "--workspace",
        type=Path,
        default=Path(__file__).resolve().parents[2],
        help="Path to the minimax_h3 workspace",
    )
    args = parser.parse_args()
    output = Path(__file__).resolve().parents[1] / "examples"
    write(output / "exact_audio_lock_trace_api.json", build_exact(load(args.workspace / EXACT_SOURCE)))
    write(output / "source_video_repaint_trace_api.json", build_t8(load(args.workspace / T8_SOURCE)))


if __name__ == "__main__":
    main()
