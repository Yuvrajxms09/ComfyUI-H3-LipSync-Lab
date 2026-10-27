from __future__ import annotations

import json
from pathlib import Path
import sys

import torch


ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from trace_core import (  # noqa: E402
    TraceSession,
    build_summary,
    parse_thresholds,
    resolve_indices,
    segment_metrics,
    source_drift_metrics,
)


def test_resolve_indices() -> None:
    assert resolve_indices("auto", 50, default_fractions=(0.1, 0.5, 0.9)) == [5, 24, 44]
    assert resolve_indices("0,-1,24", 50, default_fractions=(0.1, 0.5, 0.9)) == [0, 24, 49]
    assert resolve_indices("all", 3, default_fractions=(0.1, 0.5, 0.9)) == [0, 1, 2]


def test_segment_metrics_are_exact_for_identity() -> None:
    source = torch.arange(24, dtype=torch.float32).reshape(6, 4)
    rows = segment_metrics(source, source.clone(), [(0, 2, "text"), (2, 4, "audio"), (4, 6, "video")])
    assert {row["segment_kind"] for row in rows} == {"text", "audio", "video"}
    assert all(row["residual_rms"] == 0.0 for row in rows)
    assert all(abs(row["cosine"] - 1.0) < 1e-6 for row in rows)


def test_segment_metrics_group_repeated_kinds() -> None:
    source = torch.ones(1, 6, 2)
    output = source.clone()
    output[:, 4:] += 1
    rows = segment_metrics(source, output, [(0, 2, "ref_img"), (2, 4, "audio"), (4, 6, "ref_img")])
    ref = next(row for row in rows if row["segment_kind"] == "ref_img")
    assert ref["token_count"] == 4
    assert ref["residual_rms"] > 0


def test_source_drift_identity_and_change() -> None:
    source = torch.ones(1, 24, 2, 4, 4)
    summary, frames = source_drift_metrics(source.clone(), source)
    assert summary["relative_l2"] == 0.0
    assert summary["token_fraction_gt_0.02"] == 0.0
    assert len(frames) == 2

    changed = source.clone()
    changed[:, :, 1, 2:, 2:] += 1
    summary, frames = source_drift_metrics(changed, source)
    assert summary["relative_l2"] > 0
    assert summary["token_fraction_gt_0.02"] > 0
    assert frames[1]["relative_l2"] > frames[0]["relative_l2"]


def test_trace_session_writes_bounded_artifacts(tmp_path: Path) -> None:
    session = TraceSession(
        base_dir=tmp_path,
        run_tag="exact/ref2va",
        workflow_label="exact_ref2va_good_sync",
        steps_spec="auto",
        capture_unconditional=False,
        thresholds=parse_thresholds("0.02,0.1"),
        score_floor=1e-4,
        source_video=torch.ones(1, 24, 2, 4, 4),
        attention_queries_per_stream=4,
        static_manifest={"selected_blocks": [5, 24, 44]},
    )
    run = session.begin(latent_shapes=None)
    session.record_guided_video(
        torch.ones(1, 24, 2, 4, 4), sigma=0.5, step=1, source_kind="guided_x0"
    )
    output = session.end()
    assert output == run.output_dir
    assert (output / "manifest.json").exists()
    assert (output / "step_metrics.csv").exists()
    manifest = json.loads((output / "manifest.json").read_text())
    assert manifest["status"] == "complete"
    assert manifest["workflow_label"] == "exact_ref2va_good_sync"


def test_build_summary_groups_block_and_segment() -> None:
    rows = [
        {"block": 5, "segment_kind": "video", "relative_residual_rms": 1.0, "cosine": 0.9},
        {"block": 5, "segment_kind": "video", "relative_residual_rms": 3.0, "cosine": 0.7},
    ]
    summary = build_summary(rows, [])
    assert summary["block_segment_means"][0]["mean_relative_residual_rms"] == 2.0
