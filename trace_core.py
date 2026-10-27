from __future__ import annotations

import csv
import json
import math
import os
import re
import threading
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Sequence

import torch
import torch.nn.functional as F


DEFAULT_BLOCK_FRACTIONS = (0.10, 0.50, 0.90)
DEFAULT_STEP_FRACTIONS = (0.10, 0.50, 0.90)
DEFAULT_THRESHOLDS = (0.02, 0.05, 0.10, 0.20)


def safe_tag(value: str, fallback: str = "h3_trace") -> str:
    cleaned = re.sub(r"[^A-Za-z0-9._-]+", "_", (value or "").strip()).strip("._-")
    return cleaned[:80] or fallback


def resolve_indices(
    spec: str,
    total: int,
    *,
    default_fractions: Sequence[float],
) -> list[int]:
    if total <= 0:
        return []
    normalized = (spec or "auto").strip().lower()
    if normalized in {"", "auto"}:
        values = [round(fraction * (total - 1)) for fraction in default_fractions]
    elif normalized == "all":
        values = list(range(total))
    else:
        values = []
        for raw in normalized.split(","):
            raw = raw.strip()
            if not raw:
                continue
            index = int(raw)
            if index < 0:
                index += total
            if not 0 <= index < total:
                raise ValueError(f"index {raw!r} is outside [0, {total})")
            values.append(index)
    return sorted(set(int(value) for value in values if 0 <= int(value) < total))


def parse_thresholds(spec: str) -> tuple[float, ...]:
    if not (spec or "").strip():
        return DEFAULT_THRESHOLDS
    values = sorted({float(value.strip()) for value in spec.split(",") if value.strip()})
    if not values or values[0] < 0:
        raise ValueError("source-drift thresholds must be non-negative")
    return tuple(values)


def _scalar(value: torch.Tensor) -> float:
    return float(value.detach().to(device="cpu", dtype=torch.float64))


def _metric_row(
    source: torch.Tensor,
    output: torch.Tensor,
    *,
    kind: str,
    token_count: int,
) -> dict[str, Any]:
    source_f = source.float()
    output_f = output.float()
    delta = output_f - source_f
    count = max(1, source_f.numel())
    source_sq = source_f.square().sum()
    output_sq = output_f.square().sum()
    delta_sq = delta.square().sum()
    source_rms = torch.sqrt(source_sq / count)
    output_rms = torch.sqrt(output_sq / count)
    residual_rms = torch.sqrt(delta_sq / count)
    denom = torch.sqrt(source_sq * output_sq).clamp_min(1e-12)
    cosine = (source_f * output_f).sum() / denom
    return {
        "segment_kind": kind,
        "token_count": int(token_count),
        "element_count": int(source_f.numel()),
        "input_rms": _scalar(source_rms),
        "output_rms": _scalar(output_rms),
        "residual_rms": _scalar(residual_rms),
        "relative_residual_rms": _scalar(residual_rms / source_rms.clamp_min(1e-8)),
        "cosine": _scalar(cosine),
    }


def segment_metrics(
    source: torch.Tensor,
    output: torch.Tensor,
    segments: Iterable[tuple[int, int, str]],
) -> list[dict[str, Any]]:
    if source.shape != output.shape or source.ndim not in {2, 3}:
        raise ValueError(
            "H3 block metrics require matching [sequence, hidden] or "
            "[batch, sequence, hidden] tensors; "
            f"got {tuple(source.shape)} and {tuple(output.shape)}"
        )
    grouped: dict[str, list[tuple[int, int]]] = {}
    sequence = source.shape[-2]
    for start, stop, kind in segments:
        start, stop = int(start), int(stop)
        if start < 0 or stop > sequence or stop <= start:
            raise ValueError(f"invalid packed segment {(start, stop, kind)!r} for sequence {sequence}")
        grouped.setdefault(str(kind), []).append((start, stop))

    rows: list[dict[str, Any]] = []
    for kind, spans in grouped.items():
        sequence_dim = source.ndim - 2
        if source.ndim == 2:
            source_parts = [source[start:stop] for start, stop in spans]
            output_parts = [output[start:stop] for start, stop in spans]
        else:
            source_parts = [source[:, start:stop] for start, stop in spans]
            output_parts = [output[:, start:stop] for start, stop in spans]
        source_view = (
            source_parts[0]
            if len(source_parts) == 1
            else torch.cat(source_parts, dim=sequence_dim)
        )
        output_view = (
            output_parts[0]
            if len(output_parts) == 1
            else torch.cat(output_parts, dim=sequence_dim)
        )
        rows.append(
            _metric_row(
                source_view,
                output_view,
                kind=kind,
                token_count=sum(stop - start for start, stop in spans),
            )
        )
    return rows


def source_drift_metrics(
    video: torch.Tensor,
    source: torch.Tensor,
    *,
    thresholds: Sequence[float] = DEFAULT_THRESHOLDS,
    score_floor: float = 1e-4,
) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    if video.shape != source.shape or video.ndim != 5:
        raise ValueError(
            "source drift requires matching [batch, channels, time, height, width] latents; "
            f"got {tuple(video.shape)} and {tuple(source.shape)}"
        )
    if score_floor <= 0:
        raise ValueError("score_floor must be positive")

    video_f = video.float()
    source_f = source.to(device=video.device, dtype=torch.float32)
    delta = video_f - source_f
    source_sq = source_f.square().sum()
    delta_sq = delta.square().sum()
    cosine = (video_f * source_f).sum() / torch.sqrt(
        video_f.square().sum() * source_sq
    ).clamp_min(1e-12)

    error_energy = delta.square().mean(dim=1, keepdim=True)
    source_energy = source_f.square().mean(dim=1, keepdim=True)
    if error_energy.shape[-2] % 2 or error_energy.shape[-1] % 2:
        raise ValueError("H3 source latent height and width must be divisible by two")
    token_error = F.avg_pool3d(error_energy, kernel_size=(1, 2, 2), stride=(1, 2, 2)).sqrt()
    token_scale = F.avg_pool3d(source_energy, kernel_size=(1, 2, 2), stride=(1, 2, 2)).sqrt()
    relative = token_error / (token_scale + float(score_floor))
    flat = relative.flatten()

    summary: dict[str, Any] = {
        "relative_l2": _scalar(torch.sqrt(delta_sq) / torch.sqrt(source_sq).clamp_min(score_floor)),
        "cosine": _scalar(cosine),
        "token_relative_mean": _scalar(flat.mean()),
        "token_relative_p50": _scalar(torch.quantile(flat, 0.50)),
        "token_relative_p90": _scalar(torch.quantile(flat, 0.90)),
        "token_relative_p99": _scalar(torch.quantile(flat, 0.99)),
        "token_relative_max": _scalar(flat.max()),
    }
    for threshold in thresholds:
        summary[f"token_fraction_gt_{threshold:g}"] = _scalar((flat > threshold).float().mean())

    frame_rows = []
    for frame in range(video.shape[2]):
        frame_delta = delta[:, :, frame]
        frame_source = source_f[:, :, frame]
        frame_rows.append(
            {
                "latent_frame": frame,
                "relative_l2": _scalar(
                    torch.linalg.vector_norm(frame_delta)
                    / torch.linalg.vector_norm(frame_source).clamp_min(score_floor)
                ),
                "residual_rms": _scalar(frame_delta.square().mean().sqrt()),
            }
        )
    return summary, frame_rows


def summarize_mask(mask: Any) -> dict[str, Any] | None:
    if mask is None:
        return None
    if getattr(mask, "is_nested", False):
        return {f"stream_{index}": summarize_mask(value) for index, value in enumerate(mask.unbind())}
    if not torch.is_tensor(mask):
        return {"type": type(mask).__name__}
    value = mask.detach().float()
    return {
        "shape": list(value.shape),
        "min": _scalar(value.min()),
        "max": _scalar(value.max()),
        "mean": _scalar(value.mean()),
        "nonzero_fraction": _scalar((value != 0).float().mean()),
    }


def latent_metadata(latent: Any) -> dict[str, Any] | None:
    if not isinstance(latent, dict):
        return None
    samples = latent.get("samples")
    if getattr(samples, "is_nested", False):
        sample_shapes = [list(value.shape) for value in samples.unbind()]
    elif torch.is_tensor(samples):
        sample_shapes = [list(samples.shape)]
    else:
        sample_shapes = []
    return {
        "sample_shapes": sample_shapes,
        "noise_mask": summarize_mask(latent.get("noise_mask")),
    }


def _atomic_text(path: Path, value: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_suffix(path.suffix + ".tmp")
    temporary.write_text(value, encoding="utf-8")
    os.replace(temporary, path)


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        _atomic_text(path, "")
        return
    fields: list[str] = []
    for row in rows:
        for key in row:
            if key not in fields:
                fields.append(key)
    temporary = path.with_suffix(path.suffix + ".tmp")
    with temporary.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


@dataclass
class TraceRun:
    output_dir: Path
    manifest: dict[str, Any]
    latent_shapes: Any = None
    block_rows: list[dict[str, Any]] = field(default_factory=list)
    attention_rows: list[dict[str, Any]] = field(default_factory=list)
    step_rows: list[dict[str, Any]] = field(default_factory=list)
    frame_rows: list[dict[str, Any]] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)


class TraceSession:
    def __init__(
        self,
        *,
        base_dir: Path,
        run_tag: str,
        workflow_label: str,
        steps_spec: str,
        capture_unconditional: bool,
        thresholds: Sequence[float],
        score_floor: float,
        source_video: torch.Tensor | None,
        attention_queries_per_stream: int,
        static_manifest: dict[str, Any],
    ) -> None:
        self.base_dir = Path(base_dir)
        self.run_tag = safe_tag(run_tag)
        self.workflow_label = workflow_label
        self.steps_spec = steps_spec
        self.capture_unconditional = capture_unconditional
        self.thresholds = tuple(thresholds)
        self.score_floor = float(score_floor)
        self.source_video = None if source_video is None else source_video.detach()
        self.attention_queries_per_stream = int(attention_queries_per_stream)
        self.static_manifest = static_manifest
        self.run: TraceRun | None = None
        self._lock = threading.Lock()
        self._source_cache: dict[tuple[str, torch.dtype], torch.Tensor] = {}

    def begin(self, latent_shapes: Any = None) -> TraceRun:
        with self._lock:
            if self.run is not None:
                raise RuntimeError("MiniMax H3 trace session is already active")
            stamp = time.strftime("%Y%m%d-%H%M%S")
            suffix = f"{time.time_ns() % 1_000_000:06d}"
            output_dir = self.base_dir / f"{self.run_tag}_{stamp}_{suffix}"
            manifest = dict(self.static_manifest)
            manifest.update(
                {
                    "schema": "minimax_h3_lipsync_trace/v1",
                    "status": "running",
                    "workflow_label": self.workflow_label,
                    "run_tag": self.run_tag,
                    "started_unix": time.time(),
                    "steps_spec": self.steps_spec,
                    "capture_unconditional": self.capture_unconditional,
                    "source_drift_thresholds": list(self.thresholds),
                    "score_floor": self.score_floor,
                }
            )
            self.run = TraceRun(output_dir=output_dir, manifest=manifest, latent_shapes=latent_shapes)
            output_dir.mkdir(parents=True, exist_ok=False)
            self.write(complete=False)
            return self.run

    def end(self, *, error: BaseException | None = None) -> Path | None:
        with self._lock:
            run = self.run
            if run is None:
                return None
            run.manifest["status"] = "failed" if error is not None else "complete"
            run.manifest["finished_unix"] = time.time()
            if error is not None:
                run.manifest["error"] = f"{type(error).__name__}: {error}"
            self.write(complete=True)
            self.run = None
            self._source_cache.clear()
            return run.output_dir

    def selected_step(self, transformer_options: dict[str, Any]) -> tuple[bool, int, float]:
        sigma_tensor = transformer_options.get("sigmas")
        if sigma_tensor is None:
            return False, -1, math.nan
        sigma = float(sigma_tensor.flatten()[0])
        schedule = transformer_options.get("sample_sigmas")
        if schedule is None:
            return self.steps_spec.strip().lower() == "all", -1, sigma
        schedule_cpu = schedule.detach().flatten().to(device="cpu", dtype=torch.float32)
        step = int(torch.argmin((schedule_cpu - sigma).abs()))
        total = max(1, schedule_cpu.numel() - 1)
        selected = resolve_indices(
            self.steps_spec,
            total,
            default_fractions=DEFAULT_STEP_FRACTIONS,
        )
        return step in selected, step, sigma

    def branch_allowed(self, transformer_options: dict[str, Any]) -> tuple[bool, int]:
        values = transformer_options.get("cond_or_uncond") or [0]
        branch = int(values[0])
        return self.capture_unconditional or branch == 0, branch

    def source_for(self, video: torch.Tensor) -> torch.Tensor | None:
        if self.source_video is None:
            return None
        key = (str(video.device), torch.float32)
        source = self._source_cache.get(key)
        if source is None:
            source = self.source_video.to(device=video.device, dtype=torch.float32)
            self._source_cache[key] = source
        return source

    def record_block(
        self,
        *,
        block_index: int,
        source: torch.Tensor,
        output: torch.Tensor,
        segments: Iterable[tuple[int, int, str]],
        transformer_options: dict[str, Any],
    ) -> None:
        run = self.run
        if run is None:
            return
        selected, step, sigma = self.selected_step(transformer_options)
        allowed, branch = self.branch_allowed(transformer_options)
        if not selected or not allowed:
            return
        with torch.no_grad():
            rows = segment_metrics(source, output, segments)
        for row in rows:
            row.update(
                {
                    "block": int(block_index),
                    "step": int(step),
                    "sigma": sigma,
                    "branch": branch,
                }
            )
            run.block_rows.append(row)

    def record_attention(
        self,
        *,
        q: torch.Tensor,
        k: torch.Tensor,
        segments: Iterable[tuple[int, int, str]],
        transformer_options: dict[str, Any],
    ) -> None:
        run = self.run
        if run is None:
            return
        selected, step, sigma = self.selected_step(transformer_options)
        allowed, branch = self.branch_allowed(transformer_options)
        if not selected or not allowed:
            return
        block_index = int(transformer_options.get("block_index", -1))
        if block_index not in run.manifest.get("selected_blocks", []):
            return
        if q.ndim != 4 or k.ndim != 4 or q.shape[:3] != k.shape[:3]:
            message = f"attention capture expected matching [B,H,S,D] Q/K, got {tuple(q.shape)} and {tuple(k.shape)}"
            if message not in run.warnings:
                run.warnings.append(message)
            return

        spans = [(int(start), int(stop), str(kind)) for start, stop, kind in segments]
        sequence = q.shape[2]
        if any(start < 0 or stop > sequence or stop <= start for start, stop, _kind in spans):
            raise ValueError("packed attention layout does not match Q/K sequence length")
        target_spans = {
            kind: (start, stop)
            for start, stop, kind in spans
            if kind in {"audio", "video"}
        }
        with torch.no_grad():
            key_f = k[0].float()
            scale = 1.0 / math.sqrt(q.shape[-1])
            for query_kind in ("audio", "video"):
                span = target_spans.get(query_kind)
                if span is None:
                    continue
                start, stop = span
                count = min(self.attention_queries_per_stream, stop - start)
                indices = torch.linspace(
                    start,
                    stop - 1,
                    count,
                    device=q.device,
                    dtype=torch.float32,
                ).round().to(torch.long)
                query = q[0, :, indices].float()
                probability = torch.softmax(
                    torch.matmul(query, key_f.transpose(-1, -2)) * scale,
                    dim=-1,
                )
                grouped: dict[str, torch.Tensor] = {}
                for key_start, key_stop, key_kind in spans:
                    mass = probability[..., key_start:key_stop].sum(dim=-1).mean()
                    grouped[key_kind] = grouped.get(key_kind, mass.new_zeros(())) + mass
                for key_kind, mass in grouped.items():
                    run.attention_rows.append(
                        {
                            "block": block_index,
                            "step": step,
                            "sigma": sigma,
                            "branch": branch,
                            "query_kind": query_kind,
                            "key_kind": key_kind,
                            "attention_mass": _scalar(mass),
                            "query_count": count,
                            "head_count": int(q.shape[1]),
                        }
                    )

    def record_guided_video(self, video: torch.Tensor, sigma: float, step: int, source_kind: str) -> None:
        run = self.run
        if run is None:
            return
        source = self.source_for(video)
        if source is None:
            return
        try:
            with torch.no_grad():
                summary, frames = source_drift_metrics(
                    video,
                    source,
                    thresholds=self.thresholds,
                    score_floor=self.score_floor,
                )
        except ValueError as error:
            message = str(error)
            if message not in run.warnings:
                run.warnings.append(message)
            return
        summary.update({"step": int(step), "sigma": float(sigma), "source_kind": source_kind})
        run.step_rows.append(summary)
        for frame in frames:
            frame.update({"step": int(step), "sigma": float(sigma), "source_kind": source_kind})
            run.frame_rows.append(frame)

    def write(self, *, complete: bool) -> None:
        run = self.run
        if run is None:
            return
        manifest = dict(run.manifest)
        manifest["warnings"] = list(run.warnings)
        manifest["counts"] = {
            "block_rows": len(run.block_rows),
            "attention_rows": len(run.attention_rows),
            "step_rows": len(run.step_rows),
            "frame_rows": len(run.frame_rows),
        }
        summary = build_summary(run.block_rows, run.step_rows, run.attention_rows)
        _atomic_text(run.output_dir / "manifest.json", json.dumps(manifest, indent=2, sort_keys=True))
        _atomic_text(run.output_dir / "summary.json", json.dumps(summary, indent=2, sort_keys=True))
        _write_csv(run.output_dir / "block_metrics.csv", run.block_rows)
        _write_csv(run.output_dir / "attention_metrics.csv", run.attention_rows)
        _write_csv(run.output_dir / "step_metrics.csv", run.step_rows)
        _write_csv(run.output_dir / "frame_metrics.csv", run.frame_rows)
        _atomic_text(run.output_dir / "report.md", build_report(manifest, summary, complete=complete))


def build_summary(
    block_rows: Sequence[dict[str, Any]],
    step_rows: Sequence[dict[str, Any]],
    attention_rows: Sequence[dict[str, Any]] = (),
) -> dict[str, Any]:
    grouped: dict[tuple[int, str], list[dict[str, Any]]] = {}
    for row in block_rows:
        grouped.setdefault((int(row["block"]), str(row["segment_kind"])), []).append(row)
    blocks = []
    for (block, kind), rows in sorted(grouped.items()):
        blocks.append(
            {
                "block": block,
                "segment_kind": kind,
                "samples": len(rows),
                "mean_relative_residual_rms": sum(
                    float(row["relative_residual_rms"]) for row in rows
                )
                / len(rows),
                "mean_cosine": sum(float(row["cosine"]) for row in rows) / len(rows),
            }
        )
    latest = dict(step_rows[-1]) if step_rows else None
    attention_grouped: dict[tuple[int, str, str], list[float]] = {}
    for row in attention_rows:
        key = (int(row["block"]), str(row["query_kind"]), str(row["key_kind"]))
        attention_grouped.setdefault(key, []).append(float(row["attention_mass"]))
    attention = [
        {
            "block": block,
            "query_kind": query_kind,
            "key_kind": key_kind,
            "samples": len(values),
            "mean_attention_mass": sum(values) / len(values),
        }
        for (block, query_kind, key_kind), values in sorted(attention_grouped.items())
    ]
    return {
        "block_segment_means": blocks,
        "attention_means": attention,
        "latest_source_drift": latest,
        "block_row_count": len(block_rows),
        "attention_row_count": len(attention_rows),
        "step_row_count": len(step_rows),
    }


def build_report(manifest: dict[str, Any], summary: dict[str, Any], *, complete: bool) -> str:
    lines = [
        "# MiniMax H3 AV internal trace",
        "",
        f"- Status: `{manifest.get('status', 'unknown')}`",
        f"- Workflow label: `{manifest.get('workflow_label', '')}`",
        f"- Run tag: `{manifest.get('run_tag', '')}`",
        f"- Selected blocks: `{manifest.get('selected_blocks', [])}`",
        f"- Step selection: `{manifest.get('steps_spec', '')}`",
        f"- Output-neutral instrumentation: `true`",
        "",
        "## Captures",
        "",
        f"- Block/segment rows: {summary.get('block_row_count', 0)}",
        f"- Cross-segment attention rows: {summary.get('attention_row_count', 0)}",
        f"- Guided source-drift rows: {summary.get('step_row_count', 0)}",
    ]
    warnings = manifest.get("warnings") or []
    if warnings:
        lines.extend(["", "## Warnings", ""] + [f"- {warning}" for warning in warnings])
    if not complete:
        lines.extend(["", "The run is active; files are a partial checkpoint."])
    lines.extend(
        [
            "",
            "## Interpretation boundary",
            "",
            "Residual energy identifies where two trajectories differ; it does not by itself prove a block causes lip-sync. Use matched runs and a later causal ablation before modifying inference.",
            "",
        ]
    )
    return "\n".join(lines)
