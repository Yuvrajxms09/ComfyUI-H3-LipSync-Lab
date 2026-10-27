from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

import torch

import comfy.patcher_extension
import comfy.utils
import folder_paths
from comfy_api.latest import ComfyExtension, io

from .trace_core import (
    DEFAULT_BLOCK_FRACTIONS,
    TraceSession,
    latent_metadata,
    parse_thresholds,
    resolve_indices,
)


LOG_PREFIX = "[H3 LipSync Lab]"
WRAPPER_KEY = "h3_lipsync_internal_trace"


def _source_video(latent: Any) -> torch.Tensor | None:
    if latent is None:
        return None
    if not isinstance(latent, dict):
        raise ValueError("source_video_latent must be a ComfyUI LATENT mapping")
    samples = latent.get("samples")
    if getattr(samples, "is_nested", False):
        parts = samples.unbind()
        if not parts:
            raise ValueError("source_video_latent contains no streams")
        samples = parts[0]
    if not torch.is_tensor(samples) or samples.ndim != 5 or samples.shape[1] != 24:
        raise ValueError(
            "source_video_latent must contain an H3 video latent [B,24,T,H,W]; "
            f"got {getattr(samples, 'shape', None)}"
        )
    return samples


def _step_from_sigma(model_options: dict[str, Any], sigma: float) -> int:
    transformer_options = model_options.get("transformer_options", {})
    schedule = transformer_options.get("sample_sigmas")
    if schedule is None:
        return -1
    schedule_cpu = schedule.detach().flatten().to(device="cpu", dtype=torch.float32)
    return int(torch.argmin((schedule_cpu - sigma).abs()))


def _unpack_video(packed: Any, latent_shapes: Any) -> torch.Tensor:
    if not torch.is_tensor(packed):
        raise ValueError("expected a packed sampler latent tensor")
    if not latent_shapes:
        raise ValueError("sampling run did not expose latent_shapes")
    streams = comfy.utils.unpack_latents(packed, latent_shapes)
    if not isinstance(streams, (list, tuple)) or not streams:
        raise ValueError("packed sampler latent has no H3 video stream")
    video = streams[0]
    if not torch.is_tensor(video) or video.ndim != 5:
        raise ValueError("unpacked H3 video stream is invalid")
    return video


def _make_block_probe(session: TraceSession, block_index: int, previous_patch):
    def patch(args, extra):
        transformer_options = args["transformer_options"]
        selected, _step, _sigma = session.selected_step(transformer_options)
        allowed, _branch = session.branch_allowed(transformer_options)
        should_measure = session.run is not None and selected and allowed
        # Stock H3 updates the packed stream in place. Clone only at selected
        # measurement points or the pre-block state would be overwritten.
        source = args["img"].detach().clone() if should_measure else None
        output = previous_patch(args, extra) if previous_patch is not None else extra["original_block"](args)
        result = output["img"]
        if source is not None:
            try:
                session.record_block(
                    block_index=block_index,
                    source=source,
                    output=result,
                    segments=args["layout"].segments,
                    transformer_options=transformer_options,
                )
            except Exception:
                logging.exception("%s block %d measurement failed", LOG_PREFIX, block_index)
                run = session.run
                if run is not None:
                    message = f"block {block_index} measurement failed; see server log"
                    if message not in run.warnings:
                        run.warnings.append(message)
        return output

    return patch


def _make_outer_wrapper(session: TraceSession):
    def wrapper(executor, *args, **kwargs):
        session.begin(latent_shapes=kwargs.get("latent_shapes"))
        error = None
        result = None
        try:
            result = executor(*args, **kwargs)
            run = session.run
            if run is not None and session.source_video is not None:
                try:
                    video = _unpack_video(result, run.latent_shapes)
                    session.record_guided_video(video, sigma=0.0, step=-1, source_kind="final_sample")
                except Exception as capture_error:
                    message = f"final sample source-drift capture failed: {capture_error}"
                    if message not in run.warnings:
                        run.warnings.append(message)
            return result
        except BaseException as caught:
            error = caught
            raise
        finally:
            path = session.end(error=error)
            if path is not None:
                logging.info("%s trace report -> %s", LOG_PREFIX, path)

    return wrapper


def _make_post_cfg_observer(session: TraceSession):
    def observer(args):
        denoised = args["denoised"]
        run = session.run
        if run is None or session.source_video is None:
            return denoised
        try:
            sigma_tensor = args.get("sigma")
            if sigma_tensor is None:
                raise ValueError("post-CFG observer received no sigma")
            sigma = float(sigma_tensor.flatten()[0])
            video = _unpack_video(denoised, run.latent_shapes)
            step = _step_from_sigma(args.get("model_options", {}), sigma)
            session.record_guided_video(video, sigma=sigma, step=step, source_kind="guided_x0")
        except Exception as error:
            message = f"guided source-drift capture failed: {error}"
            if message not in run.warnings:
                run.warnings.append(message)
            logging.warning("%s %s", LOG_PREFIX, message)
        return denoised

    return observer


def _make_attention_override(session: TraceSession):
    def override(original_attention, q, k, v, heads, *args, **kwargs):
        transformer_options = kwargs.get("transformer_options") or {}
        layout = transformer_options.get("minimax_h3_layout")
        if layout is not None:
            try:
                session.record_attention(
                    q=q,
                    k=k,
                    segments=layout.segments,
                    transformer_options=transformer_options,
                )
            except Exception:
                logging.exception("%s attention measurement failed", LOG_PREFIX)
                run = session.run
                if run is not None:
                    message = "attention measurement failed; see server log"
                    if message not in run.warnings:
                        run.warnings.append(message)
        return original_attention(q, k, v, heads, *args, **kwargs)

    return override


class MiniMaxH3AVInternalTrace(io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="MiniMaxH3AVInternalTrace",
            display_name="MiniMax H3 AV Internal Trace",
            category="model/patch/minimax",
            description=(
                "Output-neutral H3 block/step instrumentation for comparing lip-sync and "
                "source-video preservation. Writes bounded aggregate reports under output/h3_lipsync_lab/."
            ),
            inputs=[
                io.Model.Input("model"),
                io.String.Input("workflow_label", default="unlabelled_h3_workflow"),
                io.String.Input("run_tag", default="h3_trace"),
                io.String.Input(
                    "blocks",
                    default="auto",
                    tooltip="auto selects early/middle/late blocks; use all or a comma list such as 5,24,44.",
                ),
                io.String.Input(
                    "steps",
                    default="auto",
                    tooltip="auto selects early/middle/late denoising steps; use all or a comma list.",
                ),
                io.Boolean.Input("capture_unconditional", default=False),
                io.Boolean.Input(
                    "capture_cross_stream_attention",
                    default=True,
                    tooltip="Measure bounded audio/video query attention mass on selected blocks and steps.",
                ),
                io.Int.Input(
                    "attention_queries_per_stream",
                    default=16,
                    min=1,
                    max=128,
                    tooltip="Evenly sampled target-audio and target-video queries per selected block/step.",
                ),
                io.String.Input("source_drift_thresholds", default="0.02,0.05,0.10,0.20"),
                io.Float.Input("score_floor", default=0.0001, min=1e-8, max=1.0, step=0.0001),
                io.Latent.Input(
                    "source_video_latent",
                    optional=True,
                    tooltip="Optional VAE-encoded source video used only as a source-preservation baseline.",
                ),
                io.Latent.Input(
                    "target_av_latent",
                    optional=True,
                    tooltip="Optional target AV latent used only to record stream shapes and noise-mask statistics.",
                ),
            ],
            outputs=[io.Model.Output()],
        )

    @classmethod
    def execute(
        cls,
        model,
        workflow_label,
        run_tag,
        blocks,
        steps,
        capture_unconditional,
        capture_cross_stream_attention,
        attention_queries_per_stream,
        source_drift_thresholds,
        score_floor,
        source_video_latent=None,
        target_av_latent=None,
    ) -> io.NodeOutput:
        diffusion_model = model.get_model_object("diffusion_model")
        model_blocks = getattr(diffusion_model, "blocks", None)
        if model_blocks is None or not model_blocks:
            raise ValueError("MiniMax H3 AV Internal Trace requires an H3 model with DiT blocks")

        selected_blocks = resolve_indices(
            blocks,
            len(model_blocks),
            default_fractions=DEFAULT_BLOCK_FRACTIONS,
        )
        thresholds = parse_thresholds(source_drift_thresholds)
        source = _source_video(source_video_latent)
        static_manifest = {
            "model_class": f"{type(diffusion_model).__module__}.{type(diffusion_model).__name__}",
            "block_count": len(model_blocks),
            "selected_blocks": selected_blocks,
            "source_video_shape": None if source is None else list(source.shape),
            "target_av_latent": latent_metadata(target_av_latent),
        }
        session = TraceSession(
            base_dir=Path(folder_paths.get_output_directory()) / "h3_lipsync_lab",
            run_tag=run_tag,
            workflow_label=workflow_label,
            steps_spec=steps,
            capture_unconditional=bool(capture_unconditional),
            thresholds=thresholds,
            score_floor=float(score_floor),
            source_video=source,
            attention_queries_per_stream=int(attention_queries_per_stream),
            static_manifest=static_manifest,
        )

        patched = model.clone()
        replacements = (
            model.model_options.get("transformer_options", {})
            .get("patches_replace", {})
            .get("dit", {})
        )
        for block_index in selected_blocks:
            key = ("double_block", block_index)
            patched.set_model_patch_replace(
                _make_block_probe(session, block_index, replacements.get(key)),
                "dit",
                "double_block",
                block_index,
            )

        transformer_options = patched.model_options.setdefault("transformer_options", {})
        existing_attention_override = transformer_options.get("optimized_attention_override")
        if capture_cross_stream_attention and existing_attention_override is None:
            transformer_options["optimized_attention_override"] = _make_attention_override(session)
            static_manifest["attention_capture"] = "enabled"
        elif capture_cross_stream_attention:
            static_manifest["attention_capture"] = "disabled_existing_attention_override"
            static_manifest.setdefault("configuration_warnings", []).append(
                "Cross-stream attention capture was disabled because the model already has an optimized-attention override. Block and source-drift metrics remain enabled."
            )
        else:
            static_manifest["attention_capture"] = "disabled_by_user"

        comfy.patcher_extension.add_wrapper_with_key(
            comfy.patcher_extension.WrappersMP.OUTER_SAMPLE,
            WRAPPER_KEY,
            _make_outer_wrapper(session),
            patched.model_options,
            is_model_options=True,
        )
        post_cfg = list(patched.model_options.get("sampler_post_cfg_function", []))
        post_cfg.append(_make_post_cfg_observer(session))
        patched.model_options["sampler_post_cfg_function"] = post_cfg

        logging.info(
            "%s armed label=%s blocks=%s steps=%s source=%s",
            LOG_PREFIX,
            workflow_label,
            selected_blocks,
            steps,
            None if source is None else tuple(source.shape),
        )
        return io.NodeOutput(patched)


class H3LipSyncLabExtension(ComfyExtension):
    async def get_node_list(self):
        return [MiniMaxH3AVInternalTrace]


async def comfy_entrypoint() -> H3LipSyncLabExtension:
    return H3LipSyncLabExtension()
