# ComfyUI-H3-LipSync-Lab

Output-neutral MiniMax H3 instrumentation for comparing a good-lip-sync Ref2VA trajectory with a source-preserving FL2VA/source-latent trajectory.

This is a measurement node, not a lip-sync modifier. Its default mode uses official ComfyUI model-patcher seams and returns the original tensors unchanged.

## Installation

```bash
cd ComfyUI/custom_nodes
git clone https://github.com/Yuvrajxms09/ComfyUI-H3-LipSync-Lab.git
```

Restart ComfyUI after cloning. The pack has no third-party Python dependencies beyond ComfyUI and PyTorch.

It was developed against ComfyUI v0.39.0 and requires the native MiniMax H3 implementation with packed-layout metadata, DiT block replacements, sampler wrappers, and `comfy_api.latest` custom-node support.

## Node

`MiniMax H3 AV Internal Trace`

Connect the model that feeds the guider. Optionally connect:

- `source_video_latent`: the source video encoded by the H3 video VAE, used only as the preservation baseline;
- `target_av_latent`: the exact joint AV latent sent to the sampler, used only to record stream and noise-mask metadata.

The node writes one directory per sample under:

`ComfyUI/output/h3_lipsync_lab/`

Artifacts:

- `manifest.json`
- `block_metrics.csv`
- `attention_metrics.csv`
- `step_metrics.csv`
- `frame_metrics.csv`
- `summary.json`
- `report.md`

Default capture is deliberately bounded: early/middle/late blocks and early/middle/late sampling steps, conditional branch only. Use `all` only for a deliberate diagnostic run.

Cross-stream attention capture is experimental and disabled by default. ComfyUI's optimized-attention override path can bypass H3's preferred attention dispatch, so it is not part of the strict output-neutral baseline. If explicitly enabled, it measures bounded target-audio/target-video query attention mass by packed key category. If another attention override is already present, capture is disabled rather than replacing it; block and source-drift metrics still run.

## Integration order

Place the trace after model/sampler patches and immediately before the guider. Existing H3 `dit/double_block` replacements found at execution time are chained and measured; they are not discarded.

For T8's dual-clock graph, route `MiniMaxH3DualClockSamplerT8`'s patched model output through the trace and then into `BasicGuider`. Keep its sampler and sigma outputs connected directly to `SamplerCustomAdvanced`.

For Exact Audio Lock, route the loaded Ref2VA model through the trace and then into `BasicGuider`. The audio-lock latent still connects directly to `SamplerCustomAdvanced`.

## Validation

CPU-only core tests:

```bash
python -m pytest -q tests
```

The tests cover selection parsing, packed-segment aggregation, source-latent drift math, and bounded artifact writing. GPU/H3 inference parity remains a separate validation gate.

The repository's initial publication was syntax-checked and its example API graphs were structurally validated. Tensor tests were not executed on the authoring host because that host's default Python environment did not include PyTorch or pytest.

## Boundaries

- no masking, inpainting, compositing, donor frames, or ControlNet;
- no checkpoint or sampler changes;
- no full activation dumps;
- no edits to ComfyUI or third-party custom nodes;
- high residual energy is correlation, not proof of a lip-sync-specific block.
