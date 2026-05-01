# Video Input Protocol

Default protocol id: `eo_ir_full_video_1fps_512px_v1`

This benchmark stores the video input policy inside every exported request under
`request.video_input` and `provider_payload.video_input`. Runtime flags may
override the policy for constrained local debugging, but official results should
report the policy used by the model.

## Official API Policy

- Clip policy: use the full decoded video, with no task-specific temporal clipping.
- Sampling: uniform sampling at `1.0 fps`.
- Frame cap: no cap by default. If a provider-specific adapter applies a cap, it
  must write that cap into the prediction metadata and the result should be
  reported as a separate constrained setting.
- Resolution: resize with aspect ratio preserved and cap each frame at
  `262144` pixels, equivalent to a 512 x 512 area budget.
- Audio: disabled unless an audio benchmark setting is explicitly added.
- EO+IR order: `Video 1 = EO`, `Video 2 = IR`.
- Leakage policy: no gold answer, rationale, or annotation time-reference
  metadata is included in the model prompt or provider payload.

## Provider Mapping

- Gemini: pass `videoMetadata.fps = 1.0`; use default media resolution unless a
  lower-resolution constrained setting is explicitly declared.
- GPT/OpenAI-style image input: extract frames at `1.0 fps`, resize frames using
  the resolution policy, then send those frames as image inputs.
- Local Qwen/InternVL/VideoLLaMA: read `fps` and pixel budget from
  `provider_payload.video_input` by default; command-line flags are overrides
  for capacity checks and must be reported separately.

## Constrained Local Runs

If full `1.0 fps` input does not fit on the local GPU, report the run as
constrained, for example:

```text
constrained local: fps=0.05, max_pixels=25088
```

Do not mix constrained local scores with official API-policy scores.
