# TubeletGraph batch inference and speedups

This page covers how to run TubeletGraph over the shared MOSCATO videos on
Modal, what was done to make each run faster, and what a full run costs.

## What a batch run produces

A batch run saves TubeletGraph's raw output for every annotated
(video, object) pair. The MOSCATO adapter is skipped by default, because labels
are matched downstream with a separate Qwen embedding space. Each run writes
these files to the `tubelet-data` Volume under
`/<video_id>/runs/<run-name>-<object-slug>/`:

- `predictions/` holds TubeletGraph's per-frame masks and its own VLM
  state-change text.
- `run_manifest.json` records the clip, sampling, commits and speedup flags.
- `tubeletgraph.log` is the full pipeline log, including speedup cache stats.

Pass `"map_states": true` in a batch entry to also run the MOSCATO adapter.

## Speedups

All of these are on by default. Each one leaves TubeletGraph's output unchanged.

| Technique | What it removes | Where |
|---|---|---|
| SAM2 image-encoder feature cache | Re-encoding every frame on every tracking pass | `scripts/tubelet_speedups.py`, patched into SAM2 |
| One shared SAM2 predictor | Building the same Hiera-L model three times per video | same |
| One frame load per video | Loading all frames into SAM2 three times | same |
| CropFormer per-frame cache | Re-segmenting frames already done for another object of the same video | patched into CropFormer's script |
| Parallel batch on Modal | Running videos one after another | `--batch-file` in `modal_tubelet.py` |

How each technique works:

- **Feature cache.** SAM2 normally keeps image features only for the latest
  frame. TubeletGraph resets SAM2 and re-propagates to the end of the clip for
  each round of newly found objects, so the frozen encoder re-ran on every
  frame of every pass. The encoder never sees prompts or memory, so a frame's
  features are the same on every pass. Features stay on the GPU up to 24 GiB
  and spill to CPU memory up to 24 GiB. Beyond that, frames are recomputed as
  before.
- **Shared predictor and frames.** TubeletGraph's tubelet stage builds three
  SAM2 predictors with the same weights, and each one reloads the video. They
  now share one model, one frame tensor and one feature cache.
- **CropFormer cache.** CropFormer segments each frame on its own. Its result
  is stored on the Volume, keyed by the frame's bytes and the CropFormer
  config. The batch generator snaps every clip start onto the frame-stride
  grid, so all objects of a video sample the same frames. The batch runner
  also starts one object per video first, so later objects find the cache
  already filled.
- **Parallel batch.** Each (video, object) runs in its own A100 container.
  All runs start at once, or in waves of `--max-parallel`.

Not done, and why:

- **Preloading CropFormer across videos.** Each container runs one video, so
  each container loads the model once either way.
- **Turning off SAM2 multimask.** It saves little, because the extra masks
  share the same computation, and it can lower recall in the spatial check.
- **bf16 or `torch.compile` for SAM2.** These change the numbers TubeletGraph
  computes. They would need their own accuracy check.

Flags for comparison or debugging:

- `--no-sam2-speedups` turns off the SAM2 and CropFormer caches for a run.
- `TUBELETGRAPH_FEATURE_CACHE_GPU_GB` and `TUBELETGRAPH_FEATURE_CACHE_CPU_GB`
  set the feature-cache budgets.

### Measured on S12

The test was an A/B run on S12 bread slices: 100 frames at 5 fps, with
intermediates saved. Both runs were on an A100 80GB on 2026-10-30.

| | Speedups off | Speedups on |
|---|---|---|
| Output files | 8 | 8, byte-identical |
| SAM2 tracking time | 173 s | 134 s |
| Frame loads | 3 | 1 |
| Multimask pass | 0.17 s/frame | 0.05 s/frame |
| Re-tracking pass with 6 objects | 0.50 s/frame | 0.37 s/frame |
| Wall time | 6.0 min | 5.2 min |

The feature cache served 412 of 511 encoder calls on this clip. Savings grow
with clip length. In longer clips, most time goes to the repeated re-tracking
passes that the cache speeds up.

## Cost of the full run

The full run covers 41 videos with 458 annotated objects: 29 CMU videos with
259 objects and 12 EGTEA videos with 199 objects. CMU is sampled at 5 fps,
matching the S12 run. EGTEA is sampled at 1 fps, because its clips run about
12 minutes per object. At 5 fps, later-object tracking grows with the square
of clip length, and single runs would exceed Modal's 12-hour limit.

An A100 container costs about $3.39 per hour. That is $2.50 for the GPU, $0.38
for 8 CPU cores and $0.51 for 64 GiB of memory, at Modal list prices.

| | Runs | GPU-hours | Cost |
|---|---|---|---|
| CMU at 5 fps | 259 | about 320 | about $1,080 |
| EGTEA at 1 fps | 199 | about 270 | about $920 |
| **Total** | 458 | about 590 | **about $2,000** |
| Same run without speedups | 458 | about 920 | about $3,100 |

The model behind these figures was fitted to the full S12 run and matches it
within 7%. It assumes about 3 objects per re-tracking pass. Busy scenes can
track more, so treat the total as roughly $1,500 to $3,000. TubeletGraph's own
GPT-4.1 calls are billed separately by OpenAI and are small next to GPU cost.

Wall time depends on your Modal plan's GPU limit. The longest single run is
about 8 hours.

| Plan | GPU limit | Approximate wall time |
|---|---|---|
| Starter | 10 | about 2.5 days |
| Team | 50 | about 12 hours |

Budget about $2,500. That covers the estimate plus about 25% for a pilot,
failed runs and image builds. Set a workspace spending limit in the Modal
dashboard before launching.

### Bringing the cost down

These levers are priced against the same 458-run batch. Every option keeps the
speedups above on.

| Option | GPU-hours | Cost | Trade-off |
|---|---|---|---|
| Current plan | about 590 | about $2,000 | none |
| 32 GiB containers instead of 64 | about 590 | about $1,850 | long CMU clips at 5 fps spill the feature cache past 32 GiB |
| Skip objects whose state never changes | about 550 | about $1,870 | 119 objects lose raw output |
| Re-track every 60 frames instead of 30 | about 360 | about $1,230 | new pieces are found up to 60 sampled frames later |
| CMU at 2 fps instead of 5 | about 355 | about $1,200 | coarser timing, and different sampling from the S12 run |
| CMU at 2 fps, 32 GiB | about 355 | about $1,110 | as above |
| CMU at 2 fps, re-track every 60, 32 GiB | about 230 | about $710 | both trade-offs |
| No later-object tracking | about 145 | about $490 | misses pieces that appear after frame 0, such as cut fruit |

The cheapest options cost accuracy. Run a pilot of a few objects at both
settings and compare before choosing. Modal also runs startup and academic
credit programs that are worth applying to before a run this size.

How to set each option:

- **Frame rate.** Run `scripts/make_batch.py --cmu-fps 2`.
- **Re-tracking interval.** Set `"collect_spacing": 60` in the batch file's
  `defaults`.
- **Memory.** Set `MOSCATO_GPU_MEMORY_MB=32768` in the shell that runs
  `modal run`.
- **Later-object tracking.** Set `"later_tracking": false` in `defaults`.

## Running it

1. **Create the R2 secret.** Fill in `.env.chefost-r2`, which is gitignored,
   then run:

   ```bash
   .venv/bin/modal secret create chefost-r2 --from-dotenv .env.chefost-r2
   ```

2. **Fetch the R2 manifest and write the batch file.** The manifest maps
   each video to its key in the bucket. CMU videos sit under `_copies/MP4/`,
   and EGTEA videos sit at the top level.

   ```bash
   .venv/bin/modal run modal_tubelet.py --r2-manifest-out r2_manifest.json
   python3 scripts/make_batch.py --data-repo ../chefOST-data \
     --manifest r2_manifest.json --out configs/batch_moscato_cmu_egtea.json
   ```

3. **Prepare frames from R2.** This uses CPU only and decodes each full video
   once into `/<video_id>/frames/` on `tubelet-data`. It reports each object
   as `ready` or `needs_mask`. It was verified on S08 Eggs: 6,243 frames
   decoded, matching the ground truth. A folder holding a hand-cut clip, such
   as S12's, is refused rather than reused. Set `"overwrite": true` on that
   video's entries to rebuild it.

   ```bash
   .venv/bin/modal run modal_tubelet.py \
     --batch-file configs/batch_moscato_cmu_egtea.json --prepare-only
   ```

4. **Upload one first-frame mask per object.** Upload each to
   `/<video_id>/masks/<object-slug>.png` on `tubelet-data`. The mask is drawn
   on frame `clip_start_global` of the full video, which is
   `/<video_id>/frames/<clip_start_global:07d>.jpg`. Use `scripts/make_mask.py`
   as for S12. Pixel values must be 0 and 1.

5. **Run a pilot, then the full batch.** A pilot batch file with two or three
   objects costs a few dollars and checks timing on your account.

   ```bash
   .venv/bin/modal run modal_tubelet.py \
     --batch-file configs/batch_moscato_cmu_egtea.json --max-parallel 10
   ```

Objects without a mask are skipped and listed as `needs_mask`. Rerunning the
same command does not re-decode videos, because existing frames are reused.

## Known gaps

- **Masks are manual.** TubeletGraph needs a first-frame mask per object, and
  MOSCATO has no masks. That is 458 masks for the full run. They are the main
  remaining cost in time.
- **EgoPER is missing.** Its videos are not on R2 yet.
- **Temporal alignment is unverified upstream.** The manifest marks every
  video's alignment as unverified. Decoded frame counts match the
  ground-truth lengths for all 41 videos, and annotation indices are used as
  source frame indices.
