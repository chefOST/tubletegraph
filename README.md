# TubeletGraph / ChefOST

ChefOST evaluates object-state tracking in procedural cooking videos. Use this repository for TubeletGraph work.

## Metrics

For each object, `P_t` is the predicted state set and `T_t` the ground-truth state set at frame `t`.

- Precision: fraction of predicted states that are correct, `|P_t ∩ T_t| / |P_t|`, averaged over frames with predictions.
- Recall (detail): fraction of ground-truth states recovered, `|P_t ∩ T_t| / |T_t|`, averaged over frames where the object is present.

Average per-frame ratios equally; missing predictions give zero recall when GT states exist. Record the dataset, split and sampling; document empty-GT handling and aggregation across objects/videos. Keep published paper results distinct from ChefOST experiments.

## Context

Read the [project goals](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#project-goals), then the [TubeletGraph paper notes](https://github.com/chefOST/.github/blob/main/docs/PaperNotes.md#6-tubeletgraph) relevant to your task. Use the [organization guide](https://github.com/chefOST/.github/blob/main/profile/README.md) for shared project context and the [data repository](https://github.com/chefOST/chefOST-data) for shared annotations and vocabularies.

## Videos

Shared MOSCATO ground-truth videos are on R2. See [video access](https://github.com/chefOST/chefOST-data#videos) for the folder location and Modal/local setup. Read-only credentials were sent in the private team Discord channel. Follow this model's preprocessing and save outputs in your own storage.

## Experiment tracking

Follow [W&B setup](weights_and_biases/README.md) to configure the appropriate team and project. Verify the destination used by your actual run command before reporting results.

## TubeletGraph x MOSCATO: CMU S12 sandwich

This branch runs TubeletGraph on the exact CMU-MMAC video represented by the
only matching MOSCATO CMU annotation entry requested for this experiment:

```text
Video / MOSCATO key: S12_Sandwich_7150991-2470
AVI: S12_Sandwich_Video/S12_Sandwich_7150991-2470.avi
Ground truth: annotations/ground_truth/CMU/gt_annotations_cmu.json
Initial target: bread slices
Global clip: [2824, 6714)
Local frame 0: raw/MOSCATO frame 2824
```

The AVI and MOSCATO bundle are local data inputs and are intentionally ignored
by Git. The experiment manifest is
`configs/s12_sandwich_7150991-2470.json`.

### Verified alignment

The source AVI is H.264, 800 x 600, 30 FPS, and 6,714 raw frames. The MOSCATO
`object` array and all five object-state timelines are also 6,714 frames. Never
use `feature_len` (2,239) as the raw frame count.

Run the check again at any time:

```bash
python3 scripts/inspect_video.py
```

### Prepare the one-object clip

Install local tools (FFmpeg must also be on `PATH`):

```bash
uv venv --python 3.12 .venv
source .venv/bin/activate
uv pip install -r requirements-moscato.txt
```

Decode every raw frame first, then select and zero-base the configured clip:

```bash
python3 scripts/prepare_frames.py
```

This creates:

```text
work/S12_Sandwich_7150991-2470/
├── raw_frames/       # 0000000.jpg ... 0006713.jpg
├── frames/           # local 0 = global 2824
├── first_frame.jpg
└── metadata.json
```

Create the checked-in starting mask geometry:

```bash
python3 scripts/make_mask.py \
  --image work/S12_Sandwich_7150991-2470/frames/0000000.jpg \
  --polygons-json configs/s12_bread_slices_mask.json \
  --out work/S12_Sandwich_7150991-2470/mask.png
```

The result must be an 800 x 600 indexed-palette PNG containing only values 0
and 1. The Modal runner also normalizes legacy grayscale masks to indexed mode
without changing their object IDs. Visually inspect the mask before spending GPU
time. To redraw it interactively, omit `--polygons-json`; press `n` to close one
polygon and begin another.

### Modal, OpenAI, and W&B environments

TubeletGraph is installed and run only in the CUDA Modal image defined by
`modal_tubelet.py`. The key is never placed in a file or image layer.

One-time setup:

```bash
python3 -m pip install modal
modal setup
modal volume create tubelet-data
modal volume create tubelet-ground-truth
modal secret create tubelet-openai OPENAI_API_KEY="$OPENAI_API_KEY"
modal secret create wandb WANDB_API_KEY="$WANDB_API_KEY"
```

The local Python environment is `.venv/`. Runtime credentials are separate
Modal environments: `tubelet-openai` injects `OPENAI_API_KEY` only into the GPU
inference function, while `wandb` injects `WANDB_API_KEY` only into the CPU
evaluation function. Neither key is copied into an image or committed to Git.

Upload the prepared frames and mask:

```bash
modal volume put \
  tubelet-data \
  work/S12_Sandwich_7150991-2470/frames \
  /S12_Sandwich_7150991-2470/frames

modal volume put \
  tubelet-data \
  work/S12_Sandwich_7150991-2470/mask.png \
  /S12_Sandwich_7150991-2470/mask.png

modal volume put \
  tubelet-data \
  vocabulary/CMU/state_dict.json \
  /S12_Sandwich_7150991-2470/state_dict.json

modal volume put \
  tubelet-ground-truth \
  annotations/ground_truth/CMU/gt_annotations_cmu.json \
  /S12_Sandwich_7150991-2470/ground_truth.json

# After verifying the upload above, remove the legacy inference-volume copy.
modal volume rm \
  tubelet-data \
  /S12_Sandwich_7150991-2470/ground_truth.json
```

Ground truth lives in the separate `tubelet-ground-truth` Volume. That Volume is
mounted only by the evaluation function; it is not mounted in the GPU inference
function. The runner also refuses inference if a legacy ground-truth file is
found on `tubelet-data`.

First run TubeletGraph's bundled example after the secret exists:

```bash
modal run modal_tubelet.py --smoke-test
```

Only after that succeeds, run S12:

```bash
modal run modal_tubelet.py \
  --video-id S12_Sandwich_7150991-2470 \
  --fps 30
```

The runner verifies CUDA, mask shape/values, and contiguous frame names before
starting. The image pins TubeletGraph commit
`fdb05b6fbd7f4644aea990bf967cc18d82bf291b` and uses an A100 80 GB GPU. It
persists predictions, raw VLM responses, mapped event JSON, visualizations, a
run log, and a manifest beneath:

```text
/S12_Sandwich_7150991-2470/runs/<run-name>/
```

Use `modal volume ls tubelet-data` to find the run and `modal volume get` to
download it. Do not calculate MOSCATO metrics before visually validating that
the tracked mask stays on the bread.

### Convert and evaluate a validated run

The Modal runner automatically writes `tubelet_events.draft.json` and the
closed-vocabulary `tubelet_events.json` beside the run manifest. The following
commands are the reproducible fallback if the adapter needs to be rerun from a
downloaded raw prediction.

Locate the VLM-enriched prediction JSON under the run's `predictions/` tree and
build a draft that preserves every raw TubeletGraph node:

```bash
python3 scripts/build_event_draft.py \
  --prediction <run>/predictions/custom-S12_Sandwich_7150991-2470-Ours_gpt-4.1/S12_Sandwich_7150991-2470_1.json \
  --out work/S12_Sandwich_7150991-2470/tubelet_events.draft.json
```

The draft deliberately contains null state fields. Run the fixed closed-list
adapter with the OpenAI key supplied as a runtime environment variable (or run
this command inside Modal with the same secret):

```bash
python3 scripts/map_event_states.py \
  --draft work/S12_Sandwich_7150991-2470/tubelet_events.draft.json \
  --state-dict vocabulary/CMU/state_dict.json \
  --frames work/S12_Sandwich_7150991-2470/frames \
  --out work/S12_Sandwich_7150991-2470/tubelet_events.json
```

The adapter sees one event-aligned image, TubeletGraph semantic text, and the
allowed state names. It does not load `gt_annotations_cmu.json`, and video IDs,
filenames, local frame numbers, and absolute frame numbers are excluded from the
model prompt. Missing semantic text becomes an explicit abstention rather than a
generated state. Image hashes, semantic evidence, raw responses, mapped labels,
and abstentions remain in the output for audit.

Evaluate only after inspecting those mappings:

```bash
python3 scripts/evaluate.py \
  --annotations annotations/ground_truth/CMU/gt_annotations_cmu.json \
  --state-dict vocabulary/CMU/state_dict.json \
  --events work/S12_Sandwich_7150991-2470/tubelet_events.json \
  --out-csv results/S12_Sandwich_7150991-2470/bread_slices.csv
```

For a completed Modal run, calculate the same metrics and publish the final
results, threshold curve, per-frame table, and reproducibility artifact to W&B:

```bash
modal run modal_tubelet.py \
  --evaluate-run-name <validated-run-name>
```

Results go to the `chefOST` entity, in the `tubeletgraph - full evals` project
by default. Add `--experiment` for a development run, which logs to
`tubeletgraph-experiments` instead. Each run is tagged automatically with its
eval kind, video, session, object, fps, stride, later-tracking setting, adapter
input mode and TubeletGraph commit. Add extra tags with
`--wandb-tags "prompt-v2,rerun"`. The command prints the final four scores and W&B run URL,
and saves the evaluation beneath the completed run's `evaluation/` directory
on the `tubelet-data` Volume.

### Batch inference over the shared videos

To run TubeletGraph over all 41 MOSCATO videos on R2, with raw output and the
SAM2 and CropFormer speedups, see
[docs/batch_inference_and_speedups.md](docs/batch_inference_and_speedups.md).
It also covers cost, credentials and masks.

### Existing evaluator diagnostics

The [project metrics](https://github.com/chefOST/tubletegraph/blob/main/README.md#metrics)
are frame-averaged precision and recall (detail). This branch's scripts still
report legacy hit accuracy and micro-averaged scores; they do not yet implement
the project metrics. Existing diagnostic rules:

- One evaluation unit is one target object at one raw video frame.
- Local frame `i` maps to MOSCATO frame `2824 + i`.
- The initial predicted state propagates to the first transition; transition
  states forward-fill until the next transition.
- A one-state prediction is correct when it is present in MOSCATO's state list.
- Frames whose MOSCATO state list is empty are excluded from adjective metrics.
- TubeletGraph/VLM text must be mapped to the checked-in MOSCATO vocabulary
  without consulting the ground-truth timeline.
- Frames without TubeletGraph semantic text are abstentions and count as
  incorrect; coverage and abstained-frame count are reported with the metrics.
- `word_accuracy/accuracy` is exact per-frame state-label hit accuracy.
- `word_accuracy/precision` and `word_accuracy/f1` are micro-averaged over the
  canonical MOSCATO state vocabulary.
- `word_accuracy/f1_max` is the best micro-F1 from global thresholds 0.1 through
  0.9. The current adapter emits one hard state rather than calibrated scores,
  so its one-hot score sweep is reported explicitly and F1-max will ordinarily
  match micro-F1.

The original detailed setup notes are retained in
`tubeletgraph_moscato_cmu_setup.md` as local source material.
