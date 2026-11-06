# One-video run

Status: completed on 2026-11-06. All four object artifacts and periodic
tubelet snapshots are verified in W&B; the collector stopped and no inference
containers remain active. W&B reports US$6.08 estimated compute including the
US$3 prior-compute allowance; API/build/storage charges are separate.
ChefOST metric evaluation remains unscored.
[W&B run](https://wandb.ai/chefOST/tubeletgraph%20-%20full%20evals/runs/vie9lddr),
[Modal app](https://modal.com/apps/justintwx7/main/ap-l6praOQO1rKvdcucE8O6Cv).
Current user limit: below US$40; configured total US$39. The [budgeted runner](../run_budgeted_video.py) verifies
`chefOST / tubeletgraph - full evals` before launching; access was verified
with W&B account `justintwx7`. See [destinations](../weights_and_biases/README.md).

Video: `S15_Salad_7150991-451`, four annotated objects at 5 fps with
later-object tracking and existing speedups. Its 3,888 source frames are on
`tubelet-data`; source metadata reports 30 fps and 800 × 600 pixels.

The user approved visible-frame initialization. The reviewed mask geometry,
image/mask hashes and exclusions are in
[initialization metadata](../configs/s15_salad_7150991-451_masks.json);
the [batch](../configs/batch_s15_salad_visible_start.json) uses those starts.
All four 800 × 600 indexed masks passed shape/value checks.

| Object | Initial frame | Omitted lead-in |
|---|---|---|
| Bowl | 654 | 4.2 s |
| Knife | 1152 | 20.8 s |
| Mayonnaise | 3024 | 3.8 s |
| Vegetables | 846 | 0.2 s |

This retains 7,026 of 7,896 annotated-span object-frames (89.0%). The omitted
870 object-frames are not inferred. Masks are assistant-traced visible
contours, not dataset segmentation ground truth; packaging and motion blur
make the vegetable and dressing boundaries approximate.

The runner uses one GPU job at a time, no configured retries and a two-hour
per-object timeout. The US$39 total allocates US$3 to prior compute (recorded
estimate: US$2.53), US$21 to remaining compute and US$15 to API/build/storage
and controller costs. This is a time-based safeguard, not an invoice or an
account-wide billing cap. See [Modal list rates](https://modal.com/pricing).

[Recovery coordinator](../resume_budgeted_video.py) verifies completed outputs
and runs on Modal, so local client disconnection does not interrupt sequencing.
Run state is persisted under `/<video>/orchestration/` on `tubelet-data`.

W&B receives raw predictions, manifests, logs and completion status.
The [periodic collector](../scripts/watch_tubelets.py) checks every 60 seconds
and uploads changed intermediate tubelets, candidate/target mask contact sheets
and source metadata under the same run’s `Files/tubelet_snapshots/`. The first
knife and mayonnaise snapshots were verified on W&B. It runs as a detached
local CPU process, so the local machine must remain awake; GPU inference is
not restarted. Collector logs/PID are in
`results/S15_Salad_7150991-451/periodic-tubelets/`. Scoring
remains separate; the existing legacy evaluator is not used as ChefOST
precision/recall. See the [batch protocol](batch_inference_and_speedups.md).

Validated: thirteen runner, batch-loader and mask-format tests passed. Runner
coverage includes wrong destination, missing masks, cancellation at the
time limit, and artifact publication.

Snapshot publication: two tests passed for unchanged-output skipping and
attaching compressed predictions, previews and metadata to the existing run.

[Media backfill](../scripts/log_tubelet_media.py) added W&B image panels:
`tubelets/<object>` shows each of the four final target tracks;
`candidate_tubelets/` shows the 128 saved proposal tracks in paged galleries.
`tubelet_progress/` shows captured progress and `tubelet_gallery/all_tracks`
is the searchable table. Green masks show one track per image, with local and
source frame indices. Candidate proposals are not validated target descendants.
These panels were backfilled from saved outputs without new GPU inference.
Two rendering tests verify per-track sampling and source-frame alignment.
