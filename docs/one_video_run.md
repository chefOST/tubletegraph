# One-video run

Status: launched on 2026-11-06; completion pending.
[W&B run](https://wandb.ai/chefOST/tubeletgraph%20-%20full%20evals/runs/vie9lddr),
[Modal app](https://modal.com/apps/justintwx7/main/ap-lejox3B5EHUNPQlx7B144o).
User authorized US$70. The [budgeted runner](../run_budgeted_video.py) verifies
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

The runner uses one GPU job at a time, no configured retries, a two-hour
per-object timeout, and a time-based US$55 compute allowance. US$15 is reserved
for API/build/storage; this is a planning safeguard, not an invoice or an
account-wide billing cap. Four two-hour jobs would cost about US$27.12 at the
checked [Modal list rates](https://modal.com/pricing), excluding startup.

W&B receives raw predictions, manifests, logs and completion status. Scoring
remains separate; the existing legacy evaluator is not used as ChefOST
precision/recall. See the [batch protocol](batch_inference_and_speedups.md).

Validated: ten runner, batch-loader and mask-format tests passed. Runner
coverage includes wrong destination, missing masks, cancellation at the
time limit, and artifact publication.
