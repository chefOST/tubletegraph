# TubeletGraph × MOSCATO (CMU-MMAC) — End-to-End Setup

Historical setup notes for the single-state prototype. The example evaluator below
reports hit accuracy and micro/macro scores; it does not implement the current
[project metrics](https://github.com/chefOST/tubletegraph/blob/main/README.md#metrics):
frame-averaged precision and recall (detail). See the [branch README](README.md#existing-evaluator-diagnostics)
for existing evaluator diagnostics.

Original experiment setup:

> Run **TubeletGraph** on the exact CMU-MMAC videos used by **MOSCATO**, convert TubeletGraph's sparse state changes into one state prediction per frame, then compare those predictions against MOSCATO's frame-level state annotations.

---

## 0. Important terminology

The `.avi` file is **not** the ground truth by itself.

You have three separate things:

```text
CMU-MMAC .avi video
        │
        ├──→ TubeletGraph → predictions
        │
        └──→ MOSCATO annotation JSON → ground truth
```

So:

- **Input video:** exact CMU-MMAC `.avi`
- **Ground truth:** `gt_annotations_cmu.json`
- **Vocabulary / normalization:** `state_dict.json`
- **Video-to-annotation mapping:** `files_dict.json`
- **Prediction method:** TubeletGraph

Your comparison is:

```text
TubeletGraph predicted state at frame t
                vs.
MOSCATO ground-truth state(s) at frame t
```

---

# 1. Recommended architecture if you are on a Mac

TubeletGraph's official environment is CUDA/NVIDIA based. Do the lightweight work on your Mac and run TubeletGraph on Linux + NVIDIA.

```text
Mac
├── CMU-MMAC .avi files
├── MOSCATO annotations/vocabulary
├── frame extraction
├── first-frame mask creation
└── evaluation scripts

        upload frames + mask
                 │
                 ▼

Linux NVIDIA GPU
(Modal / university server / RunPod / etc.)
├── TubeletGraph
├── SAM2
├── CropFormer
├── FC-CLIP
└── GPT-4.1 call
        │
        ▼
TubeletGraph output
        │
        ▼
Mac
└── temporal densification + MOSCATO evaluation
```

Do **not** make Apple Metal/MPS the first target. The official TubeletGraph installation is tested with:

- Python 3.10
- PyTorch 2.7.0 + CUDA 12.6
- torchvision 0.22.0 + CUDA 12.6
- RTX A6000

Source: https://github.com/YihongSun/TubeletGraph

---

# 2. Suggested project structure

Create a project folder:

```bash
mkdir -p tubelet-moscato
cd tubelet-moscato
```

Use this layout:

```text
tubelet-moscato/
├── README.md
│
├── moscato/
│   ├── annotations/
│   │   └── ground_truth/
│   │       └── CMU/
│   │           ├── gt_annotations_cmu.json
│   │           └── gt_annotations_cmu_idx.json
│   │
│   └── vocabulary/
│       └── CMU/
│           ├── files_dict.json
│           ├── state_dict.json
│           ├── object_dict.json
│           └── action_dict.json
│
├── cmu_videos/
│   └── S48_Salad_7150991-222.avi
│
├── work/
│   └── S48_Salad_7150991-222/
│       ├── frames/
│       ├── first_frame.jpg
│       ├── mask.png
│       ├── metadata.json
│       ├── tubelet_events.json
│       └── comparison.csv
│
└── scripts/
    ├── inspect_video.py
    ├── make_mask.py
    └── evaluate.py
```

---

# 3. Unzip the MOSCATO files

Your archives contain the CMU files at these paths:

```text
annotations/ground_truth/CMU/gt_annotations_cmu.json
annotations/ground_truth/CMU/gt_annotations_cmu_idx.json

vocabulary/CMU/files_dict.json
vocabulary/CMU/state_dict.json
vocabulary/CMU/object_dict.json
vocabulary/CMU/action_dict.json
```

Example:

```bash
unzip annotations-20260814T191719Z-1-001.zip -d moscato/
unzip vocabulary-20260814T191804Z-1-001.zip -d moscato/
```

Depending on how you unzip them, you may end up with:

```text
moscato/annotations/...
moscato/vocabulary/...
```

That is the structure assumed below.

---

# 4. Match a CMU video to MOSCATO

MOSCATO's `files_dict.json` already identifies the exact CMU-MMAC video.

For example:

```text
MOSCATO key:
S48_Salad_7150991-222

CMU-MMAC video:
S48_Salad_7150991-222.avi
```

The filename stem must match the MOSCATO key exactly.

## Quick check

```bash
python - <<'PY'
import json
from pathlib import Path

files_dict = json.load(open("moscato/vocabulary/CMU/files_dict.json"))
gt = json.load(open("moscato/annotations/ground_truth/CMU/gt_annotations_cmu.json"))

video = Path("cmu_videos/S48_Salad_7150991-222.avi")
video_id = video.stem

print("Video ID:", video_id)
print("In files_dict:", video_id in files_dict)
print("In ground truth:", video_id in gt)

if video_id in files_dict:
    print("Original MOSCATO path:", files_dict[video_id]["video_path"])

if video_id in gt:
    print("MOSCATO objects:", list(gt[video_id]["state"].keys()))
    print("GT timeline length:", len(gt[video_id]["object"]))
PY
```

For the example MOSCATO file, the annotated objects include things such as:

```text
vegetables
knife
vegetable peeler
mayonnaise
bowl
cutting board
```

---

# 5. Verify video FPS and frame count

Install FFmpeg on your Mac:

```bash
brew install ffmpeg
```

Inspect the source video:

```bash
ffprobe \
  -v error \
  -select_streams v:0 \
  -show_entries stream=avg_frame_rate,r_frame_rate,nb_frames,width,height \
  -of json \
  cmu_videos/S48_Salad_7150991-222.avi
```

You care about:

- FPS
- number of raw frames
- width
- height

## Critical alignment check

The MOSCATO ground-truth timeline must align to the raw video frames you evaluate.

Check:

```bash
python - <<'PY'
import json

VIDEO_ID = "S48_Salad_7150991-222"

gt = json.load(open(
    "moscato/annotations/ground_truth/CMU/gt_annotations_cmu.json"
))

print("MOSCATO GT frames:", len(gt[VIDEO_ID]["object"]))

for obj, states in gt[VIDEO_ID]["state"].items():
    assert len(states) == len(gt[VIDEO_ID]["object"])
print("All state timelines have matching lengths.")
PY
```

**Do not use `feature_len` from `files_dict.json` as the raw frame count.**

MOSCATO's `files_dict.json` also contains feature metadata. The feature sequence can have a lower temporal sampling rate than the original video. Your state ground truth is in `gt_annotations_cmu.json`, so use the length of those arrays for your frame-level evaluation.

---

# 6. Choose one target object first

Do not try to benchmark every object and every video immediately.

Start with:

```text
1 video
×
1 object
```

For example:

```text
Video:
S48_Salad_7150991-222.avi

Object:
vegetables
```

TubeletGraph needs a segmentation mask of the target object in the **first frame of the TubeletGraph input clip**.

This creates an important constraint:

> The target object must actually be visible in the first frame you give TubeletGraph.

If the object is not visible at raw video frame 0, start TubeletGraph from a later frame.

---

# 7. Decide whether to run the full video or a clip

## Option A — object visible in raw frame 0

Use:

```text
clip_start_global = 0
```

and extract the entire video.

## Option B — object appears later

Choose a global raw frame where the object is clearly visible:

```text
clip_start_global = 2500
```

Then TubeletGraph local frame 0 corresponds to MOSCATO global frame 2500:

```text
Tubelet local frame 0    → MOSCATO frame 2500
Tubelet local frame 1    → MOSCATO frame 2501
Tubelet local frame 100  → MOSCATO frame 2600
```

Always map with:

```python
global_frame = clip_start_global + local_frame
```

This offset is essential.

---

# 8. Extract frames from the AVI

Create the work folder:

```bash
VIDEO_ID="S48_Salad_7150991-222"

mkdir -p "work/$VIDEO_ID/frames"
```

## Full video

Preserve every decoded frame:

```bash
ffmpeg \
  -i "cmu_videos/$VIDEO_ID.avi" \
  -vsync 0 \
  -start_number 0 \
  "work/$VIDEO_ID/frames/%07d.jpg"
```

You should get:

```text
0000000.jpg
0000001.jpg
0000002.jpg
...
```

That is deliberate: local filename index = Tubelet local frame index.

## If starting from a later raw frame

The safest approach for exact raw-frame alignment is to decode all frames first, then select/copy the range you want.

Example, start at raw frame 2500:

```bash
mkdir -p "work/$VIDEO_ID/clip_frames"

python - <<'PY'
from pathlib import Path
import shutil

src = Path("work/S48_Salad_7150991-222/frames")
dst = Path("work/S48_Salad_7150991-222/clip_frames")

START = 2500
END = 5000  # exclusive; change as needed

dst.mkdir(parents=True, exist_ok=True)

for local_idx, global_idx in enumerate(range(START, END)):
    source = src / f"{global_idx:07d}.jpg"
    target = dst / f"{local_idx:07d}.jpg"
    if not source.exists():
        break
    shutil.copy2(source, target)

print("Done")
PY
```

Now:

```text
clip_frames/0000000.jpg = raw global frame 2500
clip_frames/0000001.jpg = raw global frame 2501
...
```

Keep `START=2500` recorded for evaluation.

---

# 9. Check extracted frame count against MOSCATO

Run:

```bash
python - <<'PY'
import json
from pathlib import Path

VIDEO_ID = "S48_Salad_7150991-222"

frames = sorted(Path(f"work/{VIDEO_ID}/frames").glob("*.jpg"))

gt = json.load(open(
    "moscato/annotations/ground_truth/CMU/gt_annotations_cmu.json"
))

gt_len = len(gt[VIDEO_ID]["object"])

print("Extracted frames:", len(frames))
print("MOSCATO GT frames:", gt_len)

if len(frames) == gt_len:
    print("PASS: raw frames and MOSCATO timeline align in length.")
else:
    print("WARNING: frame counts differ. Do not evaluate until you understand why.")
PY
```

Do not silently resample one array to match the other.

---

# 10. Create the TubeletGraph first-frame mask

TubeletGraph expects:

```text
0   = background
1   = target object 1
2   = target object 2
...
255 = ignore
```

For the first experiment, use **one target object**:

```text
0 = background
1 = target object
```

The mask must:

- be PNG
- have the same width and height as the first input frame
- be a single-channel integer image
- use pixel value `1` over the target object

---

# 11. Simple local mask drawing tool

Save as:

```text
scripts/make_mask.py
```

```python
import argparse
from pathlib import Path

import cv2
import numpy as np


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--image", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    image = cv2.imread(args.image)
    if image is None:
        raise FileNotFoundError(args.image)

    points = []
    preview = image.copy()

    def redraw():
        nonlocal preview
        preview = image.copy()

        for p in points:
            cv2.circle(preview, p, 4, (0, 0, 255), -1)

        if len(points) > 1:
            cv2.polylines(
                preview,
                [np.array(points, np.int32)],
                False,
                (0, 255, 0),
                2,
            )

        cv2.imshow("Mask: click polygon | ENTER save | r reset | q quit", preview)

    def on_mouse(event, x, y, flags, param):
        if event == cv2.EVENT_LBUTTONDOWN:
            points.append((x, y))
            redraw()

    cv2.namedWindow("Mask: click polygon | ENTER save | r reset | q quit")
    cv2.setMouseCallback(
        "Mask: click polygon | ENTER save | r reset | q quit",
        on_mouse,
    )

    redraw()

    while True:
        key = cv2.waitKey(20) & 0xFF

        if key in (10, 13):  # enter
            if len(points) < 3:
                print("Need at least 3 polygon points.")
                continue

            mask = np.zeros(image.shape[:2], dtype=np.uint8)
            cv2.fillPoly(
                mask,
                [np.array(points, np.int32)],
                1,
            )

            Path(args.out).parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(args.out, mask)

            print(f"Saved: {args.out}")
            print("Unique mask values:", np.unique(mask))
            break

        elif key == ord("r"):
            points.clear()
            redraw()

        elif key == ord("q"):
            break

    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
```

Install OpenCV locally:

```bash
pip install opencv-python
```

Run it:

```bash
python scripts/make_mask.py \
  --image work/S48_Salad_7150991-222/frames/0000000.jpg \
  --out work/S48_Salad_7150991-222/mask.png
```

If you made a clipped input, use:

```text
clip_frames/0000000.jpg
```

instead.

## Verify the mask

```bash
python - <<'PY'
from PIL import Image
import numpy as np

mask = np.array(Image.open(
    "work/S48_Salad_7150991-222/mask.png"
))

print(mask.shape)
print(np.unique(mask))
PY
```

For one object, you want:

```text
[0 1]
```

---

# 12. Install TubeletGraph on Linux + NVIDIA

Official repository:

https://github.com/YihongSun/TubeletGraph

Clone:

```bash
git clone --recurse-submodules https://github.com/YihongSun/TubeletGraph/
cd TubeletGraph
```

Create the environment:

```bash
conda create -n tubeletgraph python=3.10 -y
conda activate tubeletgraph
```

Install the tested PyTorch/CUDA versions:

```bash
pip install \
  torch==2.7.0 \
  torchvision==0.22.0 \
  --index-url https://download.pytorch.org/whl/cu126
```

Install TubeletGraph requirements and checkpoints:

```bash
pip install -r requirements.txt
bash thirdparty/setup_ckpts.sh
```

---

# 13. Install SAM2

From the TubeletGraph root:

```bash
cd thirdparty/sam2

pip install -e .
pip install -e ".[notebooks]"
python setup.py build_ext --inplace

cd ../..
```

---

# 14. Install CropFormer / Detectron2

From TubeletGraph root:

```bash
cd thirdparty

git clone https://github.com/facebookresearch/detectron2.git

python -m pip install -e detectron2 --no-build-isolation

ln -s "$(pwd)"/Entity/Entityv2/CropFormer \
  detectron2/projects/CropFormer

cd detectron2/projects/CropFormer/mask2former/modeling/pixel_decoder/ops

bash make.sh

cd ../../../../../../../..
```

If you get a `libstdc++` mismatch, the TubeletGraph README suggests:

```bash
conda install -c conda-forge libstdcxx-ng
```

---

# 15. Install FC-CLIP

```bash
cd thirdparty/fc-clip
pip install -r requirements.txt
cd ../..
```

---

# 16. Configure the OpenAI API key

TubeletGraph's official setup requires an OpenAI API key for GPT-4.1 state reasoning.

For the current shell:

```bash
export OPENAI_API_KEY="YOUR_KEY"
```

Verify only that it exists:

```bash
python - <<'PY'
import os
print(bool(os.environ.get("OPENAI_API_KEY")))
PY
```

Do not print or commit the actual key.

---

# 17. Verify CUDA before running TubeletGraph

```bash
nvidia-smi
```

Then:

```bash
python - <<'PY'
import torch

print("Torch:", torch.__version__)
print("CUDA available:", torch.cuda.is_available())

if torch.cuda.is_available():
    print("GPU:", torch.cuda.get_device_name(0))
    print("VRAM GB:", torch.cuda.get_device_properties(0).total_memory / 1e9)
PY
```

Do not continue until:

```text
CUDA available: True
```

---

# 18. Test TubeletGraph's bundled example first

Before introducing CMU/MOSCATO, make sure the repository itself works:

```bash
python quick_run.py \
  --input_dir assets/example/0334_cut_fruit_1 \
  --input_mask assets/example/0334_cut_fruit_1_0000000.png
```

The official repo says custom predictions/visualizations are written under paths similar to:

```text
_pred_out/predictions/custom-<video>-Ours_gpt-4.1
```

If the bundled example fails, fix TubeletGraph before debugging your CMU data.

---

# 19. Run your CMU clip through TubeletGraph

Upload/copy these two inputs to the Linux GPU machine:

```text
frames/
mask.png
```

Then:

```bash
python quick_run.py \
  --input_dir /path/to/frames \
  --input_mask /path/to/mask.png \
  --fps 30
```

Replace `30` with the actual source FPS you measured.

TubeletGraph's quick run expects:

```text
--input_dir    directory of individual video frames
--input_mask   first-frame prompt mask
--fps          visualization/video FPS
```

It generates:

- tracked-object video visualization
- state graph visualization
- internal prediction artifacts

---

# 20. Do not evaluate yet — visually validate the run

For the first video/object, check:

```text
[ ] Correct object is selected in the first frame
[ ] Tracking stays on the object
[ ] Pieces created after transformations are recovered
[ ] State changes are at approximately reasonable times
[ ] State descriptions are semantically reasonable
```

If tracking is wrong, MOSCATO accuracy is meaningless.

---

# 21. Convert TubeletGraph output into a simple event JSON

For the first prototype, do not spend hours reverse-engineering every TubeletGraph internal artifact.

Create a clean intermediate representation:

```text
work/<VIDEO_ID>/tubelet_events.json
```

Example:

```json
{
  "video_id": "S48_Salad_7150991-222",
  "object": "vegetables",
  "clip_start_global": 0,
  "num_local_frames": 9693,
  "initial_state": "peeled",
  "transitions": [
    {
      "local_frame": 3451,
      "state": "chopped"
    },
    {
      "local_frame": 8568,
      "state": "mayonnaise-mixed"
    }
  ]
}
```

This file represents **TubeletGraph predictions**, not MOSCATO.

For a clipped run starting at raw frame 2500:

```json
{
  "video_id": "S48_Salad_7150991-222",
  "object": "vegetables",
  "clip_start_global": 2500,
  "num_local_frames": 3000,
  "initial_state": "peeled",
  "transitions": [
    {
      "local_frame": 951,
      "state": "chopped"
    }
  ]
}
```

Global transition frame:

```python
global_frame = clip_start_global + local_frame
```

---

# 22. Mapping TubeletGraph text to MOSCATO vocabulary

TubeletGraph may describe a state using free text.

MOSCATO expects canonical states.

Your CMU vocabulary file has:

```text
state_dict.json
```

with mappings including:

```python
state_dict["all2one"]
```

Use it to normalize known synonyms.

Example:

```python
canonical = state_dict["all2one"].get(raw_state, raw_state)
```

## Important

A complete TubeletGraph sentence such as:

```text
"the vegetables have been chopped into small pieces"
```

will usually **not** match an `all2one` dictionary key directly.

You therefore need a fixed adapter that maps TubeletGraph's description to exactly one MOSCATO state.

Two defensible choices:

### Choice A — constrain the TubeletGraph/VLM output

Give the VLM the allowed MOSCATO states and require:

```text
Return exactly one state from the supplied vocabulary.
```

### Choice B — post-process TubeletGraph's raw state description

Use a fixed classification prompt after TubeletGraph.

Always save:

```text
raw TubeletGraph description
mapped MOSCATO state
```

so your evaluation is auditable.

Do not silently hand-edit predictions after looking at the MOSCATO ground truth.

---

# 23. Temporal densification rule

This is the core solution to:

> MOSCATO is per-frame but TubeletGraph is sparse/transition-based.

Use:

```text
initial state
    ↓
cover clip start → first transition - 1

transition state #1
    ↓
forward-fill until transition #2 - 1

transition state #2
    ↓
forward-fill until transition #3 - 1

...
```

Example:

```text
TubeletGraph events

local frame 0     initial = peeled
local frame 900   → chopped
local frame 1800  → mixed
```

Dense result:

```text
0–899      peeled
900–1799   chopped
1800–end   mixed
```

This is better described as:

> **Initial-state propagation + transition-state forward filling**

Do **not** blindly backward-fill every transition.

---

# 24. Historical evaluation policy for MOSCATO

MOSCATO can contain multiple state labels for one object at one frame.

Example:

```python
gt_states = ["full", "mayonnaise-mixed"]
```

If TubeletGraph must output **one adjective**, use:

```python
correct = predicted_state in gt_states
```

Example:

```text
prediction = mayonnaise-mixed
GT         = [full, mayonnaise-mixed]

→ correct
```

## Empty MOSCATO state frames

Some frames have:

```python
[]
```

Recommended default:

> Do not include a frame in adjective accuracy if MOSCATO supplies no ground-truth state for that object at that frame.

Otherwise you would be penalizing a required single-adjective predictor against a frame with no target adjective.

If your supervisor explicitly defines empty frames as a `"none"` class, change the evaluation accordingly.

---

# 25. Historical evaluation script

Save this as:

```text
scripts/evaluate.py
```

```python
import argparse
import csv
import json
from pathlib import Path

from sklearn.metrics import precision_recall_fscore_support


def normalize_state(state, all2one):
    state = state.strip().lower()

    if state in all2one:
        return all2one[state]

    # If it is already a canonical state but not repeated in all2one.
    canonical_states = set(all2one.values())
    if state in canonical_states:
        return state

    raise ValueError(
        f"State {state!r} is not recognized by MOSCATO state_dict. "
        "Map TubeletGraph output to a canonical MOSCATO state before evaluation."
    )


def densify(events, all2one):
    n = events["num_local_frames"]

    initial = normalize_state(events["initial_state"], all2one)
    pred = [initial] * n

    transitions = sorted(
        events.get("transitions", []),
        key=lambda x: x["local_frame"],
    )

    for transition in transitions:
        frame = int(transition["local_frame"])
        state = normalize_state(transition["state"], all2one)

        if not 0 <= frame < n:
            raise ValueError(
                f"Transition frame {frame} is outside [0, {n})."
            )

        for i in range(frame, n):
            pred[i] = state

    return pred


def main():
    parser = argparse.ArgumentParser()

    parser.add_argument("--annotations", required=True)
    parser.add_argument("--state-dict", required=True)
    parser.add_argument("--events", required=True)
    parser.add_argument("--out-csv", required=True)

    args = parser.parse_args()

    annotations = json.load(open(args.annotations))
    state_dict = json.load(open(args.state_dict))
    events = json.load(open(args.events))

    video_id = events["video_id"]
    object_name = events["object"]
    clip_start = int(events.get("clip_start_global", 0))

    if video_id not in annotations:
        raise KeyError(f"{video_id} not found in MOSCATO annotations.")

    video_gt = annotations[video_id]

    if object_name not in video_gt["state"]:
        raise KeyError(
            f"{object_name!r} not found for {video_id}. "
            f"Available: {list(video_gt['state'])}"
        )

    gt_timeline = video_gt["state"][object_name]

    all2one = {
        k.strip().lower(): v.strip().lower()
        for k, v in state_dict["all2one"].items()
    }

    pred_local = densify(events, all2one)

    rows = []
    hits = 0
    evaluated = 0

    # For multilabel precision/recall/F1.
    canonical = sorted(set(state_dict["s2i"].keys()))
    label_to_idx = {label: i for i, label in enumerate(canonical)}

    y_true = []
    y_pred = []

    for local_frame, pred in enumerate(pred_local):
        global_frame = clip_start + local_frame

        if global_frame >= len(gt_timeline):
            break

        gt_states = [
            normalize_state(s, all2one)
            for s in gt_timeline[global_frame]
        ]

        # Default policy: skip frames with no ground-truth adjective.
        if not gt_states:
            rows.append({
                "local_frame": local_frame,
                "global_frame": global_frame,
                "prediction": pred,
                "gt_states": "",
                "evaluated": 0,
                "correct": "",
            })
            continue

        correct = pred in gt_states

        evaluated += 1
        hits += int(correct)

        true_vec = [0] * len(canonical)
        pred_vec = [0] * len(canonical)

        for gt_state in gt_states:
            if gt_state in label_to_idx:
                true_vec[label_to_idx[gt_state]] = 1

        if pred in label_to_idx:
            pred_vec[label_to_idx[pred]] = 1

        y_true.append(true_vec)
        y_pred.append(pred_vec)

        rows.append({
            "local_frame": local_frame,
            "global_frame": global_frame,
            "prediction": pred,
            "gt_states": "|".join(gt_states),
            "evaluated": 1,
            "correct": int(correct),
        })

    if evaluated == 0:
        raise RuntimeError("No annotated frames were evaluated.")

    hit_accuracy = hits / evaluated

    p_micro, r_micro, f1_micro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="micro",
        zero_division=0,
    )

    p_macro, r_macro, f1_macro, _ = precision_recall_fscore_support(
        y_true,
        y_pred,
        average="macro",
        zero_division=0,
    )

    out = Path(args.out_csv)
    out.parent.mkdir(parents=True, exist_ok=True)

    with out.open("w", newline="") as f:
        writer = csv.DictWriter(
            f,
            fieldnames=[
                "local_frame",
                "global_frame",
                "prediction",
                "gt_states",
                "evaluated",
                "correct",
            ],
        )
        writer.writeheader()
        writer.writerows(rows)

    print()
    print("Video:", video_id)
    print("Object:", object_name)
    print("Clip start global frame:", clip_start)
    print("Evaluated frames:", evaluated)
    print("Correct frames:", hits)
    print(f"Frame hit accuracy: {hit_accuracy:.4f}")
    print()
    print(f"Micro precision: {p_micro:.4f}")
    print(f"Micro recall:    {r_micro:.4f}")
    print(f"Micro F1:        {f1_micro:.4f}")
    print()
    print(f"Macro precision: {p_macro:.4f}")
    print(f"Macro recall:    {r_macro:.4f}")
    print(f"Macro F1:        {f1_macro:.4f}")
    print()
    print("CSV:", out)


if __name__ == "__main__":
    main()
```

Install the evaluation dependency:

```bash
pip install scikit-learn
```

Run:

```bash
python scripts/evaluate.py \
  --annotations moscato/annotations/ground_truth/CMU/gt_annotations_cmu.json \
  --state-dict moscato/vocabulary/CMU/state_dict.json \
  --events work/S48_Salad_7150991-222/tubelet_events.json \
  --out-csv work/S48_Salad_7150991-222/comparison.csv
```

---

# 26. What your output should look like

Console:

```text
Video: S48_Salad_7150991-222
Object: vegetables
Clip start global frame: 0
Evaluated frames: ...
Correct frames: ...
Frame hit accuracy: ...

Micro precision: ...
Micro recall: ...
Micro F1: ...

Macro precision: ...
Macro recall: ...
Macro F1: ...
```

CSV:

```csv
local_frame,global_frame,prediction,gt_states,evaluated,correct
0,0,peeled,,0,
1,1,peeled,,0,
...
2613,2613,peeled,peeled,1,1
...
3451,3451,chopped,chopped,1,1
...
```

---

# 27. Modal setup if you do not have an NVIDIA Linux machine

If you already have a university/server NVIDIA GPU, skip this section.

## Install Modal locally on your Mac

```bash
python3 -m pip install modal
modal setup
```

## Create a persistent volume

```bash
modal volume create tubelet-data
```

Upload your frames and mask:

```bash
modal volume put \
  tubelet-data \
  work/S48_Salad_7150991-222/frames \
  /S48_Salad_7150991-222/frames
```

```bash
modal volume put \
  tubelet-data \
  work/S48_Salad_7150991-222/mask.png \
  /S48_Salad_7150991-222/mask.png
```

## Store the API key as a Modal Secret

Create a Modal Secret named:

```text
tubelet-openai
```

containing:

```text
OPENAI_API_KEY=...
```

Use the Modal dashboard or Modal secret CLI. Do not hard-code the key into the image.

## GPU choice

A reasonable starting point is:

```text
A100
```

Modal currently supports GPU names including A10, L40S, A100, H100 and others.

Current docs:

https://modal.com/docs/guide/gpu

---

# 28. Minimal Modal environment definition

TubeletGraph has several native/CUDA dependencies, so treat this as a starting template rather than assuming every machine will build identically.

Create:

```text
modal_tubelet.py
```

```python
import modal

app = modal.App("tubeletgraph-moscato")

data_volume = modal.Volume.from_name(
    "tubelet-data",
    create_if_missing=True,
)

image = (
    modal.Image.from_registry(
        "nvidia/cuda:12.6.3-devel-ubuntu22.04",
        add_python="3.10",
    )
    .entrypoint([])
    .apt_install(
        "git",
        "ffmpeg",
        "build-essential",
        "ninja-build",
        "libgl1",
        "libglib2.0-0",
    )
    .run_commands(
        "git clone --recurse-submodules "
        "https://github.com/YihongSun/TubeletGraph /opt/TubeletGraph",
        "cd /opt/TubeletGraph && "
        "pip install torch==2.7.0 torchvision==0.22.0 "
        "--index-url https://download.pytorch.org/whl/cu126",
        "cd /opt/TubeletGraph && pip install -r requirements.txt",
        "cd /opt/TubeletGraph && bash thirdparty/setup_ckpts.sh",
        "cd /opt/TubeletGraph/thirdparty/sam2 && pip install -e .",
        'cd /opt/TubeletGraph/thirdparty/sam2 && pip install -e ".[notebooks]"',
        "cd /opt/TubeletGraph/thirdparty/sam2 && "
        "python setup.py build_ext --inplace",
        "cd /opt/TubeletGraph/thirdparty && "
        "git clone https://github.com/facebookresearch/detectron2.git",
        "cd /opt/TubeletGraph/thirdparty && "
        "python -m pip install -e detectron2 --no-build-isolation",
        "cd /opt/TubeletGraph/thirdparty && "
        "ln -s $(pwd)/Entity/Entityv2/CropFormer "
        "detectron2/projects/CropFormer",
        "cd /opt/TubeletGraph/thirdparty/detectron2/projects/"
        "CropFormer/mask2former/modeling/pixel_decoder/ops && bash make.sh",
        "cd /opt/TubeletGraph/thirdparty/fc-clip && "
        "pip install -r requirements.txt",
    )
)

@app.function(
    image=image,
    gpu="A100",
    timeout=60 * 60,
    volumes={"/data": data_volume},
    secrets=[modal.Secret.from_name("tubelet-openai")],
)
def run_tubelet(
    video_id: str,
    fps: int = 30,
):
    import os
    import subprocess

    frames = f"/data/{video_id}/frames"
    mask = f"/data/{video_id}/mask.png"

    subprocess.run(
        [
            "python",
            "quick_run.py",
            "--input_dir",
            frames,
            "--input_mask",
            mask,
            "--fps",
            str(fps),
        ],
        cwd="/opt/TubeletGraph",
        check=True,
    )

    print("TubeletGraph finished.")
    print("Inspect /opt/TubeletGraph/_pred_out for results.")


@app.local_entrypoint()
def main(
    video_id: str = "S48_Salad_7150991-222",
    fps: int = 30,
):
    run_tubelet.remote(video_id, fps)
```

Run:

```bash
modal run modal_tubelet.py \
  --video-id S48_Salad_7150991-222 \
  --fps 30
```

### Note about results

The example above mounts your input volume, but TubeletGraph's default `_pred_out` lives inside the container.

For a durable workflow, either:

1. modify the Modal wrapper so `_pred_out` is copied into `/data/<VIDEO_ID>/tubelet_output/` before exit, or
2. configure TubeletGraph's output directory to point into the mounted volume.

Do that once the bundled example successfully runs.

---

# 29. First milestone — stop here until this works

Your first milestone should be exactly:

```text
S48_Salad_7150991-222.avi
        ↓
extract exact raw frames
        ↓
choose one MOSCATO object
        ↓
make first-frame mask
        ↓
TubeletGraph succeeds
        ↓
visually inspect state graph
        ↓
create tubelet_events.json
        ↓
forward-fill to per-frame predictions
        ↓
compare against gt_annotations_cmu.json
        ↓
one accuracy number
```

Do not batch all 30 CMU videos until this works.

---

# 30. Then scale to multiple objects/videos

Once one object works:

```text
for each MOSCATO CMU video:
    verify exact AVI
    for each target object:
        choose valid clip start
        create prompt mask
        run TubeletGraph
        map output → MOSCATO states
        densify predictions
        evaluate
```

Store results like:

```text
results/
├── S48_Salad_7150991-222/
│   ├── vegetables.csv
│   ├── bowl.csv
│   └── ...
├── S29_Salad_7150991-1/
│   └── ...
└── summary.csv
```

Summary:

```csv
video_id,object,evaluated_frames,hit_accuracy,micro_f1,macro_f1
S48_Salad_7150991-222,vegetables,...,...,...,...
...
```

---

# 31. Historical prototype methodology

These were the original prototype choices. Use the current project metric definitions for new benchmark reporting:

## A. What is one evaluation unit?

Recommended:

```text
one object × one raw video frame
```

## B. What happens when MOSCATO has multiple labels?

Recommended for a single-adjective TubeletGraph prediction:

```python
correct = prediction in gt_states
```

## C. What happens when MOSCATO has `[]`?

Recommended:

```text
skip the frame for adjective accuracy
```

unless your assignment explicitly defines `"none"`.

## D. How are TubeletGraph transitions made dense?

Recommended:

```text
initial state covers clip start → first transition
each transition state forward-fills → next transition
```

## E. How do clipped TubeletGraph frames map to MOSCATO?

Always:

```python
global_frame = clip_start_global + local_frame
```

## F. How are TubeletGraph descriptions converted to states?

Use one fixed adapter and never change it based on ground-truth results.

---

# 32. Sanity checks before trusting metrics

Before reporting accuracy/F1:

```text
[ ] Exact MOSCATO video ID matches AVI stem
[ ] Raw extracted frame count checked against GT timeline
[ ] No hidden 10-fps vs 30-fps confusion
[ ] Target object is visible in TubeletGraph first frame
[ ] Mask dimensions equal frame dimensions
[ ] Mask values are integer IDs, usually {0,1}
[ ] TubeletGraph tracks the intended object
[ ] Clip-start global frame is recorded
[ ] Transition timestamps use local TubeletGraph frame numbering
[ ] Local → global frame offset is applied exactly once
[ ] TubeletGraph state strings are normalized to MOSCATO vocabulary
[ ] Empty GT frame handling is consistent
[ ] Multi-label GT handling is consistent
[ ] Raw TubeletGraph output is retained for audit
```

---

# 33. Useful commands

## Find all CMU video IDs MOSCATO expects

```bash
python - <<'PY'
import json

d = json.load(open("moscato/vocabulary/CMU/files_dict.json"))

for video_id in d:
    print(video_id + ".avi")
PY
```

## List MOSCATO objects for one video

```bash
python - <<'PY'
import json

VIDEO_ID = "S48_Salad_7150991-222"

gt = json.load(open(
    "moscato/annotations/ground_truth/CMU/gt_annotations_cmu.json"
))

print(*gt[VIDEO_ID]["state"].keys(), sep="\n")
PY
```

## Find state change boundaries in MOSCATO for debugging

```bash
python - <<'PY'
import json

VIDEO_ID = "S48_Salad_7150991-222"
OBJECT = "vegetables"

gt = json.load(open(
    "moscato/annotations/ground_truth/CMU/gt_annotations_cmu.json"
))

states = gt[VIDEO_ID]["state"][OBJECT]

last = None

for frame, state in enumerate(states):
    current = tuple(state)

    if current != last:
        if current or last:
            print(frame, state)

        last = current
PY
```

This is useful for debugging alignment.

Do **not** use these ground-truth transition boundaries to modify TubeletGraph's predictions.

---

# 34. Final experiment flow

```text
                           ┌──────────────────────┐
                           │ CMU-MMAC exact .avi  │
                           └──────────┬───────────┘
                                      │
                         decode every raw frame
                                      │
                  ┌───────────────────┴────────────────────┐
                  │                                        │
                  ▼                                        ▼
       TubeletGraph input                           MOSCATO JSON
       frames + mask                                per-frame GT
                  │                                        │
                  ▼                                        │
       sparse state transitions                            │
                  │                                        │
                  ▼                                        │
       map to MOSCATO vocabulary                           │
                  │                                        │
                  ▼                                        │
       initial-state propagation                           │
       + transition forward fill                           │
                  │                                        │
                  ▼                                        ▼
       one predicted adjective       ↔        GT state list(s)
       per evaluated frame
                  │
                  └────────────────────┬───────────────────┘
                                       ▼
                               frame hit accuracy
                               precision / recall
                               micro-F1 / macro-F1
```

---

# 35. Official references

TubeletGraph repository:

https://github.com/YihongSun/TubeletGraph

TubeletGraph project page:

https://tubelet-graph.github.io/

Modal GPU documentation:

https://modal.com/docs/guide/gpu

Modal CUDA documentation:

https://modal.com/docs/guide/cuda

Modal Volume CLI:

https://modal.com/docs/cli/latest/volume

---

# 36. What to do right now

Do these in order:

```text
1. Put one exact CMU AVI in cmu_videos/
2. Verify its filename stem exists in gt_annotations_cmu.json
3. Check FPS + raw frame count
4. Extract all raw frames with zero-based filenames
5. Compare extracted count to MOSCATO GT timeline length
6. Pick one MOSCATO object
7. Pick a clip start where that object is visible
8. Make mask.png for clip local frame 0
9. Get TubeletGraph bundled example working on NVIDIA
10. Run TubeletGraph on the CMU frames/mask
11. Inspect tracking + state graph
12. Convert predicted states into tubelet_events.json
13. Run scripts/evaluate.py
14. Inspect comparison.csv around every predicted transition
15. Only then scale up
```
