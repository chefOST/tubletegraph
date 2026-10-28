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
