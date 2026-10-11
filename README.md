# TubeletGraph / ChefOST

ChefOST evaluates object-state tracking in procedural cooking videos. Use this repository for TubeletGraph work.

## Metrics

Precision is the fraction of evaluated frames with at least one predicted label matching ground truth:

`precision = frames with ≥1 matching predicted label / evaluated frames`

Detail is a desired metric; its definition and scoring protocol are pending. Record the dataset, split and sampling alongside results. Keep published paper results distinct from ChefOST experiments, and label proposed changes to metric definitions explicitly.

## Context

Read the [project goals](https://github.com/chefOST/.github-private/blob/main/docs/PaperNotes.md#project-goals), then the [TubeletGraph paper notes](https://github.com/chefOST/.github-private/blob/main/docs/PaperNotes.md#6-tubeletgraph) relevant to your task. Use the [organization guide](https://github.com/chefOST/.github-private/blob/main/profile/README.md) for shared project context and the [data repository](https://github.com/chefOST/chefOST-data) for shared annotations and vocabularies.

## Experiment tracking

Follow [W&B setup](weights_and_biases/README.md) to configure the appropriate team and project. Verify the destination used by your actual run command before reporting results.
