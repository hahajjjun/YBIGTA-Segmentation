## Problem

Panoptic Segmentation = instance segmentation + semantic segmentation

https://www.v7labs.com/blog/panoptic-segmentation-guide

## Contribution

- First end-to-end model for panoptic segmentation. That is, no hand-coded priors like box detection.
- PQ-style loss function via bipartite matching between predicted masks and ground truth masks
- Dual-path transformer for CNN to read and write global memory with no layer restriction. This is a novel way to combine CNN with transformers.
- Closed the gap between box-based and box-free methods.

