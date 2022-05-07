## Problem

Panoptic Segmentation = instance segmentation + semantic segmentation

https://www.v7labs.com/blog/panoptic-segmentation-guide

## Summary

- First end-to-end model for panoptic segmentation. That is, no hand-coded priors like box detection.
- PQ-style loss function via bipartite matching between predicted masks and ground truth masks
- Dual-path transformer for CNN to read and write global memory with no layer restriction. This is a novel way to combine CNN with transformers.
- Closed the gap between box-based and box-free methods.

? Possible classes include thing classes (instance segments), stuff classes (semantic segments), and a empty class. This removes the need for merging operators.

## Previous Approaches

<p align="center">
<img src="">
</p>

Previously, panoptic segmentation has been addressed with tree of surrogate tasks. This proved to be too tedious, so recent works are focused on simplifying this pipeline.

- Box-based panoptic segmentation

First detect bounding boxes, then predict a mask for each box.

Examples: Mask R-CNN, FPN

The instance segments (thing) and semantic segments (stuff) are later merged to produce panoptic segmentation.

Worth mentioning: DETR extended box-based methods with transformer-based end-to-end detector

- Bos-free panoptic segmentation

First perform semantic segmentation, then and group the instance pixels. The grouping techniques include: instance center regression, pixel affinity.

Worth mentioning: Axial-Deeplab equipped fully axial attention-backbone.

<p align="center">
<img src="Reviews/Max-DeepLab/Assets/previous.png">
</p>

Box-based methods fail due to low confidence, while box-free methods fail due to complexities such as overlapping center.

## MaX Deeplab
