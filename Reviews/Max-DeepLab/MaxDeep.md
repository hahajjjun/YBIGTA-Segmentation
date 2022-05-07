## Problem

Panoptic Segmentation = instance segmentation + semantic segmentation

https://www.v7labs.com/blog/panoptic-segmentation-guide
<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/740c6b01e5788fc677846fb404a0f9ad910bf0d9/Reviews/Max-DeepLab/Assets/panoptic.png" width="600px">
</p>

## Summary

- First end-to-end model for panoptic segmentation. That is, no hand-coded priors like box detection.
- PQ-style loss function via bipartite matching between predicted masks and ground truth masks
- Dual-path transformer for CNN to read and write global memory with no layer restriction. This is a novel way to combine CNN with transformers.
- Closed the gap between box-based and box-free methods.

? Possible classes include thing classes (instance segments), stuff classes (semantic segments), and a empty class. This removes the need for merging operators.

## Previous Approaches

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/740c6b01e5788fc677846fb404a0f9ad910bf0d9/Reviews/Max-DeepLab/Assets/surrogate.png" width="700px">
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
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/98b3a26ec3a169a990aa1df6e33e08d4c162eca6/Reviews/Max-DeepLab/Assets/previous.png" width="700px">
</p>

Box-based methods fail due to low confidence, while box-free methods fail due to complexities such as overlapping center.

# MaX Deeplab

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/6b606900ce561c41763b8fcf30ebcfe85687067e/Reviews/Max-DeepLab/Assets/equation2.png" width="300px">
</p>

Where N denotes all possible classes. For inference, we apply argmax for label and class respectively.

MaX-Deeplab does not require hand-crafted priors.

## PQ-Style Loss

PQ (panoptic quality) = RQ * SQ

Here, RQ (recognition quality) = is the class correct?, and SQ (segmentation quality) = is the mask correct?

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/fdf2fe0aa45ca6163c93c43ec6ed6cf26f34f1fa/Reviews/Max-DeepLab/Assets/equation9.png" width="300px">
</p>

In specific, our goal is to maximize this PQ-style objective. Notice that **both qualities** should be close to 1 in order to achieve high score.

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/e24bdd19e86a91d04edb4e89873f378ce11fa5a6/Reviews/Max-DeepLab/Assets/equation10.png" width="300px">
</p>

Notice that the product rule has been used to calculate the loss. The **log p** term is common in cross-entropy terms and scales great for optimization.

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/3bc3bf9922dd915f6c4ccd68be21142d1a0d1c61/Reviews/Max-DeepLab/Assets/equation11.png" width="300px">
</p>

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/3bc3bf9922dd915f6c4ccd68be21142d1a0d1c61/Reviews/Max-DeepLab/Assets/equation12.png" width="280px">
</p>

We construct the negative tasks for unmatched masks, and finally construct the grand loss.

## Architecture

### Dual Path Transformer

Transformer and the CNN are integrated.

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/4f5839ebdaaa2c624c491e02da407fac35a6a4b1/Reviews/Max-DeepLab/Assets/architecture.png" width="600px">
</p>

Observe that the pixel path accepts 2D feature H x W x d_in, while the memory path accepts 1D global feature N x d_in. For each path, we compute q, k, and v, where

- q, queries
- k, keys
- v, values

Convolution operation is applied for the pixel path, while FC for the memory path.

Notice the four connections: M2P, M2M, P2M, P2P

P2P: Axial attention, adopted from Axial-Deeplab. Notice two attention layers in the bottleneck.

<p align="center">
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/4f5839ebdaaa2c624c491e02da407fac35a6a4b1/Reviews/Max-DeepLab/Assets/axial.png" height="200px">
</p>

P2M, M2P, M2M are performed with softmax function applied over the entire N prediction set. The specific components involved are displayed in the figure

- P2M: q^p, k^m, v^m
- M2P & M2M: q^m, k^pm, v^pm, where k^pm = [k^p, k^m].T, v^pm = [v^p, v^m].T

### Stacked decoder

Notice the lined box is stacked L times. In the experiment, stacking L=2 saturated training performance.

### Output heads

We use two 2FC layers for both mask predictions and class predictions.

- Mask prediction: softmax(f (NxD) * g (DxH/4xW/4) ), m (NxH/4xW/4)
- Class prediction: 2FC(f (NxD)) -> p_hat (NxC)

For mask prediction, the authors use bilinear interpolation to upsample the feature map to original resolution.

## Auxiliary Losses

- Pixel-wise instance discrimination loss
- Mask-ID Cross entropy: per-pixel cross-entropy loss
- Semantic segmentation loss
