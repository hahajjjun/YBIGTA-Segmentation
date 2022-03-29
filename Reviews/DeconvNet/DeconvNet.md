# DeconvNet

## Problem

The paper proposes few limitations of FCN.

<p align="center">
<img src="https://pseudo-lab.github.io/SegCrew-Book/_images/deconv1.png">
</p>
1. The network is sensitive to scaling
  - Skip connection does not fully address the problem
2. Detailed structures of object are ignored due to coarse* label map*
  - FCN employs 16x16 label map, which results in unclear boundary information

*coarse*: fancy way of saying low-resolution
*label map*: input to the deconvolutional layer.

## Solution

1. Multi Layer Deconvolution Network


https://pseudo-lab.github.io/SegCrew-Book/_images/deconv2.png


2. Instance wise segmentation

3. Ensembling with FCN
