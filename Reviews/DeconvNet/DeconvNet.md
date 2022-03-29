# DeconvNet

## Problem

The paper proposes few limitations of FCN.

<p align="center">
<img src="https://pseudo-lab.github.io/SegCrew-Book/_images/deconv1.png", width="600dpi">
</p>
1. The network is sensitive to scaling
  - Skip connection does not fully address the problem
2. Detailed structures of object are ignored due to coarse* label map*
  - FCN employs 16x16 label map, which results in unclear boundary information

*coarse*: fancy way of saying low-resolution
*label map*: input to the deconvolutional layer.

## Solution

1. Multi Layer Deconvolution Network

<p align="center">
<img src="https://pseudo-lab.github.io/SegCrew-Book/_images/deconv2.png", width="600dpi">
</p>

Unpooling: Max-Pooling에서 골랐던 값의 index를 기억한 뒤, 그 index 의 값은 채우고 나머지는 0

Image size가 match해야하기 때문에, conv layer에서 padding size를 1로 설정함.

2. Instance wise segmentation

3. Ensembling with FCN
