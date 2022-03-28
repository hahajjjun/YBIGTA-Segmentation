# 2D U-Net

## Problem

Previously, Ciresan et al. used sliding-window setup to train segmentation model by providing a local patch around that pixel as input.

Advantages
- Localizable network
- More training data (in patches) compared to the number of training images

Disadvantages
- Training is slow because the network should iterate over each patch
  * It took 170 (N1, w=65), ~ 340 (N4, w=95) minutes per epoch, where training = 30 epochs.
- Because patches have pixels in common, there is redundancy. Alternatively, applying transformation is a better option.
- Increase in the localization accuracy came at the cost of context encoding.
  * Larger patch -> Worse localization
  * Smaller patch -> Lose context

The authors discard the idea of using patches everywhere; instead, they decide to improve upon FCN.

## Comparison with FCN

### FCN

<img src="https://www.researchgate.net/publication/327521314/figure/fig1/AS:668413361930241@1536373572028/Fully-convolutional-neural-network-architecture-FCN-8.ppm" width=600dpi>

- The number of upsampling layers does not match the number of downsampling layers
- Uses bilinear interpolation for upsampling the convoloved image -> not learnable
- variants of FCN-[FCN 16s and FCN 8s] add the skip connections from lower layers to make the output robust to scale changes

### U-Net

<img src="https://miro.medium.com/max/1200/1*qNdglJ1ORP3Gq77MmBLhHQ.png" width=600dpi>

- Symmetric upsampling & downsampling layers
- Uses skip connections and concatenation instead of adding up
- Interpolation is learnable

## Techniques

### [Overlap title strategy](https://joungheekim.github.io/2020/09/28/paper-review/)

<img src="https://joungheekim.github.io/img/in-post/2020/2020-09-28/overlap_tile.png" width=800dpi>

U-Net does not use padding in its architecture, so the output image differs in size with the input image.

This strategy cuts the image in overlapping manner so that the output will fully cover the image.

### Mirroring Extrapolation

<img src="https://joungheekim.github.io/img/in-post/2020/2020-09-28/mirroring.png" width=800dpi>

Overlap title strategy does not provide solution in the border. Authors additionally implement mirroring extrapolation, exploting the symmetric nature of biomedical cells.

### Weight Loss

For the cell segmentation challenge, it is important to identify the boundaries. The authors gave higher weights to the pixels near the boundaries with the following loss.

<math xmlns="http://www.w3.org/1998/Math/MathML">
  <mi>w</mi>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>=</mo>
  <msub>
    <mi>w</mi>
    <mi>c</mi>
  </msub>
  <mo stretchy="false">(</mo>
  <mi>x</mi>
  <mo stretchy="false">)</mo>
  <mo>+</mo>
  <msub>
    <mi>w</mi>
    <mn>0</mn>
  </msub>
  <mo>&#x22C5;<!-- ⋅ --></mo>
  <mi>e</mi>
  <mi>x</mi>
  <mi>p</mi>
  <mo stretchy="false">(</mo>
  <mo>&#x2212;<!-- − --></mo>
  <mfrac>
    <mrow>
      <mo stretchy="false">(</mo>
      <msub>
        <mi>d</mi>
        <mn>1</mn>
      </msub>
      <mo stretchy="false">(</mo>
      <mi>x</mi>
      <mo stretchy="false">)</mo>
      <mo>+</mo>
      <msub>
        <mi>d</mi>
        <mn>2</mn>
      </msub>
      <mo stretchy="false">(</mo>
      <mi>x</mi>
      <mo stretchy="false">)</mo>
      <msup>
        <mo stretchy="false">)</mo>
        <mn>2</mn>
      </msup>
    </mrow>
    <mrow>
      <mn>2</mn>
      <msup>
        <mi>&#x03C3;<!-- σ --></mi>
        <mn>2</mn>
      </msup>
    </mrow>
  </mfrac>
  <mo stretchy="false">)</mo>
</math>

<img src="https://joungheekim.github.io/img/in-post/2020/2020-09-28/weight_map.png" width=800dpi>

## Augmentation

Rotation, Shift, Elastic distortion

## Training

Use small batch size, but large image size to fully utilize GPU memory.

# 3D U-Net

Replaces all 2D operations with 3D counterparts.

## Purpose

Labeling volumetric data is tedious -> Create a deep network that learns to generate dense volumetric segmentations with 2D training set.

## Architecture

<img src="https://miro.medium.com/max/1200/1*iC_mVWonWI8dILrWflKr3Q.png">

- Avoids bottlenecks by doubling the channel before max-pooling
- Batch normalization for faster convergence
- Weighted softmax loss function
  * For sparsely annotated dataset, set the weights of unlabled pixels to zero

### Transformation

Besides rotation, scaling, and gray value augmentation, apply a smooth dense deformation field.

<img src="https://www.researchgate.net/publication/329716031/figure/fig5/AS:731591672360980@1551436455993/Training-data-augmentation-through-random-smooth-elastic-deformation-a-Upper-left-Raw.jpg">

