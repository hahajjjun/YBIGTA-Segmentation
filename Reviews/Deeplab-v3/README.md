# Rethinking Atrous Convolution for Semantic Image Segmentation

How does semantic segmentation differ from image classification / detection?

-> Whereas image classification / detection tasks prioritize global context, segmentation heavily relies on details and spatial information due to its pixel-scale classification

## Problems

1. Reduced feature resolution makes dense prediction tasks difficult

  - Detailed spatial information is desired

2. Objects are scaled differently (small to big)

## Previous approches

Four types of FCN that takes advantage of contextual information

1. Image Pyramid
    - Apply the same model to multi-scale inputs

2. Encoder Decoder
    - Encoder: Reduce spatial dimension -> capture longer range information
    - Decoder: Employ deconvolution, reuse the pooling indices, etc. -> Recover object details and spatial information

3. Context Module
    - Lays out extra modules in cascade* to encode long range context

*cascade*: use the same architecture with increasing IoU thresholds. (https://www.youtube.com/watch?v=1_-HfZcERJk)
<p align="center">
<img src = "http://www.svcl.ucsd.edu/projects/cascade-rcnn/img/faster2cascade.png" width = "500dp"></img>
</p>
<div align="center">Fig 1. Cascaded CNN architecture</div>

4. Spatial Pyramid Pooling
    - Uses spatial pyramid pooling
    - **DeepLabv2 uses atrous spatial pyramid pooling**

*Spatial pyramid pooling*: Divide the feature maps into layers and apply max pooling seperately, finally connecting them into a single vector
<p align="center">
<img src = "https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-21_at_3.05.44_PM.png" width = "500dp"></img>
</p>
<div align="center">Fig 2. Spatial Pyramid Pooling</div>



## DeepLab v3

Use atrous convolution to enlarge receptive field

Bilinear interpolation to recover spatial information

Keep the ground truths intact and upsample the final logits

<p align="center">
<img src = "https://gaussian37.github.io/assets/img/vision/segmentation/aspp/5.png" width = "500dp"></img>
</p>
<div align="center">Fig 3. Standard convolution (above) vs. Atrous convolution</div>

Whereas in standard convolution, which results in greater output stride, atrous convolution can maintain output stride without increasing conputations.

*output stride*: The ratio of input resolution to output resolution.

<p align="center">
<img src = "https://gaussian37.github.io/assets/img/vision/segmentation/aspp/6.png" width = "500dp"></img>
</p>
<div align="center">Fig 4. Atrous spatial pyramid pooling (ASPP)</div>

  - 1x1 convolution for conserving context
  - Parallel pooling with different rates
  - Batch normalization for each convolution layer
