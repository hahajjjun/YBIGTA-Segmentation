# Rethinking Atrous Convolution for Semantic Image Segmentation

## Problems

1. Reduced feature resolution makes dense prediction tasks difficult

  - Detailed spatial information is desired

2. Objects are scaled differently (small to big)

## Previous approches

Four types of FCN that takes advantage of contextual information

Image Pyramid
  - Apply the same model to multi-scale inputs

Encoder Decoder
  - Encoder: Reduce spatial dimension -> capture longer range information
  - Decoder: Employ deconvolution, reuse the pooling indices, etc. -> Recover object details and spatial information

Context Module
  - Lays out extra modules in cascade* to encode long range context

*cascade*: use the same architecture with increasing IoU thresholds. (https://www.youtube.com/watch?v=1_-HfZcERJk)
<p align="center">
<img src = "http://www.svcl.ucsd.edu/projects/cascade-rcnn/img/faster2cascade.png" width = "500dp"></img>
</p>
<div align="center">Fig 1. Cascaded CNN architecture</div>

Spatial Pyramid Pooling
  - Uses spatial pyramid pooling
  - **DeepLabv2 uses atrous spatial pyramid pooling**

*Spatial pyramid pooling*: Divide the feature maps into layers and apply max pooling seperately, finally connecting them into a single vector
<p align="center">
<img src = "https://production-media.paperswithcode.com/methods/Screen_Shot_2020-06-21_at_3.05.44_PM.png" width = "500dp"></img>
</p>
<div align="center">Fig 2. Spatial Pyramid Pooling</div>

## Solutions
