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
<p align = "center">
<img src = "https://www.google.com/url?sa=i&url=http%3A%2F%2Fwww.svcl.ucsd.edu%2Fprojects%2Fcascade-rcnn%2F&psig=AOvVaw1loxOlenpqe3MEQ49FN-eM&ust=1646817888113000&source=images&cd=vfe&ved=0CAsQjRxqFwoTCLCysfCYtvYCFQAAAAAdAAAAABAD" width = "500dp"></img>
</p>
<div align="center">Fig 1. Cascaded CNN architecture</div>

Spatial Pyramid Pooling
  - Uses spatial pyramid pooling
  - 

## Solutions
