# Context Encoding for Semantic Segmentation

Can we leverage the context encoding of classic approaches with the power of deep learning?

## Previous approaches

<p align=center>
<img src="https://i2.wp.com/zhangbin0917.github.io/2018/06/11/Context-Encoding-for-Semantic-Segmentation/08.png", width=600dpi></img>
Fig 1. Misclassification of FCN due to lack of global context
</p>

- Deeplabv3: Atrous Spatial Pyramid Pooling

- PSPNet: Spatial Pyramid Pooling during upsampling

- Classical context encoding: Extraction of hand-engineered features -> visual vocabulary (dictionary) -> description of **feature statistics** with classic encoders

## Suggestion

1. *Context Encoding Module incorporating Semantic Encoding Loss (SE-loss)*

2. *Context Encoding Network (EncNet)*

## Structure

1. 
## Problem

FCN finds difficulty in capturing the context of the image

What if you see foreign objects in the scene?

did not bother much
