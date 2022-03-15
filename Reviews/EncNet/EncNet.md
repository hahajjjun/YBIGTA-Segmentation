# Context Encoding for Semantic Segmentation

Given the image of a room, it is unlikely that we will find the presence of vehicles.

Can we leverage the context encoding of classic approaches with the power of deep learning?

## Previous approaches

<p align=center>
<img src="https://i2.wp.com/zhangbin0917.github.io/2018/06/11/Context-Encoding-for-Semantic-Segmentation/08.png", width=600dpi></img>
</p>
<p align=center>Fig 1. Misclassification of FCN due to lack of global context. In row 1, sand is classified as Earth; row 2, scyscraper as building.</p>

- Deeplabv3: Atrous Spatial Pyramid Pooling

- PSPNet: Spatial Pyramid Pooling during upsampling

- Classical context encoding: Extraction of hand-engineered features -> visual vocabulary (dictionary) -> description of **feature statistics** with classic encoders

## Suggestion

1. *Context Encoding Module incorporating Semantic Encoding Loss (SE-loss)*

2. *Context Encoding Network (EncNet)*

## Structure

<p align=center>
<img src="https://miro.medium.com/max/1400/1*eu3ntqcGxsBPIZ3E3Y47Lw.png", width=600dpi></img>
</p>
<p align=center>Fig 2. EncNet</p>

- A pre-trained ResNet is used to extract dense convolutional feature maps
- Context Encoding Module is built on top to capture encoded semantics and capture **scaling factors**

`
class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, lateral=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead(2048, self.nclass, se_loss=se_loss,
                            lateral=lateral, norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer)
            
`
## Problem

FCN finds difficulty in capturing the context of the image

What if you see foreign objects in the scene?

did not bother much
