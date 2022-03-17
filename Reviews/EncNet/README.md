# Context Encoding for Semantic Segmentation

Given the image of a room, it is unlikely that we will find the presence of vehicles.

Can we leverage the context encoding of classic approaches with the power of deep learning?

Bag of words: https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision

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

## 관련지식 [Updated!]

- [SENet](https://jayhey.github.io/deep%20learning/2018/07/18/SENet/)

<p align=center>
<img src="https://i.imgur.com/ufAxbPN.png", width=900dpi></img>
</p>
<p align=center>Fig 2. SE block in application</p>

Use global average pooling to **squeeze** the feature map into channel descriptor. Then, calculate channel-wize dependencies to **excite** back.

In EncNet, the encoding layer is responsible for squeezing, which squeezes [512, 42, 63] feature label into [1, 512] fc connected layer. The encoding follows these steps:

    1. BxDxHxW => Bx(HW)xD
    
    `X = en.view(1, 512, -1).transpose(1, 2).contiguous()`
    
    2. 
    
<img src="https://render.githubusercontent.com/render/math?math=e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}>

- SIFT 

## Structure

<p align=center>
<img src="https://miro.medium.com/max/1400/1*eu3ntqcGxsBPIZ3E3Y47Lw.png", width=600dpi></img>
</p>
<p align=center>Fig 2. EncNet</p>

- A pre-trained ResNet is used to extract dense convolutional feature maps
- Context Encoding Module is built on top to capture encoded semantics and capture **scaling factors**

```python

model = get_encnet('pcontext', 'resnet50s', pretrained=True, root=root, aux=True,
                      base_size=520, crop_size=480, **kwargs)
                      
model = EncNet(datasets[dataset.lower()].NUM_CLASS, backbone=backbone, root=root, **kwargs)

'''
nclass: 데이터에 나오는 클래스는 총 몇개냐?
backbone: 무슨 backbone을 쓸거냐?
aux: aux layer 쓸거냐?
se_loss: encoding 쓸거냐?
lateral: ?
'''
class EncNet(BaseNet):
    def __init__(self, nclass, backbone, aux=True, se_loss=True, lateral=False,
                 norm_layer=SyncBatchNorm, **kwargs):
        super(EncNet, self).__init__(nclass, backbone, aux, se_loss,
                                     norm_layer=norm_layer, **kwargs)
        self.head = EncHead(2048, self.nclass, se_loss=se_loss,
                            lateral=lateral, norm_layer=norm_layer,
                            up_kwargs=self._up_kwargs)
        if aux:
            self.auxlayer = FCNHead(1024, nclass, norm_layer=norm_layer) # training에 도움을 주기 위한 layer.
            
    def forward(self, x):
        imsize = x.size()[2:]
        features = self.base_forward(x) # 먼저 Resnet50에 한바퀴 돌림
        
        -------------------------------------
        def base_forward(self, x):
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4 # fully connected 는 사용하지 않는 모습
        
        class ResNet(nn.Module):
        
            """
            Reference:
            - He, Kaiming, et al. "Deep residual learning for image recognition." Proceedings of the IEEE conference on computer vision and pattern recognition. 2016.
            - Yu, Fisher, and Vladlen Koltun. "Multi-scale context aggregation by dilated convolutions."
            """
            
            def __init__(self, block, ...,):
                self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
                self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
                if dilated or dilation == 4: # dialated = True
                    self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                                   dilation=2, norm_layer=norm_layer,
                                                   dropblock_prob=dropblock_prob)
                    self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                                   dilation=4, norm_layer=norm_layer,
                                                   dropblock_prob=dropblock_prob)

        -------------------------------------

        x = list(self.head(*features)) # 각각 64
        
        -------------------------------------
        class EncHead(nn.Module):
            def __init__(self, in_channels, out_channels, se_loss=True, lateral=True,
                         norm_layer=None, up_kwargs=None):
                super(EncHead, self).__init__()
                self.se_loss = se_loss
                self.lateral = lateral
                self.up_kwargs = up_kwargs
                self.conv5 = nn.Sequential(
                    nn.Conv2d(in_channels, 512, 3, padding=1, bias=False),
                    norm_layer(512),
                    nn.ReLU(inplace=True))
                if lateral:
                    self.connect = nn.ModuleList([
                        nn.Sequential(
                            nn.Conv2d(512, 512, kernel_size=1, bias=False),
                            norm_layer(512),
                            nn.ReLU(inplace=True)),
                        nn.Sequential(
                            nn.Conv2d(1024, 512, kernel_size=1, bias=False),
                            norm_layer(512),
                            nn.ReLU(inplace=True)),
                    ])
                    self.fusion = nn.Sequential(
                            nn.Conv2d(3*512, 512, kernel_size=3, padding=1, bias=False),
                            norm_layer(512),
                            nn.ReLU(inplace=True))
                self.encmodule = EncModule(512, out_channels, ncodes=32,
                    se_loss=se_loss, norm_layer=norm_layer)
                self.conv6 = nn.Sequential(nn.Dropout(0.1, False),
                                           nn.Conv2d(512, out_channels, 1))
            def forward(self, *inputs):
                feat = self.conv5(inputs[-1])
                if self.lateral:
                    c2 = self.connect[0](inputs[1])
                    c3 = self.connect[1](inputs[2])
                    feat = self.fusion(torch.cat([feat, c2, c3], 1))
                outs = list(self.encmodule(feat))
                outs[0] = self.conv6(outs[0])
                return tuple(outs)
        -------------------------------------
        
        x[0] = F.interpolate(x[0], imsize, **self._up_kwargs) # Down/up samples the input to either the given size or the given scale_factor
        if self.aux:
            auxout = self.auxlayer(features[2])
            auxout = F.interpolate(auxout, imsize, **self._up_kwargs)
            x.append(auxout)
        return tuple(x)
        
        
            
```
## Problem

FCN finds difficulty in capturing the context of the image

What if you see foreign objects in the scene?

did not bother much
