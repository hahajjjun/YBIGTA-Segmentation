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
        
        '''
        def base_forward(self, x):
            x = self.pretrained.conv1(x)
            x = self.pretrained.bn1(x)
            x = self.pretrained.relu(x)
            x = self.pretrained.maxpool(x)
            c1 = self.pretrained.layer1(x)
            c2 = self.pretrained.layer2(c1)
            c3 = self.pretrained.layer3(c2)
            c4 = self.pretrained.layer4(c3)
        return c1, c2, c3, c4
        
        self.layer1 = self._make_layer(block, 64, layers[0], norm_layer=norm_layer, is_first=False)
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2, norm_layer=norm_layer)
        if dilated or dilation == 4: # dialated = True
            self.layer3 = self._make_layer(block, 256, layers[2], stride=1,
                                           dilation=2, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
            self.layer4 = self._make_layer(block, 512, layers[3], stride=1,
                                           dilation=4, norm_layer=norm_layer,
                                           dropblock_prob=dropblock_prob)
                                           
        self.fc = nn.Linear(512 * block.expansion, num_classes), block.expansion = 4 # bottleneck
        '''

        x = list(self.head(*features))
        x[0] = F.interpolate(x[0], imsize, **self._up_kwargs)
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
