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

## 관련지식 [Updated!]

### - [SENet](https://jayhey.github.io/deep%20learning/2018/07/18/SENet/)

<p align=center>
<img src="https://i.imgur.com/ufAxbPN.png", width=900dpi></img>
</p>
<p align=center>Fig 2. SE block in application</p>

Use global average pooling to **squeeze** the feature map into channel descriptor. Then, calculate channel-wize dependencies to **excite** back.

In EncNet, the encoding layer is responsible for squeezing, which squeezes [512, 42, 63] feature label into [1, 512] fc connected layer. The encoding architecture is explained under **structure**


### - Encoding Module (Deep Texture Encoding Network)

<p align=center>
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/9fde1e414e4d0fc4f6f08719504e19df07f19b0a/Reviews/EncNet/Assets/Encoding%20Layer.png", width=600dpi></img>
</p>
<p align=center>Fig 3. The Encoding Layer learns an inherent Dictionary. The Residuals are calculated by pairwise difference between visual descriptors of the input and the codewords of the dictionary.</p>

What does this layer actually do?

This layer gets input, like [1, 2646, 512] above (remember, it's a feature map after downsampling few times with ResNet), and tries to find which visual descriptor corresponds to which codeword.

In fact, given a set of N visual descriptors X = {X_1, ..., X_N} (N = 2646) and a learned codebook with K codewords C = {c_1, ... c_K} (K = 32), we assign a descriptor X_i to each codeword c_k with the corresponding weight a_ik.

In Korean, 각각의 descriptor이 어느 codeword 에 해당하는지 assign 해주는 것이다.

* Descriptor, codeword 는 무엇인가?

[SIFT](https://bskyvision.com/21) 참조. 기본적으로 CNN 은 scale, rotation, affine transformation 등에 민감하다. 고전적인 (그래봐야 20년정도 전에..) 방법론에는 크기 등의 요소에 영향을 받지 않는 이미지의 고유 (invariant) 특징을 추출해내려는 노력이 담겨있는데, 이중 대표적인 것이 SIFT이다. Descriptor은 image features 와 비슷하다고 생각할 수 있다.

눈치챘겠지만, codeword 는 미리 만들어놓은 "이미지 특징 사전"이며, 자주 등장하는 feature들을 descriptor의 형태로 저장해 놓은 것이다. 따라서, **우리의 목적은 descriptor들이 어떤 codeword에 해당하는지 매핑하는 것이다.**

이를 이용한 [Spatial Transform Network](https://towardsdatascience.com/spatial-transformer-networks-b743c0d112be)도 있는데, 여기선 생략.

* 비슷한 접근은?

    + Dictionary Learning

    K means 는 dictionary 를 배워서 hard-assign 한다. 즉, { X_1: c_3, X_2: c_7, ..., X_2646: c_27 } 같은 결과가 나온다. 즉, smoothing factor s_k -> inf이며, encoder의 식은
    
    <p align=center>
    <img src="https://render.githubusercontent.com/render/math?math=e_{ik} = \frac{1}{32}*(x_i - d_k)", style="width:15%;"></img>
    </p>
    
    으로 바뀌며, Codeword 는 각 군집의 중심으로 표현될 것이다. 문제는, codeword 가 많아질 수록 겹치는 feature도 생기게 될텐데, 이 때 hard-assigning을 하면 이러한 겹침 현상을 해결해줄 수 없다.
    
    이러한 단점을 보완하고자 GMM (Gaussian Mixture Model)은 확률적으로 feature destribution을 계산한다. Encoding layer은 GMM의 간소화된 버전이라고 생각할 수 있는데, 이 때 32개의 클러스터는 각각 다른 scaling (s_k) 를 가지게 된다.

    + [Bag of Words](https://en.wikipedia.org/wiki/Bag-of-words_model_in_computer_vision)

BoW 도 K-means 처럼 각각의 descriptor 을 가장 가까운 codeword로 hard-assign 하고, 그 codeword가 나타난 빈도를 합쳐서 그 image가 무엇인지 알아낸다.

<p align=center>
<img src="https://github.com/hahajjjun/YBIGTA-Segmentation/blob/cdbb0c74bae35b202e2dc5285f3582cd66fd2ec9/Reviews/EncNet/Assets/Bag%20of%20Words.png", width=400dpi></img>
</p>
<p align=center>Fig 4. Bag of Words, Image Patches</p>

이런식으로, 이미지에서 각 descriptor에 해당하는 image patch를 확인하고,

<p align=center>
<img src="https://github.com/ovysotska/in_simple_english/raw/59f3d0816418a786bfcce74e3227c71223a4e06f//data/bag_of_words/histogram_comparisons.png", width=800dpi></img>
</p>
<p align=center>Fig 5. Bag of Words, Histogram Comparisons</p>

다른 이미지와 비교해본다. 이미지 1번과 이미지 4번이 같은 이미지로 분류된다. 그렇다면 residual은 어떻게 될까?

<p align=center>
<img src="https://github.com/ovysotska/in_simple_english/raw/59f3d0816418a786bfcce74e3227c71223a4e06f//data/bag_of_words/cost_matrix_cosine.png", width=800dpi></img>
</p>
<p align=center>Fig 6. Bag of Words, Cosine Similarity</p>

이런식으로 각자 cosine similarity 를 계산한다. 일치하면 잔차가 0인것을 확인 가능. EncNet에서는 fc 로 대체한다.

## Structure

<p align=center>
<img src="https://miro.medium.com/max/1400/1*eu3ntqcGxsBPIZ3E3Y47Lw.png", width=600dpi></img>
</p>
<p align=center>Fig 2. EncNet</p>

- A pre-trained ResNet is used to extract dense convolutional feature maps
- Context Encoding Module is built on top to capture encoded semantics and capture **scaling factors**

### Encoding Layer

#### 1. BxDxHxW => Bx(HW)xD
    
Flatten the 2D image to 1D. 

```python
X = en.view(1, 512, -1).transpose(1, 2).contiguous()
```

The resulting shape is [1, 2646, 512].
    
#### 2. encode the feature map

First, calculate the residual encoder of each "pixel" with

<p align=center>
<img src="https://render.githubusercontent.com/render/math?math=e_{ik} = \frac{exp(-s_k\|r_{ik}\|^2)}{\sum_{j=1}^K exp(-s_j\|r_{ij}\|^2)} r_{ik}", style="width:30%;"></img>
</p>

where the residual is calculated by 

<p align=center>
<img src="https://render.githubusercontent.com/render/math?math=r_{ik} = x_i - d_k", style="width:15%;"></img>
</p>

```python 
A = F.softmax(scaled_l22(X, model.head.encmodule.encoding[3].codewords, model.head.encmodule.encoding[3].scale), dim=2)
```

The resulting shape is [1, 2646, 32]. Then, aggregate the residuals

<p align=center>
<img src="https://render.githubusercontent.com/render/math?math=e_k=\sum_{i=1}^Ne_{ik}", style="width:15%;"></img>
</p>

```python
E = aggregate(A, X, model.head.encmodule.encoding[3].codewords)
```

The resulting shape of encoder, E, is [1, 32, 512]. This means that E = [E_1, E_2, ..., E_K]

#### 3. average the codewords

Normalize, ReLU, then mean().

The resulting shape is [1, 512]. At this stage, each channel contains information about codeword features. This is the long white stick that appears right after "encode" triangle in figure 2.

#### 4. fc layer and channel-wise multiplication

```python
gamma = model.head.encmodule.fc(en)
y = gamma.view(b, c, 1, 1)

outputs = [F.relu_(feat + feat * y)]
```

Finally, apply fully connected layer, and do channel-wise multiplication to the original feature.

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
