# TransFuse: Fusing Transformers and CNNs for Medical Image Segmentation(2021 Feb)

### Overview

- Segmentation Task의 본질 : Global context를 효과적으로 모델링하면서도, low-level detail을 살려 fine-grained map을 만드는 것
- **CNN: long-range relation modeling의 어려움, deeper layer로 갈수록 localized detail loss due to aggressive downsampling operations**
- **Transformer: Attention 덕분에 long-range dependency model은 문제되지 않으며, skip connection 덕분에 deeper layer에서도 input information의 topology를 일부 보존**
- Thus CNN & Transformer are complementary → Fuse them in parallel fashion!
- Multi-level feature를 효과적으로 사용하기 위해 Novel fusion technique BiFusion을 제안
- 2D, 3D medical image segmentation(poly, skin lesion, hip, prostate)에서 SOTA 달성

### Inefficient global context capturing at CNN

- CNN이 global context를 학습하는 방법 :
    - Bigger and bigger receptive field
    - Deep layers and aggressive downsampling required
    - Deep layer에서 발생하는 문제는 *ResNet Paper*가 잘 설명한 바 있음
        - Low-level detailed features are washed out by consecutive multiplications. “feature reuse problem”
        - Reduced spatial resolution hinders dense prediction.
        - Heavy deep nets training are unstable & easily overfitting
        - Local Self-attention added for CNN can model global context, but due to its computational complexity → only applied to low-resolution maps
- Transformer
    - SETR: conventional encoder-decoder CNN(U-Net 계열)의 decoder를 transformer 대체하여 Natural Image Segmentation에서 SOTA 달성
    - 그러나 Fine-grained detail capturing에 어려움을 겪음
- CNN - Transformer를 같이 사용하는 모델이 이전에 있었나? TransUNet!
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/168423886-556f4581-5512-4fd4-b8de-2c60f9d3cf6a.png)
    
    - which first utilizes CNNs to extract low-level
    features and then passed through transformers to model global interaction
    - 대부분은 CNN은 Transformer 층으로 바꾸거나 Sequential하게 stack하는 정도를 제안함

### Architecture

- TransFuse는 ***Parallel Transformer/CNN*** pathway + ***BiFusion*** module for feature fusion을 제안함.
    
    ![Untitled](https://user-images.githubusercontent.com/75057952/168423887-49ce3478-fc92-4475-ba3a-47aa81eedd41.png)
    

### 01. Transformer Pathway

- Encoder
    - Input image → Divided intp S=16 patches → flatted & passed into a linear embedding with output dimension D0(learnable, positional encoding 대신 embedding))
    - resulting embeddings → input to Transformer encoder(MSA & MLP)
- Decoder
    - progressive upsampling (PUP) method, as in *SETR*(Transformer decoder를 사용하지 않음)
    - output from encoder → reshaped as a 2D feature map with D0 channels
    - upsampling-convolution layers to recover the spatial resolution

### 02. CNN Pathway(그냥 일반적인 CNN과 동일)

### 03. Fusion Network(BiFusion)

- Channel, Spatial Attention + Dim-reduced residual connection

![Untitled](https://user-images.githubusercontent.com/75057952/168423888-997f5a8c-3d8b-41bc-acbc-9e4ba19507a9.png)

![Untitled](https://user-images.githubusercontent.com/75057952/168423889-38cb4d72-b0d5-4e79-8db9-1bcd1c095052.png)

### 04. Attention Gated network with BiFusion output maps

![Untitled](https://user-images.githubusercontent.com/75057952/168423890-6b644cf4-ba4d-4f82-b56c-9ce324890e43.png)

![Untitled](https://user-images.githubusercontent.com/75057952/168423882-c4f7f349-9658-4bd4-9e5d-0352d241794e.png)

### Loss function

- $L = αL(G, head(f^2))+βL(G, head(f^0))+γL(G, head(t^2))$

### Results

![Untitled](https://user-images.githubusercontent.com/75057952/168423883-b7f79564-cee8-4908-b539-df33a4b2000f.png)

![Untitled](https://user-images.githubusercontent.com/75057952/168423885-61e50997-c66c-4708-b686-a91809a9aa2c.png)
