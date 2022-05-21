CoTr: 4 March 2021 (first on 3D)

UNETR: 9 Oct 2021

## Background

명목상으론 기존 FCNN 기반 모델의 lack of long-range dependency 를 지적하지만.. 그냥 U-net에 transformer을 빨리 갖다붙이고 싶었던 것 같음.

## Architecture

<div align="center">
<img src="https://images.velog.io/images/khp3927/post/1484adf2-ffe1-443b-bcf6-694e186bde24/Untitled%20(12).png" width="700px">
  <p>어? 어디서 본것같다.</p>
</div>

<div align="center">
<img src="../Transformer/Assets/vit.gif" width="500px"></img>
  <p>ViT 를 그대로 베껴왔다!</p>
</div>

다른 점은 Nx(P^3 \dot C) 개의 patches 로 나누는 것. 그 이후엔 각각의 patch에 linear projection을 하여 embedding layer로 보냄.

ViT와 마찬가지로, spatial information을 위해 learnable positional embedding 을 추가.

Class token은 따로 사용하지 않으며 (segmentation), 12개의 stacked transformer을 지나게 된다.

<div align="center">
<img src="https://images.velog.io/images/khp3927/post/34393459-c3e0-4d42-9ed8-847bdbe1e6fb/Untitled%20(13).png" width="700px">
  <p>UNETR Architecture</p>
</div>

Multi-head Self Attention, Multi Layer Perceptron은 아래와 같이 정의된다.

<img src="./assets/encoder.png" width="500px">

이 때, 3의 배수마다 skip connection 을 사용해 decoder의 input으로 활용된다.

## Loss function

<img src="./assets/loss.png" width="500px">

## Model comparison

<div align="center">
<img src="./assets/comparison.png" width="800px">
</div>

## Implementation Details
- Batch size: 6
- Optimizer: AdamW
- Learning rate: 0.0001
- Iteration: 20000
- Backbone: ViT-B16
- L=12, K=768, P=16×16×16 (32 일때와 비교하여 약간의 성능향상이 있었다)
- Augmentation: Random rotation(90°, 180°, 270°), Random flip(axial, sagittal, coronal views), Random scale, Shift intensity
- Ensemble: Five-fold cross-validation
