## Significance

First model to fully utilize transformer in image dataset

Done by splitting images into patches (just as tokens)

Vs. ResNet

Lower accuracies in the mid-sized datasets
- Transformers lack inductive biases

Excellent accuracies in larger datasets
- Pre-training at sufficient scale and transfer-learning to specific tasks

## Background

Inductive Bias

"어떤 작업에 대한 가정이 있으면 더 수월해짐"

CNN - locality
RNN - sequential

ViT 는 상대적으로 intuctive bias가 부족.

## Architecture

<img src="/Assets/vit.gif" width="500px"></img>

Reshape HxWxC -> Nx(P^2xC), where H*W=P^2*N

