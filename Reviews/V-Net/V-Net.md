# V-Net

## Problem

Previous CNN approaches are limited to slice-wise processing

## Solution

Apply volumetric convolutions, use custom dice loss.

## Architecture

<p align="center">
<img src="https://miro.medium.com/max/1400/1*rcT-PbkROWrSg0PRqO-KAA.png">
</p>

Some notable differences

1. Residual Function

Input of each layer is used to the output of the layer (element-wise sum). This is called residual function and aids convergence

2. Conv3d

Unlike U-net, V-net uses 5x5x5 kernels

3. No Batch normalization?

Most unofficial implementations employ batch normalization for performance

4. PReLU?

V-Net does not use ReLU, which may cause [Dying ReLU](https://brunch.co.kr/@kdh7575070/27) problem.

<p align="center">
<img src="https://gaussian37.github.io/assets/img/dl/concept/prelu/prelu.png">
</p>

5. Dice loss layer

                                                                             

```python
sdf
```
                                                                                       
