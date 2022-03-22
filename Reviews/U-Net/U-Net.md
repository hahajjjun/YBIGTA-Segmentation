# U-Net

## Problem

Previously, Ciresan et al. used sliding-window setup to train segmentation model by providing a local patch around that pixel as input.

Advantages
- Localizable network
- More training data (in patches) compared to the number of training images

Disadvantages
- Training is slow because the network should iterate over each patch
  * It took 170 (N1, w=65), ~ 340 (N4, w=95) minutes per epoch, where training = 30 epochs.
- Because patches have pixels in common, there is redundancy. Alternatively, applying transformation is a better option.
- Increase in the localization accuracy came at the cost of context encoding.

The authors discard the idea of using patches; instead, they decide to improve upon FCN.

## Comparison with FCN

### FCN

<img src=https://www.researchgate.net/publication/327521314/figure/fig1/AS:668413361930241@1536373572028/Fully-convolutional-neural-network-architecture-FCN-8.ppm>

It upsamples only once. i.e. it has only one layer in the decoder
The original implementation github repo uses bilinear interpolation for upsampling the convoloved image. That is there is no learnable filter here
variants of FCN-[FCN 16s and FCN 8s] add the skip connections from lower layers to make the output robust to scale changes

### U-Net

<img src=https://miro.medium.com/max/1200/1*qNdglJ1ORP3Gq77MmBLhHQ.png>
multiple upsampling layers
uses skip connections and concatenates instead of adding up
uses learnable weight filters instead of fixed interpolation technique
