### Brief Landscape of Segmentation Tasks
#### 0. Anatomy of Segmentation Task
- Segmentation Task consists of ...
  - Semantic label of pixels
  - Instance ID for distinguishment of different instances
- Semantic Segmentation & Panoptic Segmentation ignores instance ID, only considering pixel labels and does not specify instaces
- Instance Segmentation & Panoptic Segmentation both segment each object instance in an image. 
  - However, the difference lies in the handling of overlapping segments.
  - Instance segmentation permits overlapping segments while the panoptic segmentation task allows assigning a unique semantic label and a unique instance-id each pixel of the image. <br/>
<p align = "center"><img src = "https://media.vlpt.us/images/babydeveloper/post/9eb91603-7cb8-4412-a4df-01679fa2a65e/image.png" width = "500dp"></img></p>
<p align = "center"><img src = "https://media.vlpt.us/images/babydeveloper/post/e90c2c4e-044b-4bf6-9120-6141ebc9d012/image.png" width = "500dp"></img></p>

---

#### 1. Semantic Segmentation
- Reference : [Arxiv Survey](https://arxiv.org/pdf/2001.05566.pdf)
- Definition : 의미적, 인지적 단위로 구분하여 분할하다.
- Semantic Segmentation is like pixel-wise classification, dense prediction for every pixel.
- Convolution Networks for Feature Extraction & Upsampling methods for Dense prediction and preserve resolution.
- Thus, performance of semantic segmentation relies on...
  - Fancy architecture for high-resolution feature extraction & large receptive field.
    - Remark : Is large receptive field always good?
  - Upsampling techniques for resolution preservation.
 
---

#### 2. Instance Segmentation

---

#### 3. Panonptic Segmentation
