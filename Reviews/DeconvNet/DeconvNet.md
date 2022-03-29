# DeconvNet

## Problem

The paper proposes few limitations of FCN.

<p align="center">
<img src="https://pseudo-lab.github.io/SegCrew-Book/_images/deconv1.png", width="600dpi">
</p>
1. The network is sensitive to scaling
  - Skip connection does not fully address the problem
2. Detailed structures of object are ignored due to coarse* label map*
  - FCN employs 16x16 label map, which results in unclear boundary information

*coarse*: fancy way of saying low-resolution
*label map*: input to the deconvolutional layer.

## Solution

1. Multi Layer Deconvolution Network

<p align="center">
<img src="https://pseudo-lab.github.io/SegCrew-Book/_images/deconv2.png", width="900dpi">
</p>


<p align="center">
<img src="https://pseudo-lab.github.io/SegCrew-Book/_images/deconv3.png", width="500dpi">
</p>

Unpooling: Max-Pooling에서 골랐던 값의 index를 기억한 뒤, 그 index 의 값은 채우고 나머지는 0

Deconvolution: Bilinear interpolation 대신에 implement한 upsampling method

```cpp
template <typename Dtype>
void DeconvolutionLayer<Dtype>::Forward_cpu(const vector<Blob<Dtype>*>& bottom,
      const vector<Blob<Dtype>*>& top) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  for (int i = 0; i < bottom.size(); ++i) {
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* top_data = top[i]->mutable_cpu_data();
    for (int n = 0; n < this->num_; ++n) {
      this->backward_cpu_gemm(bottom_data + bottom[i]->offset(n), weight,
          top_data + top[i]->offset(n));
      if (this->bias_term_) {
        const Dtype* bias = this->blobs_[1]->cpu_data();
        this->forward_cpu_bias(top_data + top[i]->offset(n), bias);
      }
    }
  }
}

template <typename Dtype>
void DeconvolutionLayer<Dtype>::Backward_cpu(const vector<Blob<Dtype>*>& top,
      const vector<bool>& propagate_down, const vector<Blob<Dtype>*>& bottom) {
  const Dtype* weight = this->blobs_[0]->cpu_data();
  Dtype* weight_diff = this->blobs_[0]->mutable_cpu_diff();
  for (int i = 0; i < top.size(); ++i) {
    const Dtype* top_diff = top[i]->cpu_diff();
    const Dtype* bottom_data = bottom[i]->cpu_data();
    Dtype* bottom_diff = bottom[i]->mutable_cpu_diff();
    // Bias gradient, if necessary.
    if (this->bias_term_ && this->param_propagate_down_[1]) {
      Dtype* bias_diff = this->blobs_[1]->mutable_cpu_diff();
      for (int n = 0; n < this->num_; ++n) {
        this->backward_cpu_bias(bias_diff, top_diff + top[i]->offset(n));
      }
    }
    if (this->param_propagate_down_[0] || propagate_down[i]) {
      for (int n = 0; n < this->num_; ++n) {
        // Gradient w.r.t. weight. Note that we will accumulate diffs.
        if (this->param_propagate_down_[0]) {
          this->weight_cpu_gemm(top_diff + top[i]->offset(n),
              bottom_data + bottom[i]->offset(n), weight_diff);
        }
        // Gradient w.r.t. bottom data, if necessary, reusing the column buffer
        // we might have just computed above.
        if (propagate_down[i]) {
          this->forward_cpu_gemm(top_diff + top[i]->offset(n), weight,
              bottom_diff + bottom[i]->offset(n),
              this->param_propagate_down_[0]);
        }
      }
    }
  }
}
```

Image size가 match해야하기 때문에, conv layer에서 padding size를 1로 설정함.

2. Instance wise segmentation

3. Ensembling with FCN
