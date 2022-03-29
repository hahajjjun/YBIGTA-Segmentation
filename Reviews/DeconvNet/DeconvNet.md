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

2. Instance wise segmentation, ensembling with FCN

Generate a number of **candidate proposals**, then **aggregate** the results

```matlab
% padding for easy cropping    
[img_height, img_width, ~] = size(I);
pad_offset_col = img_height;
pad_offset_row = img_width;

% pad every images(I, cls_seg, inst_seg...) to make cropping easy
padded_I = padarray(I,[pad_offset_row, pad_offset_col]);
padded_result_base = padarray(result_base,[pad_offset_row, pad_offset_col]);
padded_prob_base = padarray(prob_base,[pad_offset_row, pad_offset_col]);
padded_cnt_base = padarray(cnt_base, [pad_offset_row, pad_offset_col]);
norm_padded_prob_base = padarray(norm_prob_base,[pad_offset_row, pad_offset_col]);
norm_padded_prob_base(:,:,1) = eps;

padded_frame_255 = 255-padarray(uint8(ones(size(I,1),size(I,2))*255),[pad_offset_row, pad_offset_col]);

padded_result_base = padded_result_base + padded_frame_255;

%% load extended bounding box
cache = load(sprintf(edgebox_cache_path, ids{i})); % boxes_padded
boxes_padded = cache.boxes_padded;

numBoxes = size(boxes_padded,1);    
cnt_process = 1;
for bidx = 1:numBoxes  
    box = boxes_padded(bidx,:);
    box_wd = box(3)-box(1)+1;
    box_ht = box(4)-box(2)+1;

    if min(box_wd, box_ht) < 112, continue; end   

    input_data = preprocess_image_bb(padded_I, boxes_padded(bidx,:), config.im_sz); 
    cnn_output = caffe('forward', input_data);

    segImg = permute(cnn_output{1}, [2, 1, 3]);
    segImg = imresize(segImg, [box_ht, box_wd], 'bilinear');

    % accumulate prediction result
    cropped_prob_base = padded_prob_base(box(2):box(4),box(1):box(3),:);
    padded_prob_base(box(2):box(4),box(1):box(3),:) = max(cropped_prob_base,segImg);

    if mod(cnt_process, 10) == 0, fprintf(',%d', cnt_process); end
    if cnt_process >= config.max_proposal_num
        break;
    end

    cnt_process = cnt_process + 1;
end

%% save DeconvNet prediction score
deconv_score = padded_prob_base(pad_offset_row:pad_offset_row+size(I,1)-1,pad_offset_col:pad_offset_col+size(I,2)-1,:);

%% load fcn-8s score
fcn_cache = load(sprintf(fcn_score_path, ids{i}));
fcn_score = fcn_cache.score;

%% ensemble
zero_mask = zeros(size(fcn_score));
fcn_score = max(zero_mask,fcn_score);

ens_score = deconv_score .* fcn_score;
[ens_segscore, ens_segmask] = max(ens_score, [], 3);
ens_segmask = uint8(ens_segmask-1);
```

