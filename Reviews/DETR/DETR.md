# DETR(End-to-End Object Detection with Transformers, ECCV 2020)

### 01. Panoptic Segmentation?

![Untitled](https://user-images.githubusercontent.com/75057952/168423807-0a6d7db7-9a49-461c-8a03-d6c894d38b3c.png)

- Object detection task = Bbox regression + label 예측
- NMS(Non-maximum suppression)을 이용한 anchor 기반의 Bounding box selection method는 heuristic 방법론들을 주로 사용함
    - Is it bad? No! 실제로 DETR은 Mask R-CNN과 성능이 비슷하거나 낮음
    - End-to-end가 가지는 이점이 있기 때문에 prior knowledge 혹은 heuristic algorithm에 의존하지 않는 방법론을 제시한 것이 DETR
    - *“Direct set prediction approach to bypass the surrogate tasks”*

### 02. Overview

![Untitled](https://user-images.githubusercontent.com/75057952/168423793-37c3168e-0a75-420a-b468-7aaf2ab79532.png)

- Feature Extraction with CNN
- Bbox predictions with transformer encoder-decoder
- 예측된 Bbox와 Ground Truth 간의 Bipartite graph matching을 통해 최종적으로 classify(Ground Truth의 어느 Bounding box에 해당하는지)하는 loss를 제안**(Bipartite matching loss)**
- 장점?
    - Custom block을 전혀 사용하지 않음
    - Scalability(Swin transformer, ViT 등으로 확장)
    - 마지막에 Mask head만 달아주면 panoptic segmentation 수행 가능

### 03. Architecture

![Untitled](https://user-images.githubusercontent.com/75057952/168423798-57e63934-3214-4201-b313-f8b2a6173d0f.png)

- 각 decoder layer에서 **병렬적으로** *N* object를 디코딩
- Autoregressive하지 않음

### 04. Loss function

- **Object detection set prediction loss**
    - loss는 multiple-object 관점에서 **ground-truth와 prediction 간에 이상적인 'bipartite matching'**
     을 생성하고, 그 후 **object 단위에서의 loss**를 최적화
    - Bounding box끼리도 matching하기 위해서 모든 permutation을 탐색
    - 이러한 permutation simga를 찾고자 함
        
        ![Untitled](https://user-images.githubusercontent.com/75057952/168423799-f3df0a90-0a05-4628-a352-0dd3196302b9.png)
        
- **Hungarian algorithm**
    - 그리고 각각의 L_match : class prediction + bounding box similarity를 모두 반영한 loss
        
        ![Untitled](https://user-images.githubusercontent.com/75057952/168423801-e2f1ab9a-0250-43ea-97e4-3f0790856a77.png)
        
        - Heuristic assignment loss와도 유사함, 대신 non-duplicates
    - 모든 Pair에 대해서 Hungarian loss를 계산
        
        ![Untitled](https://user-images.githubusercontent.com/75057952/168423802-f8f4fe94-1608-4b2c-880a-72f9e0a6d880.png)
        
        - 여기서 bounding box loss는 아래와 같이 L1 norm + IOU로 계산됨
            
            ![Untitled](https://user-images.githubusercontent.com/75057952/168423803-13346231-db42-4b2e-baaf-665cb8c0e553.png)