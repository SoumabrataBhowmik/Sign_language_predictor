**FusionASLModel: High-Performance Sign Language Recognition**

**Dual-Backbone CNN-ViT Feature Fusion for ASL Alphabet Classification**

This project introduces the FusionASLModel, a novel deep learning architecture that combines the local feature extraction of EfficientNetB5 with the global contextual understanding of a Vision Transformer (ViT-Base). By leveraging a dual-backbone approach, the model achieves near-perfect classification for American Sign Language (ASL) gestures.

**Key Features**

Hybrid Architecture: Fuses a pre-trained EfficientNetB5 [1] and ViT-Base [2] to capture both fine-grained textures and long-range spatial relationships.

Strategic Fine-Tuning: Targets only the last two blocks of each backbone to maximize transfer learning efficiency and prevent catastrophic forgetting.

Exceptional Accuracy: Achieved a 1.0000 Accuracy and 1.0000 Macro F1-Score on the ASL Alphabet test set.

Robust Data Pipeline: Utilizes aggressive transformations including RandomPerspective, ColorJitter, and RandomErasing for high generalization.

**Model Architecture**

The architecture consists of two concurrent streams whose features are concatenated and passed into a custom classification head.

Technical Specifications

CNN Backbone: EfficientNetB5 extracting 2,048-dimensional local features (f_CNN).

ViT Backbone: ViT-Base/Patch16_224 extracting 768-dimensional global features (f_ViT).

Fused Feature Vector: A 2,816-dimensional vector (f_fused) formed via concatenation.

Classification Head: Three linear layers (2816 → 1024 → 512 → 29) with ReLU activations and Dropout(0.6) for regularization.

<p align="center">
  <img src="Architecture_Diagram.jpg" width="600" title="FusionASLModel Architecture">
</p>

**Dataset**

The model was trained using the ASL Alphabet Dataset [3], which contains 87,000 balanced images.

Alphabet Classes: 26 letters (A-Z) with 3,000 images each.

Control Classes: SPACE, DELETE, and NOTHING (3,000 images each).

Input Size: 224 x 224 pixels (resized from 200 x 200).

**Performance & Results**

The model demonstrated rapid convergence, reaching perfect validation metrics by Epoch 4.

Training Summary
| Epoch | Train Loss | Val Loss | Train Acc | Val Acc | Train F1 | Val F1 |
| :--- | :---: | :---: | :---: | :---: | :---: | :---: |
| 1 | 1.9273 | 0.0777 | 0.4464 | 0.9878 | 0.4469 | 0.9878 |
| 2 | 0.2366 | 0.0018 | 0.9380 | 0.9999 | 0.9380 | 0.9999 |
| 3 | 0.1190 | 0.0005 | 0.9693 | 0.9999 | 0.9693 | 0.9999 |
| 4 | 0.0855 | 0.0003 | 0.9775 | 1.0000 | 0.9775 | 1.0000 |
| 5 | 0.0740 | 0.0002 | 0.9803 | 1.0000 | 0.9803 | 1.0000 |
| 6 | 0.0701 | 0.0001 | 0.9816 | 1.0000 | 0.9816 | 1.0000 |


**Final Test Metrics**

| Test Set Metric | Value |
| :--- | :--- |
| **Accuracy** | 1.0000 |
| **Macro F1 Score** | 1.0000 |
| **Test Loss** | 0.0002 |


**References**

[1] Tan, M., & Le, Q. V. (2019). EfficientNet: Rethinking Model Scaling for Convolutional Neural Networks. ICML. 

[2] Dosovitskiy, A., et al. (2021). An Image Is Worth 16x16 Words: Transformers for Image Recognition at Scale. ICLR. 

[3] ASL Alphabet Dataset. Kaggle.


## Model Architecture

The architecture utilizes a dual-backbone fusion approach, processing the input through a Convolutional Neural Network (CNN) and a Vision Transformer (ViT) in parallel before concatenating their feature maps for the final classification.

```mermaid
flowchart TD
    %% Node Styling
    classDef inputNode fill:#3498db,stroke:#2980b9,stroke-width:2px,color:#fff;
    classDef prepNode fill:#9b59b6,stroke:#8e44ad,stroke-width:2px,color:#fff;
    classDef backboneNode fill:#c0392b,stroke:#a93226,stroke-width:2px,color:#fff;
    classDef featureNode fill:#e74c3c,stroke:#c0392b,stroke-width:2px,color:#fff;
    classDef fusionNode fill:#e67e22,stroke:#d35400,stroke-width:2px,color:#fff;
    classDef fusedFeatNode fill:#a04000,stroke:#873600,stroke-width:2px,color:#fff;
    classDef denseNode fill:#ecf0f1,stroke:#bdc3c7,stroke-width:2px,color:#2c3e50;
    classDef outputNode fill:#2ecc71,stroke:#27ae60,stroke-width:2px,color:#fff;

    %% Workflow
    Input["Input Image<br>(RGB)"]:::inputNode --> Prep["Preprocessing / Transformation<br>(Resize 224x224, Normalization)"]:::prepNode
    
    Prep -->|Transformed Image| CNN_Stream
    Prep -->|Transformed Image| ViT_Stream

    %% CNN Stream Branch
    subgraph CNN_Stream ["CNN Stream"]
        direction TB
        EffNet["EfficientNetB5 Backbone<br>(Frozen except last 2 blocks)"]:::backboneNode
        CNN_Feat["CNN Features<br>(f_CNN, Dim: 2048)"]:::featureNode
        EffNet --> CNN_Feat
    end

    %% ViT Stream Branch
    subgraph ViT_Stream ["ViT Stream"]
        direction TB
        ViT["ViT-Base/Patch16 Backbone<br>(Frozen except last 2 blocks)"]:::backboneNode
        ViT_Feat["ViT Features<br>(f_ViT, Dim: 768)"]:::featureNode
        ViT --> ViT_Feat
    end

    %% Fusion and Classification
    CNN_Feat -->|Concatenation| Concat{"Concatenation"}
    ViT_Feat -->|Concatenation| Concat

    Concat --> FusionHead["Fusion & Classification Head"]:::fusionNode

    subgraph Classification_Head ["Classification Head"]
        direction TB
        FusionHead --> FusedFeat["Fused Features<br>(f_fused, Dim: 2816)"]:::fusedFeatNode
        FusedFeat --> Dense1["Linear (2816 -> 1024)<br>ReLU, Dropout(0.6)"]:::denseNode
        Dense1 --> Dense2["Linear (1024 -> 512)<br>ReLU, Dropout(0.6)"]:::denseNode
        Dense2 --> Dense3["Linear (512 -> 29)"]:::denseNode
        Dense3 --> Output["29 Output Classes<br>(A-Z, Nothing, Space)"]:::outputNode
    end
