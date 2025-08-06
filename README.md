# Alzheimer's Disease Classification using Deep Learning and Neuroimaging

## Project Overview

This graduation project focuses on the classification of Alzheimer's Disease (AD) using advanced deep learning techniques applied to neuroimaging data. The project implements multiple state-of-the-art architectures to distinguish between Cognitive Normal (CN), Mild Cognitive Impairment (MCI), and Alzheimer's Disease (AD) subjects using MRI brain scans.

## Project Structure

```
├── PreProcessing/                          # Data preprocessing and preparation
│   ├── Isolation_of_Amyloid_Plaques_and_Neurofibrillary_Tangles.ipynb
│   └── Onisleme_fastsurfer.ipynb          # FastSurfer segmentation pipeline
├── BasicCNNmodel/                         # Basic CNN implementation
│   ├── basiccnnham.ipynb                  # Basic CNN with preprocessing
│   └── basiccnnpre.ipynb                  # Basic CNN without preprocessing
├── DenseNet/                              # DenseNet architecture implementations
│   ├── aparc.ipynb                        # DenseNet on aparc segmentation
│   ├── ham.ipynb                          # DenseNet on raw data
│   ├── k-fold.ipynb                       # K-fold cross-validation
│   ├── orig.ipynb                         # DenseNet on original images
│   └── skull.ipynb                        # DenseNet on skull-stripped images
├── Inception/                             # Inception architecture implementations
│   ├── inception2foldham.ipynb            # 2-fold Inception validation
│   ├── inceptionskulls.ipynb             # Inception on skull-stripped images
│   └── preinception2folds.ipynb          # Inception with preprocessing
├── ResNet3D/                              # 3D ResNet implementation
│   └── resnetunpre.ipynb                  # ResNet3D on unprocessed data
├── r2plus1D/                              # R(2+1)D implementation
│   └── r2plus1dkfold.ipynb               # R(2+1)D with k-fold validation
├── MachineLearningAlgorithms/             # Traditional ML on deep features
│   ├── mldense.ipynb                      # ML on DenseNet features
│   ├── mlinception.ipynb                  # ML on Inception features
│   └── resnetml.ipynb                     # ML on ResNet features
├── Poster/                                # Project presentation
│   └── Poster_20011909_20011008.pptx
└── Rapor/                                 # Project report
    └── Rapor_20011909_20011008.pdf
```

## Dataset

The project utilizes the **ADNI (Alzheimer's Disease Neuroimaging Initiative)** dataset, containing MRI brain scans from three categories:
- **CN (Cognitive Normal)**: Healthy control subjects
- **MCI (Mild Cognitive Impairment)**: Subjects with mild cognitive decline
- **AD (Alzheimer's Disease)**: Subjects diagnosed with Alzheimer's disease

### Data Preprocessing

1. **FastSurfer Segmentation**: Automated brain segmentation using the FastSurfer pipeline
2. **ROI Extraction**: Isolation of specific brain regions associated with Alzheimer's pathology:
   - **Amyloid Plaques (Aβ)**: Precuneus, Posterior Cingulate, Frontal Cortex
   - **Neurofibrillary Tangles (NFT)**: Entorhinal, CA1, Amygdala
3. **Normalization**: Robust intensity normalization using percentile-based scaling
4. **Augmentation**: Spatial transformations and intensity variations for training

## Deep Learning Architectures

### 1. Basic CNN
- Custom 3D convolutional neural network
- Multiple convolutional blocks with batch normalization
- Dropout regularization for overfitting prevention
- Cross-validation with dynamic hyperparameter adjustment

### 2. DenseNet3D
- 3D adaptation of DenseNet architecture
- Dense connectivity between layers for feature reuse
- Growth rate optimization for memory efficiency
- Multiple variants tested on different preprocessing approaches

### 3. Inception3D
- 3D version of Inception (GoogLeNet) architecture
- Multi-scale feature extraction using parallel convolutions
- Inception modules with 1x1x1, 3x3x3, and 5x5x5 kernels
- Efficient computation through dimensionality reduction

### 4. ResNet3D
- 3D ResNet with skip connections
- Residual learning to address vanishing gradient problem
- Pre-trained weights adaptation for medical imaging
- Deep architecture (18+ layers) for complex pattern recognition

### 5. R(2+1)D
- Factorized 3D convolutions for video understanding
- Separates spatial and temporal convolutions
- Reduced computational complexity
- Effective for sequential brain scan analysis

## Machine Learning Integration

The project implements a hybrid approach combining deep learning feature extraction with traditional machine learning classifiers:

- **Feature Extraction**: Pre-trained deep networks extract high-level features
- **Classical ML**: Random Forest, SVM, Gradient Boosting, KNN, Logistic Regression, Naive Bayes, Decision Trees, Neural Networks
- **Hyperparameter Optimization**: Grid search with cross-validation
- **Ensemble Methods**: Combination of multiple classifiers for improved performance

## Key Features

### Advanced Preprocessing
- **FastSurfer Integration**: Automated brain segmentation and parcellation
- **ROI-based Analysis**: Focus on Alzheimer's-specific brain regions
- **Multi-modal Processing**: Support for different imaging modalities

### Robust Training
- **K-fold Cross-validation**: Ensures model generalization
- **Class Balancing**: Weighted sampling for imbalanced datasets
- **Early Stopping**: Prevents overfitting with patience mechanism
- **Learning Rate Scheduling**: Adaptive learning rate adjustment

### Comprehensive Evaluation
- **Multiple Metrics**: Accuracy, Precision, Recall, F1-score
- **Confusion Matrix Analysis**: Detailed classification performance
- **Statistical Significance**: Cross-validation with confidence intervals
- **Comparative Analysis**: Performance comparison across architectures

## Technical Implementation

### Framework and Libraries
- **PyTorch**: Deep learning framework
- **nibabel**: NIfTI image processing
- **scikit-learn**: Machine learning algorithms
- **FastSurfer**: Brain segmentation pipeline
- **matplotlib/seaborn**: Visualization
- **tqdm**: Progress tracking

### Hardware Requirements
- **GPU**: CUDA-compatible for deep learning training
- **Memory**: Minimum 16GB RAM for 3D volume processing
- **Storage**: Sufficient space for ADNI dataset (>100GB)

### Data Pipeline
```python
# Example data loading and preprocessing
class MRIDataset(Dataset):
    def __init__(self, root_dir, target_shape=(32, 112, 112)):
        # Load NIfTI files
        # Apply normalization
        # Resize to target dimensions
        
    def __getitem__(self, idx):
        # Load volume
        # Apply transformations
        # Return tensor and label
```

## Results and Performance

The project evaluates multiple architectures across different preprocessing strategies:

1. **Raw Data Performance**: Direct classification on original MRI volumes
2. **Preprocessed Data**: Classification after FastSurfer segmentation
3. **ROI-specific Analysis**: Performance on isolated brain regions
4. **Cross-validation Results**: Statistical significance of findings

### Evaluation Metrics
- **Accuracy**: Overall classification correctness
- **Sensitivity**: True positive rate for each class
- **Specificity**: True negative rate for each class
- **F1-Score**: Harmonic mean of precision and recall
- **AUC**: Area under ROC curve for multi-class classification

## Usage Instructions

### 1. Environment Setup
```bash
pip install torch torchvision nibabel scikit-learn scipy matplotlib seaborn tqdm
```

### 2. Data Preparation
```python
# Run FastSurfer preprocessing
# Extract ROIs using provided scripts
# Organize data into CN/MCI/AD folders
```

### 3. Model Training
```python
# Choose architecture (DenseNet, ResNet3D, etc.)
# Configure hyperparameters
# Run training with cross-validation
```

### 4. Evaluation
```python
# Load trained models
# Run inference on test set
# Generate performance reports
```

## Research Contributions

1. **Comprehensive Comparison**: Systematic evaluation of multiple deep learning architectures for AD classification
2. **ROI-based Analysis**: Novel approach focusing on pathology-specific brain regions
3. **Hybrid Methodology**: Integration of deep learning feature extraction with traditional ML classifiers
4. **Preprocessing Pipeline**: Standardized preprocessing workflow using FastSurfer
5. **Cross-validation Framework**: Robust evaluation methodology ensuring statistical significance

## Future Work

- **Longitudinal Analysis**: Temporal progression modeling
- **Multi-modal Fusion**: Integration of different imaging modalities
- **Attention Mechanisms**: Focus on diagnostically relevant regions
- **Federated Learning**: Collaborative training across institutions
- **Clinical Validation**: Real-world deployment and validation


## References

- ADNI Dataset: http://adni.loni.usc.edu/
- FastSurfer: https://github.com/Deep-MI/FastSurfer
- PyTorch: https://pytorch.org/

## License

This project is for academic purposes. The ADNI dataset usage follows the ADNI Data Use Agreement. Please cite appropriately if using this work for research purposes.


---
