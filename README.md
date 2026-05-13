AI-Generated Image Detection via Spectral Analysis

This project investigates whether frequency-domain features can help distinguish real images from AI-generated images. We use the CIFAKE dataset and compare pixel-based baselines, FFT/DCT spectral feature models, and CNN-based models with spectral feature fusion.

The main goal is to understand whether hidden frequency artifacts can provide useful information for AI-generated image detection, especially when images are degraded by common real-world transformations such as JPEG compression, noise, blur, and downscaling.


Project Overview

AI-generated images are becoming harder to identify through visual inspection alone. Real and synthetic images can look very similar at the pixel level, especially in small image settings.

Instead of only analysing raw image pixels, this project studies the image in the frequency domain using:

- Fast Fourier Transform (FFT)
- Discrete Cosine Transform (DCT)

The overall pipeline is:

Input Image → FFT/DCT Transformation → Spectral Features → Classifier → Real/Fake Prediction  Dataset

We use the CIFAKE dataset, which contains real and AI-generated images. The task is binary classification:
* Real image
* AI-generated image
All images are resized to:
32 × 32 × 3  The processed dataset is expected to be stored as:  data/cifake_processed.npz  The .npz file should contain the following arrays:
X_train, y_train
X_val, y_val
X_test, y_test  The dataset file is not included in this repository because processed dataset files can be large. It should be placed locally before running the notebook.
  Repository Structure

ai-image-detection-spectral-analysis/
│
├── README.md
├── requirements.txt
├── .gitignore
│
├── notebooks/
│   └── 682Project_v5_best.ipynb
│
├── report/
│   └── Project_Title.pdf
│
└── results/  Methods

This project compares both image-space and frequency-space approaches.

1. Pixel Baseline
Raw image pixels are flattened into one-dimensional vectors and used as input features for a Logistic Regression baseline.
For a 32×32 RGB image, the flattened pixel vector has: 32 × 32 × 3 = 3072 features

2. FFT Features
FFT is used to convert images from the spatial domain to the frequency domain.
The image is converted to grayscale, a 2D FFT is applied, and the log-magnitude spectrum is computed. Frequency energy is then summarised using radial frequency bins.
FFT features help capture global frequency patterns and possible artifacts in AI-generated images.

3. DCT Features
DCT is used to extract compact frequency coefficients from images.
DCT features are useful because they are related to compression and image structure. They can capture information that may not be obvious from raw pixel values alone.

4. Combined FFT + DCT Features
FFT and DCT features are concatenated to form a combined spectral feature vector.
This combined representation is used with lightweight classifiers and CNN-based spectral fusion models.
 Models Compared
The following models are compared:
* Pixel Logistic Regression baseline
* Logistic Regression with FFT features
* Logistic Regression with FFT + DCT features
* MLP with FFT + DCT features
* Vanilla CNN
* CNN + FFT
* CNN + FFT + DCT spectral fusion model
 CNN Architecture
The Vanilla CNN learns visual/spatial features directly from the input image.
 The general flow is:

Input Image → Convolution Blocks → Pooling → Dense Layer → Real/Fake Output  The CNN captures visual image patterns such as edges, textures, and spatial structures.
  CNN + FFT + DCT Spectral Fusion

The spectral fusion model uses two parallel branches:
Image Branch
The CNN branch learns visual features from the original image.
Spectral Branch
The spectral branch processes FFT/DCT frequency features using an MLP-style feature branch.
The two feature representations are then concatenated and passed through a final classifier.
The general flow is:
Input Image
   ├── CNN Visual Branch → Visual Features
   └── FFT/DCT Feature Branch → Spectral Features

Visual Features + Spectral Features → Concatenation → Classifier → Real/Fake Output 
This allows the model to use both image-space and frequency-space information.
  Robustness Testing
The models are evaluated on both clean images and degraded images.
The degradation conditions include:
* JPEG compression
* Gaussian noise
* Blur
* Downscaling
These transformations simulate real-world image changes that commonly happen when images are uploaded, shared, resized, or compressed.
Robustness testing helps evaluate whether spectral features remain useful beyond clean test data.  Evaluation Metrics
The following metrics are used:
* Accuracy
* Precision
* Recall
* F1-score
* ROC-AUC
Accuracy measures overall correctness.
Precision measures how many predicted positives are actually correct.
Recall measures how many actual positives are correctly detected.
F1-score balances precision and recall.
ROC-AUC measures how well the model separates real and fake images across classification thresholds.


How to Run
1. Install dependencies
Install the required Python packages:

pip install -r requirements.txt  2. Add the processed dataset
Place the processed dataset file in:
 data/cifake_processed.npz  3. Run the notebook
Open the notebook:
notebooks/682Project_v5_best.ipynb


Run all cells in order.
The notebook performs:
* Dataset loading
* FFT/DCT feature extraction
* Baseline model training
* CNN model training
* Robustness evaluation
* Result table generation
* Plot and heatmap generation
  Results Summary
CNN-based models perform strongly on clean test data.
FFT and DCT spectral features improve the performance of lightweight classifiers compared to using raw pixel features alone.
The CNN + FFT + DCT spectral fusion model is useful for robustness testing because it combines visual information from the CNN branch with frequency-domain information from the spectral branch.
Overall, the results suggest that spectral features can provide complementary information for AI-generated image detection, especially when images undergo common degradations.

Key Takeaway
The main takeaway is that frequency-domain features are useful complementary signals for AI-generated image detection.
CNN models remain strong for clean image classification, while FFT/DCT-based spectral features help analyze robustness under degraded image conditions.
 Future Improvements
Possible future improvements include:
* Training for more epochs
* Tuning learning rate, batch size, and dropout
* Testing stronger CNN backbones
* Improving spectral fusion architecture
* Testing on larger and higher-resolution datasets
* Evaluating on images generated by newer AI models
* Applying degradation-aware training
  Authors
Pradnya R Apte Sharath Srivatsan Manivannan
University of Massachusetts Amherst Manning College of Information and Computer Sciences



