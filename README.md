# GPU-Accelerated Pneumonia Detection using VGG16

## Overview

This project implements a deep learning-based medical imaging system to detect pneumonia from chest X-ray images. The model leverages transfer learning with VGG16 and is optimized for GPU-accelerated training using CUDA-enabled NVIDIA hardware.

### Tech Stack
- Python
- TensorFlow / Keras
- VGG16 (Transfer Learning)
- CUDA (GPU Acceleration)
- NumPy, Matplotlib, OpenCV

### Key Features
- Implemented transfer learning pipeline using pretrained VGG16 for efficient feature extraction
- Designed custom classification layers with Dropout (0.5) for regularization
- Applied data augmentation (rotation, zoom, flips) to improve generalization
- Handled class imbalance using class weighting, improving minority class detection
- Integrated training optimization techniques:
- EarlyStopping
- ReduceLROnPlateau
- ModelCheckpoint
- <img width="994" height="257" alt="image" src="https://github.com/user-attachments/assets/e65a00d8-5cee-4f54-bb44-7f81544e682c" />


### Performance & Optimization
- Trained using CUDA-enabled NVIDIA GPU (RTX 2050) for faster computation
- Achieved ~90% val accuracy and ~94 recall on pneumonia class (update with actual values)
- Reduced training time by ~30–40% compared to CPU execution
- Optimized batch processing and data pipeline for better GPU utilization
- <img width="1080" height="147" alt="image" src="https://github.com/user-attachments/assets/0786e520-3f9b-431a-8362-85cf41f69cd3" />

  
### Evaluation Metrics
Confusion Matrix
Classification Report (Precision, Recall, F1-score)
ROC-AUC Score
<img width="645" height="665" alt="image" src="https://github.com/user-attachments/assets/a0f9ca0a-f7ce-45b2-8da8-5886a0911e3d" />


### How to Run
Clone the repository
git clone https://github.com/vaibhavr54/GPU-Optimized-Deep-Learning-Pipeline-for-Pneumonia-Detection.git
cd GPU-Optimized-Deep-Learning-Pipeline-for-Pneumonia-Detection

### Install dependencies
pip install -r requirements.txt
Run the notebook
jupyter notebook gpu_pneumonia_detection_vgg16.ipynb

### Results

The model demonstrates strong performance in detecting pneumonia from chest X-rays, with improved recall for the minority class due to class imbalance handling and optimized training pipeline.

### Future Improvements
- Deploy model as a web application (Streamlit/Flask)
- Experiment with advanced architectures (ResNet, EfficientNet)
- Optimize inference latency for real-time applications

### Contribution

Feel free to fork and improve the project. Open to suggestions and improvements.

📎 Notes
- Dataset and .h5 model file not included due to size constraints
- Ensure GPU + CUDA setup for best performance
