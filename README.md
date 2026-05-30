# DeepNumberVision
```text
      ___           ___           ___     
     /\  \         /\__\         /\__\    
    /::\  \       /::|  |       /:/  /    
   /:/\:\  \     /:|:|  |      /:/  /     
  /:/  \:\__\   /:/|:|  |__   /:/__/  ___ 
 /:/__/ \:|__| /:/ |:| /\__\  |:|  | /\__\
 \:\  \ /:/  / \/__|:|/:/  /  |:|  |/:/  /
  \:\  /:/  /      |:/:/  /   |:|__/:/  / 
   \:\/:/  /       |::/  /     \::::/__/  
    \::/__/        /:/  /       ~~~~      
     ~~            \/__/                  
```
Deep learning computer vision pipeline built with PyTorch and OpenCV that recognizes multi digit strings drawn or uploaded by the user. 

This network achieves 99.2%+ accuracy on the MNIST handwriting benchmark.

## Features

**Multi Digit Support:** Uses OpenCV to isolate individual digits from a single image string.
**Deep CNN:** Custom PyTorch dual block Convolutional Neural Network using hierarchical geometric feature maps.
**Smart Preprocessing:** Contrast inversion handling and Gaussian stroke smoothing to mimic ink artifacts.
**Web Interface Layout:** A clean drag and drop web portal with Gradio.

## Pipeline

1. **Input:** User uploads a flat image containing an arbitrary amount of numerical characters.
2. **Computer Vision:**
   - Grayscale translation and background normalization.
   - Binary threshold filtering to isolate text.
   - Bounding box sorted horizontally from left to right.
3. **Deep Learning:**
   - Bounding crops are padded and downscaled to a standard 28x28 matrix.
   - Features pass through Conv2D , BatchNorm , ReLU , MaxPool , Dropout.
4. **Output:** Inference predictions are stitched together back into a unified string.

---

## Installation

Please ensure you have Python 3.10+ installed.

1. **Clone the repo:**
   ```bash
   git clone [https://github.com/sinidiem/DeepNumberVision.git](https://github.com/sinidiem/DeepNumberVision.git)
   cd DeepNumberVision
2. **Configure virtual environment:**
    ```bash
    python -m venv .venv
    source .venv/bin/activate
3. **Dependencies:**
    ```bash
    pip install -r requirements.txt

---

## Usage

1. **Train:**
    Runs training sequence using AdamW regularization and real time image augmentation.
    ```bash
    python cnn.py
2. **Verify:**
    Tests saved weights against MNIST dataset.
    ```bash
    python evaluate.py
3. **Launch:**
    Launch the local hosting server interface. Local network URL (http://127.0.0.1:7860).
    ```bash
    python app.py