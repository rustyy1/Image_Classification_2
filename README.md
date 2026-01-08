# Image Classification Project - Cat Breed Classifier

This project is an image classification application designed to identify different breeds of cats. It utilizes a deep learning model based on **MobileNetV2** tailored for efficiency and accuracy. The project includes scripts for training the model and a web-based interface for easy interaction.

## 📂 Project Structure

The project files are organized within the `code` directory:

```
Image_Classification_2/
├── code/
│   ├── data/
│   │   ├── train/          # Training images organized by breed
│   │   └── val/            # Validation images
│   ├── models/             # Saved trained models
│   ├── app.py              # Gradio web application for inference
│   ├── train.py            # Script to train the model
│   ├── gputest.py          # Utility to check GPU availability
│   ├── organize_dataset.py # Script for dataset organization
│   └── requirements.txt    # Python dependencies
└── README.md
```

## 🚀 Getting Started

### Prerequisites

*   Python 3.10 or higher
*   Git

### Installation

1.  **Clone the repository:**
    ```bash
    git clone https://github.com/rustyy1/Image_Classification_2.git
    cd Image_Classification_2
    ```

2.  **Install Dependencies:**
    It is recommended to use a virtual environment.
    ```bash
    pip install -r code/requirements.txt
    ```
    *Note: The project requires `tensorflow`, `gradio`, `numpy`, and `pillow`.*

## 🛠️ Usage

### 1. Training the Model

To train the model from scratch using the provided dataset:

```bash
python code/train.py
```

*   **Model Architecture:** MobileNetV2 (Pre-trained on ImageNet) with custom top layers.
*   **Parameters:**
    *   Image Size: 224x224
    *   Batch Size: 16
    *   Epochs: 15
    *   Classes: 35
*   **Output:** The best model is saved as `models/cat_breed_model.keras` and the final model as `models/cat_breed_model_final.keras`.

### 2. Running the Application

To launch the web interface for classifying images:

```bash
python code/app.py
```

This will start a local Gradio server (usually at `http://127.0.0.1:7860`). You can upload an image of a cat, and the model will predict its breed along with the top 5 confidence scores.

### 3. GPU Verification

To verify if TensorFlow can detect your GPU:

```bash
python code/gputest.py
```

## 📊 Dataset

The dataset is expected to be in the `code/data` directory, split into `train` and `val` folders. Each subfolder corresponds to a specific cat breed.

*   **Classes Supported:** Abyssinian, Bengal, Birman, Bombay, British Shorthair, Egyptian Mau, Maine Coon, Persian, Ragdoll, Russian Blue, Siamese, Sphynx, and more.

## 📝 Notes

*   **Hardcoded Paths:** Please ensure your project is cloned to a path that matches the script configurations or update the `DATA_DIR` and model paths in `train.py` and `app.py` if you encounter "File not found" errors on a different machine.
