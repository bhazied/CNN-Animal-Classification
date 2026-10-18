# CNN Animal Classification

A deep learning project using **PyTorch** to build a Convolutional Neural Network (CNN) pipeline designed to classify animals. This repository implements data loading, custom preprocessing, modular architecture design, and a complete training/testing pipeline.

## 📁 Repository Structure

```text
├── cv_pipeline/            # Custom computer vision pipeline module (Data loaders, Models)
├── animal_mnist.npz        # Compressed NumPy array containing the animal image dataset
├── helpers.py              # Utility functions for visualization, logging, or metric tracking
├── main.py                 # Main orchestration script for training and evaluation
├── requirements.txt        # Project package dependencies
└── .gitignore              # Git ignore file
```

## 🚀 Getting Started

Follow these steps to set up and run the animal classification pipeline locally.

### 1. Prerequisites
Ensure you have Python 3.8+ installed on your system. 

### 2. Clone the Repository
```bash
git clone https://github.com
cd CNN-Animal-Classification
```

### 3. Install Dependencies
It is highly recommended to use a virtual environment:

```bash
# Create a virtual environment
python -m venv venv

# Activate the virtual environment
# On Windows:
venv\Scripts\activate
# On macOS/Linux:
source venv/bin/activate

# Install required packages
pip install -r requirements.txt
```

*Note: Ensure your `requirements.txt` includes core packages like `torch`, `torchvision`, `numpy`, and `matplotlib`.*

### 4. Running the Pipeline
To start processing the dataset, training the CNN model, and running evaluations, execute the main driver script:

```bash
python main.py
```

## 🛠️ Pipeline Details

* **Data Management (`animal_mnist.npz`)**: The dataset is packaged into a compact `.npz` format. The custom data pipeline inside `cv_pipeline/` handles unpacking, parsing, applying PyTorch tensor transforms, and wrapping tensors into manageable batches via `DataLoader`.
* **Modular CNN Architecture**: The networks are modularly separated to easily tweak convolutional layers, activation functions (ReLU), pooling mechanisms, and fully connected linear layers.
* **Orchestration (`main.py` & `helpers.py`)**: Manages the forward passes, loss computation, backpropagation optimization steps, and utilizes helper functions to evaluate validation accuracy or display performance graphs.

## 📜 License
This project is open-source and available under the MIT License.