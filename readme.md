# CNN Animal Classification

A modular deep learning project built with **PyTorch** to construct, train, and evaluate a Convolutional Neural Network (CNN) pipeline for animal image classification.

## 📁 Repository Structure

```text
├── cv_pipeline/              # Core computer vision engineering modules
│   ├── __init__.py           # Makes cv_pipeline a Python package
│   ├── dataset.py            # Extracts .npz arrays & creates PyTorch Custom Dataset instances
│   ├── dataloader.py         # Configures data batching, shuffling, and worker threads
│   ├── transforms.py         # Handles data augmentation (resizing, normalization, flips)
│   ├── models.py             # Defines the modular CNN model architectures
│   └── engine.py             # Contains train_step, test_step, and loop orchestrations
├── animal_mnist.npz          # Compressed NumPy array containing the animal image data
├── helpers.py                # Utility functions for plotting metrics and handling logs
├── main.py                   # Main orchestration script to run execution workflows
├── requirements.txt          # Python project package dependencies
└── .gitignore                # Git tracked exceptions
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

### 3. Setup Environment & Install Dependencies
It is highly recommended to use an isolated Python virtual environment:

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

### 4. Run the Classification Pipeline
To unpack the dataset, initialize the data loaders, build the model network, and start training, run:

```bash
python main.py
```

## 🛠️ Modular Pipeline Architecture (`cv_pipeline/`)

To keep the deep learning code scalable and maintainable, the components inside the `cv_pipeline/` directory are completely decoupled:

### 🔹 Data Ingestion (`dataset.py` & `dataloader.py`)
* Parses raw data arrays from `animal_mnist.npz`.
* Implements a custom PyTorch class subclassing `torch.utils.data.Dataset` to fetch items lazily.
* Wraps datasets inside `torch.utils.data.DataLoader` to handle automatic mini-batch processing, multi-threaded worker loading, and dataset shuffling for training cycles.

### 🔹 Augmentation & Preprocessing (`transforms.py`)
* Applies essential validation mapping like Tensor casting (`ToTensor()`) and RGB channel scaling normalization.
* Houses configurable image augmentation pipelines (e.g., random cropping, horizontal flipping, or color jittering) to improve model generalization and reduce overfitting.

### 🔹 Architecture Design (`models.py`)
* Builds clean, flexible convolutional neural networks subclassing `torch.nn.Module`.
* Separates individual architectural blocks (Convolutional feature extractors, non-linear activation layers like `ReLU`, downsampling `MaxPool2d` layers, and Fully-Connected dense classifier heads).

### 🔹 Execution Engine (`engine.py`)
* Abstracts the core execution phases of machine learning tasks.
* Implements the forward pass, loss calculation (e.g., CrossEntropyLoss), backpropagation tracking, and weight optimization steps using optimizers like Adam or SGD.

## 📜 License
This project is open-source and available under the MIT License.
