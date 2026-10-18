import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
from torch.utils.data import DataLoader, random_split
from cv_pipeline import Downloader, ClassifyModel, ClassifyDataSet, ClassifyDataLoader, VisionTransformer
DATASET_URL = "https://zenodo.org/records/20965195/files/animal_mnist.npz"
NPZ_PATH = "animal_mnist.npz"
EXTRACT_DIR = "./data"
BATCH_SIZE = 32
LEARNING_RATE = 0.001
EPOCHS = 3
def run_pipeline():
    downloader = Downloader(DATASET_URL, NPZ_PATH, EXTRACT_DIR)
    images, labels, classes = downloader.get_data()
    print(images.shape)
    print(labels.shape)
    num_class = len(classes)
    model = ClassifyModel(num_class)
    #dataset = ClassifyDataSet(images, labels,classes)
    vision_transformers =  VisionTransformer()

    num_samples = len(images)
    indices = np.arange(num_samples)
    np.random.seed(42) 
    np.random.shuffle(indices)
    train_end = int(0.8 * num_samples) # 80%
    train_idx = indices[:train_end] # the reste
    eval_idx = indices[train_end:]    
    x_train, y_train = images[train_idx], labels[train_idx]
    x_eval, y_eval = images[eval_idx], labels[eval_idx]
    train_dataset = ClassifyDataSet(x_train, y_train, classes, vision_transformers.train_transform)
    print(train_dataset[0])
    eval_dataset = ClassifyDataSet(x_eval, y_eval, classes, vision_transformers.main_transform)
    # data_loader  = ClassifyDataLoader()  
    # train_dataset, val_dataset = random_split(dataset, [total_train, total_eval])
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    eval_dataloader = DataLoader(eval_dataset, batch_size=BATCH_SIZE, shuffle=False)
    loss_function = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE)
    model.fit(train_dataloader, eval_dataloader, loss_function, optimizer, num_epochs=EPOCHS)

if __name__ == "__main__":
    run_pipeline()


