from fileinput import filename
import os
import requests
import zipfile
from tqdm import tqdm
import numpy as np
from pathlib import Path
class Downloader():
    def __init__(self, dataset_url, npz_path, extract_dir):
        self.dataset_url = dataset_url
        self.npz_path = npz_path    
        self.extract_dir = extract_dir
        self.download_dataset()

    def download_dataset(self):
        print (f"Start Downloading dataset from {self.dataset_url} ....")
        file_path = Path(self.extract_dir) / self.npz_path
        if(not os.path.isfile(file_path)):
            response = requests.get(self.dataset_url, stream=True)
            size = int(response.headers.get('content-length', 0))
            with open(file_path, 'wb') as file, tqdm(
                desc=self.npz_path,
                total=  size,
                unit='iB',
                unit_scale=True,
                unit_divisor=1024,
            ) as bar:
                for data in response.iter_content(chunk_size=1024):
                    size = file.write(data)
                    bar.update(size)

            # print (f"Extraction of dataset to {self.extract_dir} ....")
            # with zipfile.ZipFile(self.zip_path, 'r') as zip_ref:
            #     zip_ref.extractall(self.extract_dir)
            # os.remove(self.zip_path)
        else:
            print(f"Dataset already exists in {self.extract_dir}, skipping download.")

    def get_data(self):
        file_path = Path(self.extract_dir) / self.npz_path
        with np.load(file_path, allow_pickle=True) as data:
            images = data["X"]
            labels = data["y"]
            classes = list(data["class_names"])
        return images, labels, classes
            



