import torch
from tqdm import tqdm
import torch.nn
import os
from torch.utils.data import Dataset, DataLoader
import numpy as np
import config
import rasterio as rio

class MyImageFolder(Dataset):
    def __init__(self, root_dir):
        super(MyImageFolder, self).__init__()
        self.data = []
        self.root_dir = root_dir
        self.class_names = os.listdir(root_dir)

        for index, name in enumerate(self.class_names):
            self.data.append((name, index))

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        img_file, label = self.data[index]

        img_path = os.path.join(self.root_dir, img_file)

        original = rio.open(img_path)
        image = np.array(original.read())
        image = np.transpose(image, (1, 2, 0))

        both_transform = config.both_transforms(image=image)["image"]
        low_res = config.lowres_transform(image=both_transform)["image"]
        high_res = config.highres_transform(image=both_transform)["image"]
        return low_res, high_res

def test():
    dataset = MyImageFolder(root_dir="data")
    loader = DataLoader(dataset, batch_size=8)

    for low_res, high_res in loader:
        print(low_res.shape)
        print(high_res.shape)

if __name__ == "__main__":
    test()
