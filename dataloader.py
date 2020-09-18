from __future__ import print_function, division

from torch.utils.data import Dataset
import numpy as np
from PIL import Image

class MyDataset(Dataset):
    def __init__(self, label_path, transform = None, target_transform = None):
        super(MyDataset, self).__init__()
        lines = open(label_path, 'r')
        imgs = []
        for line in lines:
            line = line.rstrip()
            labels = line.split()

            image_path = labels[0]
            label = labels[1]
            label = float(label)
            #label = np.asarray(label, dtype=np.float32)

            imgs.append((image_path, label))

        self.imgs = imgs
        self.transform = transform
        self.target_transform = target_transform

    def __getitem__(self, index):
        image, label = self.imgs[index]
        img = Image.open(image).convert('RGB')
        if self.transform is not None:
            img = self.transform(img)

        return img, label

    def __len__(self):
        return len(self.imgs)
