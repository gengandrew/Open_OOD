from torch.utils.data import Dataset
from PIL import Image
import io
import os
import random
import numpy as np
import pickle


def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img


class Places(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(Places, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()

        for line in lines:
            image = line.strip()
            self.images.append(image)
            self.cls_idx.append(0)
            self.classes.add(0)
        self.num = len(self.images)
        # self.classes = len(self.cls_set)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, self.cls_idx[idx]
