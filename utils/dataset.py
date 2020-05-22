# dataset
from os.path import splitext
from os import listdir
import numpy as np
from glob import glob
import torch
from torch.utils.data import Dataset
import logging
from PIL import Image


class BasicDataset(Dataset):
    def __init__(self, root):
        self.root = root
        data_file = open(root+'Dataset/'+'label_data.csv')
        self.data = data_file.readlines()

    def __len__(self):
        return len(self.data)

    @classmethod
    def preprocessDepth(cls, pil_img):

        img_nd = np.array(pil_img)
        img_nd = 255 - img_nd

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans
    
    def preprocess(cls, pil_img):

        img_nd = np.array(pil_img)

        if len(img_nd.shape) == 2:
            img_nd = np.expand_dims(img_nd, axis=2)

        # HWC to CHW
        img_trans = img_nd.transpose((2, 0, 1))
        if img_trans.max() > 1:
            img_trans = img_trans / 255
        return img_trans

    def __getitem__(self, i):
        idx = self.data[i].split(';')
        bg = Image.open(self.root + idx[0].replace('"',''))
        fgbg = Image.open(self.root + idx[1].replace('"',''))
        mask = Image.open(self.root + idx[2].replace('"',''))
        depth = Image.open(self.root + idx[3].replace('"','').replace('\n',''))
        bg = self.preprocess(bg)
        fgbg = self.preprocess(fgbg)
        mask = self.preprocess(mask)
        depth =self.preprocessDepth(depth)
        return {'bg' : torch.from_numpy(bg), 'fgbg': torch.from_numpy(fgbg), 'mask': torch.from_numpy(mask), 'depth': torch.from_numpy(depth)}
