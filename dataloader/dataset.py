import os
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np
from PIL import Image, ImageFile
import random
import pickle

# ImageFile.LOAD_TRUNCATED_IMAGES = True

class ClrDataset(Dataset):
    """Contrastive Learning Representations Dataset."""
    print("Contrastive Learning Representations Dataset - ClrDataset")
    def __init__(self, 
                csv_file, 
                img_root_dir,
                img_root_dir_test,
                input_shape, 
                img_path_col, 
                text_col, 
                text_from_files, 
                text_root_dir,
                mode,
                transform=None):
        """
        Args:
            csv_file (string): Path to the csv file with annotations.
            img_root_dir (string): Directory with all the images.
            input_shape: shape of input image
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.clr_frame = pd.read_csv(csv_file, nrows=10000)   #read only first 2000 rows
        # self.clr_frame = pd.read_csv(csv_file)   #read only first 2000 rows
        self.img_root_dir = img_root_dir
        self.img_root_dir_test = img_root_dir_test
        self.transform = transform
        self.input_shape = input_shape
        self.img_path_col = int(img_path_col)
        self.text_col = int(text_col)
        self.text_from_files = text_from_files
        self.mode = mode
        self.text_root_dir = text_root_dir

    def __len__(self):
        return len(self.clr_frame)

    def __getitem__(self, idx):

        if torch.is_tensor(idx):
            idx = idx.tolist()

        if self.mode == 'train':

            img_name = os.path.join(self.img_root_dir, self.clr_frame.iloc[idx, self.img_path_col])
            image = Image.open(img_name)

            if self.input_shape[2] == 3:
                image = image.convert('RGB')

            #chooosig a phrase
            if not self.text_from_files:
                # print('image: ', img_name)
                text = self.clr_frame.iloc[idx, self.text_col]
                text = text.replace("\n", "")
                ls_text = text.split(".")
                if '' in ls_text:
                    ls_text.remove('')
                phrase = random.choice(ls_text)

            else:
                text_path = os.path.join(self.text_root_dir,
                                         self.clr_frame.iloc[idx, self.text_col]
                                        )
                with open(text_path) as f:
                    content = f.readlines()
                content = content.replace("\n", "")
                ls_text = content.split(".")
                if '' in ls_text:
                    ls_text.remove('')
                phrase = random.choice(ls_text)

            sample = {'image': image, 'phrase': phrase}
        else :

            img_name = os.path.join(self.img_root_dir_test, self.clr_frame.iloc[idx, self.img_path_col])
            image = Image.open(img_name)

            if self.input_shape[2] == 3:
                image = image.convert('RGB')

            sample = {'image': image, 'img_id': img_name}

        if self.transform:
            sample = self.transform(sample)

        # print('image: ', img_name, 'phrase: ', phrase)
        # print('get sample')

        return sample