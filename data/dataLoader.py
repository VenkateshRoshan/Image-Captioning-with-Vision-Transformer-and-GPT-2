import numpy as np
import os
import cv2
from PIL import Image
from torchvision import transforms
import pandas as pd

class dataLoader:
    def __init__(self, path):
        self.path = path
        self.img_path = path + 'images/'
        self.caption_path = path + 'captions.csv'
        self.img_list = os.listdir(self.img_path)
        self.caption_dict = self.get_caption_dict()
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor()
        ])

    def get_caption_dict(self):
        caption_dict = {}
        df = pd.read_csv(self.caption_path, delimiter=',')
        for i in range(len(df)):
            img_name = df.iloc[i, 0]
            caption = df.iloc[i, 1]
            caption_dict[img_name] = caption
        return caption_dict
    
    def get_image(self, img_name):
        img = Image.open(self.img_path + img_name)
        img = self.transform(img)
        return img
    
    def get_caption(self, img_name):
        return self.caption_dict[img_name]
    
    def get_batch(self, batch_size):
        batch = np.random.choice(self.img_list, batch_size)
        images = []
        captions = []
        for img_name in batch:
            images.append(self.get_image(img_name))
            captions.append(self.get_caption(img_name))
        return images, captions
    
    def get_all(self):
        images = []
        captions = []
        for img_name in self.img_list:
            images.append(self.get_image(img_name))
            captions.append(self.get_caption(img_name))
        return images, captions