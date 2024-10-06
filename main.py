import numpy as np
import os
import cv2
from PIL import Image
from matplotlib import pyplot as plt

from config.config import Config
from data.dataLoader import dataLoader

if __name__ == '__main__':
    dl = dataLoader(Config.DATASET_PATH)
    images, captions = dl.get_all()
    print('Number of images:', len(images))
    print('Number of captions:', len(captions))