import os
import time
from matplotlib import image
import numpy as np
import cv2
import glob

IMAGE_DIR = 'data/test'

def read_images():
    images = []
    for file in os.listdir(IMAGE_DIR):
        img = cv2.imread(os.path.join(IMAGE_DIR, file))
        if img is not None:
            images.append(os.path.join(IMAGE_DIR, file))

    return images

def main():
    images = read_images()
    print(images)

if __name__ == "__main__":
    main()
