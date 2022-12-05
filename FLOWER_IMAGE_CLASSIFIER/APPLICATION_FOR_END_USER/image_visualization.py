import numpy as np
import torch
from get_input_args import get_input_args
from image_process import process_image
import os
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
image_dir = get_input_args().dir4
images = os.listdir(image_dir)
image = random.choice(images)
image_path = f'{image_dir}/{image}'
image = process_image(image_path)

def imshow(image, ax=None):
    """Imshow for Tensor."""
    if ax is None:
        fig, ax = plt.subplots()
    
    # PyTorch tensors assume the color channel is the first dimension
    # but matplotlib assumes is the third dimension
    image = image.numpy().transpose((2, 1, 0))
    
    # Undo preprocessing
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    image = std * image + mean
    
    # Image needs to be clipped between 0 and 1 or it looks like noise when displayed
    image = np.clip(image, 0, 1)
    
    ax.imshow(image)
    ax.show()
    

imshow(image,ax=plt)