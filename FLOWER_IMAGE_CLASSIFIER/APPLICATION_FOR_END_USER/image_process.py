import numpy as np
from PIL import Image
import os
import random
from get_input_args import get_input_args
import torch
image_dir = get_input_args().flower_input
images = os.listdir(image_dir)
image = random.choice(images)
image_path = f'{image_dir}/{image}'

def process_image(image_path):
    
    ''' Scales, crops, and normalizes a PIL image for a PyTorch model,
        returns an Numpy array
    '''
    
    # TODO: Process a PIL image for use in a PyTorch model
    im = Image.open(image_path,'r')
    im = im.resize((256,290))
    pil_image = im.crop((32,66,256,290))
    np_image = np.array(pil_image)/np.max(np.array(pil_image)[:,:,:])
    
    mean = np.array([0.485, 0.456, 0.406])
    std = np.array([0.229, 0.224, 0.225])
    
    norm_image = (np_image - mean) / std
    trans_image = norm_image.transpose(2,1,0)
    tensor_image = torch.from_numpy(trans_image)

    
    
    return tensor_image
    
    
    
process_image(image_path)