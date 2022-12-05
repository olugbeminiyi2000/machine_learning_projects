import numpy as np
import torch
from torch import nn
from torch import optim
import torch.nn.functional as F
from torchvision import datasets, transforms, models
from get_input_args import get_input_args
import numpy as np
from PIL import Image
import os
import random
from get_input_args import get_input_args
import torch
from train import train_process
from image_process import process_image
import json
import matplotlib.pyplot as plt

#getting category names
categories = get_input_args().category_name
with open(categories, 'r') as f:
    cat_to_name = json.load(f)
    
#image path
image_dir = get_input_args().flower_input
images = os.listdir(image_dir)
image = random.choice(images)
image_path = f'{image_dir}/{image}'

#get gpu or cpu train_on
train_on = get_input_args().gpu
device = torch.device(train_on if torch.cuda.is_available() else "cpu")


#get the checkpoint
checkpoint_file = get_input_args().checkpoint

def load_checkpoint(filepath):
    checkpoint = torch.load(filepath)
    #this load the checkpoint dictionary that
    #as been saved as a file checkpoint.pth
    
    model = checkpoint['model_name']
    model.load_state_dict(checkpoint['state_dict'])
    
    return model
model = load_checkpoint(checkpoint_file)

#top probabilities
topk = get_input_args().topk


def predict(image_path, model, topk):
    ''' Predict the class (or classes) of an image using a trained deep learning model.
    '''
    
    # TODO: Implement the code to predict the class from an image file
    inputs = process_image(image_path)
    inputs = inputs.view(1,inputs.shape[0],inputs.shape[1],inputs.shape[2])
    inputs = inputs.to(device=device, dtype=torch.float)
    logps = model.forward(inputs)
    ps = torch.exp(logps)
    ps = ps.topk(topk)
    
    return ps

probs, classes = predict(image_path,model,topk)
probas = []
for i in range(probs.shape[1]):
    for j in range(1):
        probas.append(round(float(probs[j,i]) * 100 * np.exp(-2),2))
        
cate = classes.squeeze().tolist()
category = [cat_to_name[str(i)] for i in cate]

print(probas)
print(category)

plt.bar(category, probas)

# Labeling the axes
plt.xlabel('category')
plt.ylabel('probas')
plt.xticks(rotation=90)

# Dsiplay the plot
plt.show()




