import torch
import torchvision.transforms as transforms
import numpy as np
import pandas as pd
def load_image(images, path = None):
    # Raise error if both path and images are provided
    if(path!=None and images!=None):
        raise ValueError("Image batches cannot be passed when path is provided")

    # If path is provided
    elif(path!=None):
        images = cv2.imread(path)
        images = cv2.cvtColor(images, cv2.COLOR_BGR2RGB)
        images = preprocess_image(images).unsqueeze(0)

    # If batches of image is provided
    if(images!=None):
        if(isinstance(images, torch.Tensor) == False):
            raise ValueError("reveal() expects images to be of type tensor with shape (B,C,H,W)")    
    # If None then raise errors
    else:
        raise ValueError("Either path or images need to be provided to reveal visualization.")

    return images

def preprocess_image(images):
        '''
            Description:
            Takes in an images and applies some transformations to it

            Args:
            images -> np.ndarray

        '''
        if(isinstance(images, np.ndarray)):
            transform = transforms.Compose([
                            transforms.ToPILImage(),
                            transforms.Resize((224, 224)),
                            transforms.ToTensor(),
                            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
            ])

            images = transform(images)
            return images

        else:
            raise AttributeError("Preprocessing requires a np.ndarray")